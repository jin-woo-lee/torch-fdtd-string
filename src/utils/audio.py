import math
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from einops import rearrange

eps = np.finfo(np.float32).eps

def calculate_rms(amp):
    if isinstance(amp, torch.Tensor):
        return amp.pow(2).mean(-1, keepdim=True).pow(.5)
    elif isinstance(amp, np.ndarray):
        return np.sqrt(np.mean(np.square(amp), axis=-1) + eps)
    else:
        raise TypeError(f"argument 'amp' must be torch.Tensor or np.ndarray. got: {type(amp)}")

def dB2amp(dB):
    return np.power(10., dB/20.)

def amp2dB(amp):
    return 20. * np.log10(amp)

def rms_normalize(wav, ref_dBFS=-23.0, skip_nan=True):
    exists_nan = np.isnan(np.sum(wav))
    if not skip_nan:
        assert not exists_nan, np.isnan(wav)
    if exists_nan:
        return wav, 1.
    # RMS normalize
    # value_dBFS = 20*log10(rms(signal) * sqrt(2)) = 20*log10(rms(signal)) + 3.0103
    rms = calculate_rms(wav)
    if isinstance(ref_dBFS, torch.Tensor):
        ref_linear = torch.pow(10, (ref_dBFS-3.0103)/20.)
    else:
        ref_linear = np.power(10, (ref_dBFS-3.0103)/20.)
    gain = ref_linear / (rms + eps)
    wav = gain * wav
    return wav, gain

def ell_infty_normalize(wav, skip_nan=True):
    if isinstance(wav, np.ndarray):
        ''' numpy '''
        exists_nan = np.isnan(np.sum(wav))
        if not skip_nan:
            assert not exists_nan, np.isnan(wav)
        if exists_nan:
            return wav, 1.
        maxv = np.max(np.abs(wav), axis=-1)
        # 1 if maxv == 0 else 1. / maxv
        if len(list(maxv.shape)) == 0:
            gain = 1 if maxv==0 else 1. / maxv
        else:
            gain = 1. / maxv; gain[maxv==0] = 1
    elif isinstance(wav, torch.Tensor):
        ''' torch '''
        exists_nan = torch.isnan(wav.sum())
        if not skip_nan:
            assert not exists_nan, torch.isnan(wav)
        if exists_nan:
            return wav, 1.
        maxv = wav.abs().max(-1).values.unsqueeze(-1)
        # 1 if maxv == 0 else 1. / maxv
        gain = torch.where(maxv.eq(0),
            torch.ones_like(maxv), 1. / maxv)
    else:
        assert False, wav
    wav = gain * wav
    return wav, gain

def dB_RMS(wav):
    if isinstance(wav, torch.Tensor):
        return 20 * torch.log10(calculate_rms(wav))
    elif isinstance(wav, np.ndarray):
        return 20 * np.log10(calculate_rms(wav))

def mel_basis(sr, n_fft, n_mel):
    return librosa.filters.mel(sr=sr,n_fft=n_fft,n_mels=n_mel,fmin=0,fmax=sr//2,norm=1)

def inv_mel_basis(sr, n_fft, n_mel):
    return librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mel, norm=None, fmin=0, fmax=sr//2,
    ).T

def lin_to_mel(linspec, sr, n_fft, n_mel=80):
    basis = mel_basis(sr, n_fft, n_mel)
    return basis @ linspec

def save_waves(est, save_dir, sr=16000):
    data = []
    batch_size = inp.shape[0]
    for b in range(batch_size):
        est_wav = est[b,0].squeeze()
        wave_path = f"{save_dir}/{b}.wav"
        sf.write(wave_path, est_wav, samplerate=sr)

def get_inverse_window(forward_window, frame_length, frame_step):
    denom = torch.square(forward_window)
    overlaps = -(-frame_length // frame_step)  # Ceiling division.
    denom = F.pad(denom, (0, overlaps * frame_step - frame_length))
    denom = denom.reshape(overlaps, frame_step)
    denom = denom.sum(0, keepdims=True)
    denom = denom.tile(overlaps, 1)
    denom = denom.reshape(overlaps * frame_step)
    return forward_window / denom[:frame_length]

def state_to_wav(state, normalize=True, sr=48000):
    ''' state: (Bs, Nt, Nx) '''
    assert len(list(state.shape)) == 3, state.shape
    Nt = state.size(1)
    vel = ((state.narrow(1,1,Nt-1) - state.narrow(1,0,Nt-1)) * sr).sum(-1)
    return ell_infty_normalize(vel)[0] if normalize else vel

def state_to_spec(x, window):
    ''' x: (Bs, Nt, Nx, Ch)
        -> (Bs, Nt, Nx, Ch*n_fft*2)
    '''
    Bs, Nt, Nx, Ch = x.shape
    n_ffts = window.size(-1)
    n_freq = n_ffts // 2 + 1
    hop_length = n_ffts // 4
    x = rearrange(x, 'b t x c -> (b x c) t')
    s = torch.stft(x, n_ffts, hop_length=hop_length, window=window)
    s = rearrange(s, '(b x c) f t k -> b t x (c f k)',
        b=Bs, x=Nx, c=Ch, f=n_freq, k=2)
    return s

def spec_to_state(x, window, length):
    ''' x: (Bs, Nt, Nx, Ch*n_fft*2)
        -> (Bs, Nt, Nx, Ch)
    '''
    Bs, Nt, Nx, _ = x.shape
    n_ffts = window.size(-1)
    n_freq = n_ffts // 2 + 1

    x = rearrange(x, 'b t x (c f k) -> (b x c) f t k', f=n_freq, k=2)
    x = torch.istft(x, n_ffts, length=length, window=window)
    x = rearrange(x, '(b x c) t -> b t x c', b=Bs, x=Nx)
    return x


def to_spec(x, window, reduce_channel=True):
    ''' x: (Bs, Nt)
        -> (Bs, Nt, Nf*2) if reduce_channel==True
        -> (Bs, Nt, Nf,2) otherwise
    '''
    Bs, Nt = x.shape
    n_ffts = window.size(-1)
    n_freq = n_ffts // 2 + 1
    hop_length = n_ffts // 4
    s = torch.stft(x, n_ffts, hop_length=hop_length, window=window)
    s = s.transpose(1,2)
    if reduce_channel:
        s = rearrange(s, 'b t f k -> b t (f k)',
            b=Bs, f=n_freq, k=2)
    return s

def from_spec(x, window, length):
    ''' x: (Bs, Nt, Nf*2)
        -> (Bs, Nt)
    '''
    Bs, Nt, _ = x.shape
    n_ffts = window.size(-1)
    n_freq = n_ffts // 2 + 1

    x = rearrange(x, 'b t (f k) -> b f t k', f=n_freq, k=2)
    x = torch.istft(x, n_ffts, length=length, window=window)
    return x

def adjust_gain(y, x, minmax, ref_dBFS=-23.0):
    ran_gain = (minmax[1] - minmax[0]) * torch.rand_like(y.narrow(-1,0,1)) + minmax[0]
    ref_linear = np.power(10, (ref_dBFS-3.0103)/20.)
    ran_linear = torch.pow(10, (ran_gain-3.0103)/20.)
    x_rms = calculate_rms(x)
    y_rms = calculate_rms(y)
    x_gain = ref_linear / (x_rms + eps)
    y_gain = ref_linear / (y_rms + eps)

    y_xscale = y * y_gain / x_gain
    return y_xscale / ran_linear

def degrade(x, rir, noise):
    ''' x    : (Bs, Nt)
        rir  : (Bs, Nt)
        noise: (Bs, Nt)
    ''' 
    x_pad = F.pad(x,   (0,rir.size(-1)))
    w_pad = F.pad(rir, (0,rir.size(-1)))
    x_fft = torch.fft.rfft(x_pad)
    w_fft = torch.fft.rfft(w_pad)
    wet_x = torch.fft.irfft(x_fft * w_fft).narrow(-1,0,x.size(-1))

    y = adjust_gain(wet_x, x, [-0, 30]) # ser
    n = adjust_gain(noise, y, [10, 30]) # snr
    return y + n

def T60_to_sigma(T60, f_0, K):
    ''' T60 : (Bs, 2, 2)  [[T60_freq_1, T60_1], [T60_freq_2, T60_2]]
        f_0 : (Bs, Nt, 1) fundamental frequency
        K   : (Bs, Nt, 1) kappa (K == gamma * kappa_rel)
     -> sig : (Bs, Nt, 2)
    '''
    gamma = f_0 * 2
    freq1, time1 = T60.narrow(1,0,1).chunk(2,-1)
    freq2, time2 = T60.narrow(1,1,1).chunk(2,-1)

    zeta1 = - gamma.pow(2) + (gamma.pow(4) + 4 * K.pow(2) * (2 * math.pi * freq1).pow(2)).pow(.5)
    zeta2 = - gamma.pow(2) + (gamma.pow(4) + 4 * K.pow(2) * (2 * math.pi * freq2).pow(2)).pow(.5)
    sig0 = - zeta2 / time1 + zeta1 / time2
    sig0 = 6 * math.log(10) * sig0 / (zeta1 - zeta2)

    sig1 = 1 / time1 - 1 / time2
    sig1 = 6 * math.log(10) * sig1 / (zeta1 - zeta2)

    sig = torch.cat((sig0, sig1), dim=-1)
    return sig


