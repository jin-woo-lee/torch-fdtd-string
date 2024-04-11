import torch
import numpy as np
import librosa
import soundfile as sf

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
        maxv = np.max(np.abs(wav))
        gain = 1 if maxv == 0 else 1. / maxv
    elif isinstance(wav, torch.Tensor):
        ''' torch '''
        exists_nan = torch.isnan(wav.sum())
        if not skip_nan:
            assert not exists_nan, torch.isnan(wav)
        if exists_nan:
            return wav, 1.
        maxv = wav.abs().max(-1).values.unsqueeze(-1)
        gain = 1 if maxv == 0 else 1. / maxv
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

def state_to_wav(state, normalize=True, sr=48000):
    ''' state: (Bs, Nt, Nx) '''
    assert len(list(state.shape)) == 3, state.shape
    Nt = state.size(1)
    vel = ((state.narrow(1,0,Nt-1) - state.narrow(1,1,Nt-1)) * sr).sum(-1)
    return ell_infty_normalize(vel)[0] if normalize else vel
