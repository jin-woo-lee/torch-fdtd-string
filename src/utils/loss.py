import math
import torch
import abc
import torch.nn as nn
import torch.nn.functional as F
from auraloss.freq import MultiResolutionSTFTLoss
from einops import rearrange
from src.utils import misc as ms
from src.utils import audio as audio

def rms(x):
    out_shape = [x.size(0)] + [1] * len(list(x.shape)[1:])
    x = x.flatten(1)
    x_rm = x.pow(2).mean(1, keepdim=True)
    x_rms = torch.where(x_rm.eq(0), torch.ones_like(x_rm), x_rm.sqrt())
    return x_rms.view(out_shape)

def stft(x, n_fft, hop_length):
    w = torch.hann_window(n_fft).to(x.device)
    x = torch.stft(x, n_fft, hop_length, window=w)
    if x.shape[-1] == 2: x = torch.view_as_complex(x)
    return x.transpose(-1,-2) # (batch_size, frames, num_freqs)

def melscale(mag, n_fft, n_mel, sr):
    mel_fbank = audio.mel_basis(sr, n_fft, n_mel)
    mel_basis = torch.from_numpy(mel_fbank).to(mag.device)
    mel = torch.matmul(mel_basis, mag.transpose(-1,-2)).transpose(-1,-2)
    return mel

def stft_loss(x, y, n_fft=1024, n_mel=128, sr=48000, eps=1e-5):
    ''' x: (batch_size, n_samples)
        y: (batch_size, n_samples)
    '''
    n_fft = n_fft if x.size(1) > n_fft else x.size(1)
    hop_length = n_fft // 4
    x_linmag = stft(x, n_fft, hop_length).abs()
    y_linmag = stft(y, n_fft, hop_length).abs()
    x_logmag = 20 * torch.log10(x_linmag + eps)
    y_logmag = 20 * torch.log10(y_linmag + eps)
    x_linmel = melscale(x_linmag, n_fft, n_mel, sr)
    y_linmel = melscale(y_linmag, n_fft, n_mel, sr)
    x_logmel = 20 * torch.log10(x_linmel + eps)
    y_logmel = 20 * torch.log10(y_linmel + eps)
    def l1_dist(x, y):
        return (x - y).abs().flatten(1).mean(1)
    scores = dict(
        linmag=l1_dist(x_linmag, y_linmag),
        logmag=l1_dist(x_logmag, y_logmag),
        linmel=l1_dist(x_linmel, y_linmel),
        logmel=l1_dist(x_logmel, y_logmel),
    )
    return scores

def mse_loss(preds, target):
    return F.mse_loss(preds, target)

def dirichlet_bc(u, dim=-1):
    ''' u: (batch_size, time, space) '''
    u_D = u.roll(1,dim).narrow(dim,0,2) # (b, t, x=0,-1)
    return u_D.abs().mean()

def pde_loss(
        #---------- 
        # string parameters 
        ut, zt, u0, f0, kappa, alpha, sig0, sig1, masks,
        #---------- 
        # excitation parameters
        bow_params, hammer_params,
        #---------- 
        # grids and metrics
        x, t, f_ic, f_bc, f_r, w_ic=1., w_bc=1., w_r=1.,
        #---------- 
    ):
    ''' ut: (batch, time, space) '''
    est_u0 = ut.narrow(1,0,1)
    val_ic = f_ic(est_u0, u0)
    val_bc = f_bc(ut)
    val_r, results = f_r(
        ut, zt, x, t, f0, kappa, alpha, sig0, sig1,
        masks, bow_params, hammer_params)
    return w_ic * val_ic \
         + w_bc * val_bc \
         + w_r  * val_r, results

def si_sdr(reference_signal, estimated_signal, scaling=True, eps=None):
    ''' reference_signal: (batch_size, channels, time)
        estimated_signal: (batch_size, channels, time)
        -> SISDR calculated for the last dim (batch_size, channels)
    '''
    eps = torch.finfo(reference_signal.dtype).eps if eps is None else eps
    batch_size = estimated_signal.shape[0]

    if scaling:
        num = torch.sum(reference_signal*estimated_signal, dim=-1, keepdim=True) + eps
        den = reference_signal.pow(2).sum(-1, keepdim=True) + eps
        a = num / den
    else:
        a = torch.ones_like(reference_signal)

    e_true = a * reference_signal
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum(dim=-1) + eps
    Snn = (e_res**2).sum(dim=-1) + eps

    SDR = 10 * torch.log10(Sss / Snn)
    return SDR

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric = nn.MSELoss()

    def forward(self, preds, target):
        preds  =  preds.permute(0,3,1,2)
        target = target.permute(0,3,1,2)
        return self.metric(preds, target)

class FkLoss(nn.Module):
    def __init__(self, scale=1., weight=1.):
        super().__init__()
        self.scale = scale
        self.weight = weight
        self.metric = nn.L1Loss()

    def forward(self, preds_fk, target_fk):
        w = torch.ones_like(target_fk).cumsum(-1).flip(-1) / target_fk.size(-1)
        scale = self.scale * w
        preds_fk  = scale * preds_fk
        target_fk = scale * target_fk
        #print("fk", self.metric(preds_fk, target_fk))
        return self.weight * self.metric(preds_fk, target_fk)

class ModeFreqLoss(nn.Module):
    def __init__(self, scale=1., weight=1., sr=48000):
        super().__init__()
        self.sr = sr
        self.scale = scale
        self.weight = weight
        self.metric = nn.L1Loss()

    def forward(self, preds_freq, target_fk):
        #w = torch.ones_like(target_fk).cumsum(-1).flip(-1) / target_fk.size(-1)
        #scale = self.scale * w
        preds_freq  = self.scale * preds_freq
        target_fk   = self.scale * target_fk
        return self.weight * self.metric(preds_freq, target_fk)

class ModeAmpsLoss(nn.Module):
    def __init__(self, scale=1., weight=1.):
        super().__init__()
        self.scale = scale
        self.weight = weight
        self.metric = nn.L1Loss()

    def forward(self, preds_coef, target_ck):
        preds_coef  = self.scale * preds_coef
        target_ck   = self.scale * target_ck
        return self.weight * self.metric(preds_coef, target_ck)

class L1Loss(nn.Module):
    def __init__(self, weight=1., scale_invariance=False):
        super().__init__()
        self.si = scale_invariance
        self.weight = weight
        self.metric = nn.L1Loss()

    def forward(self, preds, target):
        if self.si:
            eps = torch.finfo(target.dtype).eps
            preds_rms = preds.pow(2).mean(-1, keepdim=True).clamp(min=eps).sqrt()
            target_rms = target.pow(2).mean(-1, keepdim=True).clamp(min=eps).sqrt()
            preds = preds / preds_rms
            target = target / target_rms
        return self.weight * self.metric(preds, target)

class SISDR(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric = si_sdr

    def forward(self, preds, target):
        preds  = rearrange( preds, 'b (1 t) -> b 1 t')
        target = rearrange(target, 'b (1 t) -> b 1 t')
        value  = self.metric(preds, target, eps=1e-8)
        return - value.mean() / 20

class FFTLoss(nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.weight = weight
        self.metric = nn.L1Loss()

    def forward(self, preds, target):
        preds  = torch.fft.rfft( preds)
        target = torch.fft.rfft(target)
        return self.weight * self.metric(preds, target)

class MRSTFT(nn.Module):
    def __init__(self, input_scale=5., weight=1., **kwargs):
        super().__init__()
        self.scale = input_scale
        self.weight = weight
        self.metric = MultiResolutionSTFTLoss(**kwargs)

    def forward(self, preds, target):
        target = target * self.scale
        preds = preds * self.scale
        if len(list(preds.shape)) == 4:
            preds  = rearrange(preds,  'b t x c -> b (c x) t')
            target = rearrange(target, 'b t x c -> b (c x) t')
        elif len(list(preds.shape)) == 2:
            preds = preds.unsqueeze(1)
            target = target.unsqueeze(1)
        else:
            assert len(list(preds.shape)) == 3, preds.shape
        return self.weight * self.metric(preds, target)

class PDELoss(nn.Module):
    def __init__(self, f_ic, f_bc, f_r, w_ic=1., w_bc=1., w_r=1.):
        super().__init__()
        self.f_ic = f_ic; self.f_bc = f_bc; self.f_r  = f_r 
        self.w_ic = w_ic; self.w_bc = w_bc; self.w_r  = w_r 

    def forward(self,
            pde_preds,
            u0, f0, kappa, alpha, sig0, sig1,
            bow_mask, hammer_mask,
            x_B, v_B, F_B, ph0_B, ph1_B, wid_B,
            x_H, v_H, u_H, w_H, M_H, a_H,
            xt, tt,
        ):
        ''' pde_preds: (batch, time, space, 2) '''
        ut, zt = pde_preds.chunk(2, dim=-1) # (Bs, Nt, Nx, 1)
        ut = ut.squeeze(-1)
        zt = zt.squeeze(-1)
        ms = [bow_mask, hammer_mask]
        bp = [x_B, v_B, F_B, ph0_B, ph1_B, wid_B]
        hp = [x_H, v_H, u_H, w_H, M_H, a_H]
        return pde_loss(
            #---------- 
            ut, zt, u0, f0, kappa, alpha, sig0, sig1,
            ms, bp, hp, xt, tt,
            #---------- 
            self.f_ic, self.f_bc, self.f_r,
            self.w_ic, self.w_bc, self.w_r,
        )

class BCLoss(nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.weight = weight
        self.metric = nn.L1Loss()

    def forward(self, preds_bc):
        target = torch.zeros_like(preds_bc)
        return self.weight * self.metric(preds_bc, target)

class ICLoss(nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.weight = weight
        self.metric = nn.L1Loss()

    def forward(self, preds_ic, target_ic):
        return self.weight * self.metric(preds_ic, target_ic)

class F0Loss(nn.Module):
    def __init__(self, scale=10., weight=1.):
        super().__init__()
        self.scale = scale
        self.weight = weight
        self.metric = nn.L1Loss()

    def forward(self, preds_f0, target_f0):
        ''' (Bs, Nt) '''
        target_mean = target_f0.mean()
        target_f0 = target_f0 - target_mean
        preds_f0  =  preds_f0 - target_mean
        target_std = target_f0.std()
        target_f0 = target_f0 / target_std
        preds_f0  =  preds_f0 / target_std

        preds_f0 = preds_f0 * self.scale
        target_f0 = target_f0 * self.scale
        return self.weight * self.metric(preds_f0, target_f0)

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

class DisLoss(nn.Module):
    def __init__(self):
        ''' Discriminator Loss '''
        super().__init__()

    def forward(self, real_disc, fake_disc):
        loss_real = sum([adv_loss(d, 1) for d in real_disc]) / len(real_disc)
        loss_fake = sum([adv_loss(d, 0) for d in fake_disc]) / len(fake_disc)
        return loss_real + loss_fake

class GenLoss(nn.Module):
    def __init__(self):
        ''' Generator Loss '''
        super().__init__()

    def forward(self, fake_disc):
        return sum([adv_loss(d, 1) for d in fake_disc]) / len(fake_disc)








