import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.utils import audio as audio

class Synthesizer(nn.Module):
    """ Synthesizer Network """
    def __init__(self,
            embed_dim=64,
            x_scale=1, t_scale=1,
            gamma_scale=0, kappa_scale=0, alpha_scale=0, sig_0_scale=0, sig_1_scale=0,
            **kwargs):
        super().__init__()
        self.sr=kwargs['sr']
        hidden_dim=kwargs['hidden_dim']
        self.n_modes = kwargs['n_modes']
        inharmonic = kwargs['harmonic'].lower() == 'inharmonic'

        self.x_scale = x_scale
        self.t_scale = t_scale
        self.gamma_scale = gamma_scale
        self.kappa_scale = kappa_scale
        self.alpha_scale = alpha_scale
        self.sig_0_scale = sig_0_scale
        self.sig_1_scale = sig_1_scale

        from src.model.nn.blocks import RFF, ModeEstimator
        n_feats = 7
        self.material_encoder = RFF([1.]*n_feats, embed_dim // 2)
        feature_size = embed_dim * n_feats
        self.mode_estimator = ModeEstimator(
            self.n_modes, embed_dim, kappa_scale, gamma_scale,
            inharmonic=inharmonic,
        )
        if inharmonic:
            from src.model.nn.dmsp import DMSP
            self.net = DMSP(
                embed_dim=embed_dim,
                hidden_size=hidden_dim,
                n_features=n_feats,
                n_modes=kwargs['n_modes'],
                n_bands=kwargs['n_bands'],
                block_size=kwargs['block_size'],
                sampling_rate=kwargs['sr'],
            )
        else:
            from src.model.nn.ddsp import DDSP
            self.net = DDSP(
                feature_size=feature_size,
                hidden_size=hidden_dim,
                n_modes=kwargs['n_modes'],
                n_bands=kwargs['n_bands'],
                block_size=kwargs['block_size'],
                sampling_rate=kwargs['sr'],
                fm=kwargs['ddsp_frequency_modulation'],
            )

    def forward(self, params, pitch, initial):
        ''' params : input parameters
            pitch  : fundamental frequency in Hz
            initial: initial condition
        '''
        space, times, kappa, alpha, t60, mode_freq, mode_coef = params

        f_0 = pitch.unsqueeze(2)    # (b, frames, 1)
        times = times.unsqueeze(-1) # (b, sample, 1)
        kappa = kappa.unsqueeze(-1) # (b, 1, 1)
        alpha = alpha.unsqueeze(-1) # (b, 1, 1)
        space = space.unsqueeze(-1) # (b, 1, 1)
        gamma = 2*f_0               # (b, frames, 1)
        omega = f_0 / self.sr * (2*math.pi) # (b, t, 1)
        relf0 = omega - omega.narrow(1,0,1) # (b, t, 1)

        in_coef, in_freq = self.mode_estimator(initial, space, kappa, gamma.narrow(1,9,1))
        mode_coef = in_coef if mode_coef is None else mode_coef
        mode_freq = in_freq if mode_freq is None else mode_freq
        mode_freq = mode_freq + relf0 # linear FM

        Nt = times.size(1)     # total number of samples
        Nf = mode_freq.size(1) # total number of frames
        frames = self.get_frame_time(times, Nf)

        space = space.repeat(1,f_0.size(1),1) # (b, frames, 1)
        alpha = alpha.repeat(1,f_0.size(1),1) # (b, frames, 1)
        kappa = kappa.repeat(1,f_0.size(1),1) # (b, frames, 1)
        sigma = audio.T60_to_sigma(t60, f_0, 2*f_0*kappa) # (b, frames, 2)

        # fourier features
        feat = [space, frames, kappa, alpha, sigma, gamma]
        feat = self.normalize_params(feat)
        feat = self.material_encoder(feat) # (b, frames, n_feats * embed_dim)

        damping = torch.exp(- frames * sigma.narrow(-1,0,1))
        mode_coef = mode_coef * damping
        ut, ut_freq, ut_coef = self.net(feat, mode_freq, mode_coef, frames, alpha, omega, Nt)
        return ut, [in_freq, in_coef], [ut_freq, ut_coef]

    def get_frame_time(self, times, Nf):
        t_0 = times.narrow(1,0,1) # (Bs, 1, 1)
        t_k = torch.ones_like(t_0).repeat(1,Nf,1).cumsum(1) / self.sr
        t_k = t_k + t_0 # (Bs, Nt, 1)
        return t_k

    def normalize_params(self, params):
        def rescale(var, scale):
            minval = min(scale)
            denval = max(scale) - minval
            return (var - minval) / denval
        space, times, kappa, alpha, sigma, gamma = params
        sig_0, sig_1 = sigma.chunk(2, -1)
        space = rescale(space, self.x_scale)
        times = rescale(times - max(self.t_scale), self.t_scale)
        kappa = rescale(kappa, self.kappa_scale)
        alpha = rescale(alpha, self.alpha_scale)
        sig_0 = rescale(sig_0, self.sig_0_scale)
        sig_1 = rescale(sig_1, self.sig_1_scale)
        gamma = rescale(gamma, self.gamma_scale)
        sigma = torch.cat((sig_0, sig_1), dim=-1)
        return torch.cat([space, times, kappa, alpha, sigma, gamma], dim=-1)



