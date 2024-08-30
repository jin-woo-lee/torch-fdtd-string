import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from src.utils import misc as utils
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

def swish(x):
    return x * torch.sigmoid(x)

def get_activation(name):
    if name is None or name == 'linear':
        return nn.Identity(), 'linear'
    elif name.lower() == 'relu':
        return nn.ReLU(), 'relu'
    elif name.lower() == 'leaky_relu':
        return nn.LeakyReLU(), 'leaky_relu'
    elif name.lower() == 'tanh':
        return nn.Tanh(), 'tanh'
    elif name.lower() == 'sin':
        return torch.sin, 'tanh'
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid(), 'sigmoid'
    elif name.lower() == 'swish':
        return swish, 'linear'
    else:
        raise NotImplementedError(name)

def apply_gain(x, gain, fn=None):
    gain = fn(gain) if fn is not None else gain
    x_list = x.chunk(len(gain), -1)
    x_list = [gain[i] * x_i for i, x_i in enumerate(x_list)]
    return torch.cat(x_list, dim=-1)

class FMBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, num_features):
        super().__init__()
        concat_size = embed_dim * num_features + embed_dim
        feature_dim = embed_dim * num_features
        self.rff2 = RFF2(input_dim, embed_dim//2)
        self.tmlp = mlp(concat_size, feature_dim, 5)
        self.proj = nn.Linear(concat_size, 2*input_dim)
        self.activation = nn.GLU(dim=-1)

        gain_in = torch.randn(num_features) / 2
        gain_out = torch.Tensor([0.1])
        self.register_parameter('gain_in', nn.Parameter(gain_in, requires_grad=True))
        self.register_parameter('gain_out', nn.Parameter(gain_out, requires_grad=True))

    def forward(self, input, feature, slider, omega):
        ''' input  : (B T input_dim)
            feature: (B T feature_dim)
            slider : (B T 1)
        '''
        _input = input / (1.3*math.pi) - 1
        _input = self.rff2(_input)
        feature = apply_gain(feature, self.gain_in, torch.tanh)

        x = torch.cat((_input, feature), dim=-1)
        x = torch.cat((self.tmlp(x), _input), dim=-1)
        x = self.activation(self.proj(x))

        gate = torch.tanh((slider - 1) * self.gain_out)
        return input + omega * x * gate

class AMBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, num_features):
        super().__init__()
        concat_size = embed_dim * num_features + embed_dim
        feature_dim = embed_dim * num_features
        self.rff2 = RFF2(input_dim, embed_dim//2)
        self.tmlp = mlp(concat_size, feature_dim, 5)
        self.proj = nn.Linear(concat_size, 2*input_dim)
        self.activation = nn.GLU(dim=-1)

        gain_in = torch.randn(num_features) / 2
        self.register_parameter('gain_in', nn.Parameter(gain_in, requires_grad=True))

    def forward(self, input, feature, slider):
        ''' input  : (B T input_dim)
            feature: (B T feature_dim)
            slider : (B T 1)
        '''
        _input = input * 110 - 0.55
        _input = self.rff2(_input)
        feature = apply_gain(feature, self.gain_in, torch.tanh)

        x = torch.cat((_input, feature), dim=-1)
        x = torch.cat((self.tmlp(x), _input), dim=-1)
        x = self.activation(self.proj(x))

        return input * (1 + x)

class ModBlock(nn.Module):
    def __init__(self, input_dim, feature_dim, embed_dim):
        super().__init__()
        cat_size = 1+feature_dim
        self.tmlp = mlp(cat_size, feature_dim, 2)
        self.proj = nn.Linear(cat_size, 2)
        self.activation = nn.GLU(dim=-1)

    def forward(self, input, feature, slider):
        ''' input  : (B T input_dim)
            feature: (B T feature_dim)
            slider : (B T 1)
        '''
        input   =   input.unsqueeze(-1) # (B T input_dim 1)
        feature = feature.unsqueeze(-2).repeat(1,1,input.size(-2),1)
        x = torch.cat((input, feature), dim=-1)
        x = torch.cat((self.tmlp(x), input), dim=-1)
        x = self.activation(self.proj(x))
        return (input * (1 + x)).squeeze(-1)

def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        #net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.PReLU())
    return nn.Sequential(*net)

class RFF2(nn.Module):
    """ Random Fourier Features Module """
    def __init__(self, input_dim, embed_dim, scale=1.):
        super().__init__()
        N = torch.ones((input_dim, embed_dim)) / input_dim / embed_dim
        N = nn.Parameter(N, requires_grad=False)
        e = torch.Tensor([scale])
        e = nn.Parameter(e, requires_grad=True)
        self.register_buffer('N', N)
        self.register_parameter('e', e)

    def forward(self, x):
        ''' x: (Bs, Nt, input_dim)
            -> (Bs, Nt, embed_dim)
        '''
        B = self.e * self.N
        x_embd = utils.fourier_feature(x, B)
        return x_embd

class RFF(nn.Module):
    """ Random Fourier Features Module """
    def __init__(self, scales, embed_dim):
        super().__init__()
        input_dim = len(scales)
        N = torch.randn(input_dim, embed_dim)
        N = nn.Parameter(N, requires_grad=False)
        e = torch.Tensor(scales).view(-1,1)
        e = nn.Parameter(e, requires_grad=True)
        self.register_buffer('N', N)
        self.register_parameter('e', e)

    def forward(self, x):
        ''' x: (Bs, Nt, input_dim)
            -> (Bs, Nt, input_dim*embed_dim)
        '''
        xs = x.chunk(self.N.size(0), -1) # (Bs, Nt, 1) * input_dim
        Ns = self.N.chunk(self.N.size(0), 0) # (1, embed_dim) * input_dim
        Bs = [torch.pow(10, self.e[i]) * N for i, N in enumerate(Ns)]
        x_embd = [utils.fourier_feature(xs[i], B) for i, B in enumerate(Bs)]
        return torch.cat(x_embd, dim=-1)

class ModeEstimator(nn.Module):
    def __init__(self, n_modes, hidden_dim, kappa_scale=None, gamma_scale=None, inharmonic=True, sr=48000):
        super().__init__()
        self.sr = sr
        self.kappa_scale = kappa_scale
        self.gamma_scale = gamma_scale
        self.rff = RFF([1.]*5, hidden_dim//2)
        self.a_mlp = mlp(5*hidden_dim, hidden_dim, 2)
        self.a_proj = nn.Linear(hidden_dim, n_modes)
        self.tanh = nn.Tanh()
        if inharmonic:
            self.f_mlp = mlp(5*hidden_dim, hidden_dim, 2)
            self.f_proj = nn.Linear(hidden_dim, n_modes)
            self.sigmoid = nn.Sigmoid()
        else:
            self.f_mlp = None
            self.f_proj = None
            self.sigmoid = nn.Sigmoid()

    def forward(self, u_0, x_p, kappa, gamma):
        ''' u_0   : (b, 1, x)
            x_p   : (b, 1, 1)
            kappa : (b, 1, 1)
            gamma : (b, 1, 1)
        '''
        p_x = torch.argmax(u_0, dim=-1, keepdim=True) / 255. # (b, 1, 1)
        p_a = torch.max(u_0, dim=-1, keepdim=True).values / 0.02 # (b, 1, 1)
        kappa = self.normalize_kappa(kappa)
        gamma = self.normalize_gamma(gamma)
        con = torch.cat((p_x, p_a, x_p, kappa, gamma), dim=-1) # (b, 1, 5)
        con = self.rff(con) # (b, 1, 3*hidden_dim)

        mode_amps = self.a_mlp(con) # (b, 1, k)
        mode_amps = self.tanh(1e-3 * self.a_proj(mode_amps)) # (b, 1, m)

        if self.f_mlp is not None:
            mode_freq = self.f_mlp(con) # (b, 1, k)
            mode_freq = 0.3 * self.sigmoid(self.f_proj(mode_freq)) # (b, 1, m)
            mode_freq = mode_freq.cumsum(-1)
        else:
            int_mults = torch.ones_like(mode_amps).cumsum(-1) # (b, 1, k)
            omega = gamma / self.sr * (2*math.pi)
            mode_freq = omega * int_mults

        return mode_amps, mode_freq

    def normalize_gamma(self, x):
        if self.gamma_scale is not None:
            minval = min(self.gamma_scale)
            denval = max(self.gamma_scale) - minval
            x = (x - minval) / denval
        return x

    def normalize_kappa(self, x):
        if self.kappa_scale is not None:
            minval = min(self.kappa_scale)
            denval = max(self.kappa_scale) - minval
            x = (x - minval) / denval
        return x

