import torch
import numpy as np
import torch.nn.functional as F

def constant(f0, n, dtype=None):
    ''' f0 (batch_size,)
        n  (int)
    '''
    return f0.unsqueeze(-1) * torch.ones(1,n, dtype=dtype)

def linear(f1, f2, n):
    ''' f1 (batch_size,)
        f2 (batch_size,)
        n  (int)
    '''
    out = torch.cat((f1.unsqueeze(-1),f2.unsqueeze(-1)), dim=-1)             # (batch_size, 2)
    out = F.interpolate(out.unsqueeze(1), size=n, mode='linear', align_corners=True).squeeze(1)  # (batch_size, n)
    return out

def glissando(f1, f2, n, mode='linear'):
    if mode == 'linear':
        return linear(f1, f2, n)
    else:
        raise NotImplementedError(mode)

def vibrato(f0, k, mf=[3,5], ma=0.05, ma_in_hz=False):
    ''' f0 (batch_size, n)
        k  (int): 1/sr
        mf (list): modulation frequency ([min, max])
        ma (float): modulation amplitude (in Hz)
        ma_in_hz (bool): ma is given in Hz (else: ma is given as a weighting factor of f0)
    '''
    ff = f0.narrow(-1,0,1)
    def get_new_vibrato(f0, k, mf, ma, ma_in_hz):
        mod_frq = mf[1] * torch.rand_like(ff) + mf[0] # (B, 1)
        mod_amp = ma * torch.rand_like(ff) # (B, 1)

        nt = f0.size(-1)  # total time
        vt = torch.floor((nt // 2) * torch.rand(f0.size(0)).view(-1,1))  # vibrato time
        t = torch.ones_like(f0).cumsum(-1)
        m = t.gt(vt) # mask `t` for n <= vt
        vibra = m * mod_amp * (1 - torch.cos(2 * np.pi * mod_frq * (t - vt) * k)) / 2
        if not ma_in_hz: vibra *= f0
        return vibra * torch.randn_like(ff).sign()
    return f0 + get_new_vibrato(f0, k, mf, ma, ma_in_hz)

def triangle_with_velocity(vel, n, sr_t, sr_x, max_u=.1):
    ''' vel    (batch_size,) velocity
        n      (int) number of samples
        sr_t   (int) sampling rate in time
        sr_x   (int) sampling rate in space
        max_u  (float) maximum displacement
    '''
    vel = vel.view(-1,1) * sr_x / sr_t    # m/s to non-dimensional quantity
    vel = vel * torch.ones_like(vel).repeat(1,n)
    u_H = torch.relu(max_u - (max_u - vel.cumsum(1)).abs() - vel)
    u_H = u_H.pow(5).clamp(max=0.01)
    return u_H



