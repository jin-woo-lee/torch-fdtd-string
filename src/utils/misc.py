import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

chars = [c for c in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ']

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def batchify(x, batch_size, n_samples):
    pass

def random_str(length=8):
    return "".join(np.random.choice(chars, length))

def sqrt(x):
    return x.pow(.5) if isinstance(x, torch.Tensor) else x**.5

def soft_bow(v_rel, a=100):
    return np.sqrt(2*a) * v_rel * torch.exp(-a * v_rel**2 + 0.5)

def hard_bow(v_rel, a=5, eps=0.1, hard_sign=True):
    sign = torch.sign(v_rel) if hard_sign else torch.tanh(100 * v_rel)
    return sign * (eps + (1-eps) * torch.exp(-a * v_rel.abs()))

def raised_cosine(N, h, ctr, wid, n):
    ''' N      (int): number of maximal samples in space
        h    (float): spatial grid cell width
        ctr  (B,1,1): center points for each batch
        wid  (B,1,1): width lengths for each batch
        n       (B,): number of actual samples in space
    '''
    xax = torch.linspace(h, 1, N).to(ctr.device).view(1,-1,1)   # (1, N, 1)
    ctr = (ctr * n / N)
    wid = wid / N
    ind = torch.sign(torch.relu(-(xax - ctr - wid / 2) * (xax - ctr + wid / 2)))
    out = 0.5 * ind * (1 + torch.cos(2 * np.pi * (xax - ctr) / wid))
    return out / out.abs().sum(1, keepdim=True)  # (batch_Size, N, 1)

def floor_dirac_delta(n, ctr, N):
    ''' torch::Tensor n,        // number of samples in space
        torch::Tensor ctr,      // center point of raised cosine curve
        int N
    ''' 
    xax = torch.ones_like(ctr).view(-1,1,1).repeat(1,N,1).cumsum(1) - 1
    idx = torch.floor(ctr * n).view(-1,1,1)
    #return torch.floor(xax).eq(idx).to(n.dtype())  # (batch_size, N, 1)
    return torch.floor(xax).eq(idx)  # (batch_size, N, 1)

def triangular(N, n, p_x, p_a):
    ''' N    (int): number of maximal samples in space
        n    (B,  1, 1): number of actual samples in space
        p_x  (B, Nt, 1): peak position
        p_a  (B, Nt, 1): peak amplitude
    '''
    vel_l = torch.where(p_x.le(0), torch.zeros_like(p_x), p_a / p_x / n)
    vel_r = torch.where(p_x.le(0), torch.zeros_like(p_x), p_a / (1-p_x) / n)
    vel_l = ((vel_l * torch.ones_like(vel_l).repeat(1,1,N)).cumsum(2) - vel_l).clamp(min=0)
    vel_r = ((vel_r * torch.ones_like(vel_r).repeat(1,1,N)).cumsum(2) - vel_r * (N-n+1)).clamp(min=0).flip(2)
    tri = torch.minimum(vel_l, vel_r)
    assert not torch.isnan(tri).any(), torch.isnan(tri.flatten(1).sum(1))
    return tri

def pre_shaper(x, sr, velocity=10):
    w = torch.tanh(torch.ones_like(x).cumsum(-1) / sr * velocity)
    return w * x

def post_shaper(x, sr, pulloff, velocity=100):
    offset = x.size(-1) - int(sr * pulloff)
    w = torch.tanh(torch.ones_like(x).cumsum(-1) / sr * velocity).flip(-1)
    w = F.pad(w.narrow(-1,offset,w.size(-1)-offset), (0,offset))
    return w * x

def random_uniform(floor, ceiling, size=None, weight=None, dtype=None):
    if not isinstance(size, tuple): size = (size,)
    if weight is None: weight = torch.ones(size, dtype=dtype)
    # NOTE: torch.rand(..., dtype=dtype) for dtype \in [torch.float32, torch.float64]
    #       can result in different random number generation
    #       (for different precisions; despite fixiing the random seed.)
    return (ceiling - floor) * torch.rand(size=size).to(dtype) * weight + floor

def equidistant(floor, ceiling, steps, dtype=None):
    return torch.linspace(floor, ceiling, steps).to(dtype)

def get_masks(model_name, bs, disjoint=True):
    ''' setting `disjoint=False` enables multiple excitations allowed
        (e.g., bowing over hammered strings.) While this could be a
        charming choice, but it can also drive the simulation unstable.
    '''
    # boolean mask that determines whether to impose each excitation
    if model_name.endswith('bow'):
        bow_mask    = torch.ones( size=(bs,)).view(-1,1,1)
        hammer_mask = torch.zeros(size=(bs,)).view(-1,1,1)
    elif model_name.endswith('hammer'):
        bow_mask    = torch.zeros(size=(bs,)).view(-1,1,1)
        hammer_mask = torch.ones( size=(bs,)).view(-1,1,1)
    elif model_name.endswith('pluck'):
        bow_mask    = torch.zeros(size=(bs,)).view(-1,1,1)
        hammer_mask = torch.zeros(size=(bs,)).view(-1,1,1)
    else:
        bow_mask    = torch.rand(size=(bs,)).gt(0.5).view(-1,1,1)
        hammer_mask = torch.rand(size=(bs,)).gt(0.5).view(-1,1,1)
        if disjoint:
            both_are_true = torch.logical_and(
                torch.logical_or(bow_mask, hammer_mask),
                torch.logical_or(bow_mask, hammer_mask.logical_not())
            )
            hammer_mask[both_are_true] = False
        bow_mask    = bow_mask.view(-1,1,1)
        hammer_mask = hammer_mask.view(-1,1,1)
    return [bow_mask, hammer_mask]

def f0_interpolate(f0_1, n_frames, tmax):
    t_0 = np.linspace(0, tmax, n_frames)
    t_1 = np.linspace(0, tmax, f0_1.shape[0])
    return np.interp(t_0, t_1, f0_1)

def minmax_normalize(x, dim=-1):
    x_min = x.min(dim, keepdim=True).values
    x = x - x_min
    x_max = x.max(dim, keepdim=True).values
    x = x / x_max
    return x

def get_minmax(x):
    if np.isnan(x.sum()):
        return None, None
    return np.nan_to_num(x.min()), np.nan_to_num(x.max())

def batched_index_select(input, dim, index):
    Nx = len(list(input.shape))
    expanse = [-1 if k==(dim % Nx) else 1 for k in range(Nx)]
    tiler = [1 if k==(dim % Nx) else n for k, n in enumerate(input.shape)]
    index = index.view(expanse).tile(tiler)
    return torch.gather(input, dim, index)

def random_index(max_N, idx_N):
    if max_N < idx_N:
        # choosing with replacement
        return torch.randint(0, max_N, (idx_N,))
    else:
        # choosing without replacement
        return torch.randperm(max_N)[:idx_N]

def ell_infty_normalize(x, normalize_dims=1):
    eps = torch.finfo(x.dtype).eps
    x_shape = list(x.shape)
    m_shape = x_shape[:normalize_dims] + [1] * (len(x_shape) - normalize_dims)
    x_max = x.abs().flatten(normalize_dims).max(normalize_dims).values + eps
    x_gain =  1. / x_max.view(m_shape)
    return x * x_gain, x_gain

def sinusoidal_embedding(x, n, gain=10000, dim=-1):
    ''' let `x` be normalized to be in the nondimensional (0 ~ 1) range '''
    assert n % 2 == 0, n
    x = x.unsqueeze(-1)
    shape = [1] * len(list(x.shape)); shape[dim] = -1 # e.g., [1,1,-1]
    half_n = n // 2

    expnt = torch.arange(half_n, device=x.device, dtype=x.dtype).view(shape)
    _embed = torch.exp(expnt * -(np.log(gain) / (half_n - 1)))
    _embed = torch.exp(expnt * -(np.log(gain) / (half_n - 1)))
    _embed = x * _embed
    emb = torch.cat((torch.sin(_embed), torch.cos(_embed)), dim)
    return emb # list(x.shape) + [n]

def fourier_feature(x, B):
    ''' x: (Bs, ..., in_dim)
        B: (in_dim, out_dim)
    '''
    if B is None:
        return x
    else:
        x_proj = (2.*np.pi*x) @ B
        return torch.cat((torch.sin(x_proj), torch.cos(x_proj)), dim=-1)

def save_simulation_data(directory, excitation_type, overall_results, constants):
    os.makedirs(directory, exist_ok=True)
    string_params   = overall_results.pop('string_params')
    hammer_params   = overall_results.pop('hammer_params')
    bow_params      = overall_results.pop('bow_params')
    simulation_dict = overall_results
    string_dict = {
        'kappa': string_params[0],
        'alpha': string_params[1],
        'u0'   : string_params[2],
        'v0'   : string_params[3],
        'f0'   : string_params[4],
        'pos'  : string_params[5],
        'T60'  : string_params[6],
        'target_f0': string_params[7],
    }
    hammer_dict = {
        'x_H'  : hammer_params[0],
        'v_H'  : hammer_params[1],
        'u_H'  : hammer_params[2],
        'w_H'  : hammer_params[3],
        'M_r'  : hammer_params[4],
        'alpha': hammer_params[5],
    }
    bow_dict = {
        'x_B'  : bow_params[0],
        'v_B'  : bow_params[1],
        'F_B'  : bow_params[2],
        'phi_0': bow_params[3],
        'phi_1': bow_params[4],
        'wid_B': bow_params[5],
    }

    def sample(val):
        try:
            _val = val.item(0)
        except AttributeError as err:
            if isinstance(val, float) or isinstance(val, int):
                _val = val
            else:
                raise err
        return _val
    short_configuration = {
        'excitation_type': excitation_type,
        'theta_t' : constants[1],
        'lambda_c': constants[2],
    }
    short_configuration['value-string'] = {}
    for key, val in string_dict.items():
        short_configuration['value-string'].update({ key : sample(val) })
    short_configuration['value-hammer'] = {}
    for key, val in hammer_dict.items():
        short_configuration['value-hammer'].update({ key : sample(val) })
    short_configuration['value-bow'] = {}
    for key, val in bow_dict.items():
        short_configuration['value-bow'].update({ key : sample(val) })

    np.savez_compressed(f'{directory}/simulation.npz', **simulation_dict)
    np.savez_compressed(f'{directory}/string_params.npz', **string_dict)
    np.savez_compressed(f'{directory}/hammer_params.npz', **hammer_dict)
    np.savez_compressed(f'{directory}/bow_params.npz',    **bow_dict)

    with open(f"{directory}/simulation_config.yaml", 'w') as f:
        yaml.dump(short_configuration, f, default_flow_style=False)


if  __name__=='__main__':
    N = 10
    B = 1
    h = 1 / N
    ctr = 0.5 * torch.ones(B).view(-1,1,1)
    wid = 1 * torch.ones(B).view(-1,1,1)
    n = N * torch.ones(B)
    ''' N      (int): number of maximal samples in space
        h    (float): spatial grid cell width
        ctr  (B,1,1): center points for each batch
        wid  (B,1,1): width lengths for each batch
        n       (B,): number of actual samples in space
    '''
    c = raised_cosine(N, h, ctr, wid, n)
    print(c.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(c[0,:,0])
    plt.savefig('asdf.png')

