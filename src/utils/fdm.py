import math
import torch
import numpy as np
import torch.nn.functional as F
#from src.utils.misc import raised_cosine, sqrt
def sqrt(x):
    return x.pow(.5) if isinstance(x, torch.Tensor) else x**.5

def tridiagonal_inverse(X, check_tridiagonal=False):
    n = X.size(0); ks = torch.arange(n).to(X.device) + 1; jk = torch.outer(ks,ks)
    a = X[1,0]; b = X[0,0]; c = X[0,1]
    if check_tridiagonal:
        assert list(X.shape) == [n,n], X.shape
        assert torch.allclose(torch.diag(X, diagonal=-1), a * torch.ones(n-1)) \
           and torch.allclose(torch.diag(X, diagonal= 0), b * torch.ones(n)) \
           and torch.allclose(torch.diag(X, diagonal=+1), c * torch.ones(n-1)) \
           and all([torch.allclose(torch.diag(X, diagonal=i), torch.zeros(n-i)) for i in range(2,n)])
    lam = b + (a+c) * torch.cos(ks * np.pi / (n+1))
    L = torch.diag(1 / lam)
    V = np.sqrt(2 / (n+1)) * torch.sin(jk * np.pi / (n+1))
    return V @ L @ V.T

def I(n, diagonal=0):
    l = n - abs(diagonal)
    i = torch.ones(l)
    return torch.diag(i, diagonal)

def Dxx(n, h):
    Dx = I(n, +1) - 2*I(n) + I(n, -1)
    Dx = Dx / (h**2)
    return Dx

def Dxxxx(n, h):
    Dx = I(n, +2) - 4*I(n, +1) + 6*I(n) - 4*I(n, -1) + I(n, -2)
    Dx = Dx / (h**4)
    return Dx

def D(n, xd, h):
    if xd == 'x-':
        Dx = I(n) - I(n, -1)
        Dx = Dx / h
    elif xd == 'x+':
        Dx = I(n, +1) - I(n)
        Dx = Dx / h
    elif xd == 'xc':
        Dx = I(n, +1) - I(n, -1)
        Dx = Dx / h
    elif xd == 'xx':
        Dx = I(n, +1) - 2*I(n) + I(n, -1)
        Dx = Dx / (h**2)
    elif xd == 'xxxx':
        Dx = I(n, +2) - 4*I(n, +1) + 6*I(n) - 4*I(n, -1) + I(n, -2)
        Dx = Dx / (h**4)
    else:
        assert False
    return Dx


def displacement_update(u, n):
    # interpolate?
    return F.interpolate(u.view(1,1,-1), size=n, mode='linear').view(-1)
    # pad/trim raises artifact waves
    #if u.size(-1) < n:
    #    return F.pad(u, (0,n-u.size(-1)))
    #else:
    #    return u[...,:n]

def bow_term_rhs(N, h, k, u1, u2, x_B, v_B, F_B, wid, friction_fn):
    rc = raised_cosine(N-1, h, x_B, wid)
    rc = rc / rc.abs().sum()
    I = rc
    J = rc / h
    v_rel = I @ (u1 - u2) / k - v_B      # using explicit scheme
    Gamma = J * F_B * friction_fn(v_rel)
    return - k**2 * Gamma, v_rel

def initialize_state(u0, v0, Nt, Nx_t, Nx_l, k, dtype=None):
    ''' u0   (batch_size, Nt, Nx_t) predefined displacement for each time
        v0   (batch_size, Nt, Nx_t) predefined displacement for each time
        Nt   (int) number of samples in time
        Nx_t (int) number of transverse samples in space
        Nx_l (int) number of longitudinal samples in space
        k    (int) temporal spacing
        ---
        state_t (batch_size, Nt, Nx_t+1) 
        state_l (batch_size, Nt, Nx_l+1) 
    '''
    batch_size = u0.size(0)
    u0 = torch.from_numpy(u0, dtype=dtype) if isinstance(u0, np.ndarray) else u0
    v0 = torch.from_numpy(v0, dtype=dtype) if isinstance(v0, np.ndarray) else v0

    u1 = u0 + k * v0
    u2 = u0

    state_t = torch.zeros(batch_size, Nt, Nx_t+1, dtype=dtype)
    state_l = torch.zeros(batch_size, Nt, Nx_l+1, dtype=dtype)
    state_t[:,:-1,:] = u2[:,:-1,:]
    state_t[:,+1:,:] = u1[:,:-1,:]
    return state_t, state_l

def get_derived_vars(f0, kappa_rel, k, theta_t, lambda_c, alpha):
    # Derived variables
    gamma = 2 * f0                           # set parameters
    kappa = gamma * kappa_rel                # stiffness parameter
    IHP = (np.pi * kappa / gamma)**2         # inharmonicity parameter (>0); eq 7.21
    K = sqrt(IHP) * (gamma / np.pi)          # set parameters
    if isinstance(lambda_c, torch.Tensor):
        lambda_c = torch.relu(lambda_c - 1) + int(1) # make sure >= 1
    else:
        lambda_c = int(1) if lambda_c <= 1 else lambda_c

    h = lambda_c * sqrt( \
        (gamma**2 * k**2 + sqrt(gamma**4 * k**4 + 16 * K**2 * k**2 * (2 * theta_t - 1))) \
      / (2 * (2 * theta_t - 1)) \
    )
    N_t = torch.floor(1/h) if isinstance(h, torch.Tensor) else int(1 / h)
    h_t = 1 / N_t

    h = lambda_c * gamma * alpha * k
    N_l = torch.floor(1/h) if isinstance(h, torch.Tensor) else int(1 / h)
    h_l = 1 / N_l

    return gamma, K, N_t, h_t, N_l, h_l

def get_theta(kappa_max, f0_inf, sr, lambda_c=1):
    ''' theta gets larger as...
        (1) f0 gets larger
        (2) kappa gets smaller
    '''
    gamma = 2 * f0_inf
    kappa = gamma * kappa_max
    k = 1 / sr

    R = ((gamma**4 * k**2 + 4*kappa**2 * math.pi**2) / (gamma**4 * k**2))**.5
    S = gamma**4 * k**2 * lambda_c**2 / (4 * kappa**2 * math.pi**4)
    expr_1 = 2 * S * lambda_c**2 * (R-1)**2
    expr_2 = math.pi**2 * S * (R-1)
    theta = 0.5 + expr_1 + expr_2
    assert theta < 1, theta
    
    return theta

def stiff_string_modes(f0, kappa_rel, p_max=1):
    ''' Returns a list of modes of an ideal lossless stiff string.
        This inharmonicity factor `B` is valid only if kappa_rel is small.
        c.f.;
          Fletcher `Normal Vibration Frequencies of a Stiff Piano String` 
          Bilbao `Numerical Sound Synthesis` (pp. 176)
    '''
    B = (np.pi * kappa_rel)**2

    modes = []
    factor = []
    for p in range(1,p_max+1):
        w_p = p * (1 + (2/np.pi) * B**.5 + 4/np.pi**2 * B) * (1 + B*p**2)**.5
        factor.append(w_p)
        modes.append(f0 * w_p) 
    return modes, factor

if __name__=='__main__':
    sr = 48000
    k = 1/sr
    #tt = 0.5 + 2/(np.pi**2)

    #for f0 in [20,40,80,160,320]:
    #    _, _, N_t, _, N_l, _ = get_derived_vars(f0=f0, kappa_rel=0.03, k=k, theta_t=tt, alpha=1)
    #    print(f0, N_t, N_l)
    #for al in [1,2,3,4]:
    #    _, _, N_t, _, N_l, _ = get_derived_vars(f0=96, kappa_rel=0.03, k=k, theta_t=tt, alpha=al)
    #    print(al, N_t, N_l)

    #------------------------------ 

    #def vibrato(f0, k, mf=3, ma=0.01, upward=True, ma_in_Hz=False, dtype=None):
    #    nt = f0.size(-1)  # total time
    #    vt = torch.floor((nt // 2) * torch.rand(f0.size(0)).view(-1,1))  # vibrato time
    #    t = torch.ones_like(f0).cumsum(-1)
    #    m = t.gt(vt) # mask `t` for n <= vt
    #    vibra = m * ma * (1 - torch.cos(2 * np.pi * mf * (t - vt) * k)) / 2
    #    if not ma_in_Hz: vibra *= f0
    #    return f0 + vibra if upward else f0 - vibra

    #import matplotlib.pyplot as plt

    #f0 = 40 * torch.ones(sr).view(1,-1)
    #f0 = vibrato(f0, 1/sr)

    #_, _, N_t, _, N_l, _ = get_derived_vars(f0=f0, kappa_rel=0., k=k, theta_t=tt, lambda_c=1, alpha=10)

    #fig, ax = plt.subplots(nrows=3)
    #ax[0].plot(f0[0])
    #ax[1].plot(N_t[0])
    #ax[2].plot(N_l[0])
    #plt.savefig('asdf.png')

    #------------------------------ 

    #tt = 1
    #------------------------------ 
    f0 = 55 * torch.ones(sr).view(1,-1)
    lam = 1.01
    als = 1
    #kappa_rel = 0.03
    #kappa_rel = 0.02
    kappa_rel = 0.01
    #------------------------------ 
    #f0 = 60. * torch.ones(sr).view(1,-1)
    #lam = 2
    #als = 5
    #kappa_rel = 0.03
    #------------------------------ 
    tt = get_theta(kappa_rel, f0.min(), sr, lam)
    _, _, N_t, _, N_l, _ = get_derived_vars(f0=f0, kappa_rel=kappa_rel, k=k, theta_t=tt, lambda_c=1, alpha=als)
    print(N_t[0,0], N_l[0,0])
    _, _, N_t, _, N_l, _ = get_derived_vars(f0=f0, kappa_rel=kappa_rel, k=k, theta_t=tt, lambda_c=lam, alpha=als)
    print(N_t[0,0], N_l[0,0])
    _, _, N_t, _, N_l, _ = get_derived_vars(f0=f0, kappa_rel=kappa_rel, k=k, theta_t=tt, lambda_c=lam**2, alpha=als)
    print(N_t[0,0], N_l[0,0])


