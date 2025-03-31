import yaml
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import scipy

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir1 = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir1)
sys.path.append(parentdir1)
sys.path.append(parentdir2)
import src.utils.misc as ms

MACHINE_EPS = 2.23e-16

def manufactured_solution(Nt, Nx, gamma, sig0, p_a, sr):
    mu = np.pi
    omega = gamma[:,None]
    sigma = sig0[None,None]
    x = np.linspace(-.5, .5, Nx)
    t = (np.ones((Nt, 1)).cumsum(0)-1) / sr
    return p_a * np.cos(mu*x)**2 * np.cos(omega * t) * np.exp(-sigma*t)

def get_data(dir_name):
    _sim = np.load(f"{dir_name}/simulation.npz")
    _str = np.load(f"{dir_name}/string_params.npz")
    return _sim['state_u'], _str['f0'], _str['kappa'], _str['T60']

#============================== 
# Lossless Non-stiff String
#============================== 

def lossless_nonstiff_solution(x, t, u0, f0):
    '''  x: (Bs,  1, Nx)
         t: (Bs, Nt,  1)
        u0: (Bs,  1, Nx)
        f0: (Bs, Nt,  1)
    '''
    Bs,  _, Nx = x.shape
    _ , Nt,  _ = t.shape
    L = x.max(-1).values.unsqueeze(-1) # (Bs, 1, 1) maximal `x` value
    c = 2*L * f0
    def b(n, x, u0):
        return 2/L * (u0 * torch.sin(n*math.pi*x/L)).mean()
    u = torch.zeros(Bs, Nt, Nx, dtype=u0.dtype, device=u0.device)
    for n in range(1,Nx+1):
        u_n = b(n,x,u0) * torch.sin(n*math.pi*x/L) * torch.cos(n*math.pi*c*t/L)
        u += u_n
    return u

def nonlinear_wave_solution(x, t, u0, f0, alpha):
    '''  x: (Bs,  1, Nx)
         t: (Bs, Nt,  1)
        u0: (Bs,  1, Nx, 2)
        f0: (Bs, Nt,  1)
        alpha: (Bs, 1,  1)
    '''
    Bs,  _, Nx = x.shape
    _ , Nt,  _ = t.shape
    u = torch.zeros(Bs, Nt, Nx, 2, dtype=u0.dtype, device=u0.device)
    _u0 = u0.select(-1,0)
    _z0 = u0.select(-1,1)
    L = x.max(-1).values.unsqueeze(-1) # (Bs, 1, 1) maximal `x` value
    cu = 2*L * f0
    cz = 2*L * f0 * alpha
    def b(n, x, u_0):
        return 2/L * (u_0 * torch.sin(n*math.pi*x/L)).mean()
    for n in range(1,Nx+1):
        u[...,0] += b(n,x,_u0) * torch.sin(n*math.pi*x/L) * torch.cos(n*math.pi*cu*t/L)
        u[...,1] += b(n,x,_z0) * torch.sin(n*math.pi*x/L) * torch.cos(n*math.pi*cz*t/L)
    return u

#def plucked_string(u0, f0, Nt, Nx, sr, L=1, device='cpu'):
def lossless_nonstiff_string(u0, f0, Nt, Nx, sr, L=1, device='cpu'):
    ''' u0: initial condition, u(x, 0)
        f0: fundamental frequency
        Nt: temporal grid size
        Nx: spatial grid size
        sr: temporal sampling rate
        L: maximal `x` value
    '''
    if isinstance(u0, np.ndarray): u0 = torch.from_numpy(u0)
    if isinstance(f0, np.ndarray): f0 = torch.from_numpy(f0)
    dtype = u0.dtype
    t = torch.arange(Nt, dtype=dtype).view(1,-1,1).to(device) / sr
    x = torch.linspace(0,L,Nx, dtype=dtype).view(1,1,-1).to(device)
    u0 = u0.unsqueeze(0).to(device)
    f0 = f0.view(1,-1,1).to(device)
    u = lossless_nonstiff_solution(x, t, u0, f0).squeeze(0)
    return u

#def interpolated_plucked_string(u0, f0, Nt, Nx, sr, L=1, order=1, device='cpu'):
def interpolated_nonstiff_string(u0, f0, Nt, Nx, sr, L=1, order=1, device='cpu'):
    if isinstance(u0, np.ndarray): u0 = torch.from_numpy(u0)
    if isinstance(f0, np.ndarray): f0 = torch.from_numpy(f0)
    dtype = u0.dtype
    ti = torch.arange(Nt, dtype=dtype).view(1,-1,1) / sr
    xi = torch.linspace(0,L,Nx, dtype=dtype).view(1,1,-1)
    new_Nx=int(order*Nx)
    xvals = np.linspace(0,L,new_Nx)
    _u0 = torch.from_numpy(ms.interpolate(
        u0.squeeze(0).numpy(), ti, xi, xvals
    )).narrow(0,0,1)
    _v = lossless_nonstiff_string(_u0, f0, Nt, new_Nx, sr, device=device).cpu().numpy()
    v = ms.interpolate(_v, ti, xvals, xi)
    return v


#============================== 
# Lossy Stiff String
#============================== 
def arange_like(x, max_k):
    ''' x: N-dim tensor
        -> N-dim tensor with shape (1, ..., 1, max_k)
    '''
    out_dim = [1] * (len(list(x.shape))-1) + [max_k]
    return torch.arange(max_k).view(out_dim).to(x.device)

def kappa_to_K(kappa_rel, gamma):
    K = gamma * kappa_rel
    return K

def T60_to_sigma(T60, gamma, K):
    # T60 = [[T60_freq_1, T60_1], [T60_freq_2, T60_2]]
    # sig1 == 0 if T60_freq_1 == T60_freq_2
    #------------------------------ 
    zeta1 = - gamma.pow(2) + (gamma.pow(4) + 4 * K.pow(2) * (2 * math.pi * T60[0,0]).pow(2)).pow(.5)
    zeta2 = - gamma.pow(2) + (gamma.pow(4) + 4 * K.pow(2) * (2 * math.pi * T60[1,0]).pow(2)).pow(.5)
    sig0 = - zeta2 / T60[0,1] + zeta1 / T60[1,1]
    sig0 = 6 * math.log(10) * sig0 / (zeta1 - zeta2)
    #------------------------------ 
    sig1 = 1 / T60[0,1] - 1 / T60[1,1]
    sig1 = 6 * math.log(10) * sig1 / (zeta1 - zeta2)
    #------------------------------ 
    #assert sig1.allclose(torch.zeros_like(sig1)), T60 # frequency independent
    return sig0

class RootFinder(object):
    def __init__(self, l, L, Nx, fn_type):
        super().__init__()
        self.l = l.narrow(1,0,1).flatten().detach().cpu().numpy()
        self.L = L
        self.fn_type = fn_type
        self.u0 = None
        self.x = None
        self.I = None

        self.mu2_to_mu1 = lambda mu2: np.sqrt(mu2**2 - 2*self.l)
        self.mu1_to_mu2 = lambda mu1: np.sqrt(mu1**2 + 2*self.l)

        self.f_even = lambda x: self.mu2_to_mu1(x) *  np.tan(self.mu2_to_mu1(x) * L/2) + x * np.tanh(x * L/2)
        self.f_odds = lambda x: x *  np.tan(self.mu2_to_mu1(x) * L/2) - self.mu2_to_mu1(x) * np.tanh(x * L/2)

        #self.s_min = -math.pi/2
        self.s_min = math.pi / 2
        #self.s_max = 20 * math.pi
        self.s_max = 100 * math.pi
        #self.s_max = Nx/2*math.pi/(L/2)
        self.s_res = int(1e6)
        mu_1, mu_2 = self.sweep(fn_type)
        self.set_mu(mu_1, mu_2)

        self.max_val, self.min_val = self.mu_2.max(), self.mu_2.min()

    def sweep(self, fn_type, peak_val=1, return_sweep=False):
        mu_1_sweep = np.linspace(self.s_min, self.s_max, self.s_res)
        mu_2_sweep = self.mu1_to_mu2(mu_1_sweep)
        if fn_type=='even':
            sweep_val = self.f_even(mu_2_sweep)
        else:
            sweep_val = self.f_odds(mu_2_sweep)
        sweep_val = np.abs(sweep_val).clip(max=peak_val)
        peak_train = peak_val - sweep_val
        peaks = scipy.signal.find_peaks(
            peak_train, height=0.1*peak_val, distance=math.pi/2)[0]
        if return_sweep:
            return [mu_1_sweep, sweep_val, peaks]
        else:
            initial_mu_1 = torch.from_numpy(mu_1_sweep[peaks])
            initial_mu_2 = torch.from_numpy(mu_2_sweep[peaks])
            initial_mu_1 = torch.sort(initial_mu_1)[0]
            initial_mu_2 = torch.sort(initial_mu_2)[0]
            return initial_mu_1, initial_mu_2

    def set_mu(self, mu_1=None, mu_2=None):
        if (mu_1 is not None) and (mu_2 is not None):
            self.mu_1 = torch.sort(mu_1)[0]
            self.mu_2 = torch.sort(mu_2)[0]
        elif mu_1 is not None:
            self.mu_1 = torch.sort(mu_1)[0]
            self.mu_2 = self.mu1_to_mu2(self.mu_1)
        elif mu_2 is not None:
            self.mu_2 = torch.sort(mu_2)[0]
            self.mu_1 = self.mu2_to_mu1(self.mu_2)
        else:
            assert False

    def find_freqs(self, fn, verbose=False, strict=True):
        result = scipy.optimize.least_squares(
            fn, self.mu_2.numpy(),
            #method='trf', bounds=(self.min_val, self.max_val), ftol=1e-15, xtol=None, gtol=None, x_scale='jac', f_scale=1e-5, loss='soft_l1',
            method='lm', ftol=MACHINE_EPS, xtol=MACHINE_EPS, gtol=MACHINE_EPS,
            verbose=2 if verbose else 0,
        )
        if strict:
            assert float(result.cost) < 1e-20, result
        self.set_mu(mu_2=torch.from_numpy(result.x))

    def cost_odds(self, b_t):
        return (sum(self.X_odds(b_t)) - self.u0).flatten()

    def cost_even(self, b_t):
        return (sum(self.X_even(b_t)) - self.u0).flatten()

    def X_odds_n(self, b_t, m1, m2):
        b_h = - np.sin(m1*self.I) / np.sinh(m2*self.I) * b_t
        tri = b_t *  np.sin(m1*self.x)
        hyp = b_h * np.sinh(m2*self.x)
        return tri + hyp

    def X_even_n(self, b_t, m1, m2):
        b_h = - np.cos(m1*self.I) / np.cosh(m2*self.I) * b_t
        tri = b_t *  np.cos(m1*self.x)
        hyp = b_h * np.cosh(m2*self.x)
        return tri + hyp

    def X_odds(self, b_t):
        out = []
        for i, (m1, m2) in enumerate(zip(self.mu_1, self.mu_2)):
            out.append(self.X_odds_n(b_t[i], m1, m2))
        return out

    def X_even(self, b_t):
        out = []
        for i, (m1, m2) in enumerate(zip(self.mu_1, self.mu_2)):
            out.append(self.X_even_n(b_t[i], m1, m2))
        return out

    def find_coeff(self, u0, x, I, verbose=False):
        self.u0 = u0.detach().cpu().numpy()
        self.x = x.detach().cpu().numpy()
        self.I = I
        if self.fn_type=='odds':
            init = [c_sin(m,self.x,self.u0,self.I) for m in self.mu_1.numpy()]
            fn = self.cost_odds
        if self.fn_type=='even':
            init = [c_cos(m,self.x,self.u0,self.I) for m in self.mu_1.numpy()]
            fn = self.cost_even
        result = scipy.optimize.least_squares(
            fn, init,
            method='lm', ftol=MACHINE_EPS, xtol=MACHINE_EPS, gtol=MACHINE_EPS,
            verbose=2 if verbose else 0,
        )
        if self.fn_type=='odds': self.X = self.X_odds(result.x)
        if self.fn_type=='even': self.X = self.X_even(result.x)



def c_sin(o, x, u0, I, dim=None):
    integral = u0 * np.sin(o*x)
    return (1/I) * np.mean(integral)

def c_cos(o, x, u0, I, dim=None):
    integral = u0 * np.cos(o*x)
    return (1/I) * np.mean(integral)

def lossy_stiff_solution(x, t, u0, f0, kappa_rel, t60, L=1, strict=True):
    '''  x: (Bs,  1, Nx)
         t: (Bs, Nt,  1)
        u0: (Bs,  1, Nx)
        f0: (Bs, Nt,  1)
    '''
    Bs,  _, Nx = x.shape
    _ , Nt,  _ = t.shape
    u = torch.zeros(Bs, Nt, Nx, dtype=u0.dtype, device=u0.device)

    gamma = 2*L * f0
    K = kappa_to_K(kappa_rel, gamma)
    assert K.gt(0).all(), [K.flatten(), kappa_rel.flatten()]
    l = gamma.pow(2) / (2 * K.pow(2))
    rf = {
        'even': RootFinder(l, L, Nx, fn_type='even'),
        'odds': RootFinder(l, L, Nx, fn_type='odds'),
    }

    def add_modes(fn, fn_type='even'):
        rf[fn_type].find_freqs(fn, strict=strict)
        rf[fn_type].find_coeff(u0, x, L/2)
        out = torch.zeros_like(u)

        sigma = T60_to_sigma(t60, gamma, K)
        mu_1 = rf[fn_type].mu_1
        shape_matrix = [dict()] * Bs
        for n, mu1_n in enumerate(mu_1):
            varsg = mu1_n**4 * K**2 + mu1_n**2 * gamma**2
            omega = torch.sqrt(varsg - sigma**2)
            sigma_t = -sigma*t
            omega_t =  omega*t
            T = torch.exp(sigma_t) * torch.cos(omega_t)
            X = rf[fn_type].X[n].to(T.device)
            out += X * T

            omega_t = omega_t.narrow(1,1,1)
            for b in range(Bs):
                omega_val = omega_t[b].flatten().item()
                amplitude = X[b].flatten(1).cpu().numpy() # (1,Nx,)
                shape_matrix[b].update({omega_val: amplitude})
        return out, mu_1, shape_matrix

    u_even, mu_even, shape_even = add_modes(rf['even'].f_even, 'even')
    u_odds, mu_odds, shape_odds = add_modes(rf['odds'].f_odds, 'odds')
    u += u_even
    u += u_odds

    shape_matrix = [dict()] * Bs
    for b in range(Bs):
        shape_matrix[b].update(shape_even[b])
        shape_matrix[b].update(shape_odds[b])

    mode_freq = []
    mode_amps = []
    for b in range(Bs):
        # sort by mode freq
        mode_dict = dict(sorted(shape_matrix[b].items(), key=lambda item: item[0]))
        # concatenate to numpy
        mode_frq = list(mode_dict.keys())
        mode_amp = list(mode_dict.values())
        mode_frq = np.array(mode_frq)[None,:] # (1,n_modes)
        mode_amp = np.concatenate(mode_amp, axis=0)[None,:,:] # (1,n_modes, Nx)
        mode_freq.append(mode_frq)
        mode_amps.append(mode_amp)
    mode_freq = np.concatenate(mode_freq, axis=0)
    mode_amps = np.concatenate(mode_amps, axis=0)
    return u, mode_freq, mode_amps

    #pb_even = rf['even'].sweep('even', return_sweep=True)
    #pb_odds = rf['odds'].sweep('odds', return_sweep=True)
    #mu_dict = dict(
    #    mu_e=mu_even, mu_o=mu_odds,
    #    pb_e=pb_even, pb_o=pb_odds,
    #)
    #return u, mu_dict


def lossy_stiff_string(u0, f0, kappa_rel, t60, Nt, Nx, sr, L=1, strict=True, device='cpu'):
    ''' u0: initial condition, u(x, 0)
        f0: fundamental frequency
        kappa_rel: relative stiffness
        t60: T60
        Nt: temporal grid size
        Nx: spatial grid size
        sr: temporal sampling rate
        L: maximal `x` value
    '''
    if isinstance(u0, np.ndarray): u0 = torch.from_numpy(u0)
    if isinstance(f0, np.ndarray): f0 = torch.from_numpy(f0)
    dtype = u0.dtype
    t = torch.arange(Nt, dtype=dtype).view(1,-1,1).to(device) / sr
    x = torch.linspace(-L/2,L/2,Nx, dtype=dtype).view(1,1,-1).to(device)
    u0 = u0.unsqueeze(0).to(device)
    f0 = f0.view(1,-1,1).to(device)
    u, mf, ma = lossy_stiff_solution(x, t, u0, f0, kappa_rel, t60, L, strict=strict)
    return u.squeeze(0), mf.squeeze(0), ma.squeeze(0)

def interpolated_stiff_string(u0, f0, kr, ts, Nt, Nx, sr, L=1, order=1, device='cpu'):
    _Nt, _Nx = u0.shape
    #if _Nt != 1: u0 = u0.narrow(Nt, 1)
    if _Nt != 1: u0 = u0[0][None,:]
    if isinstance(f0, np.ndarray): f0 = torch.from_numpy(f0)
    if isinstance(kr, np.ndarray): kr = torch.from_numpy(kr)
    if isinstance(ts, np.ndarray): ts = torch.from_numpy(ts)
    ti = torch.arange(Nt).view(1,-1,1) / sr
    xi = torch.linspace(-L/2,L/2,Nx).view(1,1,-1)
    new_Nx=int(order*Nx)
    xvals = np.linspace(-L/2,L/2,new_Nx)

    _u0 = torch.from_numpy(ms.interpolate1d(u0, xi, xvals))
    #_v, mu = lossy_stiff_string(_u0, f0, kr, ts, Nt, new_Nx, sr, device=device)
    #v = ms.interpolate(_v.cpu().numpy(), ti, xvals, xi)
    #return v, mu
    _v = lossy_stiff_string(_u0, f0, kr, ts, Nt, new_Nx, sr, device=device)[0]
    v = ms.interpolate(_v.cpu().numpy(), ti, xvals, xi)
    return v

def plot_difference(u, v, save_path, max_T=1000):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    w = u - v

    min_val = min(u.min(), v.min())
    max_val = max(u.max(), v.max())
    wval = max(np.abs(u).max(), np.abs(v).max())

    fig, ax = plt.subplots(figsize=(5,5), nrows=3, ncols=1)
    uspec = librosa.display.specshow(u.T[:,:max_T], ax=ax[0])
    vspec = librosa.display.specshow(v.T[:,:max_T], ax=ax[1])
    wspec = librosa.display.specshow(w.T[:,:max_T], ax=ax[2])
    uspec.set_clim([min_val, max_val])
    vspec.set_clim([min_val, max_val])
    wspec.set_clim([-wval/10, wval/10])

    ax[0].set_ylabel('$u$')
    ax[1].set_ylabel('$u_{\mathrm{exact}}$')
    ax[2].set_ylabel('$u - u_{\mathrm{exact}}$')
    ax[2].set_xlabel('time')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{save_path}')


    #t1 = 50
    #t2 = 100
    t1 = 5
    t2 = 20
    t3 = 1000
    fig, ax = plt.subplots(figsize=(5,5), nrows=4, ncols=2)
    ax[0,0].axhline(0, c='k', alpha=0.2)
    ax[0,0].plot(u[ 0], alpha=0.5, label="simulated")
    ax[0,0].plot(v[ 0], alpha=0.5, label="analytic")
    ax[0,1].fill_between(np.arange(w.shape[-1]), w[ 0], fc='k', alpha=0.2)
    ax[1,0].axhline(0, c='k', alpha=0.2)
    ax[1,0].plot(u[t1], alpha=0.5)
    ax[1,0].plot(v[t1], alpha=0.5)
    ax[1,0].plot(v[ 0], alpha=0.2) #
    ax[1,1].fill_between(np.arange(w.shape[-1]), w[t1], fc='k', alpha=0.2)
    ax[2,0].axhline(0, c='k', alpha=0.2)
    ax[2,0].plot(u[t2], alpha=0.5)
    ax[2,0].plot(v[t2], alpha=0.5)
    ax[2,0].plot(v[ 0], alpha=0.2) #
    ax[2,1].fill_between(np.arange(w.shape[-1]), w[t2], fc='k', alpha=0.2)
    ax[3,0].axhline(0, c='k', alpha=0.2)
    ax[3,0].plot(u[t3], alpha=0.5)
    ax[3,0].plot(v[t3], alpha=0.5)
    ax[3,0].plot(v[ 0], alpha=0.2) #
    ax[3,1].fill_between(np.arange(w.shape[-1]), w[t3], fc='k', alpha=0.2)

    fig.legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    save_name = save_path.split('/')[-1]
    plt.savefig(f'{save_path}'.replace(save_name, 'analytic-state.pdf'))



def plot_freqs(save_path, Nx, mu_e=None, mu_o=None, pb_e=None, pb_o=None, L=1):
    mu_1 = (2*torch.arange(Nx//2)+1)/2 * math.pi / (L/2)
    mu_2 = (2*torch.arange(Nx//2)+2)/2 * math.pi / (L/2)
    mu_i = torch.sort(torch.cat((mu_1, mu_2), dim=0))[0]

    fig, ax = plt.subplots(figsize=(7,3), nrows=3, height_ratios=[5,2,1])
    e_max = o_max = 0
    if pb_e is not None:
        mus = pb_e[0]
        val = pb_e[1]
        ax[0].plot(mus, val, 'b', alpha=0.5)
        for peak in pb_e[2]:
            ax[0].plot(mus[peak], val[peak], 'bx')
    if pb_o is not None:
        mus = pb_o[0]
        val = pb_o[1]
        ax[0].plot(mus, val, 'r', alpha=0.5)
        for peak in pb_o[2]:
            ax[0].plot(mus[peak], val[peak], 'rx')
    if mu_e is not None:
        for mu in mu_e:
            ax[1].axvline(mu, c='b')
        e_max = mu_e.max()
    if mu_o is not None:
        for mu in mu_o:
            ax[1].axvline(mu, c='r')
        o_max = mu_o.max()
    for mu in mu_i:
        ax[1].axvline(mu, c='k', alpha=0.5, ls=':')
        ax[2].axvline(mu, c='k')
    max_mu = max(e_max, o_max)
    ax[0].set_yscale('log')
    ax[0].set_xticks([])
    #ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    ax[0].set_xlim([-1,max_mu])
    ax[1].set_xlim([-1,max_mu])
    ax[2].set_xlim([-1,max_mu])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_path)





if __name__ == '__main__':
    import soundfile as sf
    from glob import glob
    from tqdm import tqdm

    import src.utils.audio as audio
    import src.utils.plot as plot
    import src.utils.fdm as fdm

    order = 32
    sr = 48000
    #---------- 
    dir_name = [
        'new-pluck',
    ]
    #---------- 
    plot_T = 10000

    subdirs = []
    for dn in dir_name:
        sd = sorted([d for d in glob(f'/data2/private/szin/dfdm/{dn}/*') if os.path.isdir(d) and not 'codes' in d])
        subdirs += sd

    # plot pluck
    iterator = tqdm(subdirs)
    iterator.set_description("Saving Analytic Solution")
    for subdir in iterator:
        try:
            ut, f0, kappa_rel, T60 = get_data(subdir)
        except FileNotFoundError as err:
            continue
        Nt, Nx = ut.shape
    
        # upsample for a constant spatial grid size
        Nx = 64
        k = 1 / sr
        xi = np.linspace(0,1,Nx)[None,:]
        with open(f"{subdir}/simulation_config.yaml", 'r') as f:
            constants = yaml.load(f, Loader=yaml.FullLoader)
        alpha = np.ones(1)
        theta_t  = constants["theta_t"]
        lambda_c = constants["lambda_c"]
        nx_t, _, nx_l, _ = fdm.get_derived_vars(
            torch.from_numpy(f0),
            torch.from_numpy(kappa_rel), k, theta_t, lambda_c,
            torch.from_numpy(alpha))[2:6]
        _ut = np.zeros((Nt, Nx))
        for t in range(Nt):
            _Nu = int(nx_t[t]) + 1
            _xu = np.linspace(0,1,_Nu)[None,:]
            _ut[t] += ms.interpolate1d(ut[t,:_Nu][None,:], _xu, xi)[0] # (time, Nx)
        ut = _ut

        iterator.set_postfix(directory=subdir, Nt=Nt, Nx=Nx, order=order)
        vt, mu = interpolated_stiff_string(
            ut, f0, kappa_rel, T60,
            Nt, Nx, sr, order=order, device='cuda:0')
        w = audio.state_to_wav(torch.from_numpy(vt[None,:,:])) # (Bs, Nt)
        np.savez_compressed(f'{subdir}/analytic.npz', ua=vt, order=order)
        sf.write(f'{subdir}/analytic.wav', w[0], samplerate=sr)
        plot_freqs(f'{subdir}/analytic-freqs.pdf', Nx, **mu)
        plot_difference(ut, vt, f'{subdir}/analytic.pdf', plot_T)
        plot.state_video(f'{subdir}',  [vt, ut], sr, trim_front=True, prefix='analytic')
        #============================== 
    
    
