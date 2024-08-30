import os
import shutil
import yaml
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import math

import src.utils.fdm as fdm
import src.utils.misc as ms
import src.utils.data as data
import src.utils.audio as audio
import src.model.analytic as analytic
from src.utils.analysis.frequency import compute_harmonic_parameters

def is_processed(directory, N):
    if not os.path.exists(directory): return False
    ut_list = glob(f"{directory}/ut-*.wav")
    ua_list = glob(f"{directory}/ua-*.wav")
    vt_list = glob(f"{directory}/vt.wav")
    parameters = f"{directory}/parameters.npz"
    if len(ut_list) != N: return False
    if len(ua_list) != N: return False
    if len(vt_list) != 1: return False
    if not os.path.exists(parameters): return False
    return True

def rms(x, eps=1e-18):
    mean_val = np.mean(x ** 2)
    return 1 if mean_val < eps else np.sqrt(np.mean(x ** 2))

def load_data(dirs):
    _sim = np.load(f"{dirs}/simulation.npz");     _sim_dict = dict()
    _str = np.load(f"{dirs}/string_params.npz");  _str_dict = dict()
    _bow = np.load(f"{dirs}/bow_params.npz");     _bow_dict = dict()
    _ham = np.load(f"{dirs}/hammer_params.npz");  _ham_dict = dict()

    for key in _sim.keys(): _sim_dict[key] = _sim[key]
    for key in _str.keys(): _str_dict[key] = _str[key]
    for key in _bow.keys(): _bow_dict[key] = _bow[key]
    for key in _ham.keys(): _ham_dict[key] = _ham[key]
    return _sim_dict, _str_dict, _bow_dict, _ham_dict

def remove_above_nyquist_mode(amplitudes, frequencies, sampling_rate):
    ''' amplitudes: (batch, Nt, n_harmoincs)
        frequencies: (batch, Nt, n_harmonics)
    '''
    aa = (frequencies < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa

def synth(freq, coef, damp, n_chunks=100):
    freqs = freq.chunk(n_chunks, 1)
    coefs = coef.chunk(n_chunks, 1)
    damps = damp.chunk(n_chunks, 1)
    lastf = torch.zeros_like(freqs[0])
    sols = []
    for f, c, d in zip(freqs, coefs, damps):
        fcs = f.cumsum(1) + lastf
        sol = (torch.cos(fcs) * c * d).sum(-1, keepdim=True)
        lastf = fcs.narrow(1,-1,1)
        sols.append(sol)
    return torch.cat(sols, 1)

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

def get_analytic_solution(u0, f0, kr, ts, sr, new_Nx, strict=True, device='cuda:0'):
    Nt, Nx = u0.shape

    if isinstance(u0, np.ndarray): u0 = torch.from_numpy(u0)
    if isinstance(f0, np.ndarray): f0 = torch.from_numpy(f0)
    if isinstance(kr, np.ndarray): kr = torch.from_numpy(kr)
    if isinstance(ts, np.ndarray): ts = torch.from_numpy(ts)
    dtype = u0.dtype
    ti = torch.arange(Nt, dtype=dtype).view(1,-1,1) / sr
    xi = torch.linspace(0,1,Nx, dtype=dtype).view(1,1,-1)
    xvals = np.linspace(0,1,new_Nx)
    _u0 = torch.from_numpy(ms.interpolate(
        u0.squeeze(0).numpy(), ti, xi, xvals
    )).narrow(0,0,1)

    _, mode_freq, mode_amps = analytic.lossy_stiff_string(_u0, f0, kr, ts, Nt, new_Nx, sr, strict=strict, device=device)

    return mode_freq, mode_amps


def save_upsampled_data(load_dir, save_dir, sr, Nx, strict=True):
    try:
        _sim, _str, _bow, _ham = load_data(load_dir)
    except FileNotFoundError as err:
        print("*"*30)
        print(f"File Not Found in {load_dir}")
        print("*"*30)
        return 0
   
    ut = _sim['state_u'] # (time, Nu)
    f0 = _str['f0']      # (time, )
    kr = _str['kappa'] 
    al = _str['alpha']
    ts = _str['T60']     # (2, 2)
    k = 1 / sr
    with open(f"{load_dir}/simulation_config.yaml", 'r') as f:
        constants = yaml.load(f, Loader=yaml.FullLoader)
    theta_t  = constants["theta_t"]
    lambda_c = constants["lambda_c"]
    nx_t, _, nx_l, _ = fdm.get_derived_vars(
        torch.from_numpy(f0),
        torch.from_numpy(kr), k, theta_t, lambda_c,
        torch.from_numpy(al))[2:6]

    dtype = np.float64 if ut.dtype == 'float64' else np.float32
    Nt, Nu = list(ut.shape)
    ki = max(min([5, int(min(nx_t))-1]), 1)
    xi = np.linspace(0,1,Nx)[None,:]
    ti = np.arange(Nt, dtype=dtype)[:,None] / sr

    ''' Upsample ut, zt to the spatial resolution with Nx
    '''
    if np.abs(f0 - np.mean(f0)).sum() < 0.1: # Hz
        # constant f0
        xu = np.linspace(0,1,Nu, dtype=dtype)[None,:]
        ut = ms.interpolate(ut, ti, xu, xi, kx=ki, ky=ki) # (time, Nx)
    else:
        # time-varying f0
        _ut = np.zeros((Nt, Nx))
        for t in range(Nt):
            _Nu = int(nx_t[t]) + 1
            _xu = np.linspace(0,1,_Nu, dtype=dtype)[None,:]
            _ut[t] += ms.interpolate1d(ut[t,:_Nu][None,:], _xu, xi, k=ki)[0] # (time, Nx)
        ut = _ut
    
    Na = 1024
    xa = np.linspace(0,1,Na, dtype=dtype)[None,:]
    xi = np.linspace(0,1,Nx)[None,:]

    pitch = torch.from_numpy(f0).cuda()
    kappa = torch.from_numpy(kr).view(1,1,1).cuda() # (1,1,1)
    t60_s = torch.from_numpy(ts[None,:,:]).cuda() # (1,2,2)
    times = torch.from_numpy(ti).view(1,-1,1).cuda() # (1,Nt,1)

    ''' Calculate analytic solution and downsample to the spatial resolution with Nx
    '''
    mode_freq, mode_amps = get_analytic_solution(ut, pitch, kr, ts, sr, new_Nx=Na, strict=strict) # (time, Na)
    mode_amps_nx = np.zeros((mode_amps.shape[0], Nx))
    for n in range(mode_amps.shape[0]):  # (n_modes, Na) --> (n_modes, Nx)
        mode_amps_nx[n] = ms.interpolate1d(mode_amps[n][None,:], xa, xi)[0]
    mode_amps = mode_amps_nx

    omega = pitch / sr * (2*math.pi)
    romg = (omega - omega[0]).view(1,-1,1)                       # ( 1, Nt,       1)
    mode_freq = torch.from_numpy( mode_freq[None,None,:]).cuda() # ( 1,  1, n_modes)
    mode_amps = torch.from_numpy((mode_amps.T)[:,None,:]).cuda() # (Nx,  1, n_modes)
    mode_freq_tv = mode_freq + romg  # ( 1, Nt, n_modes)

    sigma = T60_to_sigma(t60_s, pitch, 2*pitch*kappa)     # (1, Nt, 2)
    damping = torch.exp(- times * sigma.narrow(-1,0,1))     # (1, Nt, 1)

    mode_freq_hz = mode_freq_tv / (2*math.pi) * sr # (Hz)
    mode_amps_tv = remove_above_nyquist_mode(mode_amps, mode_freq_hz, sr)

    # (Nx, Nt, 1)
    ua = synth(mode_freq_tv, mode_amps_tv, damping).cpu()
    ua = ua.squeeze(-1).numpy().T # (time, Nx)

    mode_freq = mode_freq.squeeze().cpu()                # (n_modes,)
    mode_amps = mode_amps.squeeze().transpose(0,1).cpu() # (n_modes, Nx)

    uas = np.sum(ua, axis=1); _ua = uas / rms(uas)
    uts = np.sum(ut, axis=1); _ut = uts / rms(uts)
    ua_f0 = compute_harmonic_parameters(_ua, sr)['f0'] # (101,)
    ut_f0 = compute_harmonic_parameters(_ut, sr)['f0'] # (101,)

    gain = audio.ell_infty_normalize(ut.flatten())[1]
    u0 = ut[0,:][None,:]
    _str.pop("v0")
    _sim.pop("state_u")
    _sim.pop("state_z")

    vt = torch.from_numpy(ut).unsqueeze(0) # (Nt, Nx)
    vt = audio.state_to_wav(vt).squeeze(0).numpy() # (Nt)

    _sim.update(dict(ua_f0=ua_f0))
    _sim.update(dict(ut_f0=ut_f0))

    _sim.update(dict(mode_freq=mode_freq, mode_amps=mode_amps))
    _sim.update(dict(x=xi, t=ti))
    _sim.update(dict(ut=ut, ua=ua, vt=vt))
    _sim.update(dict(gain=gain.squeeze().item()))
    _str.update(dict(u0=u0))
    #---------- 
    _bow.update(dict(ph0_B=_bow.pop('phi_0')))
    _bow.update(dict(ph1_B=_bow.pop('phi_1')))
    _bow.update(dict(wid_B=_bow.pop('wid_B')))
    #---------- 
    _ham.update(dict(M_H=_ham.pop("M_r")))
    _ham.update(dict(a_H=_ham.pop("alpha")))
    #---------- 

    _ovr = {}
    _ovr.update(_sim)
    _ovr.update(_str)
    _ovr.update(_bow)
    _ovr.update(_ham)
    data.save(save_dir, _ovr)

def process(args):
    path_to_dir = os.path.join(args.task.root_dir, args.task.result_dir)
    subdirs = sorted([d for d in glob(f'{path_to_dir}/*') if os.path.isdir(d) and not 'codes' in d])

    if args.task.data_split > 1:
        subdirs = subdirs[args.task.split_n::args.task.data_split]

    iterator = tqdm(subdirs)
    iterator.set_description("Preprocess Data (Simulation --> Training)")
    for subdir in iterator:
        iterator.set_postfix(
            load_dir=subdir,
            Nx=args.task.Nx)
        save_dir = subdir.replace(args.task.result_dir, args.task.save_dir)
        os.makedirs(save_dir, exist_ok=True)

        if is_processed(save_dir, args.task.Nx): continue
        save_upsampled_data(subdir, save_dir, args.task.sr, args.task.Nx, args.task.strict)

