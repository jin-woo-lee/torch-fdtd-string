import os
import shutil
import yaml
from glob import glob
from tqdm import tqdm
import numpy as np
import torch

import src.utils.fdm as fdm
import src.utils.misc as ms
import src.utils.data as data
import src.utils.audio as audio
import src.model.analytic as analytic

def is_processed(directory, N):
    if not os.path.exists(directory): return False
    ut_list = glob(f"{directory}/ut-*.wav")
    zt_list = glob(f"{directory}/zt-*.wav")
    vt_list = glob(f"{directory}/vt.wav")
    parameters = f"{directory}/parameters.npz"
    if len(ut_list) != N: return False
    if len(zt_list) != N: return False
    if len(vt_list) != 1: return False
    if not os.path.exists(parameters): return False
    return True

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

#def save_data(dirs, _sim, _str, _bow, _ham):
#    np.savez_compressed(f"{dirs}/simulation.npz",    **_sim)
#    np.savez_compressed(f"{dirs}/string_params.npz", **_str)
#    np.savez_compressed(f"{dirs}/bow_params.npz",    **_bow)
#    np.savez_compressed(f"{dirs}/hammer_params.npz", **_ham)
#    return _sim, _str, _bow, _ham

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

    _v, om = analytic.lossy_stiff_string(_u0, f0, kr, ts, Nt, new_Nx, sr, strict=strict, device=device)
    _v = _v.cpu().numpy()
    om = om.cpu().numpy()
    return _v, om

def save_alpha_data(load_dir, save_dir, sr, Nx, batch_idx, strict=True):
    try:
        _sim, _str, _bow, _ham = load_data(load_dir)
    except FileNotFoundError as err:
        print("*"*30)
        print(f"File Not Found in {load_dir}")
        print("*"*30)
        return 0
   
    ut = _sim['state_u'] # (time, Nu)
    zt = _sim['state_z'] # (time, Nz)
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
    _,  Nz = list(zt.shape)
    #ki = min([5, Nu-1, Nz-1])
    ki = min([5, int(min(nx_t))-1, int(min(nx_l))-1])
    xi = np.linspace(0,1,Nx)[None,:]
    ti = np.arange(Nt, dtype=dtype)[:,None] / sr

    ''' Upsample ut, zt to the spatial resolution with Nx
    '''
    if np.abs(f0 - np.mean(f0)).sum() < 0.1: # Hz
        # constant f0
        xu = np.linspace(0,1,Nu, dtype=dtype)[None,:]
        xz = np.linspace(0,1,Nz, dtype=dtype)[None,:]
        ut = ms.interpolate(ut, ti, xu, xi, kx=ki, ky=ki) # (time, Nx)
        zt = ms.interpolate(zt, ti, xz, xi, kx=ki, ky=ki) # (time, Nx)
    else:
        # time-varying f0
        _ut = np.zeros((Nt, Nx))
        _zt = np.zeros((Nt, Nx))
        for t in range(Nt):
            _Nu = int(nx_t[t]) + 1
            _Nz = int(nx_l[t]) + 1
            _xu = np.linspace(0,1,_Nu, dtype=dtype)[None,:]
            _xz = np.linspace(0,1,_Nz, dtype=dtype)[None,:]
            _ut[t] += ms.interpolate1d(ut[t,:_Nu][None,:], _xu, xi, k=ki)[0] # (time, Nx)
            _zt[t] += ms.interpolate1d(zt[t,:_Nz][None,:], _xz, xi, k=ki)[0] # (time, Nx)
        ut = _ut
        zt = _zt
    
    Na = 1024
    xa = np.linspace(0,1,Na, dtype=dtype)[None,:]
    xi = np.linspace(0,1,Nx)[None,:]

    omega = np.empty(0)
    if batch_idx == 'a':
        ''' Calculate analytic solution and downsample to the spatial resolution with Nx
        '''
        ua, omega = get_analytic_solution(ut, f0, kr, ts, sr, new_Nx=Na, strict=strict) # (time, Na)
        ua = ms.interpolate(ua, ti, xa, xi)       # (time, Nx)
        ut = ua
        zt = np.zeros(ut.shape)

    gain = audio.ell_infty_normalize(ut.flatten())[1]
    u0 = ut[0,:][None,:]
    z0 = zt[0,:][None,:]
    _str.pop("v0")
    _sim.pop("state_u")
    _sim.pop("state_z")

    vt = torch.from_numpy(ut + zt).unsqueeze(0) # (Nt, Nx)
    vt = audio.state_to_wav(vt).squeeze(0).numpy() # (Nt)

    _sim.update(dict(omega=omega))
    _sim.update(dict(x=xi, t=ti))
    _sim.update(dict(ut=ut, zt=zt, vt=vt))
    _sim.update(dict(gain=gain.squeeze().item()))
    _str.update(dict(u0=u0, z0=z0))
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
    save_dirs = []
    # plot pluck
    n_span = 0
    iterator = tqdm(subdirs)
    iterator.set_description("Preprocess Data (Simulation --> Training)")
    for load_dir in iterator:
        # load_dir: {ROOT_DIR}/{PROJ_DIR}/{ITER_ID}-{BATCH_IDX}
        root_dir = '/'.join(load_dir.split('/')[:-2])
        proj_dir = load_dir.split('/')[-2]
        iter_id, batch_idx = load_dir.split('/')[-1].split('-')

        iter_dir = '/'.join([ root_dir, args.task.save_dir, iter_id, ])

        if iter_dir not in save_dirs:
            save_dirs.append(iter_dir)

        # save_dir: {ROOT_DIR}/{SAVE_DIR}/{ITER_ID}/{BATCH_IDX}
        save_dir = '/'.join([ iter_dir, batch_idx, ])
        asol_dir = '/'.join([ iter_dir, 'a', ])
        n_span = max(n_span, int(batch_idx)+2)
        if int(batch_idx) == 0:
            if not is_processed(asol_dir, args.task.Nx):
                os.makedirs(asol_dir, exist_ok=True)
                iterator.set_postfix(
                    load_dir=load_dir,
                    Nx=args.task.Nx)
                # save analytic solution
                save_alpha_data(load_dir, asol_dir,
                    args.task.sr, args.task.Nx,
                    'a', args.task.strict)

        if is_processed(save_dir, args.task.Nx): continue
        os.makedirs(save_dir, exist_ok=True)
        iterator.set_postfix(
            load_dir=load_dir,
            Nx=args.task.Nx)
        save_alpha_data(load_dir, save_dir,
            args.task.sr, args.task.Nx,
            batch_idx, args.task.strict)


    # filter saved data
    iterator = tqdm(save_dirs)
    iterator.set_description("Filter Training Data")
    for save_dir in iterator:
        subdirs = sorted(glob(f"{save_dir}/*"))
        if len(subdirs) < n_span:
            print(len(subdirs))
            print(f"{save_dir} ccontains incomplete sub-directories")
            print(subdirs)
            print(f"deleting tree...")
            shutil.rmtree(save_dir, ignore_errors=True)
            print("done")
            print("-"*30)



