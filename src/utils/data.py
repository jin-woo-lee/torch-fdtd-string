import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import src.utils.misc as ms
import glob

def load(dir_path, n_subsample=None, sr=48000, wav_keys=['ut', 'zt', 'ua'], subsample_method='sequential'):
    path_wav = sorted(glob.glob(f"{dir_path}/*.wav"))
    _ovr = {}
    for prefix in wav_keys:
        _wav = []
        _wav_path_unsrt = glob.glob(f"{dir_path}/{prefix}-*.wav")
        max_N = len(_wav_path_unsrt)
        _wav_paths = [f"{dir_path}/{prefix}-{i}.wav" for i in range(max_N)]
        if n_subsample is not None:
            if subsample_method == 'random':
                # randomly sample `n_subsample` number of wavs
                assert isinstance(n_subsample, int), n_subsample
                if max_N < n_subsample:
                    # with replacement
                    x_idx = np.random.randint(0, max_N, size=n_subsample, )
                else:
                    # without replacement
                    x_idx = np.random.permutation(max_N)[:n_subsample]
            else:
                r = np.random.randint(0, max_N-n_subsample)
                x_idx = np.array([r + i for i in range(n_subsample)])
            _wav_paths = [_wav_paths[i] for i in x_idx]
        for path in _wav_paths: _wav.append(sf.read(path)[0][:,None])
        _ovr[prefix] = np.concatenate(_wav, 1)

    _res = np.load(f"{dir_path}/parameters.npz")
    for key in _res.keys():
        _ovr[key] = _res[key]
    return _ovr

def save(dir_path, data_dict, sr=48000, chunk_length=.1):
    cl = int(sr * chunk_length)
    new_data_dict = data_dict.copy()
    for key, data in data_dict.items():
        file_path = f"{dir_path}/{key}"
        data = data_dict[key]
        if isinstance(data, float) or isinstance(data, int):
            continue
        data = data.squeeze()
        data_dim = len(list(data.shape))
        if key in ['ut', 'zt', 'ua']:
            Nt, Nx = data.shape
            assert min(Nt, Nx) > 1, [key, data.shape]
            for xi in range(Nx):
                sf.write(file_path+f"-{xi}.wav", data[:,xi], samplerate=sr, subtype='PCM_24')
            new_data_dict.pop(key)

    np.savez_compressed(f"{dir_path}/parameters.npz", **new_data_dict)

def set_length(x, size, method='pad', mode='linear', idx_x=None):
    if method == 'interpolate':
        x_shape = list(x.shape)
        if x_shape[-1] == size:
            return x
        new_shape = x_shape[:-1] + [size]; res = 3 - len(x_shape)
        unsqueezed_shape = [1]*res + x_shape if len(x_shape) < 3 else x_shape
        x = x.view(unsqueezed_shape)
        return F.interpolate(x, size=size, mode=mode).view(new_shape)

    elif method == 'pad':
        x_shape = list(x.shape)
        assert x_shape[-1] <= size, f"set Nx (={size}) geq to {x_shape[-1]}. To do this, set smaller args.task.f0_inf."
        if x_shape[-1] == size:
            return x
        new_shape = x_shape[:-1] + [size]
        new_x = torch.zeros(new_shape, device=x.device, dtype=x.dtype)
        new_x[...,:x_shape[-1]] = x
        return new_x

    elif method == 'random':
        assert idx_x is not None, idx_x
        new_x = ms.batched_index_select(x, -1, idx_x) # (Bs, Nt, size)
        return new_x

    else:
        assert False, method

def stack_batch(batch, Nx, Nt=None, Nr=None, sr=48000,
                #x_method='interpolate',  t_method='interleave',
                #x_method='interpolate',  t_method='sequential',
                x_method='random',  t_method='sequential',
                start_time=None, end_time=None):
    assert x_method in ['interpolate', 'pad', 'random'], x_method
    assert t_method in ['interpolate', 'sequential', 'interleave'], t_method
    ''' interpolate: conduct linear interpolation to subsample
        sequential : sequentially subsample without interleave
        interleave : subsample with uniform interleaving 
    '''
    keys = batch[0].keys()
    stacked_data_dict = dict()
    Bs = len(batch)

    idx_x = None
    if x_method == 'random':
        ut_shape = list(batch[0]['ut'].shape)
        idx_x = ms.random_index(ut_shape[-1], Nx)

    T = batch[0]['ut'].shape[0]
    if Nt is not None:
        if start_time is None:
            st = np.random.randint(T-Nt, size=Bs)
        else:
            assert isinstance(start_time, float), start_time
            st = int(start_time * sr) * np.ones(Bs, dtype=int)
        if end_time is None:
            et = np.random.randint(st+Nt, T, size=Bs)
            # let (et-st) to be divisible by Nt,
            # to match the shape for `interleave`
            et = Nt * ((et - st) // Nt) + st
        else:
            assert isinstance(end_time, float), end_time
            et = int(end_time * sr) * np.ones(Bs, dtype=int)
    else:
        st = np.zeros(Bs, dtype=int); Nt = T
        et = T * np.ones(Bs, dtype=int)

    time_varying_vars  = ['ut', 'zt', 'f0', 'Nu', 'Nz']
    time_varying_vars += ['x_B', 'v_B', 'F_B', 'wid_B', ]
    time_varying_vars += ['v_H', 'u_H', 'uf0']
    time_varying_vars += ['uat', 'uar', 'tt', 'tr']

    space_varying_vars  = ['ut', 'zt', 'uat', 'uar', 'u0', 'z0', 'xt', 'xr']

    for key in keys:
        data_list = [torch.from_numpy(x[key]) for x in batch]

        # randomize temporal initial point
        if key in time_varying_vars:
            if key in ['ut', 'zt', 'uf0', 'uat', 'tt']:
                # these variables will be used along with `ut` and `zt`
                # set whole length by `Nt`
                TL = Nt
            else:
                # these variables will be used along with `ur` and `zr`
                # set whole length by `Nr`
                TL = Nr
            ''' batch * (time, space) '''
            if t_method == 'sequential':
                data_list = [x.narrow(0,st[i],TL) for i, x in enumerate(data_list)] if TL is not None else None
            elif t_method == 'interpolate':
                data_list = [x.narrow(0,st[i],T-st[i]) for i, x in enumerate(data_list)]
                if len(list(data_list[0].shape)) < 2:
                    data_list = [set_length(x, TL, t_method) for i, x in enumerate(data_list)] if TL is not None else None
                else:
                    data_list = [set_length(x.transpose(0,1), TL, t_method).transpose(0,1) for i, x in enumerate(data_list)] if TL is not None else None
            elif t_method == 'interleave':
                data_list = [x.narrow(0,st[i],et[i]-st[i])[0::(et[i]-st[i]) // TL] for i, x in enumerate(data_list)] if TL is not None else None
            else:
                assert False, t_method

        # interpolate to the maximal spatial grid size
        ''' batch * (time, space) '''
        if key in space_varying_vars:
            data_list = [set_length(x, Nx, x_method, idx_x=idx_x) for x in data_list]

        data_batch = torch.stack(data_list)
        stacked_data_dict.update({key: data_batch})

    # randomized input
    if Nr is not None:
        rand_t = stacked_data_dict['tr'] \
               + torch.rand(size=(Bs,1,1)) / sr
        rand_x = stacked_data_dict['xr'] \
               + torch.rand(size=(Bs,1,1)) / Nx

        template = torch.ones_like(rand_t * rand_x) # (Bs, Nt, Nx)
        rand_t = (rand_t * template).requires_grad_()
        rand_x = (rand_x * template).requires_grad_()
        stacked_data_dict.update(dict(xr=rand_x, tr=rand_t))

    return stacked_data_dict


