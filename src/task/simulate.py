import os
import time
import glob
import tqdm
import torch
import numpy as np
import soundfile as sf
from torch.utils.cpp_extension import load as cpp_load

import src.model.simulator as simulator
import src.utils.fdm as fdm
import src.utils.misc as ms
import src.utils.plot as plot
import src.utils.audio as audio

def process(
    root_dir, state_u, state_z,
    string_params, bow_params, hammer_params,
    bow_mask, hammer_mask,
    consts, Nt, chunk_size,
    save_path=None,
    skip_nan=True,
    relative_order=4,
    surface_integral=False,
    manufactured=False,
):

    cpp_dir = f'{root_dir}/src/model/cpp'
    os.makedirs(f"{cpp_dir}/build", exist_ok=True)
    cpp_files = list(glob.glob(f"{cpp_dir}/*.cpp"))
    simulator = cpp_load(
        name="forward_fn",
        sources=cpp_files,
        verbose=False, #set verbose=True to see the build log.
        build_directory=f'{cpp_dir}/build',
    )

    def chunk(x,n,size,axis=1):
        if not isinstance(x, torch.Tensor):
            return x
        if len(list(x.shape)) > 1:
            if x.size(axis) > 2:
                x = x.narrow(axis, n, size)
        return x
    def chunk_params(params, n, chunk_size):
        params = list(params)
        params[-1] = chunk_size  # Nt -> chunk_size
        for i, p in enumerate(params):
            if isinstance(p, torch.Tensor):
                params[i] = chunk(p,n,chunk_size)
            elif not (isinstance(p, tuple) or isinstance(p, list)):
                pass
            else:
                params[i] = tuple([chunk(pp,n,chunk_size) for pp in list(p)])
        return params

    cn = 0
    total_uout = []
    total_zout = []
    total_v_r_out = []
    total_F_H_out = []
    total_u_H_out = []
    while cn < Nt-2:
        output_size = min(chunk_size, state_u.size(1) - cn)
        outputs = simulator.forward_fn(
            *chunk_params(
                (state_u, state_z,
                string_params, bow_params, hammer_params,
                bow_mask, hammer_mask,
                consts,
                relative_order, surface_integral,
                manufactured, cn,
                Nt), # keep this `Nt` as the last argument
                cn, output_size,
            )
        )
        uout, zout, c_state_u, c_state_z, v_r_out, F_H_out, u_H_out, sig0, sig1 = outputs

        # save chunked output
        state_u[:,cn+2:cn+output_size,:] = c_state_u[:,2:2+output_size,:]
        state_z[:,cn+2:cn+output_size,:] = c_state_z[:,2:2+output_size,:]
        total_uout.append(uout.narrow(1,2,output_size-2)) # (batch, time)
        total_zout.append(zout.narrow(1,2,output_size-2)) # (batch, time)
        total_v_r_out.append(v_r_out.narrow(1,2,output_size-2)) # (batch, time)
        total_F_H_out.append(F_H_out.narrow(1,2,output_size-2)) # (batch, time)
        total_u_H_out.append(u_H_out.narrow(1,2,output_size-2)) # (batch, time)

        cn += chunk_size-2

        # check chunked output
        state_is_nan = torch.isnan(c_state_u.flatten(1).sum(-1))
        if not skip_nan:
            assert not state_is_nan.any(), state_is_nan.nonzero() # print indices with nan-values

        if save_path is not None:
            _total_uout = torch.cat(total_uout, dim=1)
            _total_zout = torch.cat(total_zout, dim=1)
            for b in range(_total_uout.size(0)):
                if not state_is_nan[b]:
                    p = save_path.split('/')
                    sr = int(p.pop(-1)); sp = '/'.join(p)
                    os.makedirs(f"{sp}-{b}", exist_ok=True)
                    bitrate = 'PCM_16'
                    sf.write(f'{sp}-{b}/output-u.wav', _total_uout[b].cpu(), sr, subtype=bitrate)
                    sf.write(f'{sp}-{b}/output-z.wav', _total_zout[b].cpu(), sr, subtype=bitrate)
                    sf.write(f'{sp}-{b}/output.wav'  , _total_uout[b].cpu()\
                                                     + _total_zout[b].cpu(), sr, subtype=bitrate)
    total_uout = torch.cat(total_uout, dim=1)
    total_zout = torch.cat(total_zout, dim=1)
    total_v_r_out = torch.cat(total_v_r_out, dim=1)
    total_F_H_out = torch.cat(total_F_H_out, dim=1)
    total_u_H_out = torch.cat(total_u_H_out, dim=1)
    sig0_out = sig0 # (batch, 1, 1)
    sig1_out = sig1 # (batch, 1, 1)

    return total_uout, total_zout, \
           state_u, state_z, \
           total_v_r_out, total_F_H_out, total_u_H_out, \
           sig0_out, sig1_out

def simulate(
    root_dir, model_name, sr, theta_t, length, batch_size, f0_inf, alpha_inf, lambda_c,
    cpu=False, load_config=None, chunk_length=-1,
    save_path=None,
    string_kwargs=dict(),
    hammer_kwargs=dict(),
    bow_kwargs=dict(),
    skip_nan=True,
    precision='single',
    relative_order=4,
    surface_integral=False,
    randomize_each='batch',
    manufactured=False,
):
    # Global parameters
    k = 1 / sr
    total_size = int(length*sr)
    chunk_size = total_size if chunk_length < 0 else int(chunk_length * sr)

    # initialize models with random parameters
    if model_name.endswith('pluck'):
        pluck_batch = True
    elif model_name == 'random':
        pluck_batch = None
    else:
        pluck_batch = False

    bow_mask, hammer_mask = [m.bool() for m in ms.get_masks(model_name, batch_size)]

    pluck_mask = torch.logical_not(torch.logical_or(bow_mask, hammer_mask))
    string = simulator.String(
        k, theta_t, lambda_c, sr, length, f0_inf, alpha_inf, batch_size, precision,
        pluck_batch, pluck_mask, hammer_mask, 
        randomize_each, manufactured,
        **string_kwargs,
    )
    bow    = simulator.Bow(sr, length, batch_size, precision,
        randomize_each,
        **bow_kwargs)
    hammer = simulator.Hammer(sr, length, batch_size, precision, k,
        randomize_each,
        **hammer_kwargs)

    if load_config is not None:
        files = glob.glob(f"{load_config}/*.npy")
        for npy_path in files:
            val = np.load(npy_path)
            if val.shape[-1] < total_size:
                res = total_size - val.shape[-1]
                val = np.pad(val, (0,res), mode='edge')
            else:
                val = val[:total_size]

            target_model, target_param = npy_path.split('/')[-1].split('.')[0].split('-')
            if target_model.lower() == 'string':
                string.dump_parameter(target_param, val)
            elif target_model.lower() == 'bow':
                bow.dump_parameter(target_param, val)
            elif target_model.lower() == 'hammer':
                hammer.dump_parameter(target_param, val)
            else:
                raise NotImplementedError(target_model)

    if not cpu:
        string = string.cuda()
        bow = bow.cuda()
        hammer = hammer.cuda()
        bow_mask = bow_mask.cuda()
        hammer_mask = hammer_mask.cuda()


    string_params = string()
    bow_params    = bow()
    hammer_params = hammer()
    consts = [k, theta_t, lambda_c]

    state_u = string_params.pop(0)
    state_z = string_params.pop(0)
    target_f0 = string_params.pop(-1)
    Nt = string.Nt

    outputs = process(
        root_dir, state_u, state_z,
        string_params, bow_params, hammer_params,
        bow_mask, hammer_mask,
        consts, Nt, chunk_size,
        save_path,
        skip_nan,
        relative_order,
        surface_integral,
        manufactured,
    )
    uout, zout, state_u, state_z, v_r_out, F_H_out, u_H_out, sig0, sig1 = outputs

    return [uout, zout, state_u, state_z, v_r_out, F_H_out, u_H_out, sig0, sig1], \
           [string_params, bow_params, hammer_params, consts, target_f0], \
           [bow_mask, hammer_mask, pluck_mask]

def run(args, save_dir, model_name, n_samples):
    ''' save_dir   : save directory path
        model_name : name of the model to run
        n_samples  : number of data to simulate
    '''
    sr = args.task.sr
    #theta_t = 0.5 + 2/(np.pi**2) if args.task.theta_t is None else args.task.theta_t
    if args.task.sampling_kappa == 'fix':
        kappa_max = [args.task.string_condition[num]['kappa_fixed'] for num in range(len(args.task.string_condition)) if 'kappa_fixed' in args.task.string_condition[num].keys()][0]
    else:
        kappa_max = [args.task.string_condition[num]['kappa_max'] for num in range(len(args.task.string_condition)) if 'kappa_max' in args.task.string_condition[num].keys()][0]
    if args.task.sampling_f0 == 'fix':
        f0_min = [args.task.string_condition[num]['f0_fixed'] for num in range(len(args.task.string_condition)) if 'f0_fixed' in args.task.string_condition[num].keys()][0]
    else:
        f0_min = [args.task.string_condition[num]['f0_min'] for num in range(len(args.task.string_condition)) if 'f0_min' in args.task.string_condition[num].keys()][0]
    theta_t = fdm.get_theta(kappa_max, f0_min, sr) if args.task.theta_t is None else args.task.theta_t

    string_kwargs = dict(
        sampling_f0     = 'random' if args.task.sampling_f0     is None else args.task.sampling_f0,
        sampling_kappa  = 'random' if args.task.sampling_kappa  is None else args.task.sampling_kappa,
        sampling_alpha  = 'random' if args.task.sampling_alpha  is None else args.task.sampling_alpha,
        sampling_pickup = 'random' if args.task.sampling_pickup is None else args.task.sampling_pickup,
        sampling_T60    = 'random' if args.task.sampling_T60    is None else args.task.sampling_T60,
        precorrect      = 'random' if args.task.precorrect      is None else args.task.precorrect,
    )
    for condition_dict in args.task.string_condition:
        key = list(condition_dict.keys())[0]
        val = list(condition_dict.values())[0]
        if val is not None:
            string_kwargs.update({key: val})
    for condition_dict in args.task.pluck_condition:
        key = list(condition_dict.keys())[0]
        val = list(condition_dict.values())[0]
        if val is not None:
            string_kwargs.update({key: val})

    hammer_kwargs = dict()
    for condition_dict in args.task.hammer_condition:
        key = list(condition_dict.keys())[0]
        val = list(condition_dict.values())[0]
        if val is not None:
            hammer_kwargs.update({key: val})
    bow_kwargs = dict()
    for condition_dict in args.task.bow_condition:
        key = list(condition_dict.keys())[0]
        val = list(condition_dict.values())[0]
        if val is not None:
            bow_kwargs.update({key: val})

    time_log = []
    iterator = tqdm.tqdm(range(n_samples))
    iterator.set_description('Preparing simulation data')
    for it in iterator:
        save_path = None
        dx = str(it) if not args.task.randomize_name else ms.random_str()
        if args.task.write_during_process:
            save_path = f'{save_dir}/{dx}/{sr}'

        if args.task.measure_time and args.proc.cpu:
            torch.set_num_threads(1)
        if args.task.measure_time and not args.proc.cpu:
            s_time = torch.cuda.Event(enable_timing=True)
            e_time = torch.cuda.Event(enable_timing=True)
        else:
            st = time.time()
        with torch.no_grad():
            if args.task.measure_time and not args.proc.cpu:
                s_time.record()

            results, params, masks = simulate(
                args.cwd, model_name,
                sr, theta_t, args.task.length, args.task.batch_size,
                args.task.f0_inf, args.task.alpha_inf, args.task.lambda_c,
                args.proc.cpu, 
                args.task.load_config,
                args.task.chunk_length,
                save_path,
                string_kwargs,
                hammer_kwargs,
                bow_kwargs,
                args.task.skip_nan,
                args.task.precision,
                args.task.relative_order,
                args.task.surface_integral,
                args.task.randomize_each,
                args.task.manufactured,
            )

            if args.task.measure_time and not args.proc.cpu:
                e_time.record()
        state_u = results[2]
        state_z = results[3]
        if args.task.measure_time and not args.proc.cpu:
            torch.cuda.synchronize()
            proc_time = s_time.elapsed_time(e_time) / 1000 # ms -> s
            time_log_name = 'gpu_time'
        else:
            proc_time = time.time() - st
            time_log_name = 'cpu_time'
        time_log.append(proc_time)

        uout, zout, state_u, state_z, v_r_out, F_H_out, u_H_out, sig0, sig1 = results
        string_params, bow_params, hammer_params, consts, target_f0 = params
        bow_mask, hammer_mask, pluck_mask = masks

        iterator.set_postfix(
            avg_time=sum(time_log) / len(time_log),
            proc_time=proc_time,
        )

        with open(f"{save_dir}/{time_log_name}.txt", 'a') as f:
            f.write(f"{dx}\t{proc_time:.2f}\n")

        state_is_nan = torch.isnan(uout.flatten(1).sum(-1))
        uout = uout * state_is_nan.logical_not().unsqueeze(-1)
        is_silent = audio.dB_RMS(uout).le(args.task.silence_threshold)

        kappa = string_params[0].unsqueeze(-1)
        alpha = string_params[1].unsqueeze(-1)
        f0 = string_params[5]
        _, _, Nx_t, _, Nx_l, _ = fdm.get_derived_vars(
            f0=f0, kappa_rel=kappa, alpha=alpha,
            k=1/sr, theta_t=theta_t, lambda_c=args.task.lambda_c)

        uout = uout.squeeze(-1).cpu().numpy()
        zout = zout.squeeze(-1).cpu().numpy()
        state_u = state_u.cpu().numpy()
        state_z = state_z.cpu().numpy()
        v_r_out = v_r_out.cpu().numpy()
        F_H_out = F_H_out.cpu().numpy()
        u_H_out = u_H_out.cpu().numpy()
        sig0 = sig0.view(-1).cpu().numpy() # (batch_size)
        sig1 = sig1.view(-1).cpu().numpy() # (batch_size)
        Nx_t = Nx_t.cpu().numpy()
        Nx_l = Nx_l.cpu().numpy()

        string_params = [item.cpu().numpy() for item in string_params]
        bow_params    = [item.cpu().numpy() for item in bow_params]
        hammer_params = [item.cpu().numpy() for item in hammer_params]
        consts = [item.cpu().numpy() if isinstance(item, torch.Tensor) else item for item in consts]
        target_f0 = target_f0.cpu().numpy()

        bow_mask    = bow_mask.cpu().numpy()
        hammer_mask = hammer_mask.cpu().numpy()
        pluck_mask  = pluck_mask.cpu().numpy()

        wout = uout + zout
        if args.task.plot:
            tau = 1
            udif = (uout[:,tau:] - uout[:,:-tau]) / (tau / sr)
            zdif = (zout[:,tau:] - zout[:,:-tau]) / (tau / sr)
            wdif = (wout[:,tau:] - wout[:,:-tau]) / (tau / sr)
            uddf = (uout[:,2*tau:] - 2*uout[:,tau:-tau] + uout[:,:-2*tau]) / (2*tau / sr)
            zddf = (zout[:,2*tau:] - 2*zout[:,tau:-tau] + zout[:,:-2*tau]) / (2*tau / sr)
            wddf = (wout[:,2*tau:] - 2*wout[:,tau:-tau] + wout[:,:-2*tau]) / (2*tau / sr)
            uout_min, uout_max = ms.get_minmax(uout)
            zout_min, zout_max = ms.get_minmax(zout)
            wout_min, wout_max = ms.get_minmax(wout)
            udif_min, udif_max = ms.get_minmax(udif)
            zdif_min, zdif_max = ms.get_minmax(zdif)
            wdif_min, wdif_max = ms.get_minmax(wdif)
            uddf_min, uddf_max = ms.get_minmax(uddf)
            zddf_min, zddf_max = ms.get_minmax(zddf)
            wddf_min, wddf_max = ms.get_minmax(wddf)

        os.makedirs(f"{save_dir}", exist_ok=True)
        for b in range(args.task.batch_size):
            if state_is_nan[b]:
                # skip nan result
                continue
            if args.task.skip_silence and is_silent[b]:
                # skip silent result
                continue

            excitation_types = []
            if bow_mask[b]:
                excitation_types += ['bow']
            if hammer_mask[b]:
                excitation_types += ['hammer']
            if pluck_mask[b]:
                excitation_types += ['pluck']
            excitation_type = ','.join(excitation_types)

            # (batch, time, space)
            # print(state_u.shape, Nx_t) # +1 for the boundary condition
            state_u_b = state_u[b,:,:np.max(Nx_t[b]).astype(int)+1]
            state_z_b = state_z[b,:,:np.max(Nx_l[b]).astype(int)+1]
            simulation_dict = dict(
                uout=uout[b], zout=zout[b], state_u=state_u_b, state_z=state_z_b,
                v_r_out=v_r_out[b], F_H_out=F_H_out[b], u_H_out=u_H_out[b],
                bow_mask=bow_mask[b], hammer_mask=hammer_mask[b], pluck_mask=pluck_mask[b],
                Nx_t=Nx_t[b], Nx_l=Nx_l[b],
                sig0=sig0[b], sig1=sig1[b],
            )
            string_params_b = [item[b] for item in string_params]
            string_params_b += [target_f0[b]]
            hammer_params_b = [item[b] for item in hammer_params]
            bow_params_b    = [item[b] for item in bow_params]

            overall_results = dict()
            overall_results.update(**simulation_dict)
            overall_results.update(dict(string_params = string_params_b))
            overall_results.update(dict(hammer_params = hammer_params_b))
            overall_results.update(dict(bow_params    = bow_params_b))

            if args.task.plot or args.task.plot_state or args.task.save:
                os.makedirs(f"{save_dir}/{dx}-{b}", exist_ok=True)
            bitrate = 'PCM_24' if args.task.precision == 'double' else 'PCM_16'
            if args.task.normalize_output:
                #uout_nrm, gain = audio.rms_normalize(uout[b])
                uout_nrm, gain = audio.ell_infty_normalize(uout[b]) # \ell_\infty-norm
                zout_nrm = gain * zout[b]; wout_nrm = uout_nrm + zout_nrm
                sf.write(f'{save_dir}/{dx}-{b}/output-u.wav', uout_nrm, sr, subtype=bitrate)
                sf.write(f'{save_dir}/{dx}-{b}/output-z.wav', zout_nrm, sr, subtype=bitrate)
                sf.write(f'{save_dir}/{dx}-{b}/output.wav',   wout_nrm, sr, subtype=bitrate)
            else:
                sf.write(f'{save_dir}/{dx}-{b}/output-u.wav', uout[b], sr, subtype=bitrate)
                sf.write(f'{save_dir}/{dx}-{b}/output-z.wav', zout[b], sr, subtype=bitrate)
                sf.write(f'{save_dir}/{dx}-{b}/output.wav',   wout[b], sr, subtype=bitrate)

            if args.task.plot:
                plot.simulation_data(f'{save_dir}/{dx}-{b}', **overall_results)
                plot.rainbowgram(f'{save_dir}/{dx}-{b}/spec.pdf', uout[b], sr, colorbar=False)
                plot.rainbowgram(f'{save_dir}/{dx}-{b}/f0.pdf',   uout[b], sr, f0_input=target_f0[b], colorbar=False)

                plot.phase_diagram(f'{save_dir}/{dx}-{b}/phs.pdf',   wout[b], None, wout_min, wout_max, wdif_min, wdif_max, wddf_min, wddf_max, sr, tau, label=r'$\xi$')
                plot.phase_diagram(f'{save_dir}/{dx}-{b}/phs-u.pdf', uout[b], state_u_b, uout_min, uout_max, udif_min, udif_max, uddf_min, uddf_max, sr, tau, label='$u$')
                plot.phase_diagram(f'{save_dir}/{dx}-{b}/phs-z.pdf', zout[b], state_z_b, zout_min, zout_max, zdif_min, zdif_max, zddf_min, zddf_max, sr, tau, label='$\zeta$')


            if args.task.plot_state:
                #plot.state_video(f'{save_dir}/{dx}-{b}',  state_u_b, sr)
                plot.state_video(f'{save_dir}/{dx}-{b}',  state_u_b, sr, trim_front=True)

            if args.task.save:
                ms.save_simulation_data(f'{save_dir}/{dx}-{b}', excitation_type, overall_results, consts)

