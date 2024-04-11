import os
import shutil
import subprocess
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
from src.utils.control import *
from src.utils.misc import soft_bow, hard_bow, sinusoidal_embedding
from src.utils.audio import rms_normalize
import wandb
import soundfile as sf

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def gt_param(TF=5, sr=44100):
    sr = 44100
    NF = int(sr * TF)
    k = 1 / sr
    TRANS = int(0.05 * sr)
    x_bow = torch.linspace(0.25, 0.45, NF)
    v_bow = 0.1 * torch.tanh(torch.linspace(0., 10, NF))
    F_bow = torch.cat((
        torch.linspace(100, 120, NF//8 - TRANS), torch.zeros(TRANS),
        100 * torch.ones(NF//8 - TRANS), torch.zeros(TRANS),
        100 * torch.ones(NF//8 - TRANS), torch.zeros(TRANS),
        torch.linspace(100, 80, NF//8 - TRANS), torch.zeros(TRANS),
        80 * torch.ones(NF//4),
        torch.zeros(NF//4),
    ), dim=-1)
    
    f0 = torch.cat((
        glissando(98,110, NF//8),
        constant(130.81, NF//8),
        glissando(146.83, 164.81, NF//8),
        constant(207.65, NF//8),
        vibrato(207.65, NF//4, k, 5, 10),
        constant(207.65, NF//4),
    ), dim=-1)
    F_bow = F.pad(F_bow, (NF-F_bow.size(-1),0))
    f0 = F.pad(f0, (NF-f0.size(-1),0))
    
    #wid = torch.linspace(0.05, 0.05, NF)
    #rp = np.array([0.3, 0.7])
    #T60 = np.array([[100, 8], [2000, 5]])

    return [x_bow, v_bow, F_bow, f0]

def param(est_param, gt_param, save_path):
    e_x_bow, e_v_bow, e_F_bow, e_f0 = [item.detach().cpu().numpy() for item in est_param[:4]]
    g_x_bow, g_v_bow, g_F_bow, g_f0 = [item.cpu().numpy() for item in gt_param]

    fig, ax = plt.subplots(figsize=(7,7), nrows=4, ncols=1)

    ax[0].plot(g_x_bow, 'b:')
    ax[0].plot(e_x_bow, 'k-')
    ax[0].axhline(y=0, c='k', lw=.5)
    ax[0].set_ylabel('bow pos')
    
    ax[1].plot(g_v_bow, 'b:')
    ax[1].plot(e_v_bow, 'k-')
    ax[1].axhline(y=0, c='k', lw=.5)
    ax[1].set_ylabel('bow vel')
    
    ax[2].plot(g_F_bow, 'b:')
    ax[2].plot(e_F_bow, 'k-')
    ax[2].axhline(y=0, c='k', lw=.5)
    ax[2].set_ylabel('bow force')
    
    ax[3].plot(g_f0, 'b:')
    ax[3].plot(e_f0, 'k-')
    ax[3].axhline(y=0, c='k', lw=.5)
    ax[3].set_ylabel('f0')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def simulation_data(
        save_dir,
        uout, zout, v_r_out, F_H_out, u_H_out,
        state_u, state_z,
        string_params, bow_params, hammer_params,
        **kwargs,
    ):
    N = min(1000, uout.shape[0])
    
    kappa, alpha, u0, v0, f0, pos, T60, target_f0 = string_params
    x_b, v_b, F_b, phi_0, phi_1, wid_b = bow_params
    x_H, v_H, u_H, w_H, M_r, alpha_H = hammer_params

    max_disp = np.max(np.abs(uout[:N]))
    rels = torch.linspace(-1,1,100)
    prof = hard_bow(rels, phi_0, phi_1)

    # plot string params
    fig, ax = plt.subplots(figsize=(7,7), nrows=5, ncols=1)

    ax[0].plot(f0, 'k-')
    ax[0].axhline(y=0, c='k', lw=.5)
    ax[0].set_ylabel('f0')
    ax[0].yaxis.tick_right()
    ax[0].set_ylim([0, 500])
  
    ax[1].plot(np.linspace(0,1,state_u.shape[-1]), state_u[-1], 'k-')
    ax[1].axvline(x=pos, c='r', lw=.5); ax[1].axvline(x=x_b[-1], c='b', lw=.5)
    ax[1].set_ylabel('transverse state')
    ax[1].yaxis.tick_right()
    #ax[1].set_ylim([-max_disp, max_disp])

    ax[2].plot(np.linspace(0,1,state_z.shape[-1]), state_z[-1], 'k-')
    ax[2].axvline(x=pos, c='r', lw=.5); ax[1].axvline(x=x_b[-1], c='b', lw=.5)
    ax[2].set_ylabel('longitudinal state')
    ax[2].yaxis.tick_right()
    #ax[2].set_ylim([-max_disp, max_disp])

    ax[3].plot(np.arange(N), uout[:N], 'k-')
    ax[3].axhline(y=0, c='k', lw=.5)
    ax[3].set_ylabel('output')
    ax[3].yaxis.tick_right()
    ax[3].set_ylim([-max_disp, max_disp])

    ax[4].plot(np.arange(N), zout[:N], 'k-')
    ax[4].axhline(y=0, c='k', lw=.5)
    ax[4].set_ylabel('output')
    ax[4].yaxis.tick_right()
    #ax[4].set_ylim([-max_disp, max_disp])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/string.png")
    plt.clf()
    plt.close()

    # plot bow params
    fig, ax = plt.subplots(figsize=(7,7), nrows=3, ncols=2)

    ax[0,0].plot(x_b, 'k-')                ;  ax[0,1].plot(rels.numpy(), prof.numpy(), 'k-')
    ax[0,0].axhline(y=0, c='k', lw=.5)     ;  ax[0,1].axhline(y=0, c='k', lw=.5)
    ax[0,0].set_ylabel('bowing position')  ;  ax[0,1].set_ylabel('bow friction fn')
    ax[0,0].yaxis.tick_right()             ;  ax[0,1].yaxis.tick_right()
    ax[0,0].set_ylim([0, 1])               ;  ax[0,1].set_ylim([-1.5, 1.5])

    ax[1,0].plot(v_b, 'k-')                ;  ax[1,1].plot(np.arange(N), v_r_out[:N], 'k-')
    ax[1,0].axhline(y=0, c='k', lw=.5)     ;  ax[1,1].axhline(y=0, c='k', lw=.5)
    ax[1,0].set_ylabel('bowing velocity')  ;  ax[1,1].set_ylabel('rel vel (attack)')
    ax[1,0].yaxis.tick_right()             ;  ax[1,1].yaxis.tick_right()
    ax[1,0].set_ylim([0, 0.5])             ;  ax[1,1].set_ylim([-2, 2])

    ax[2,0].plot(F_b, 'k-')                ;  ax[2,1].plot(np.arange(N), v_r_out[-N:], 'k-')
    ax[2,0].axhline(y=0, c='k', lw=.5)     ;  ax[2,1].axhline(y=0, c='k', lw=.5)
    ax[2,0].set_ylabel('bowing force')     ;  ax[2,1].set_ylabel('rel vel (release)')
    ax[2,0].yaxis.tick_right()             ;  ax[2,1].yaxis.tick_right()
    ax[2,0].set_ylim([0, 100])             ;  ax[2,1].set_ylim([-2, 2])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bow.png")
    plt.clf()
    plt.close()


    sr = 48000
    Nt = len(v_r_out)
    Nx = state_u.shape[-1]
    a_f = (v_r_out[1:] - v_r_out[:Nt-1]) * sr
    F_f = a_f / Nx
    mu = F_f / F_b[-(Nt-1):]
    vr = v_r_out[:Nt-1]
    rels = torch.linspace(np.min(vr)-.1,np.max(vr)+.1,100)
    prof = hard_bow(rels, phi_0, phi_1)
    #prof = soft_bow(rels, phi_0)

    fig, ax = plt.subplots(figsize=(4,4), nrows=1, ncols=1)
    #ax.plot(rels.numpy(), prof.numpy(), 'r--')
    ax.fill_between(rels.numpy(), prof.numpy(), alpha=0.2, facecolor='r')
    ax.plot(vr, mu, 'k-')
    ax.axhline(y=0, c='k', lw=.5)
    ax.set_xlabel('Relative velocity')
    ax.set_ylabel('Friction coefficient')
    ax.set_ylim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bow-velforce.pdf")
    plt.clf()
    plt.close()

    # plot string params
    fig, ax = plt.subplots(figsize=(7,7), nrows=2, ncols=1)

    sr = 48000
    # ms
    t_1 = 0; Nt_1 = int(sr * t_1 * 1e-3)
    #t_2 = 3; Nt_2 = int(sr * t_2 * 1e-3)
    t_2 = 8; Nt_2 = int(sr * t_2 * 1e-3)
    time = np.linspace(t_1, t_2, Nt_2 - Nt_1)
    ax[0].plot(time, u_H_out[Nt_1:Nt_2], 'k-')
    ax[0].axhline(y=0, c='k', lw=.5)
    ax[0].set_ylabel('hammer displacement')
    ax[0].yaxis.tick_right()
    #ax[0].set_ylim([0, 0.1])

    ax[1].plot(time, F_H_out[Nt_1:Nt_2], 'k-')
    ax[1].axhline(y=0, c='k', lw=.5)
    ax[1].set_ylabel('hammer force')
    ax[1].yaxis.tick_right()
    #ax[1].set_ylim([0, 10000])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/hammer.png")
    plt.clf()
    plt.close()



#def state_video(save_dir, state_u, sr, framerate=50, trim_front=False, verbose=False, prefix=None):
def state_video(save_dir, state_u, sr, framerate=100, trim_front=False, verbose=False, prefix=None):
    if isinstance(state_u, list):
        state_v = state_u[1]
        state_u = state_u[0]
    else:
        state_v = None

    if trim_front:
        #state_u = state_u[:int(0.01 * sr)]
        state_u = state_u[:int(sr / 55)] # for 55 Hz (A1)
        state_v = state_v[:int(sr / 55)] if state_v is not None else None
        downs = int(state_u.shape[0]/framerate)
    else:
        downs = 100

    Nt, Nx = state_u.shape
    maxy = np.max(np.abs(state_u))
    locs = np.linspace(0, 1, Nx)
    for j in range(Nt // downs):

        plt.figure(figsize=(5,2))
        if state_v is not None:
            plt.plot(locs, state_v[j * downs], c='k', alpha=0.5)
        plt.plot(locs, state_u[j * downs], c='k')
        plt.xlim([0, 1])
        plt.ylim([-maxy, maxy])
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        os.makedirs(f'{save_dir}/temp', exist_ok=True)
        plt.savefig(f'{save_dir}/temp/file%02d.png' % j)
        plt.clf()
        plt.close("all")

    prefix = 'fdtd' if prefix is None else prefix
    with open(os.devnull, 'w') as devnull:
        silent_video = ['ffmpeg',
            '-framerate', f'{framerate}',
            '-i', f'{save_dir}/temp/file%02d.png',
            '-r', '30', '-pix_fmt', 'yuv420p', '-y',
            f'{save_dir}/{prefix}-silent_video.mp4']
        output_video = ['ffmpeg',
            '-i', f'{save_dir}/silent_video.mp4',
            '-i', f'{save_dir}/output.wav',
            '-c:v', 'copy', '-map', '0:v', '-map', '1:a',
            '-shortest', '-y',
            f'{save_dir}/{prefix}-output.mp4']
        silent_video +=  ['-loglevel', 'quiet'] if not verbose else []
        output_video +=  ['-loglevel', 'quiet'] if not verbose else []
        subprocess.call(silent_video, stdout=devnull)
        subprocess.call(output_video, stdout=devnull)

    shutil.rmtree(f"{save_dir}/temp")

def rainbowgram(
    save_path, out, sr, n_fft=2**13, hop_length=None,
    f0_input=None, f0_estimate=None, modes=None, colorbar=True,
):
    L = 32
    if out.shape[-1] > 2*n_fft:
        hop_length = n_fft // L if hop_length is None else hop_length
    else:
        n_fft = out.shape[-1] // 2
        hop_length = n_fft // L
    t_max = out.shape[-1] / sr

    out, gain = rms_normalize(out)
    D = librosa.stft(out, n_fft=n_fft, hop_length=hop_length, pad_mode='reflect')
    mag, phase = librosa.magphase(D)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.times_like(D, sr=sr, hop_length=hop_length)
    
    phase_exp = 2 * np.pi * np.multiply.outer(freqs, times)
    unwrapped_phase = np.unwrap((np.angle(phase)-phase_exp) / (L/4), axis=1)
    unwrapped_phase_diff = np.diff(unwrapped_phase, axis=1, prepend=0)

    alpha = librosa.amplitude_to_db(mag, ref=np.max) / 80 + 1

    #width = 2.5; height = 1.9
    width = 7; height = 7
    fig, ax = plt.subplots(figsize=(width,height))
    spec = librosa.display.specshow(
        unwrapped_phase_diff, cmap='hsv', alpha=alpha,
        n_fft=n_fft, hop_length=hop_length, sr=sr, ax=ax,
        y_axis='log', x_axis='time',
    )
    ax.set_facecolor('#000')
    if colorbar:
        cbar = fig.colorbar(spec, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ax=ax)
        cbar.ax.set(yticklabels=['$-\pi$', '$-\pi/2$', "$0$", '$\pi/2$', '$\pi$']);

    def add_plot(freqs, label=None, ls=None, lw=2., dashes=(None,None)):
        x = np.linspace(1/sr, t_max, freqs.shape[-1])
        freqs = np.interp(times, x, freqs)
        line, = ax.plot(times - times[0], freqs, label=label, color='white', lw=lw, ls=ls, dashes=dashes)
        return line

    freq_ticks = [0, 128, 512, 2048, 8192, sr // 2]
    time_ticks = [0, 1, 2]
    if f0_input is not None:
        add_plot(f0_input, "f0_input", dashes=(10,5))
        freq_ticks += [f0_input[0]]

    if f0_estimate is not None:
        add_plot(f0_estimate, "f0_estimate", dashes=(2,5))
        freq_ticks += [] if f0_input is not None else [f0_estimate[0]]

    if modes is not None:
        for im, m in enumerate(modes):
            l = add_plot(m, f"mode {im}")
            l.set_dashes([5,10,1,10])

    #ax.set_xticks(time_ticks)
    #ax.set_yticks(freq_ticks)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=-1e-6)
    plt.clf()
    plt.close("all")

def phase_diagram(
        save_path, x, s,
        xmin, xmax,
        dxmin, dxmax,
        ddxmin, ddxmax,
        sr, tau=1, label='$u$'):
    dx  = (x[tau:] - x[:-tau]) / (tau / sr)
    ddx = (x[2*tau:] - 2*x[tau:-tau] + x[:-2*tau]) / (2*tau / sr)

    if s is not None:
        if s.shape[0] > x.shape[0]:
            s = s[:x.shape[0]]
        dsdt = (s[tau:] - s[:-tau]) / (tau / sr)
        _dsdt = np.mean(np.abs(dsdt), axis=0)
        spax = np.arange(len(_dsdt))

    if s is not None:
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8,3.5), width_ratios=[4, 1])
        ax[0,0].axhline(y=0, color='gray', ls='-', lw=0.3)
        ax[0,0].plot(x, 'k-', lw=0.5)
        ax[0,0].set_xlim([0,len(x)])
        ax[0,0].set_ylim([xmin,xmax])
        ax[0,0].set_xticks([])
        ax[0,0].set_yticks([])
        #ax[0,0].set_xlabel('$t$')
        ax[0,0].set_ylabel(label)

        ax[0,1].axhline(y=0, color='gray', ls='-', lw=0.3)
        ax[0,1].axvline(x=0, color='gray', ls='-', lw=0.3)
        ax[0,1].plot(dx, x[tau:], 'k-', lw=0.5)
        ax[0,1].set_xlim([dxmin,dxmax])
        ax[0,1].set_ylim([xmin,xmax])
        ax[0,1].set_xticks([])
        ax[0,1].set_yticks([])
        #ax[0,1].set_xlabel('$d$'+label+'$/dt$')

        #state = librosa.display.specshow(s.T, cmap='coolwarm', ax=ax[1,0])
        state = librosa.display.specshow(dsdt.T, cmap='coolwarm', ax=ax[1,0])
        maxabs = np.max(np.abs(dsdt))
        state.set_clim([-maxabs, +maxabs])
        ax[1,0].set_xlim([0,x.shape[0]])
        ax[1,0].set_xlabel('$t$')
        ax[1,0].set_ylabel('$x$')

        _dsdt = np.pad( _dsdt, (1,1))
        _spax = np.pad(  spax, (1,1), mode='edge')
        ax[1,1].fill_between(+ _dsdt, _spax, alpha=0.2, facecolor='k')
        ax[1,1].fill_between(- _dsdt, _spax, alpha=0.2, facecolor='k')
        ax[1,1].axvline(x=0, color='k', ls='-', lw=1.0)
        ax[1,1].set_ylim([spax[0], spax[-1]])
        ax[1,1].set_xticks([])
        ax[1,1].set_yticks([])
        ax[1,1].set_xlabel('$d$'+label+'$/dt$')
    else:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,2), width_ratios=[4, 1])
        ax[0].axhline(y=0, color='gray', ls='-', lw=0.3)
        ax[0].plot(x, 'k-', lw=0.5)
        ax[0].set_xlim([0,len(x)])
        ax[0].set_ylim([xmin,xmax])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel(label)

        ax[1].axhline(y=0, color='gray', ls='-', lw=0.3)
        ax[1].axvline(x=0, color='gray', ls='-', lw=0.3)
        ax[1].plot(dx, x[tau:], 'k-', lw=0.5)
        ax[1].set_xlim([dxmin,dxmax])
        ax[1].set_ylim([xmin,xmax])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_xlabel('$d$'+label+'$/dt$')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.)
    plt.subplots_adjust(hspace=0.)
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.clf()
    plt.close("all")


    #fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(3.5,2))

    #ax[0].axvline(x=0, color='gray', ls='-', lw=0.3)
    #ax[0].axhline(y=0, color='gray', ls='-', lw=0.3)
    #ax[0].plot(x[2*tau:], ddx, 'k-', lw=0.5)
    #ax[0].set_xlim([xmin,xmax])
    #ax[0].set_ylim([ddxmin,ddxmax])
    #ax[0].set_xticks([])
    #ax[0].set_yticks([])
    #ax[0].set_xlabel(label)
    #ax[0].set_ylabel('$d^2$'+label+'$/dt^2$')

    #ax[1].axvline(x=0, color='gray', ls='-', lw=0.3)
    #ax[1].axhline(y=0, color='gray', ls='-', lw=0.3)
    #ax[1].plot(dx[tau:], ddx, 'k-', lw=0.5)
    #ax[1].set_xlim([dxmin, dxmax])
    #ax[1].set_ylim([ddxmin,ddxmax])
    #ax[1].set_xticks([])
    #ax[1].set_yticks([])
    #ax[1].set_xlabel('$d$'+label+'$/dt$')

    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0.)
    #plt.subplots_adjust(hspace=0.)
    #save_dir = save_path.split('/')[:-1]
    #save_name = save_path.split('/')[-1]
    #save_path_2 = '/'.join(save_dir+[save_name.replace('phs', 'dphs')])
    #plt.savefig(save_path_2, bbox_inches='tight', transparent=True)
    #plt.clf()
    #plt.close("all")


def xt_grid_embedding(save_path, x, t, embed_dim=32, t_gain=1e-6, x_gain=1e-2):
    t = t * 1000

    Bs,  _, Nx = x.shape
    Bs, Nt,  _ = t.shape
    t_embd = sinusoidal_embedding(t.unsqueeze(-1), n=embed_dim, gain=t_gain) # (Bs, 1,Nx,1,embed_dim)
    x_embd = sinusoidal_embedding(x.unsqueeze(-1), n=embed_dim, gain=x_gain) # (Bs,Nt, 1,1,embed_dim)

    t_axis = t.squeeze().detach().cpu().numpy()
    x_axis = x.squeeze().detach().cpu().numpy()
    t_embd = t_embd.squeeze().detach().cpu().numpy()
    x_embd = x_embd.squeeze().detach().cpu().numpy()
    assert len(list(t_embd.shape)) == 2, t_embd.shape
    assert len(list(x_embd.shape)) == 2, x_embd.shape
    e = np.arange(embed_dim)

    fig, ax = plt.subplots(figsize=(13,7), nrows=1, ncols=2)
    librosa.display.specshow(t_embd, ax=ax[0], x_coords=e, y_coords=t_axis)
    librosa.display.specshow(x_embd, ax=ax[1], x_coords=e, y_coords=x_axis)
    ax[0].set_title("t embed")
    ax[0].set_xlabel("embedding dim")
    ax[0].set_ylabel("time")
    ax[0].set_yticks(t_axis[0::10])

    ax[1].set_title("x embed")
    ax[1].set_xlabel("embedding dim")
    ax[1].set_ylabel("space")
    ax[1].set_yticks(x_axis[0::10])
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.)
    plt.subplots_adjust(hspace=0.)
    plt.savefig(save_path)
    plt.clf()
    plt.close("all")

def logedc(save_path, logedc, tmax):
    time = np.linspace(0, tmax, logedc.shape[0])

    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(time, logedc)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (dB)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close("all")

def f0curve(save_path, f0_input, f0_estimate, first_mode, tmax):
    time = np.linspace(0, tmax, len(f0_estimate))

    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(time, f0_input, label='$f_0$')
    ax.plot(time, f0_estimate, label='$f_0^{(\\tt est)}$')
    ax.plot(time, first_mode, label='$\hat{f_0}$')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 200)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close("all")

def spectrum(save_path, out, f0_input, f0_estimate, modes, sr, n_fft=2**14, ylabel=None):
    t_max = out.shape[-1] / sr
    n_fft = min(n_fft, out.shape[-1])
    cr = int(f0_estimate.shape[-1] / t_max)   # crepe framerate
    simulated = out[-n_fft:]
    f0_input    =    f0_input[-1]
    f0_estimate = f0_estimate[-1]
    modes = [m[-1] for m in modes]

    simulated_fr = 20 * np.log10(np.abs(np.fft.rfft(simulated, n_fft)))
    freqs = np.linspace(0, sr/2 / 1000, int(n_fft/2+1))

    n_freqs = 1024

    fig, ax = plt.subplots(figsize=(4,2))

    lw = 0.7
    ax.plot(freqs[:n_freqs], simulated_fr[:n_freqs], 'k', lw=1.)
    ax.axvline(x=f0_input / 1000, c='r', ls='-', lw=lw, label='$f_0$')
    ax.axvline(x=f0_estimate / 1000, c='g', ls='--', lw=lw, label='$f_0^{(\\tt est)}$')
    for i, m in enumerate(modes):
        if i == 0:
            ax.axvline(x=m / 1000, c='b', ls='-.', lw=lw, label='$\hat{f_p}$')
        else:
            ax.axvline(x=m / 1000, c='b', ls='-.', lw=lw)

    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    plt.xlim([0, 2])
    plt.xlabel('Frequency (kHz)')
    plt.ylabel(ylabel)

    plt.legend(ncol=3, fancybox=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    plt.close("all")


def spectrum_uz(save_path, uout, zout, f0_input, f0_estimate, modes, sr, n_fft=2**14):
    t_max = uout.shape[-1] / sr
    n_fft = min(n_fft, uout.shape[-1])
    cr = int(f0_estimate.shape[-1] / t_max)   # crepe framerate
    simulated_u = uout[-n_fft:]
    simulated_z = zout[-n_fft:]
    f0_input    =    f0_input[-1]
    f0_estimate = f0_estimate[-1]
    modes = [m[-1] for m in modes]

    simulated_fr_u = 20 * np.log10(np.abs(np.fft.rfft(simulated_u, n_fft)))
    simulated_fr_z = 20 * np.log10(np.abs(np.fft.rfft(simulated_z, n_fft)))
    freqs = np.linspace(0, sr/2 / 1000, int(n_fft/2+1))

    n_freqs = 1024

    fig, ax = plt.subplots(figsize=(2.5,2), ncols=1, nrows=2)
    #fig, ax = plt.subplots(figsize=(4,2), ncols=1, nrows=2)

    lw = 1.
    lw_fr = .5
    al = .5
    ax[0].axhline(y=0, c='k', lw=0.5, alpha=al)
    ax[0].plot(freqs[:n_freqs], simulated_fr_u[:n_freqs], 'k', lw=lw_fr)
    ax[0].axvline(x=f0_input / 1000, c='r', ls='-', lw=lw, label='$f_0$', alpha=al)
    ax[0].axvline(x=f0_estimate / 1000, c='g', ls='--', lw=lw, label='$f_0^{(\\tt est)}$', alpha=al)
    for i, m in enumerate(modes):
        if i == 0:
            ax[0].axvline(x=m / 1000, c='b', ls=':', lw=lw, label='$\hat{f_p}$', alpha=al)
        else:
            ax[0].axvline(x=m / 1000, c='b', ls=':', lw=lw, alpha=al)
    ax[0].set_xticks([0, 0.5, 1, 1.5, 2])
    ax[0].set_xlim([0, 2])
    ax[0].set_ylabel('$|u|$')
    ax[0].xaxis.set_label_position('top')
    ax[0].yaxis.tick_right()
    ax[0].xaxis.tick_top()

    ax[1].axhline(y=0, c='k', lw=0.3, alpha=al)
    ax[1].plot(freqs[:n_freqs], simulated_fr_z[:n_freqs], 'k', lw=lw_fr)
    ax[1].axvline(x=f0_input / 1000, c='r', ls='-', lw=lw, label='$f_0$', alpha=al)
    ax[1].axvline(x=f0_estimate / 1000, c='g', ls='--', lw=lw, label='$f_0^{(\\tt est)}$', alpha=al)
    for i, m in enumerate(modes):
        if i == 0:
            ax[1].axvline(x=m / 1000, c='b', ls=':', lw=lw, label='$\hat{f_p}$', alpha=al)
        else:
            ax[1].axvline(x=m / 1000, c='b', ls=':', lw=lw, alpha=al)
    ax[1].set_xticks([])
    ax[1].set_xlim([0, 2])
    ax[1].set_xlabel('Frequency (kHz)')
    ax[1].set_ylabel('$|\zeta|$')
    ax[1].yaxis.tick_right()
    #ax[1].xaxis.set_label_coords(0.2, -0.05)
    #plt.legend(loc='lower center', bbox_to_anchor=(0.7,-0.4), ncol=3, fancybox=True, handletextpad=0.1, columnspacing=1.)
    #ax[1].xaxis.set_label_coords(0.2, -0.1)
    #plt.legend(loc='lower center', bbox_to_anchor=(0.7,-0.8), ncol=3, fancybox=True, handletextpad=0.1, columnspacing=1.)
    ax[1].xaxis.set_label_coords(0.3, -0.1)
    plt.legend(loc='lower center', bbox_to_anchor=(.95,-0.5), ncol=3, fancybox=True, handlelength=1., handletextpad=0.1, columnspacing=.5, fontsize=7)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.)
    plt.subplots_adjust(hspace=0.)
    plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=-1e-6)
    plt.clf()
    plt.close("all")


def scatter_xy(save_path, x, y_dict, xlabel, ylabel, xticks=[], yticks=[]):
    fig, ax = plt.subplots(figsize=(2.5,2.5))
    for y_label in y_dict.keys():
        ax.scatter(x, y_dict[y_label], label=y_label, s=1.)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.clf()
    plt.close("all")



def scatter_pluck(save_path, total_summary, ss=.3, al=0.7):
    cmap = {
        '$|f_0^{(\\tt est)} - f_0|$'       : 'orchid',
        '$|f_0^{(\\tt est)} - \hat{f_0}|$' : 'cadetblue',
    }

    fig, ax = plt.subplots(figsize=(2., 1.1), nrows=1, ncols=3)
    f0_diffs, kappa, alpha, c0 = total_summary['f0-kappa']
    # kappa
    for y_label in f0_diffs.keys():
        ax[0].scatter(kappa, f0_diffs[y_label], c=cmap[y_label], label=y_label, s=ss, alpha=al)
    ax[0].axvline(x=5.88, c='k', ls='--', lw=0.5)
    ax[0].axhline(y=6, c='k', ls='--', lw=0.5)
    ax[0].axhline(y=1, c='k', ls='--', lw=0.5)
    ax[0].set_xlabel('$\kappa$')
    ax[0].set_ylabel('Detune')
    ax[0].set_ylim([0, 10])
    ax[0].set_xticks([2,5,8])
    ax[0].set_yticks([])
    ax[0].xaxis.tick_top()

    # alpha
    f0_diffs, kappa, alpha, c0 = total_summary['f0-alpha']
    for y_label in f0_diffs.keys():
        ax[1].scatter(alpha, f0_diffs[y_label], c=cmap[y_label], label=y_label, s=ss, alpha=al)
    ax[1].axhline(y=6, c='k', ls='--', lw=0.5)
    ax[1].axhline(y=1, c='k', ls='--', lw=0.5)
    ax[1].set_xlabel('$\\alpha$')
    ax[1].set_xlim([0.8,4.2])
    ax[1].set_ylim([0, 10])
    ax[1].set_xticks([1,2,3,4])
    ax[1].set_yticks([])
    ax[1].xaxis.tick_top()

    # c0
    f0_diffs, kappa, alpha, c0 = total_summary['f0-c0']
    c0 = [item*1e3 for item in c0]
    for y_label in f0_diffs.keys():
        ax[2].scatter(c0, f0_diffs[y_label], c=cmap[y_label], label=y_label, s=ss, alpha=al)
    ax[2].axhline(y=6, c='k', ls='--', lw=0.5)
    ax[2].axhline(y=1, c='k', ls='--', lw=0.5)
    ax[2].set_xlabel('$c_0\\times10^{3}$')
    ax[2].set_xlim([0, 11])
    ax[2].set_ylim([0, 10])
    ax[2].set_xticks([1, 4, 7, 10])
    ax[2].set_yticks([0,5,10])
    ax[2].xaxis.tick_top()
    ax[2].yaxis.tick_right()

    plt.tight_layout()
    plt.legend(loc='lower center', bbox_to_anchor=(-0.5, -1.2), ncol=2, fancybox=True, handletextpad=0.02, columnspacing=.2, markerscale=5., fontsize=7)
    plt.subplots_adjust(wspace=0.)
    plt.subplots_adjust(hspace=0.)
    plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=-1e-5)
    plt.clf()
    plt.close("all")


def scatter_pluck_front(save_path, total_summary, ss=.3, al=0.7):
    cmap = {
        '$|f_0^{(\\tt est)} - f_0|$'       : 'orchid',
        '$|f_0^{(\\tt est)} - \hat{f_0}|$' : 'cadetblue',
    }

    fig, ax = plt.subplots(figsize=(3., 1.5), nrows=1, ncols=3)
    f0_diffs, kappa, alpha, c0 = total_summary['f0-kappa']
    # kappa
    for y_label in f0_diffs.keys():
        ax[0].scatter(kappa, f0_diffs[y_label], c=cmap[y_label], label=y_label, s=ss, alpha=al)
    ax[0].axvline(x=5.88, c='k', ls='--', lw=0.5)
    ax[0].axhline(y=6, c='k', ls='--', lw=0.5)
    ax[0].axhline(y=1, c='k', ls='--', lw=0.5)
    ax[0].set_xlabel('$\kappa$')
    ax[0].set_ylabel('Detune (Hz)')
    ax[0].set_ylim([0, 10])
    ax[0].set_xticks([2,4,6,8])
    ax[0].set_yticks([])
    ax[0].xaxis.tick_top()

    # alpha
    f0_diffs, kappa, alpha, c0 = total_summary['f0-alpha']
    for y_label in f0_diffs.keys():
        ax[1].scatter(alpha, f0_diffs[y_label], c=cmap[y_label], label=y_label, s=ss, alpha=al)
    ax[1].axhline(y=6, c='k', ls='--', lw=0.5)
    ax[1].axhline(y=1, c='k', ls='--', lw=0.5)
    ax[1].set_xlabel('$\\alpha$')
    ax[1].set_xlim([0.8,4.2])
    ax[1].set_ylim([0, 10])
    ax[1].set_xticks([1,2,3,4])
    ax[1].set_yticks([])
    ax[1].xaxis.tick_top()

    # c0
    f0_diffs, kappa, alpha, c0 = total_summary['f0-c0']
    c0 = [item*1e3 for item in c0]
    for y_label in f0_diffs.keys():
        ax[2].scatter(c0, f0_diffs[y_label], c=cmap[y_label], label=y_label, s=ss, alpha=al)
    ax[2].axhline(y=6, c='k', ls='--', lw=0.5)
    ax[2].axhline(y=1, c='k', ls='--', lw=0.5)
    ax[2].set_xlabel('$c_0\\times10^{3}$')
    ax[2].set_xlim([0, 11])
    ax[2].set_ylim([0, 10])
    ax[2].set_xticks([1, 4, 7, 10])
    ax[2].set_yticks([0,2,4,6,8,10])
    ax[2].xaxis.tick_top()
    ax[2].yaxis.tick_right()

    plt.tight_layout()
    plt.legend(loc='lower center', bbox_to_anchor=(-0.5, -0.75), ncol=2, fancybox=True, handletextpad=0.1, columnspacing=1., markerscale=5.)
    plt.subplots_adjust(wspace=0.)
    plt.subplots_adjust(hspace=0.)
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.clf()
    plt.close("all")


def time_experiment(save_path, gpu_summary, cpu_summary):

    n_criteria = len(list(gpu_summary.keys()))
    fig, ax = plt.subplots(figsize=(5, 1.66), nrows=1, ncols=n_criteria)

    config = {
        'Batch size'       : [4, 16, 64, 256, 1024],
        '$N_t$'            : [0.25, 0.50, 1.00, 2.00, 4.00],
        '$N_x^{(\\tt t)}+N_x^{(\\tt l)}$' : [20, 40, 80, 160,  320],
        '$N_x^{(\\tt l)}$' : [1, 2, 3, 4],
    }
    xlims = {
        'Batch size'       : [2,1800],
        '$N_t$'            : [6000, 300000],
        #'$N_x^{(\\tt t)}$' : [15, 160],
        '$N_x^{(\\tt t)}+N_x^{(\\tt l)}$' : [70, 1900],
        '$N_x^{(\\tt l)}$' : [15, 160],
    }

    def f0_to_NtNl(f0, k=1/48000, theta_t=0.5 + 2/(np.pi**2), kappa_rel=0.03):
        gamma =  2*f0
        kappa = gamma * kappa_rel
        IHP = (np.pi * kappa / gamma)**2         # inharmonicity parameter (>0); eq 7.21
        K = pow(IHP, .5) * (gamma / np.pi)          # set parameters
        h = pow( \
            (gamma**2 * k**2 + pow(gamma**4 * k**4 + 16 * K**2 * k**2 * (2 * theta_t - 1), .5)) \
          / (2 * (2 * theta_t - 1)) \
        , .5)
        N_t = int(1/h)
        alpha = 1
        h = gamma * alpha * k
        N_l = int(1/h)
        return N_t + N_l
    def alpha_to_Nl(alpha, gamma=600, k=1/48000):
        h = gamma * alpha * k
        N_l = int(1/h)
        return N_l

    for i, criterion in enumerate(config.keys()):
        if criterion == '$N_t$':
            config[criterion] = [int(c * 48000) for c in config[criterion]]
        if criterion == '$N_x^{(\\tt t)}+N_x^{(\\tt l)}$':
            config[criterion] = [f0_to_NtNl(c) for c in config[criterion]]
        if criterion == '$N_x^{(\\tt l)}$':
            config[criterion] = [alpha_to_Nl(c) for c in config[criterion]]

    print(config)

    for i, criterion in enumerate(gpu_summary.keys()):
        conf_list = config[criterion]
        gpu_times = gpu_summary[criterion]
        cpu_times = cpu_summary[criterion]

        if i == 0:
            # divide by number of batch
            #gpu_times = [gpu_times[k] / conf_list[k] for k in range(len(gpu_times))]
            #cpu_times = [cpu_times[k] / conf_list[k] for k in range(len(cpu_times))]
            pass
        elif i > 1:
            conf_list = list(reversed(conf_list))
            gpu_times = list(reversed(gpu_times))
            cpu_times = list(reversed(cpu_times))
        gpu_times = [gpu_times[k] / gpu_times[0] for k in range(len(gpu_times))]
        cpu_times = [cpu_times[k] / cpu_times[0] for k in range(len(cpu_times))]

        lin_times = [conf_list[k] / conf_list[0] for k in range(len(gpu_times))]

        thicklw = 0.8
        ax[i].axhline(y=100,  c='lightgray', lw=thicklw, ls=':')
        ax[i].axhline(y=10,   c='lightgray', lw=thicklw, ls=':')
        ax[i].axhline(y=1,    c='lightgray', lw=thicklw, ls='-')

        ax[i].plot(conf_list[:len(cpu_times)], cpu_times, 'kD--', lw=.9, label="CPU", mfc='lightgray')
        ax[i].plot(conf_list[:len(gpu_times)], gpu_times, 'ko-',  lw=.9, label="GPU", mfc='white')

        ax[i].fill_between(conf_list[:len(cpu_times)], lin_times, alpha=.2)

        ax[i].set_xlabel(criterion)
        if i == 0:
            ax[i].set_ylabel('Relative time')

        ax[i].set_xscale('log')
        ax[i].set_yscale('log')

        ax[i].set_ylim([0.5, 1e3])

        ax[i].set_xlim(xlims[criterion])
        ax[i].xaxis.set_label_position('top')
        ax[i].yaxis.tick_right()
        if i < len(list(gpu_summary.keys()))-1:
            ax[i].set_yticks([])
        else:
            ax[i].set_yticks([1, 10, 100, 1000])

    plt.tight_layout()
    #plt.legend(loc='lower center', bbox_to_anchor=(-0.5, -0.75), ncol=2, fancybox=True, handletextpad=0.1, columnspacing=1.)
    #plt.legend(loc='lower right', ncol=2, fancybox=True)
    plt.legend(loc='upper right', ncol=2, fancybox=True)
    plt.subplots_adjust(wspace=0.)
    plt.subplots_adjust(hspace=0.)
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.clf()
    plt.close("all")


def est_tar_specs(est, tar, plot_path, wave_path, sr=16000):
    data = []
    batch_size = est["wav"].shape[0]
    for b in range(batch_size):
        logspecs = []
        difspecs = []
        
        nrows = 3 if tar is not None else 1;  ncols = 2
        height = 2*nrows; widths = 7
        specfig, ax = plt.subplots(nrows, ncols, figsize=(widths,height))

        logspecs.append(librosa.display.specshow(est["logmag"][b].numpy().T,      cmap='magma',ax=ax[0,0] if tar is not None else ax[0]))
        if tar is not None:
            diff_0 = tar["logmag"][b] - est["logmag"][b]
            logspecs.append(librosa.display.specshow(tar["logmag"][b].numpy().T,cmap='magma',ax=ax[1,0]))
            difspecs.append(librosa.display.specshow(diff_0.numpy().T,            cmap='bwr',  ax=ax[2,0]))

        logspecs.append(librosa.display.specshow(est["logmel"][b].numpy().T,      cmap='magma',ax=ax[0,1] if tar is not None else ax[1]))
        if tar is not None:
            diff_0 = tar["logmel"][b] - est["logmel"][b]
            logspecs.append(librosa.display.specshow(tar["logmel"][b].numpy().T,cmap='magma',ax=ax[1,1]))
            difspecs.append(librosa.display.specshow(diff_0.numpy().T,            cmap='bwr',  ax=ax[2,1]))

        for spec in logspecs:
            spec.set_clim([-60, 30])
        for spec in difspecs:
            spec.set_clim([-20, 20])

        titles = ['Regenerated', 'Original', 'Difference'] if tar is not None else ['Generated']
        for i, title in enumerate(titles):
            if tar is not None:
                ax[i,0].set_ylabel(title)
            else:
                ax[i].set_ylabel(title)

        specfig.tight_layout()
        specfig.subplots_adjust(wspace=0)
        specfig.subplots_adjust(hspace=0)

        specfig.savefig(plot_path)
        plt.close('all')
        plt.clf()

        statfig = None
        if len(list(est['state'].shape)) == 4:
            u_states = []
            z_states = []
            dustates = []
            dzstates = []
            
            nrows = 5 if tar is not None else 3;  ncols = 2
            height = 2*nrows; widths = 7
            statfig, ax = plt.subplots(nrows, ncols, figsize=(widths,height))
            dif = est["state"] - tar["state"]
   

            cm = 'coolwarm'
            Nx = est["state"].size(-2)
            Nt = est["state"].size(-3)
            St = int(sr * 10 / 1000)
            x_k = Nx // 2
            t_k = Nt // 2
            u_states.append(librosa.display.specshow(est["state"][b,:Nt,:,0].numpy().T, cmap=cm,ax=ax[0,0]))
            u_states.append(librosa.display.specshow(tar["state"][b,:Nt,:,0].numpy().T, cmap=cm,ax=ax[1,0]))
            dustates.append(librosa.display.specshow(dif[b,:Nt,:,0].numpy().T,          cmap=cm,ax=ax[2,0]))
    
            z_states.append(librosa.display.specshow(est["state"][b,:Nt,:,1].numpy().T, cmap=cm,ax=ax[0,1]))
            z_states.append(librosa.display.specshow(tar["state"][b,:Nt,:,1].numpy().T, cmap=cm,ax=ax[1,1]))
            dzstates.append(librosa.display.specshow(dif[b,:Nt,:,1].numpy().T,          cmap=cm,ax=ax[2,1]))

            for j in range(2):
                label_tar = 'tar' if j == 0 else None
                label_est = 'est' if j == 0 else None
                ax[3,j].axhline(0, c='k', alpha=0.4)
                ax[4,j].axhline(0, c='k', alpha=0.4)
                ax[3,j].plot(tar["state"][b,:St,x_k,j], c='k', label=label_tar)
                ax[3,j].plot(est["state"][b,:St,x_k,j], c='b', label=label_est)
                ax[4,j].plot(tar["state"][b,  t_k,:,j], c='k')
                ax[4,j].plot(est["state"][b,  t_k,:,j], c='b')


            u_max = tar["state"][b,:St,:,0].abs().max()
            z_max = tar["state"][b,:St,:,1].abs().max()
            for stat in u_states: stat.set_clim([-u_max, u_max])
            for stat in z_states: stat.set_clim([-z_max, z_max])
            for stat in dustates: stat.set_clim([-u_max/10, u_max/10])
            for stat in dzstates: stat.set_clim([-u_max/10, u_max/10])
   
            for stat in dustates: stat.set_clim([-u_max/10, u_max/10])
            for stat in dzstates: stat.set_clim([-u_max/10, u_max/10])
            for i in [3,4]:
                ax[i,0].set_xticks([]); ax[i,0].yaxis.tick_left()
                ax[i,1].set_xticks([]); ax[i,1].yaxis.tick_right()

            ax[0,0].set_ylabel("Estimate")
            ax[1,0].set_ylabel("Original")
            ax[2,0].set_ylabel("Difference")
   
            ax[0,0].set_title(r"$u$")
            ax[0,1].set_title(r"$\zeta$")

            statfig.legend()
            statfig.tight_layout()
            statfig.subplots_adjust(wspace=0)
            statfig.subplots_adjust(hspace=0)
    
            new_name_split = plot_path.split('.')
            new_name = f'{new_name_split[0]}-state.{new_name_split[1]}'
            statfig.savefig(new_name)
            plt.close('all')
            plt.clf()

        gradfig = None
        if est['du'] is not None:
            nrows = 2;  ncols = 1
            height = 5; widths = 5
            gradfig, ax = plt.subplots(nrows, ncols, figsize=(widths,height))

            ax[0].plot(est['du'][0][b].flatten(), label=r'$d^2u/dt^2$', c='k', lw=1.3)
            ax[0].plot(est['du'][1][b].flatten(), label=r'$d^2u/dx^2+\cdots$', c='r', lw=1.0)
            ax[0].set_ylabel(r'$u$')
            ax[0].legend()

            ax[1].plot(est['dz'][0][b].flatten(), label=r'$d^2\zeta/dt^2$', c='k', lw=1.3)
            ax[1].plot(est['dz'][1][b].flatten(), label=r'$d^2\zeta/dx^2+\cdots$', c='r', lw=1.0)
            ax[1].set_ylabel(r'$\zeta$')
            ax[1].legend()

            gradfig.tight_layout()
            gradfig.subplots_adjust(wspace=0)
            gradfig.subplots_adjust(hspace=0)
            new_name_split = plot_path.split('.')
            new_name = f'{new_name_split[0]}-grad.{new_name_split[1]}'
            gradfig.savefig(new_name)
            plt.close('all')
            plt.clf()

        est_wav = est["wav"][b].squeeze()
        sf.write(wave_path.replace('.wav', f"-{b}-est.wav"), est_wav, samplerate=sr)
        if tar is not None:
            tar_wav = tar["wav"][b].squeeze()
            sf.write(wave_path.replace('.wav', f"-{b}-tar.wav"), tar_wav, samplerate=sr)

        d  = [ wandb.Image(specfig) ]
        d += [ wandb.Image(statfig) ] if statfig is not None else []
        d += [ wandb.Image(gradfig) ] if gradfig is not None else []
        d += [ wandb.Audio(est_wav, sample_rate=sr) ]
        d += [ wandb.Audio(tar_wav, sample_rate=sr) ] if tar is not None else []
        data.append(d)
        plt.close()

    columns  = ["spec"]
    columns += ["state"] if statfig is not None else []
    columns += ["grad"] if gradfig is not None else []
    columns += ["regenerated", "original"]
    return {
        "columns": columns,
        "data": data,
    }


