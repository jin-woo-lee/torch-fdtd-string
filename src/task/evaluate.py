import os
import time
import glob
import tqdm
import torch
import numpy as np
import soundfile as sf

from src.utils import plot as plot
from src.utils.misc import f0_interpolate
from src.utils.analysis.frequency import compute_harmonic_parameters
from src.utils.fdm import stiff_string_modes

def evaluate(load_dir):
    data_list = sorted(glob.glob(f"{load_dir}/*/string_params.npz"))
    iterator = tqdm.tqdm(data_list)
    iterator.set_description("Evaluating")
    for path in iterator:
        iterator.set_postfix(path=path)
        subd = path.split('/')[-2]
        string_data = np.load(path)
        bow_data = np.load(path.replace('string_params.npz', 'bow_params.npz'))
        hammer_data = np.load(path.replace('string_params.npz', 'hammer_params.npz'))

        uout, sr = sf.read(path.replace('string_params.npz', 'output-u.wav'))
        zout, sr = sf.read(path.replace('string_params.npz', 'output-z.wav'))
        k       = 1 / sr
        theta_t = 0.5 + 2/(np.pi**2)

        f0_input  = string_data["f0"]
        T60       = string_data["T60"]
        kappa_rel = string_data["kappa"]
        alpha     = string_data["alpha"]
        f0_target = string_data["target_f0"]

        kappa = (2 * f0_input * kappa_rel).mean()
        modes = stiff_string_modes(f0_input, kappa_rel, 10)[0]

        h_params = compute_harmonic_parameters(uout, sr)
        f0_estimate = h_params['f0']
        f0_input_interpolated  = f0_interpolate(f0_input,  len(f0_estimate), len(uout) / sr)
        f0_target_interpolated = f0_interpolate(f0_target, len(f0_estimate), len(uout) / sr)
        modes_interpolated = [f0_interpolate(m, len(f0_estimate), len(uout) / sr) for m in modes]
        f0_diff_input  = np.mean(np.abs(f0_input_interpolated  - f0_estimate))
        f0_diff_target = np.mean(np.abs(f0_target_interpolated - f0_estimate))
        f0_diff_modes  = np.mean(np.abs(modes_interpolated[0]  - f0_estimate))

        front = int(len(f0_estimate) * 0.2)  # 0.2 sec
        f0_diff_input_front = np.mean(np.abs(f0_input_interpolated[:front] - f0_estimate[:front]))
        f0_diff_modes_front = np.mean(np.abs(modes_interpolated[0][:front] - f0_estimate[:front]))

        with open(f"{load_dir}/{subd}/string_params.txt", 'w') as f:
            f.write(f"f0 diff (input)\t{f0_diff_input:.2f}\n")
            f.write(f"f0 diff (target)\t{f0_diff_target:.2f}\n")
            f.write(f"f0 diff (modes)\t{f0_diff_modes:.2f}\n")
            f.write(f"f0 diff input front\t{f0_diff_input_front:.2f}\n")
            f.write(f"f0 diff modes front\t{f0_diff_modes_front:.2f}\n")
        #plot_spectrum_uz(f'{load_dir}/{subd}/spectrum.pdf', uout, zout, f0_input, f0_estimate, modes, sr)
        plot.rainbowgram(f'{load_dir}/{subd}/spec.pdf', uout, sr, colorbar=False)
        plot.rainbowgram(f'{load_dir}/{subd}/f0-naive.pdf',   uout, sr, f0_input=f0_input, colorbar=False)
        plot.rainbowgram(f'{load_dir}/{subd}/f0-precorrected.pdf',   uout, sr, f0_input=f0_target, colorbar=False)


