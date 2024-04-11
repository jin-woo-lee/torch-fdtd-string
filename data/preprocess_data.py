import os
import glob
import librosa
import librosa.display

import torch
import torch.nn.functional as F
import torchaudio.transforms as TAT
import torchaudio.transforms as TAF
import crepe

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def plot_spectrogram(
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

    D = librosa.stft(out, n_fft=n_fft, hop_length=hop_length, pad_mode='reflect')
    mag, phase = librosa.magphase(D)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.times_like(D, sr=sr, hop_length=hop_length)

    logmag = librosa.amplitude_to_db(mag, ref=np.max)

    #width = 2.5; height = 1.9
    width = 30; height = 5
    plt.figure(figsize=(width,height))
    spec = librosa.display.specshow(
        logmag,
        n_fft=n_fft, hop_length=hop_length, sr=sr,
        y_axis='log', x_axis='time',
    )
    if colorbar:
        cbar = plt.colorbar(spec, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.ax.set(yticklabels=['$-\pi$', '$-\pi/2$', "$0$", '$\pi/2$', '$\pi$']);

    def add_plot(freqs, label=None, ls=None, lw=2., dashes=(None,None)):
        x = np.linspace(1/sr, t_max, freqs.shape[-1])
        freqs = np.interp(times, x, freqs)
        line, = plt.plot(times - times[0], freqs, label=label, color='white', lw=lw, ls=ls, dashes=dashes)
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

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    #plt.xaxis.set_visible(False)
    #plt.yaxis.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=-1e-6)
    plt.clf()
    plt.close("all")

def spectrogram(x, n_fft=1024, hop_length=None, logscale=False):
    L = 4
    if x.shape[-1] > 2*n_fft:
        hop_length = n_fft // L if hop_length is None else hop_length
    else:
        n_fft = x.shape[-1] // 2
        hop_length = n_fft // L

    X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, pad_mode='reflect')
    mag, phase = librosa.magphase(X)
    mag = librosa.amplitude_to_db(mag, ref=np.max) if logscale else mag
    return mag, phase

def load_wav(root_dir, filename, target_sr):
    wav_path = f"{root_dir}/{filename}/input.wav"
    if os.path.exists(wav_path):
        x, sr = sf.read(wav_path)
    else:
        x, sr = librosa.load(librosa.example(filename))

    if sr != target_sr:
        resampler = TAT.Resample(sr, target_sr, resampling_method='kaiser_window')
        x = resampler(torch.from_numpy(x)).numpy()
        sr = target_sr
    if not os.path.exists(wav_path):
        os.makedirs(f'{root_dir}/{filename}', exist_ok=True)
        sf.write(f"{root_dir}/{filename}/input.wav", x, samplerate=sr)
    return x, sr

def inverse_spectrogram(mag, phase, n_fft=1024, hop_length=None):
    hop_length = n_fft // 4 if hop_length is None else hop_length
    X = mag * phase
    x = librosa.istft(X, n_fft=n_fft, hop_length=hop_length)
    return x

def get_amplitude(x):
    X_mag, X_phs = spectrogram(x) # (F, T)
    X_rms = np.sqrt(np.mean(X_mag**2, axis=0)+1e-5)
    return X_rms # (T,)

def sine_like(freqs, length, sr):
    time_axis_1 = np.arange(length) / sr
    time_axis_2 = np.linspace(1/sr, length / sr, freqs.shape[-1])
    freqs = np.interp(time_axis_1, time_axis_2, freqs)
    phase = np.cumsum(freqs)
    return np.sin(2 * np.pi * phase / sr)

def AM(x, amp, sr):
    X_mag, X_phs = spectrogram(x) # (F, T)
    X_rms = np.sqrt(np.mean(X_mag**2, axis=0, keepdims=True)+1e-5)
    X_mag = X_mag / X_rms
    X_mag = X_mag * amp[None,:]
    x = inverse_spectrogram(X_mag, X_phs)
    return x

def running_avg(x, N=1024, threshold=0.3):
    w = np.pad(np.ones(N)/N, (N,0))
    x = np.where(x > threshold, x, np.zeros(x.shape))
    x = np.convolve(x, w, mode='same')
    return x

def process_f0(root_dir, filename, target_sr):
    x, sr = load_wav(root_dir, filename, target_sr)

    f0_path = f'{root_dir}/{filename}/string-f0.npy'
    if os.path.exists(f0_path):
        f0 = np.load(f0_path)
    else:
        os.makedirs(f'{root_dir}/{filename}', exist_ok=True)
        time, f0, confidence, activation = crepe.predict(x, sr, viterbi=True)
        np.save(f0_path, f0)
    return x, f0

def process_amp(root_dir, filename, target_sr):
    x, sr = load_wav(root_dir, filename, target_sr)

    ''' get f0 '''
    f0_path = f'{root_dir}/{filename}/string-f0.npy'
    f0 = np.load(f0_path)
    if len(f0) != len(x):
        time_axis_1 = np.arange(len(x)) / sr
        time_axis_2 = np.linspace(1/sr, len(x) / sr, len(f0))
        f0 = np.interp(time_axis_1, time_axis_2, f0)
        np.save(f0_path, f0)
    
    ''' get amplitude '''
    amp_path = f'{root_dir}/{filename}/amp.npy'
    amp = get_amplitude(x)
    
    y1 = sine_like(f0,  x.shape[-1], sr)
    y2 = AM(y1, amp, sr)
    
    if len(amp) != len(x):
        time_axis_1 = np.arange(len(x)) / sr
        time_axis_2 = np.linspace(1/sr, len(x) / sr, len(amp))
        amp = np.interp(time_axis_1, time_axis_2, amp)
    
    force = running_avg(amp)
    force = 100 * (force/2+ 1e-5)**.1
    force = np.where(force > 40, force, np.zeros(force.shape))
    force_path = f'{root_dir}/{filename}/bow-F_b.npy'
    
    
    o_env = librosa.onset.onset_strength(y=x, sr=sr)
    time_axis_f = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    scale = x.shape[-1] / time_axis_f.shape[-1]
    hammer = np.zeros(x.shape[-1])
    onset_sample = np.array(onset_frames * scale).astype(int)
    hammer[onset_sample] = 1
    hammer_path = f'{root_dir}/{filename}/hammer-v_H.npy'
    
    y3 = x * running_avg(hammer)
    
    np.save(force_path, force)
    np.save(hammer_path, hammer)

    sf.write(f"{root_dir}/{filename}/sine-f0.wav",     y1, sr)
    sf.write(f"{root_dir}/{filename}/sine-f0-amp.wav", y2, sr)
    sf.write(f"{root_dir}/{filename}/sine-f0-ham.wav", y3, sr)

    return y1, y2, y3

if __name__=='__main__':
    root_dir = 'data'
    filename = 'trumpet'
    sr = 48000

    x, f0      = process_f0(root_dir, filename, sr)
    y1, y2, y3 = process_amp(root_dir, filename, sr)

    plot_spectrogram(f'{root_dir}/{filename}/spec.pdf',        x,  sr, f0_input=f0, colorbar=False)
    plot_spectrogram(f'{root_dir}/{filename}/spec-f0.pdf',     y1, sr, colorbar=False)
    plot_spectrogram(f'{root_dir}/{filename}/spec-f0-amp.pdf', y2, sr, colorbar=False)
    plot_spectrogram(f'{root_dir}/{filename}/spec-f0-ham.pdf', y3, sr, colorbar=False)


    sample_list = glob.glob(f'{root_dir}/{filename}/sample-*.wav')
    for sp in sample_list:
        x, sr = sf.read(sp)
        sample_name = sp.split('/')[-1].split('.')[0]
        plot_spectrogram(f'{root_dir}/{filename}/{sample_name}.pdf', x, sr, colorbar=False)


