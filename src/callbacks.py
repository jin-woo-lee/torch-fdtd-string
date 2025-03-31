import os
import shutil
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import numpy as np
import soundfile as sf

from src.utils import plot as plot
from src.utils import audio as audio

class PlotResults(Callback):

    def __init__(self, args):
        self.debug = args.proc.debug
        self.plot_dirs = f'plot'
        self.wave_dirs = f'wave'
        os.makedirs(f"valid/{self.plot_dirs}", exist_ok=True)
        os.makedirs(f"valid/{self.wave_dirs}", exist_ok=True)
        os.makedirs(f"test/{self.plot_dirs}", exist_ok=True)
        os.makedirs(f"test/{self.wave_dirs}", exist_ok=True)
        
        self.sr = args.task.sr
        self.n_fft = args.callbacks.plot.n_fft
        self.n_mel = args.callbacks.plot.n_mel
        self.hop_length = args.callbacks.plot.hop_length
        self.window = torch.hann_window(self.n_fft)
        mel_fbank = audio.mel_basis(args.task.sr, self.n_fft, self.n_mel)
        self.mel_basis = torch.from_numpy(mel_fbank)

    def stft(self, x):
        x = torch.stft(x, n_fft = self.n_fft, hop_length = self.hop_length, win_length = self.n_fft, window = self.window)    
        return torch.view_as_complex(x).transpose(-1,-2)

    def logmag(self, spec):
        eps = torch.finfo(spec.abs().dtype).eps
        return 20 * (spec.abs() + eps).log10()

    def logmel(self, spec):
        eps = torch.finfo(spec.abs().dtype).eps
        mag = spec.abs() + eps
        mel = torch.matmul(self.mel_basis, mag.transpose(-1,-2)).transpose(-1,-2)
        return 20 * (mel + eps).log10()

    def summary(self, outputs, prefix, epoch, it):
        plot_path = f'{prefix}/{self.plot_dirs}/{epoch}-{it}.png'
        wave_path = f'{prefix}/{self.wave_dirs}/{epoch}-{it}.wav'
        inp_wave, tar_wave, est_wave = outputs

        batch_size = est_wave.shape[0]
        N = min(batch_size, 2)
        n = np.random.randint(batch_size-N) if batch_size > N else 0
        est_wave = est_wave[n:n+N].squeeze(1)
        inp_wave = inp_wave[n:n+N].squeeze(1)
        tar_wave = tar_wave[n:n+N].squeeze(1)

        est_spec = self.stft(est_wave)
        tar_spec = self.stft(tar_wave)
        inp_spec = self.stft(inp_wave)

        est_logmag = self.logmag(est_spec)
        est_logmel = self.logmel(est_spec)
        tar_logmag = self.logmag(tar_spec)
        tar_logmel = self.logmel(tar_spec)
        inp_logmag = self.logmag(inp_spec)
        inp_logmel = self.logmel(inp_spec)

        inp = {
            "state"  : inp_wave,
            "wav"    : inp_wave,
            "logmag" : inp_logmag,
            "logmel" : inp_logmel,
        }
        est = {
            "state"  : est_wave,
            "wav"    : est_wave,
            "logmag" : est_logmag,
            "logmel" : est_logmel,
        }
        tar = {
            "state"  : est_wave,
            "wav"   : tar_wave,
            "logmag": tar_logmag,
            "logmel": tar_logmel,
        }
        return plot.est_tar_specs(est, tar, inp, plot_path, wave_path, self.sr)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            #if not trainer.sanity_checking:
            prefix = "valid" if dataloader_idx == 0 else "test"
            summary = self.summary(outputs, prefix, trainer.current_epoch, batch_idx)
            if not self.debug:
                key = "valid" if dataloader_idx==0 else "test"
                trainer.logger.log_table(key=key, **summary)            

class SaveTestResults(Callback):

    def __init__(self, args):
        self.debug = args.proc.debug
        self.load_name = args.task.load_name
        if not os.path.isabs(args.task.ckpt_dir):
            ckpt_dir = f"{args.task.root_dir}/{args.task.ckpt_dir}"
        else:
            ckpt_dir = args.task.ckpt_dir
        self.video_dirs = f"{ckpt_dir}/test/{self.load_name}/video"
        self.state_dirs = f"{ckpt_dir}/test/{self.load_name}/state"
        self.score_dirs = f"{ckpt_dir}/test/{self.load_name}/score"
        os.makedirs(self.video_dirs, exist_ok=True)
        os.makedirs(self.state_dirs, exist_ok=True)
        if os.path.exists(self.score_dirs):
            print(f"*  Score file already exists! {self.score_dirs}")
            print(f"   Replacing with a new score...")
            shutil.rmtree(self.score_dirs)
        os.makedirs(self.score_dirs, exist_ok=True)
    
    def write_eval_scores(self, scores, epoch, it):
        out_score_path = f'{self.score_dirs}/output.txt'
        byp_score_path = f'{self.score_dirs}/modals.txt'
        for i, score_path in enumerate([out_score_path, byp_score_path]):
            keys = list(scores[i].keys())
            bs = scores[i][keys[0]].shape[0]
            if not os.path.exists(score_path):
                with open(score_path, 'a+') as f:
                    f.write('\t'.join(['id'] + keys) + '\n')
            with open(score_path, 'a+') as f:
                for b in range(bs):
                    line = [f"{epoch}-{it}-{b}"] + [f"{scores[i][key][b]:.8f}" for key in keys]
                    f.write('\t'.join(line) + '\n')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        data, scores = outputs
        self.write_eval_scores(scores, trainer.current_epoch, batch_idx)

class PlotStateVideo(Callback):

    def __init__(self, args):
        self.debug = args.proc.debug
        self.load_name = args.task.load_name
        if not os.path.isabs(args.task.ckpt_dir):
            ckpt_dir = f"{args.task.root_dir}/{args.task.ckpt_dir}"
        else:
            ckpt_dir = args.task.ckpt_dir
        self.video_dirs = f"{ckpt_dir}/test/{self.load_name}/video"
        self.state_dirs = f"{ckpt_dir}/test/{self.load_name}/state"
        os.makedirs(self.video_dirs, exist_ok=True)
        os.makedirs(self.state_dirs, exist_ok=True)

        self.sr = args.task.sr

    def summary(self, outputs, epoch, it):
        k = f'{epoch}-{it}'
        inp_wave, tar_wave, est_wave = outputs

        batch_size = est_wave.shape[0]
        est = est_wave.squeeze(1).numpy().T # (Nt, Bs=Nx)
        inp = inp_wave.squeeze(1).numpy().T # (Nt, Bs=Nx)
        tar = tar_wave.squeeze(1).numpy().T # (Nt, Bs=Nx)

        sf.write(f'{self.video_dirs}/estimate.wav', est.mean(-1), samplerate=self.sr)
        sf.write(f'{self.video_dirs}/analytic.wav', inp.mean(-1), samplerate=self.sr)
        sf.write(f'{self.video_dirs}/fdtd.wav',     tar.mean(-1), samplerate=self.sr)

        np.savez_compressed(f"{self.state_dirs}/{k}.npz", analytic=inp, estimate=est, simulate=tar)
        plot.state_specs(f"{self.state_dirs}/{k}.pdf", inp, est, tar)

        plot.rainbowgram(f'{self.video_dirs}/{k}-estimate.pdf', est.mean(-1), self.sr, colorbar=False)
        plot.rainbowgram(f'{self.video_dirs}/{k}-analytic.pdf', inp.mean(-1), self.sr, colorbar=False)
        plot.rainbowgram(f'{self.video_dirs}/{k}-fdtd.pdf',     tar.mean(-1), self.sr, colorbar=False)

        plot.state_video(self.video_dirs, est, self.sr, prefix=k, trim_front=True, fname='estimate')
        plot.state_video(self.video_dirs, inp, self.sr, prefix=k, trim_front=True, fname='analytic')
        plot.state_video(self.video_dirs, tar, self.sr, prefix=k, trim_front=True, fname='fdtd')
   
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        data, scores = outputs
        self.summary(data, trainer.current_epoch, batch_idx)


class PlotRDE(Callback):

    def __init__(self, args):
        self.debug = args.proc.debug
        self.plot_dirs = f"{args.task.root_dir}/{args.task.ckpt_dir}/test/plot"
        self.wave_dirs = f"{args.task.root_dir}/{args.task.ckpt_dir}/test/wave"
        os.makedirs(self.plot_dirs, exist_ok=True)
        os.makedirs(self.wave_dirs, exist_ok=True)
        
        self.sr = args.task.sr
        self.n_fft = args.callbacks.plot.n_fft
        self.n_mel = args.callbacks.plot.n_mel
        self.hop_length = args.callbacks.plot.hop_length
        self.window = torch.hann_window(self.n_fft)
        mel_fbank = audio.mel_basis(args.task.sr, self.n_fft, self.n_mel)
        self.mel_basis = torch.from_numpy(mel_fbank)

    def stft(self, x):
        x = torch.stft(x, n_fft = self.n_fft, hop_length = self.hop_length, win_length = self.n_fft, window = self.window)    
        return torch.view_as_complex(x).transpose(-1,-2)

    def logmag(self, spec):
        eps = torch.finfo(spec.abs().dtype).eps
        return 20 * (spec.abs() + eps).log10()

    def logmel(self, spec):
        eps = torch.finfo(spec.abs().dtype).eps
        mag = spec.abs() + eps
        mel = torch.matmul(self.mel_basis, mag.transpose(-1,-2)).transpose(-1,-2)
        return 20 * (mel + eps).log10()

    def summary(self, outputs, epoch, it):
        plot_path = f'{self.plot_dirs}/rde.png'
        wave_path = f'{self.wave_dirs}/rde.wav'
        sim_list, est_list, factors = outputs
        sim_wave, est_wave = [], []
        for i, (sim, est) in enumerate(zip(sim_list, est_list)):
            sim = audio.state_to_wav(sim.sum(-1))
            est = audio.state_to_wav(est.sum(-1))
            sim_wave.append(sim)
            est_wave.append(est)

        est_list = [x.detach().squeeze().cpu() for x in est_list]
        sim_list = [x.detach().squeeze().cpu() for x in sim_list]
        est_wave = [x.detach().squeeze().cpu() for x in est_wave]
        sim_wave = [x.detach().squeeze().cpu() for x in sim_wave]
        est_spec = [self.stft(x) for x in est_wave]
        sim_spec = [self.stft(x) for x in sim_wave]
        est_logmag = [self.logmag(x) for x in est_spec] 
        est_logmel = [self.logmel(x) for x in est_spec]
        sim_logmag = [self.logmag(x) for x in sim_spec] 
        sim_logmel = [self.logmel(x) for x in sim_spec]

        est = {
            "state"  : est_list,
            "wav"    : est_wave,
            "spec"   : est_spec,
            "logmag" : est_logmag,
            "logmel" : est_logmel,
        }
        sim = {
            "state" : sim_list,
            "wav"   : sim_wave,
            "spec"  : sim_spec,
            "logmag": sim_logmag,
            "logmel": sim_logmel,
        }
        return plot.rde_specs(factors, est, sim, plot_path, wave_path, self.sr)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            if not trainer.sanity_checking:
                summary = self.summary(outputs, trainer.current_epoch, batch_idx)
                trainer.logger.log_table(key="rde", **summary)            

class SaveResults(Callback):

    def __init__(self, args):
        self.debug = args.proc.debug
        self.sr = args.task.sr
        if isinstance(args.task.testset, str) or len(args.task.testset) == 1:
            self.save_dirs = f'eval/{args.task.testset[0]}'
            print(f"... Saving results under {self.save_dirs}")
            os.makedirs(f"{self.save_dirs}", exist_ok=True)
        else:
            self.save_dirs = f'eval/{args.task.testset[0]}'
            print(f"*** Mulitiple arguments provided for --testset: {args.task.testset}")
            print(f"    Saving results under {self.save_dirs}")
            os.makedirs(f"{self.save_dirs}", exist_ok=True)

    def summary(self, outputs, epoch, it):
        outputs = outputs.detach().cpu().numpy()
        save_dir = f"{self.save_dirs}/wave"
        os.makedirs(save_dir, exist_ok=True)
        audio.save_waves(outputs, save_dir, self.sr)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.summary(outputs, trainer.current_epoch, batch_idx)

