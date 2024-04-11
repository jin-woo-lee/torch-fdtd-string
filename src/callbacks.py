import os
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback

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
        return torch.view_as_complex(x).transpose(1,2)

    def logmag(self, spec):
        eps = torch.finfo(spec.abs().dtype).eps
        return 20 * (spec.abs() + eps).log10()

    def logmel(self, spec):
        eps = torch.finfo(spec.abs().dtype).eps
        mag = spec.abs() + eps
        mel = torch.matmul(self.mel_basis, mag.transpose(1,2)).transpose(1,2)
        return 20 * (mel + eps).log10()

    def summary(self, outputs, prefix, epoch, it):
        plot_path = f'{prefix}/{self.plot_dirs}/{epoch}-{it}.png'
        wave_path = f'{prefix}/{self.wave_dirs}/{epoch}-{it}.wav'
        tar_state, est_state, du, dz = outputs
        if tar_state.shape[-1] == 2:
            assert len(list(tar_state.shape)) == 4, tar_state.shape # (Bs, Nt, Nx, 2)
            tar_wave = audio.state_to_wav(tar_state.sum(-1)) # (Bs, Nt)
            est_wave = audio.state_to_wav(est_state.sum(-1)) # (Bs, Nt)
        else:
            assert len(list(tar_state.shape)) == 2, tar_state.shape # (Bs, Nt)
            tar_wave = tar_state
            est_wave = est_state

        batch_size = est_wave.shape[0]
        N = min(batch_size, 3)
        est_wave = est_wave[:N].detach().squeeze(1).cpu()
        tar_wave = tar_wave[:N].detach().squeeze(1).cpu() if tar_wave is not None else None
        est_state = est_state[:N].detach().cpu()
        tar_state = tar_state[:N].detach().cpu() if tar_state is not None else None
        du = [x[:N].detach().cpu().numpy() for x in du] if du is not None else None
        dz = [x[:N].detach().cpu().numpy() for x in dz] if dz is not None else None

        est_spec = self.stft(est_wave)
        tar_spec = self.stft(tar_wave) if tar_wave is not None else None

        est_logmag = self.logmag(est_spec)
        est_logmel = self.logmel(est_spec)
        tar_logmag = self.logmag(tar_spec) if tar_wave is not None else None
        tar_logmel = self.logmel(tar_spec) if tar_wave is not None else None

        est = {
            "state"  : est_state,
            "wav"    : est_wave,
            "logmag" : est_logmag,
            "logmel" : est_logmel,
            "du"     : du,
            "dz"     : dz,
        }
        tar = {
            "state" : tar_state,
            "wav"   : tar_wave,
            "logmag": tar_logmag,
            "logmel": tar_logmel,
        } if tar_wave is not None else None
        return plot.est_tar_specs(est, tar, plot_path, wave_path, self.sr)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            if not trainer.sanity_checking:
                prefix = "valid" if dataloader_idx == 0 else "test"
                summary = self.summary(outputs, prefix, trainer.current_epoch, batch_idx)
                if not self.debug:
                    key = "valid" if dataloader_idx==0 else "test"
                    trainer.logger.log_table(key=key, **summary)            

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

if __name__=='__main__':

    est = {
        "wav"   : torch.empty(1,16000),
        "logmag": torch.empty(1,100,513),
        "logmel": torch.empty(1,100,128),
    }
    tar = est
    plot.est_tar_specs(est, tar, "check/check.png", "check/check.wav", 48000)
   
