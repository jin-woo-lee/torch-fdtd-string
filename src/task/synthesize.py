import os
import json
import time
import math
import glob
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from einops import rearrange
from torch.utils.data import DataLoader
import torchaudio.functional as TAF
import torchaudio.transforms as TAT

import pytorch_lightning as pl
import torchmetrics

from src.utils import vnv as vnv
from src.utils import fdm as fdm
from src.utils import misc as misc
from src.utils import data as data
from src.utils import loss as loss
from src.utils import audio as audio
from src.utils import optimizer as opt
from src.utils import objective as obj
from src.utils import diffusion as diffusion

from src.dataset import synthesize as dataset
from src.model import analytic as analytic
from src.model.nn.synthesizer import Synthesizer

class Trainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if args.proc.train and args.task.ckpt_dir is None:
            self.save_hyperparameters()
        self.sr = args.task.sr
        self.Nt = int(self.sr * args.task.train_lens)
        self.trim = True if args.task.train_lens < args.task.total_lens else False

        framework_conf = vars(args.framework).copy()
        self.framework = framework_conf.pop('_name_')

        self.n_modes = args.model.n_modes
        self.n_ffts = args.task.n_fft
        self.window = torch.hann_window(self.n_ffts).cuda()
        self.block_size=args.model.block_size

        self.plot_test_video = args.task.plot_test_video

        ''' model archcitecture setup '''
        network_arch = vars(args.model).copy()['_name_']
        self.network_arch = network_arch
        framework_conf = vars(args.framework).copy()

        self.inharmonic = args.model.harmonic == 'inharmonic'
        self.upm = args.model.use_precomputed_mode

        if args.proc.test:
            from codes.src.model.nn.synthesizer import Synthesizer
            self.model = Synthesizer(
                harmonic=args.model.harmonic,
                ddsp_frequency_modulation=args.model.ddsp_frequency_modulation,
                #------------------------------ 
                embed_dim=args.model.embed_dim,
                #------------------------------ 
                t_scale=args.model.t_scale,
                x_scale=args.model.x_scale,
                gamma_scale=args.model.gamma_scale,
                kappa_scale=args.model.kappa_scale,
                alpha_scale=args.model.alpha_scale,
                sig_0_scale=args.model.sig_0_scale,
                sig_1_scale=args.model.sig_1_scale,
                #------------------------------ 
                n_modes=args.model.n_modes,
                n_bands=args.model.n_bands,
                hidden_dim=args.model.hidden_dim,
                block_size=args.model.block_size,
                sr=self.sr,
                #------------------------------ 
            )
        else:
            from src.model.nn.synthesizer import Synthesizer
            self.model = Synthesizer(
                harmonic=args.model.harmonic,
                ddsp_frequency_modulation=args.model.ddsp_frequency_modulation,
                #------------------------------ 
                embed_dim=args.model.embed_dim,
                #------------------------------ 
                t_scale=args.model.t_scale,
                x_scale=args.model.x_scale,
                gamma_scale=args.model.gamma_scale,
                kappa_scale=args.model.kappa_scale,
                alpha_scale=args.model.alpha_scale,
                sig_0_scale=args.model.sig_0_scale,
                sig_1_scale=args.model.sig_1_scale,
                #------------------------------ 
                n_modes=args.model.n_modes,
                n_bands=args.model.n_bands,
                hidden_dim=args.model.hidden_dim,
                block_size=args.model.block_size,
                sr=self.sr,
                #------------------------------ 
            )

        print(self.model)

        ''' loss setup '''
        opt_conf = vars(args.optimizer).copy()
        sch_conf = vars(args.scheduler).copy()
        self.opt_type = opt_conf.pop('_name_')
        self.sch_type = sch_conf.pop('_name_')
        self.opt_conf = opt_conf
        self.sch_conf = sch_conf
        self.interval = args.train.interval
        size_1 = min(self.Nt, 1024)
        size_2 = 2 ** int(math.log(size_1) / math.log(2) - 1)
        size_3 = 2 ** int(math.log(size_1) / math.log(2) - 2)
        self.magspec_kwargs = dict(
            fft_sizes   = [size_1,       size_2,      size_3],
            hop_sizes   = [size_1 // 4,  size_2 // 4, size_3 // 4],
            win_lengths = [size_1,       size_2,      size_3],
            w_log_mag=.5,
            w_lin_mag=2.,
        )
        self.melspec_kwargs = dict(
            fft_sizes   = [size_1],
            hop_sizes   = [size_1 // 4],
            win_lengths = [size_1],
            w_log_mag=.5,
            w_lin_mag=2.,
            scale="mel", n_bins=128, sample_rate=self.sr,
        )
        self.loss_conf = {
            "l1"     : (loss.L1Loss(scale_invariance=True), ['preds', 'target']),
            "f0"     : (loss.F0Loss(scale=1.0, weight=10.), ['preds_f0', 'target_f0']),
            "fk"     : (loss.FkLoss(scale=1.0, weight=1.0), ['preds_fk', 'target_fk']),
            "sisdr"  : (loss.SISDR(), ['preds', 'target']),
            "fft"    : (loss.FFTLoss(10), ['preds', 'target']),
            "magspec": (loss.MRSTFT(input_scale=10., **self.magspec_kwargs).cuda(), ['preds', 'target']),
            "melspec": (loss.MRSTFT(input_scale=10., **self.melspec_kwargs).cuda(), ['preds', 'target']),
            "modefreq": (loss.ModeFreqLoss(1.), ['preds_freq', 'target_fk']),
            "modeamps": (loss.ModeAmpsLoss(scale=200., weight=20), ['preds_coef', 'target_ck']),
        }
        self.loss_criteria = args.task.loss_criteria
        self.grad_clip = args.task.grad_clip

        self.results = dict()
        self.eval_criteria = args.task.eval_criteria
        self._init_torchmetrics("train")
        self._init_torchmetrics("valid")
        self._init_torchmetrics("test")

        ''' task setup '''
        self.data_dir = args.task.load_dir
        self.load_name = args.task.load_name

        ''' validation setup '''
        self.batch_size = args.task.batch_size
        self.valid_batch_size = args.task.valid_batch_size
        self.test_batch_size = args.task.test_batch_size

        ''' proc setup '''
        self.num_workers = args.proc.num_workers

    def train_dataloader(self):
        trainset = dataset.Trainset(
            data_dir=self.data_dir, load_name=self.load_name,
            trim=self.Nt if self.trim else None,
        )
        return DataLoader(
            trainset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True, sampler=None, drop_last=True,
        )

    def val_dataloader(self):
        # Return all val + test loaders
        validset = dataset.Testset(
            data_dir=self.data_dir, load_name=self.load_name, split='valid',
        )
        valid_loader = DataLoader(
            validset, batch_size=self.valid_batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True, sampler=None, drop_last=False,
        )
        test_loader = self.test_dataloader()
        return [valid_loader, test_loader]

    def test_dataloader(self):
        testset = dataset.Testset(
            data_dir=self.data_dir, load_name=self.load_name, split='test',
        )
        return DataLoader(
            testset, batch_size=self.test_batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True, sampler=None, drop_last=False,
        )

    def configure_optimizers(self):
        optimizer = opt.get_optimizer(self.opt_type, self.parameters(), self.opt_conf)
        scheduler = opt.get_scheduler(self.sch_type, optimizer, self.sch_conf)

        optimizer = optimizer if isinstance(optimizer, list) else [optimizer]
        if scheduler is None:
            self.optimizer = optimizer
            return optimizer
        else:
            scheduler = [{
                "scheduler": scheduler,
                "interval": self.interval,
            }]
            self.optimizer = optimizer
            return optimizer, scheduler

    def compute_loss(self, prefix, data_dict):
        results = dict()
        total_loss = 0
        results.update({f"{prefix}/artifact": []})
        for criterion in self.loss_criteria:
            # e.g.; metric=F.mse_loss, kwarg_names=['estimate', 'target']
            metric, kwarg_names = self.loss_conf[criterion]
            kwargs = {k: data_dict[k] for k in kwarg_names}
            if None in kwargs.values(): continue
            loss_val = metric(**kwargs)
            if isinstance(loss_val, list) or isinstance(loss_val, tuple):
                artifact = loss_val[1:]
                loss_val = loss_val[0]
                results[f"{prefix}/artifact"].append(artifact)
            total_loss = total_loss + loss_val
            results.update({f"{prefix}/loss-{criterion}": loss_val})
        results.update({f"{prefix}/loss": total_loss})
        return results

    def compute_eval(self, prefix, data_dict):
        for criterion in self.eval_criteria:
            metric, kwarg_names = self.results[prefix][criterion]
            kwargs = {k: data_dict[k] for k in kwarg_names}
            if None in kwargs.values(): continue
            metric(**kwargs)

    def _get_eval_conf(self):
        eval_dict = {
            "magspec": (obj.MultiSpec(**self.magspec_kwargs).cuda(), ['preds', 'target']),
            "melspec": (obj.MultiSpec(**self.melspec_kwargs).cuda(), ['preds', 'target']),
            "sisdr"   : (obj.SISDR().cuda(), ['preds', 'target']),
            "modefreq": (obj.ModeFreq().cuda(), ['preds_freq', 'target_fk']),
            "modeamps": (obj.ModeAmps().cuda(), ['preds_coef', 'target_ck']),
        }
        return eval_dict

    def _init_torchmetrics(self, prefix):
        self.results[prefix] = dict()
        eval_conf = self._get_eval_conf()
        for criterion in self.eval_criteria:
            self.results[prefix].update({
                criterion: eval_conf[criterion], 
            })

    def _reset_torchmetrics(self, prefix):
        for criterion in self.results[prefix].keys():
            self.results[prefix][criterion][0].reset()

    def process_results(self, prefix):
        out = dict()
        for k, (v, n) in self.results[prefix].items():
            v = v.compute()
            if isinstance(v, dict):
                for vk, vv in v.items():
                    if not vv.isnan():
                        out.update({f"{prefix}/{vk}": vv})
            else:
                if not v.isnan():
                    out.update({f"{prefix}/{k}": v})
        return out

    def on_train_epoch_start(self):
        self._reset_torchmetrics("train")

    def on_validation_epoch_start(self):
        self._reset_torchmetrics("valid")
        self._reset_torchmetrics("test")

    def on_test_epoch_start(self):
        self._reset_torchmetrics("test")

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        gt = batch["target"].float() # (Bs, Nt)
        xg = batch["x"].float().unsqueeze(1) # (Bs,  1)
        tg = batch["t"].float().squeeze(2)  # (Bs, Nt)
        ka = batch["kappa"].float().view(-1,1)
        al = batch["alpha"].float().view(-1,1)
        f_k = batch["mode_freq"].float()  # (Bs, n_modes)
        c_k = batch["mode_coef"].float()  # (Bs, 1,  1, n_modes)
        f_k = f_k.narrow(-1,0,self.n_modes).unsqueeze(1)
        c_k = c_k.narrow(-1,0,self.n_modes).squeeze(1)
        f_0 = batch["f0"].float()  # (Bs, Nt)
        u_0 = batch["u0"].float()  # (Bs, 1, Nx)
        t60 = batch["T60"].float()  # (Bs, 2, 2)
        gt_f0 = batch["ut_f0"].float()  # (Bs, 101)
        
        f_0 = misc.downsample(f_0, factor=self.block_size)
        gt_f0 = misc.downsample(gt_f0, size=f_0.size(1)) / self.sr * (2*math.pi)

        if self.inharmonic: params = [xg, tg, ka, al, t60, f_k, c_k]
        else: params = [xg, tg, ka, al, t60, None, None]

        ut, mode_input, mode_output = self.model(params, f_0, u_0)
        in_freq, in_coef = mode_input
        ut_freq, ut_coef = mode_output
        ut_f0 = ut_freq.select(-1,0) # (Bs, n_frames)

        batch.update(dict(preds=ut, target=gt))
        batch.update(dict(preds_f0=ut_f0, target_f0=gt_f0))
        batch.update(dict(preds_fk=ut_freq.narrow(1,-1,1), target_fk=f_k))
        batch.update(dict(preds_freq=in_freq))
        batch.update(dict(preds_coef=in_coef, target_ck=c_k))

        loss_dict = self.compute_loss("train", batch)
        self.compute_eval("train", batch)
        loss_dict.pop("train/artifact")
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss_dict["train/loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        an = batch["analytic"].float() # (Bs, Nt)
        gt = batch["target"].float() # (Bs, Nt)
        xg = batch["x"].float().unsqueeze(1) # (Bs,  1)
        tg = batch["t"].float().squeeze(2) # (Bs, Nt)
        ka = batch["kappa"].float().view(-1,1)
        al = batch["alpha"].float().view(-1,1)
        f_k = batch["mode_freq"].float()  # (Bs, n_modes)
        c_k = batch["mode_coef"].float()  # (Bs, 1,  1, n_modes)
        f_k = f_k.narrow(-1,0,self.n_modes).unsqueeze(1)
        c_k = c_k.narrow(-1,0,self.n_modes).squeeze(1)
        f_0 = batch["f0"].float()  # (Bs, Nt)
        u_0 = batch["u0"].float()  # (Bs, 1, Nx)
        t60 = batch["T60"].float()  # (Bs, 2, 2)
        gt_f0 = batch["ut_f0"].float()  # (Bs, 101)
        gain = batch["gain"].float().view(-1,1)  # (Bs, 1)

        f_0 = misc.downsample(f_0, factor=self.block_size)
        gt_f0 = misc.downsample(gt_f0, size=f_0.size(1)) / self.sr * (2*math.pi)

        if dataloader_idx == 0 and self.inharmonic:
            params = [xg, tg, ka, al, t60, f_k, c_k]
        else:
             params = [xg, tg, ka, al, t60, None, None]
        ut, mode_input, mode_output = self.model(params, f_0, u_0)
        in_freq, in_coef = mode_input
        ut_freq, ut_coef = mode_output
        ut_f0 = ut_freq.select(-1,0) # (Bs, n_frames)

        batch.update(dict(preds=ut, target=gt))
        batch.update(dict(preds_f0=ut_f0, target_f0=gt_f0))
        batch.update(dict(preds_fk=ut_freq.narrow(1,-1,1), target_fk=f_k))
        batch.update(dict(preds_freq=in_freq))
        batch.update(dict(preds_coef=in_coef, target_ck=c_k))

        if dataloader_idx == 0:
            # conduct validation
            self.compute_eval("valid", batch)
        else:
            # conduct test
            self.compute_eval("test", batch)

        an *= gain
        gt *= gain
        ut *= gain

        N = min(ut.size(0), 2)
        n = np.random.randint(ut.size(0)-N) if ut.size(0) > N else 0
        return an[n:n+N].detach().cpu(), gt[n:n+N].detach().cpu(), ut[n:n+N].detach().cpu()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        an = batch["analytic"].float() # (Bs, Nt)
        gt = batch["target"].float() # (Bs, Nt)
        xg = batch["x"].float().unsqueeze(1) # (Bs,  1)
        tg = batch["t"].float().squeeze(2) # (Bs, Nt)
        ka = batch["kappa"].float().view(-1,1)
        al = batch["alpha"].float().view(-1,1)
        f_k = batch["mode_freq"].float()  # (Bs, n_modes)
        c_k = batch["mode_coef"].float()  # (Bs, 1,  1, n_modes)
        f_k = f_k.narrow(-1,0,self.n_modes).unsqueeze(1)
        c_k = c_k.narrow(-1,0,self.n_modes).squeeze(1)
        f_0 = batch["f0"].float()  # (Bs, Nt)
        u_0 = batch["u0"].float()  # (Bs, 1, Nx)
        t60 = batch["T60"].float()  # (Bs, 2, 2)
        gt_f0 = batch["ut_f0"].float()  # (Bs, 101)
        an_f0 = batch["ua_f0"].float()  # (Bs, 101)
        gain = batch["gain"].float().view(-1,1)  # (Bs, 1)


        f_0 = misc.downsample(f_0, factor=self.block_size)
        gt_f0 = misc.downsample(gt_f0, size=f_0.size(1)) / self.sr * (2*math.pi)
        an_f0 = misc.downsample(an_f0, size=f_0.size(1)) / self.sr * (2*math.pi)

        if self.upm: params = [xg, tg, ka, al, t60, f_k, c_k]
        else: params = [xg, tg, ka, al, t60, None, None]
        ut, mode_input, mode_output = self.model(params, f_0, u_0)
        in_freq, in_coef = mode_input
        ut_freq, ut_coef = mode_output
        ut_f0 = ut_freq.select(-1,0) # (Bs, n_frames)

        scores = self.summarize_eval_scores(
            params=[xg, ka, al, t60, u_0],
            result=[ut, gt],
            pitchs=[ut_f0, gt_f0],
        )
        modal_scores = self.summarize_eval_scores(
            params=[xg, ka, al, t60, u_0],
            result=[an, gt],
            pitchs=[an_f0, gt_f0],
        )


        batch.update(dict(preds=ut, target=gt))
        batch.update(dict(preds_f0=ut_f0, target_f0=gt_f0))
        batch.update(dict(preds_fk=ut_freq.narrow(1,-1,1), target_fk=f_k))
        batch.update(dict(preds_freq=in_freq))
        batch.update(dict(preds_coef=in_coef, target_ck=c_k))

        self.compute_eval("test", batch)

        an *= gain
        gt *= gain
        ut *= gain

        if self.plot_test_video:
            return [an.detach().cpu(), gt.detach().cpu(), ut.detach().cpu()], [scores, modal_scores]
        else:
            return [], [scores, modal_scores]

    def summarize_eval_scores(self, params, result, pitchs):
        xg, ka, al, t60, u_0 = params
        ut, gt = result
        ut_f0, gt_f0 = pitchs

        p_a = u_0.squeeze(1).max(-1).values
        p_x = torch.argmax(u_0.squeeze(1), dim=-1) / 255.

        si_sdr = loss.si_sdr(gt, ut) # (Bs, )
        si_sdr = loss.si_sdr(gt, ut) # (Bs, )
        sdr = loss.si_sdr(gt, ut, scaling=False) # (Bs, )
        stft_dict = loss.stft_loss(ut, gt) # (Bs, )

        detune = (ut_f0 - gt_f0).abs() / (2*math.pi) * self.sr # Hz
        detune = detune.flatten(1).mean(1, keepdim=False)

        results = dict(
            x_grid=xg.squeeze(1).detach().cpu().numpy(), # (Bs, )
            #------------------------------  
            kappa=ka.squeeze(1).detach().cpu().numpy(), # (Bs, )
            alpha=al.squeeze(1).detach().cpu().numpy(), # (Bs, )
            p_a=p_a.detach().cpu().numpy(), # (Bs, )
            p_x=p_x.detach().cpu().numpy(), # (Bs, )
            #------------------------------  
            si_sdr=si_sdr.detach().cpu().numpy(), # (Bs, )
            sdr=sdr.detach().cpu().numpy(), # (Bs, )
            logmag=stft_dict["logmag"].detach().cpu().numpy(), # (Bs, )
            #------------------------------  
            f0_error=detune.detach().cpu().numpy(), # (Bs, )
            #------------------------------  
        )
        return results

    def training_epoch_end(self, outputs):
        # Log training torchmetrics
        super().training_epoch_end(outputs)
        self.log_dict(
            self.process_results("train"),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def validation_epoch_end(self, outputs):
        # Log all validation torchmetrics
        super().validation_epoch_end(outputs)
        for name in ["valid", "test"]:
            self.log_dict(
                self.process_results(name),
                on_step=False,
                on_epoch=True, 
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

    def test_epoch_end(self, outputs):
        # Log all validation torchmetrics
        super().test_epoch_end(outputs)
        results = self.process_results("test")





