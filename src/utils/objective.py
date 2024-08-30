import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
import random
import numpy.polynomial.polynomial as poly
from einops import rearrange

import torchmetrics
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics import ScaleInvariantSignalDistortionRatio as SI_SDR
from auraloss.freq import MultiResolutionSTFTLoss as MRSTFT

from src.utils import loss as loss

# Coefficients for polynomial fitting
COEFS_SIG = np.array([9.651228012789436761e-01, 6.592637550310214145e-01,
                    7.572372955623894730e-02])
COEFS_BAK = np.array([-3.733460011101781717e+00,2.700114234092929166e+00,
                    -1.721332907340922813e-01])
COEFS_OVR = np.array([8.924546794696789354e-01, 6.609981731940616223e-01,
                    7.600269530243179694e-02])


def pesq(sr, ref, est, len_x=None, mask=None, mode='wb', return_type='sum'):
    '''
    if mask is not None:
        ref = ref[torch.where(mask == 1)[0]]
        est = est[torch.where(mask == 1)[0]]
        mask = mask[torch.where(mask == 1)[0]]
        len_x = len_x[torch.where(mask==1)[0]]
    '''
    batch_size = ref.shape[0]

    #pesq_ = torch.zeros_like(mask, dtype=torch.float)
    #run_pesq = PESQ(sr, mode)
    #for ndx in range(batch_size):
    #    if mask is None or mask[ndx] == 1:
    #        ed = int(len_x[ndx].tolist())
    #        pesq_[ndx] = run_pesq(ref[ndx, :ed], est[ndx, :ed])
    pesq_ = list()
    run_pesq = PESQ(sr, mode)
    for ndx in range(batch_size):
        ed = int(len_x[ndx].tolist())
        # TODO: handle cypesq.NoUtterancesError properly?
        try:
            p = run_pesq(ref[ndx, :ed], est[ndx, :ed])
            pesq_.append(p)
        except:
            pass
    pesq_ = torch.FloatTensor(pesq_)

    if return_type == 'sum':
        pesq_ = torch.sum(pesq_)
    elif return_type == 'mean':
        pesq_ = torch.mean(pesq_)
    else:
        pass
    return pesq_

def audio_logpowspec(audio, nfft=320, hop_length=160, sr=16000):
    powspec = (np.abs(librosa.core.stft(audio, n_fft=nfft, hop_length=hop_length)))**2
    logpowspec = np.log10(np.maximum(powspec, 10**(-12)))
    return logpowspec.transpose(0,2,1)


def dns_mos(x, len_x=None, path_model='models/dns_mos/', mos_type='sig'):

    sr = 16000
    time_target = 9
    len_samples = int(time_target * sr)
    providers = ['CPUExecutionProvider']

    if mos_type == 'sig':
        session_sig = ort.InferenceSession(os.path.join(path_model, 'sig.onnx'), providers=providers)

    session_bak_ovr = ort.InferenceSession(os.path.join(path_model, 'bak_ovr.onnx'), providers=providers)

    batch_size = x.shape[0]
    if len_x is not None:
        audio_tmp = []
        lens_tmp = len_x

        for ndx in range(batch_size):
            ed = int(len_x[ndx].tolist())
            audio = x[ndx, :ed]
            while audio.shape[-1] < len_samples:
                audio = np.append(audio, x[ndx,:ed], axis=-1)
                lens_tmp[ndx] += lens_tmp[ndx]
            audio_tmp.append(audio)

        min_len = int(lens_tmp.min())
        audio_tmp = [aud[:min_len] for aud in audio_tmp]

        audio = np.stack(audio_tmp, 0)

    else:
        audio = x
        while audio.shape[-1] < len_samples:
            audio = np.append(audio, x, axis=-1)

    num_hops = int(np.floor(audio.shape[-1]/sr) - time_target)+1
    hop_len_samples = sr

    audio_seg = []
    for idx in range(num_hops):
        audio_seg.append(audio[:,int(idx*hop_len_samples) : int((idx+time_target)*hop_len_samples)])

    len_seg = len(audio_seg)
    audio_seg = np.concatenate(audio_seg, axis=0)

    spec = audio_logpowspec(audio_seg)
    spec = spec.astype('float32')

    mos_sig = None
    if mos_type == 'sig':
        onnx_inputs_sig = {inp.name: spec for inp in session_sig.get_inputs()}
        mos_sig = poly.polyval(session_sig.run(None, onnx_inputs_sig), COEFS_SIG)
        mos_sig = mos_sig.squeeze().reshape(len_seg, batch_size).mean(0)

    onnx_inputs_bak_ovr = {inp.name: spec for inp in session_bak_ovr.get_inputs()}
    mos_bak_ovr = session_bak_ovr.run(None, onnx_inputs_bak_ovr)
    mos_bak = poly.polyval(mos_bak_ovr[0][:,1], COEFS_BAK)
    mos_ovr = poly.polyval(mos_bak_ovr[0][:,2], COEFS_OVR)

    mos_bak = mos_bak.squeeze().reshape(len_seg, batch_size).mean(0)
    mos_ovr = mos_ovr.squeeze().reshape(len_seg, batch_size).mean(0)

    return mos_sig, mos_bak, mos_ovr


def si_sdr(reference_signals, estimated_signal, source_idx=0, scaling=True):
    batch_size = estimated_signal.shape[0]
    print(reference_signals.shape)
    print(estimated_signal.shape)

    Rss = torch.bmm(reference_signals, reference_signals.permute(0,2,1))
    this_s = reference_signals[:, source_idx]

    if scaling:
        a = torch.sum(this_s*estimated_signal, dim=-1) / Rss[:, source_idx, source_idx]
    else:
        a = torch.ones(batch_size).to(Rss)

    e_true = a.unsqueeze(-1) * this_s
    e_res = estimated_signal - e_true
    print(e_res.shape)

    Sss = (e_true**2).sum(dim=-1)
    Snn = (e_res**2).sum(dim=-1)

    SDR = 10 * torch.log10(Sss/Snn)

    Rsr = torch.bmm(reference_signals, e_res.unsqueeze(-1)).squeeze(-1)
    try:
        b = torch.linalg.solve(Rss, Rsr)
    except:
        #print(Rss)
        #print('-'*30)
        #print(Rsr)
        #raise Exception
        b = torch.empty(2,2).to(Rss.device)

    e_interf = torch.bmm(b.unsqueeze(1), reference_signals).squeeze(1)
    e_artif = e_res - e_interf

    SIR = 10 * torch.log10(Sss / (e_interf**2).sum(dim=-1))
    SAR = 10 * torch.log10(Sss / (e_artif**2).sum(dim=-1))

    return SDR, SIR, SAR


def aec_mos(lpb_sig, mic_sig, enh_sig, talk_type, mos_model=None, path_model='AECMOS/AECMOS_local/', sr=16000):

    if mos_model is None:
        sys.path.append(path_model); import aecmos
        mos = aecmos.AECMOSEstimator(f"{path_model}/Run_1663915512_Stage_0.onnx")
    else:
        mos = mos_model

    MOS_1 = {}; MOS_2 = {}
    for talk in ['nst', 'st', 'dt']:
        MOS_1.update({talk: []})
        MOS_2.update({talk: []})
    batch_size, lens = enh_sig.shape
    assert lens == lpb_sig.shape[1]
    assert lens == mic_sig.shape[1]
    for n in range(batch_size):
        talk = talk_type[n]
        assert talk in ['nst', 'st', 'dt'], talk

        #++++++++++++++++++++++++++++++ 
        # https://github.com/microsoft/AEC-Challenge/tree/main/AECMOS/AECMOS_local#nb
        if talk == 'dt':
            start_point = (lens - int(15. * sr) // 2)
        elif talk == 'st':
            start_point = lens // 2
        else:
            start_point = 0
        #start_point = 0
        #++++++++++++++++++++++++++++++ 
        lpb_ = lpb_sig[n, start_point:]
        mic_ = mic_sig[n, start_point:]
        enh_ = enh_sig[n, start_point:]

        scores = mos.run(talk_type[n], lpb_, mic_, enh_)
        #print(f'The AECMOS echo score is {scores[0]}, and (other) degradation score is {scores[1]}.')
        MOS_1[talk].append(scores[0])
        MOS_2[talk].append(scores[1])

    return MOS_1, MOS_2


class AECMOS(torchmetrics.Metric):
    def __init__(self, path_model):
        super().__init__()
        sys.path.append(path_model); import aecmos
        self.mos = aecmos.AECMOSEstimator(f"{path_model}/Run_1663915512_Stage_0.onnx")

        self.map = {"doubletalk": "dt", "farend-singletalk": "st", "nearend-singletalk": "nst", }
        self.add_state("dt_echo_dmos",  default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("dt_other_dmos", default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("st_echo_mos",   default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("nst_near_mos",  default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("size_dt", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("size_st", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("size_nst", default=torch.tensor(0), dist_reduce_fx="sum")

    def filter(self, s_list):
        return [self.map[s] if s in list(self.map.keys()) else s for s in s_list]

    def update(self, near_end, far_end, estimated, scenario):
        scenario = self.filter(scenario)
        if estimated.size(1) == 2:
            estimated = estimated.narrow(1,0,1)  # enhanced
        mos_score_1, mos_score_2 = aec_mos(
            far_end.squeeze(1).detach().cpu().numpy(),
            near_end.squeeze(1).detach().cpu().numpy(),
            estimated.squeeze(1).detach().cpu().numpy(),
            talk_type=scenario,
            mos_model=self.mos,
        )
        self.dt_echo_dmos  += sum(mos_score_1['dt'])
        self.dt_other_dmos += sum(mos_score_2['dt'])
        self.st_echo_mos   += sum(mos_score_1['st'])
        self.nst_near_mos  += sum(mos_score_1['nst'])

        self.size_dt  += len(mos_score_1['dt'])
        self.size_st  += len(mos_score_1['st'])
        self.size_nst += len(mos_score_1['nst'])

    def compute(self):
        dt_echo_dmos  = self.dt_echo_dmos  / self.size_dt
        dt_other_dmos = self.dt_other_dmos / self.size_dt
        st_echo_mos   = self.st_echo_mos   / self.size_st
        nst_near_mos  = self.nst_near_mos  / self.size_nst
        return {
            "dt_echo_dmos" : dt_echo_dmos,
            "dt_other_dmos": dt_other_dmos,
            "st_echo_mos"  : st_echo_mos,
            "nst_near_mos" : nst_near_mos,
        }

class MultiSpec(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__()
        self.mrstft = MRSTFT(**kwargs)
        self.add_state("value", default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ''' (Bs, Nt, Nx, 2)
        '''
        preds  = rearrange(preds,  'b t c -> b c t').sum(-1)
        target = rearrange(target, 'b t c -> b c t').sum(-1)
        val = self.mrstft(preds, target)
        self.value += val * target.size(0)
        self.size += target.size(0)

    def compute(self):
        return self.value / self.size


class SISDR(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.sisdr = SI_SDR()
        self.add_state("value", default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ''' (Bs, Nt) '''
        val = self.sisdr(preds.flatten(1), target.flatten(1))
        self.value += val * target.size(0)
        self.size += target.size(0)

    def compute(self):
        return self.value / self.size



class ModeFreq(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.metric = nn.L1Loss()
        self.add_state("value", default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds_freq, target_fk):
        val = self.metric(preds_freq, target_fk)
        self.value += val * target_fk.size(0)
        self.size += target_fk.size(0)

    def compute(self):
        return self.value / self.size

class ModeAmps(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.metric = nn.L1Loss()
        self.add_state("value", default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds_coef, target_ck):
        val = self.metric(preds_coef, target_ck)
        self.value += val * target_ck.size(0)
        self.size += target_ck.size(0)

    def compute(self):
        return self.value / self.size



class FeatureL1(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.metric = nn.L1Loss()
        self.add_state("value", default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, **kwargs):
        val = 0
        for key in kwargs.keys():
            preds, target = kwargs[key]
            val += self.metric(preds, target)
        self.value += val * target.size(0)
        self.size += target.size(0)

    def compute(self):
        return self.value / self.size


class ESTOI(torchmetrics.Metric):
    def __init__(self, sr=16000):
        super().__init__()
        self.estoi = loss.ESTOILoss(sr=sr)
        self.add_state("value", default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.value -= self.estoi(preds, target) * target.size(0)
        self.size += target.size(0)

    def compute(self):
        return self.value / self.size

class MSE(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("value", default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.value += F.mse_loss(preds, target) * target.size(0)
        self.size += target.size(0)

    def compute(self):
        return self.value / self.size

class L1(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("value", default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.value += F.l1_loss(preds, target) * target.size(0)
        self.size += target.size(0)

    def compute(self):
        return self.value / self.size

class PDELoss(torchmetrics.Metric):
    def __init__(self, f_ic, f_bc, f_r, w_ic=1., w_bc=1., w_r=1.):
        super().__init__()
        self.metric = loss.PDELoss(
            f_ic, f_bc, f_r,
            w_ic, w_bc, w_r)
        self.add_state("value", default=torch.tensor(0.).float(), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self,
            pde_preds: torch.Tensor, # NN prediction
            #---------- 
            u0: torch.Tensor, # initial condition
            f0: torch.Tensor, # input fundamental frequency
            kappa: torch.Tensor, # string relative stiffness
            alpha: torch.Tensor, # string nonlinearity parameter
            sig0: torch.Tensor, # frequency-independent loss 
            sig1: torch.Tensor, # frequency-dependent loss 
            bow_mask: torch.Tensor, # indicates bow excitation
            hammer_mask: torch.Tensor, # indicates hammer excitation
            #---------- 
            # bow parameters
            x_B, v_B, F_B, ph0_B, ph1_B, wid_B,
            #---------- 
            # hammer parameters
            x_H, v_H, u_H, w_H, M_H, a_H,
            #---------- 
            xr: torch.Tensor, # space variable
            tr: torch.Tensor, # time variable
        ):
        val = self.metric(
            pde_preds,
            u0, f0, kappa, alpha, sig0, sig1,
            bow_mask, hammer_mask,
            x_B, v_B, F_B, ph0_B, ph1_B, wid_B,
            x_H, v_H, u_H, w_H, M_H, a_H,
            xr, tr,
        )

        self.value += val * preds.size(0)
        self.size += preds.size(0)

    def compute(self):
        return self.value / self.size



