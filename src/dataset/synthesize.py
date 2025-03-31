import torch
import numpy as np 
from torch.utils.data import DataLoader
import os 
import soundfile as sf 
import pickle
import glob
import scipy
import random
import librosa
from tqdm import tqdm
import json
import time
import torch.nn.functional as F
import torchaudio.functional as TAF
import src.utils.audio as audio
import src.utils.data as data
import src.utils.misc as ms

import os
import sys

class GenericDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            data_dir, 
            load_name, 
            split='train',
            trim=None,
            alpha=None,
            Nx=256,
        ):
        np.random.seed(0)
        self.alpha = '*' if alpha is None else alpha
        self.trim = trim if trim is not None else None

        self.keys  = ['x', 't']
        self.keys += ['kappa', 'alpha', 'f0', 'T60',]
        self.keys += ['u0']
        self.keys += ['mode_freq', 'mode_amps']
        self.keys += ['gain']
        self.keys += ['ua_f0',]
        self.keys += ['ut_f0',]
        data_expr = lambda split: f"{data_dir}/{load_name}/{split}/*/ut-0.wav"
        # set `load_name` to be the directory containing
        # data preprocessed by `src/task/process_training_data.py`
        def get_string_id(path): return path.split('/')[-2]
        def get_space_idx(path): return int(os.path.splitext(os.path.basename(path))[0].split('-')[-1])
        def get_data_list(split):
            wp = f"{data_dir}/{load_name}/{split}/*/ut-0.wav"
            total_data = [p for p in glob.glob(data_expr(split))]
            return sorted(total_data, key=lambda i: \
                (get_string_id(i), get_space_idx(i)))
        dl = get_data_list(split.lower())
        assert len(dl) > 0, f"[Loader] No data found in the directory {data_expr(split.lower())}."
        self.Nx = Nx
        self.tgt_list = dl

        self.n_data = len(self.tgt_list) * Nx

    def load_data(self, tgt_path):
        ''' simulation.npz
                uout, zout, state_u, state_z
                v_r_out, F_H_out, u_H_out,
                bow_mask, hammer_mask, pluck_mask

            string.npz
                kappa, alpha, u0, v0, f0
                pos, T60, target_f0

            bow.npz
                x_B, v_B, F_B, phi_0, phi_1, wid_B

            hammer.npz
                x_H, v_H, u_H, w_H, M_r, alpha
        '''
        # {data_dir}/{load_name}/{split}/{string_id}-*/ut-{nx}.wav
        tgt_path_list = tgt_path.split('/')
        string_dir = '/'.join(tgt_path_list[:-1])
        filename = tgt_path_list[-1]
        x_idx = int(filename.split('.')[0].split('-')[-1])

        npz_path = os.path.join(string_dir, 'parameters.npz')
        lin_path = tgt_path.replace('ut-', 'ua-')
        linear_wave = sf.read(lin_path)[0]

        Nt = len(linear_wave)
        if self.trim is not None:
            st = np.random.randint(Nt-self.trim)
            et = st + self.trim
            linear_wave = linear_wave[st:et]
            _tgt = data.load_wav(tgt_path, npz_path, [st, et], keys=self.keys)
        else:
            _tgt = data.load_wav(tgt_path, npz_path, keys=self.keys)
        xval = _tgt['x'][0,x_idx]
        coef = _tgt['mode_amps'][:,x_idx][None,None,:]
        _tgt.update(dict(x=xval))
        _tgt.update(dict(mode_coef=coef))
        _tgt.update(dict(analytic=linear_wave))
        return _tgt

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        anchor_index = index // self.Nx
        spaces_index = index % self.Nx
        anchor_path = self.tgt_list[anchor_index]
        target_path = anchor_path.replace('ut-0.wav', f'ut-{spaces_index}.wav')
        return self.load_data(target_path)

class Trainset(GenericDataset):

    def __init__(
            self,
            data_dir,
            load_name,
            trim=None,
        ):
        super().__init__(
            data_dir,
            load_name,
            split='Train',
            trim=trim,
        )
        print(f"[Loader] Train samples:")
        print(f"\t(total) {len(self)}")

class Testset(GenericDataset):

    def __init__(
            self,
            data_dir,
            load_name,
            split='Test',
            trim=None,
        ):
        super().__init__(
            data_dir,
            load_name,
            split=split,
            trim=trim,
        )
        print(f"[Loader] {split} samples:")
        print(f"\t(total) {len(self)}")


if __name__=='__main__':
    dset = Trainset('/data2/private/szin/dfdm', 'cvg-*')
    data = dset[0]
    for key, value in data.items():
        print(key, value.shape)



