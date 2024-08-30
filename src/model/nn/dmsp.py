import torch
import torch.nn as nn
from src.model.nn.blocks import FMBlock, AMBlock, ModBlock
from src.utils.ddsp import scale_function, remove_above_nyquist, upsample
from src.utils.ddsp import remove_above_nyquist_mode
from src.utils.ddsp import harmonic_synth, amp_to_impulse_response, fft_convolve
from src.utils.ddsp import modal_synth
from src.utils.ddsp import resample
import math

class DMSP(nn.Module):
    def __init__(self,
            embed_dim, hidden_size, n_features,
            n_modes, n_bands, sampling_rate, block_size,
        ):
        super().__init__()
        self.n_modes = n_modes

        self.freq_modulator = FMBlock(n_modes, embed_dim, n_features)
        self.coef_modulator = AMBlock(n_modes, embed_dim, n_features)
        self.proj_noise = nn.Linear(n_features*embed_dim, n_bands)

        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

    def forward(self, hidden, mode_freq, mode_coef, times, alpha, omega, lengths):
        ''' hidden    : (Bs,  1, hidden_size)
            mode_freq : (Bs, Nt, n_modes)
            mode_coef : (Bs,  1, n_modes)
            times     : (Bs, Nt, 1)
        '''
        freq_m = self.freq_modulator(mode_freq, hidden, alpha, omega)
        coef_m = self.coef_modulator(mode_coef, hidden, times)

        #============================== 
        # harmonic part
        #============================== 
        freqs = freq_m / (2*math.pi) * self.sampling_rate
        coef_m = remove_above_nyquist_mode(coef_m, freqs, self.sampling_rate) # (Bs, Nt, n_modes)
        freq_s = upsample(freq_m, self.block_size).narrow(1,0,lengths)
        coef_s = upsample(coef_m, self.block_size).narrow(1,0,lengths)
        harmonic = modal_synth(freq_s, coef_s, self.sampling_rate)

        #============================== 
        # noise part
        #============================== 
        param = scale_function(self.proj_noise(hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1
        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1).narrow(1,0,lengths)

        signal = harmonic + noise
        return signal.squeeze(-1), freq_m, coef_m




