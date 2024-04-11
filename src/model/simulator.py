import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import src.utils.fdm as fdm
import src.utils.misc as ms
import src.utils.control as control

class String(nn.Module):
    def __init__(
        self, k, theta_t, lambda_c, sr, length, f0_inf, alpha_inf, batch_size, precision,
        pluck_batch=False, pluck_mask=None, hammer_mask=None,
        **string_kwargs):
        super().__init__()
        ''' state_u Tensor(float) : initial transverse state   with shape (batch_size, Nt, Nx_t,)
            state_z Tensor(float) : initial longitudinal state with shape (batch_size, Nt, Nx_l,)
            kappa   Tensor(float) : relative stiffness         with shape (batch_size, )
            alpha   Tensor(float) : stiffness versus tension   with shape (batch_size, )
            u0      Tensor(float) : initial displacement       with shape (batch_size,)
            v0      Tensor(float) : initial velocity           with shape (batch_size,)
            f0      Tensor(float) : fundamental frequency      with shape (batch_size, time,)
            pos     Tensor(float) : readout position           with shape (batch_size,)
            T60     Tensor(float) : T60 against frequency      with shape (batch_size, 2, 2,)
        '''
        assert precision in ['single', 'double']
        self.dtype = torch.float64 if precision == 'double' else torch.float32
        assert alpha_inf >= 1, "alpha_inf should be greater than or equal to 1."
        Nt = int(sr * length)   # duration of simulation (samples)
        var = fdm.get_derived_vars(f0_inf, 0, k, theta_t, lambda_c, alpha_inf)
        Nx_t, Nx_l = var[2], var[4]   # maximum spatial resolution

        self.sr = sr
        self.Nt = Nt
        self.Nx_t = Nx_t
        self.Nx_l = Nx_l
        self.Bs = batch_size
        self.k = k
        self.theta_t = theta_t
        self.lambda_c = lambda_c
        self.pluck_batch = pluck_batch
        self.pluck_mask = pluck_mask.view(-1,1) if pluck_mask is not None else torch.zeros((batch_size,1), dtype=self.dtype)
        self.hammer_mask = hammer_mask.view(-1) if hammer_mask is not None else torch.zeros((batch_size,), dtype=self.dtype)
        self.f0_inf = f0_inf
        self.alpha_inf = alpha_inf

        self.plucked = None
        self.initialize_config(**string_kwargs)
        self.initialize_state()

    #def forward(self, verbose=True):
    def forward(self, verbose=False):
        if verbose:
            u0_maxvals = self.u0.flatten(1).max(dim=1).values
            u0_maxidxs = self.u0.max(dim=2).values.max(dim=1).indices / self.u0.size(1)
            index_str = '\nindex || '
            tarf0_str = 'f0    || '
            kappa_str = 'kappa || '
            alpha_str = 'alpha || '
            u0max_str = 'u0max || '
            u0idx_str = 'u0idx || '
            pmask_str = 'pmask || '
            hmask_str = 'hmask || '
            index_str += '\t| '.join([str(i.item()) for i in torch.arange(self.kappa.size(0))])
            tarf0_str += '\t| '.join([f"{f[0]:.0f}" for f in self.target_f0.tolist()])
            kappa_str += '\t| '.join([f"{f:.3f}" for f in self.kappa.tolist()])
            alpha_str += '\t| '.join([f"{f:.2f}" for f in self.alpha.tolist()])
            u0max_str += '\t| '.join([f"{f:.3f}" for f in u0_maxvals.tolist()])
            u0idx_str += '\t| '.join([f"{f:.3f}" for f in u0_maxidxs.tolist()])
            pmask_str += '\t| '.join([f"{f}"     for f in self.pluck_mask.view(-1).tolist()])
            hmask_str += '\t| '.join([f"{f}"     for f in self.hammer_mask.view(-1).tolist()])
            print(index_str)
            print(tarf0_str)
            print(kappa_str)
            print(alpha_str)
            print(u0max_str)
            print(u0idx_str)
            print(pmask_str)
            print(hmask_str)

        return [self.state_u, self.state_z,
                self.kappa, self.alpha,
                self.u0, self.v0, self.f0,
                self.pos, self.T60, self.target_f0]

    def dump_parameter(self, par, val):
        val = torch.from_numpy(val).to(self.dtype) if isinstance(val, np.ndarray) else val
        val = nn.Parameter(val, requires_grad=True).float()
        if par == 'plucked':
            self.plucked = self.pluck_mask * val.unsqueeze(0) # (1, Nt)
            self.initialize_state()
        for name, parameter in self.state_dict().items():
            if name == par:
                if name == 'f0' and self.precorrect:
                    w0 = fdm.stiff_string_modes(0, self.kappa.view(-1,1), 1)[1][0]
                    f0 = val / w0
                    assert f0.min() >= self.f0_inf, f0.min()
                    self.state_dict()[name].copy_(f0)
                else:
                    self.state_dict()[name].copy_(val)

    def initialize_config(self,
            sampling_f0='random',
            sampling_kappa='random',
            sampling_alpha='random',
            sampling_pickup='random',
            sampling_T60='random',
            #---------- 
            precorrect=True,
            #---------- 
            f0_min=27.50, f0_max=440, f0_diff_max=50, f0_mod_max=0.02, f0_fixed=20,
            kappa_min=0., kappa_max=0.08, kappa_fixed=0.08, kappa_hammer=0.,
            alpha_min=1, alpha_max=25, alpha_fixed=3.,
            pos_min=0.3, pos_max=0.7, pos_fixed=0.5,
            lossless=False,  # lossless string, without randomization
            t60_min_1=10., t60_max_1=20., t60_min_2=20., t60_max_2=20.,
            t60_fixed=20.,
            #---------- 
            sampling_p_a='random', sampling_p_x='random',
            p_a_min=0.001, p_a_max=0.01, p_a_fixed=0.01, # pluck amplitude
            p_x_min=0.100, p_x_max=0.90, p_x_fixed=0.50, # pluck position
            pluck_profile=None,
            #---------- 
        ):
        # type check
        sampling_options = ['random', 'equidist', 'fix']
        sampling_configs = [
            sampling_f0,
            sampling_kappa,
            sampling_alpha,
            sampling_pickup,
            sampling_T60,
            sampling_p_x,
            sampling_p_a,
        ]; assert set(sampling_configs).issubset(sampling_options), \
        f"{sampling_configs} should be given as {sampling_options}."

        self.precorrect = precorrect

        self.initialize_kappa(sampling_kappa, kappa_min, kappa_max, kappa_fixed, kappa_hammer)
        self.initialize_f0(sampling_f0, f0_min, f0_max, f0_diff_max, f0_mod_max, f0_fixed)
        self.initialize_alpha(sampling_alpha, alpha_min, alpha_max, alpha_fixed)
        self.initialize_pickup_position(sampling_pickup, pos_min, pos_max, pos_fixed)
        self.initialize_T60(sampling_T60, lossless, t60_min_1, t60_max_1, t60_min_2, t60_max_2, t60_fixed)

        self.sampling_p_a = sampling_p_a
        self.sampling_p_x = sampling_p_x
        self.p_a_min=p_a_min; self.p_a_max=p_a_max; self.p_a_fixed=p_a_fixed;
        self.p_x_min=p_x_min; self.p_x_max=p_x_max; self.p_x_fixed=p_x_fixed;

        try:
            assert pluck_profile in ['triangular', 'smooth', 'raised_cosine'], pluck_profile
        except AssertionError as err:
            if pluck_profile is None: pluck_profile = 'triangular'
            else: raise err
        self.pluck_profile=pluck_profile

    def initialize_state(self):
        p_a, p_x = self.initialize_pluck_amplitude()
        f0_b = self.f0.min(-1, keepdim=True).values.view(-1)
        nx_t = fdm.get_derived_vars(f0_b, self.kappa, self.k, self.theta_t, self.lambda_c, self.alpha)[2].view(-1,1,1)

        if self.pluck_profile == 'triangular':
            # triangular initial condition
            u0 = ms.triangular(self.Nx_t+1, nx_t+1, p_x, p_a) # (B, T, Nx_t)
        elif self.pluck_profile == 'smooth':
            # smooth, sine squared profile
            tr = ms.triangular(self.Nx_t+1, nx_t+1, p_x, torch.ones_like(p_x))
            u0 = p_a * torch.sin(tr*math.pi/2).pow(2) # (B, T, Nx_t)
        else:
            # raised cosine profile
            u0 = ms.raised_cosine(self.Nx_t+1, 1/self.Nx_t,
                p_x.narrow(1,0,1), nx_t.div(10, rounding_mode='trunc'),
                nx_t.flatten()+1).transpose(1,2) * torch.sign(p_x)

        v0 = torch.zeros_like(u0)

        state_u, state_z = fdm.initialize_state(u0, v0, self.Nt, self.Nx_t, self.Nx_l, self.k, dtype=self.dtype)
        self.register_buffer('u0',  nn.Parameter(u0,  requires_grad=False))
        self.register_buffer('v0',  nn.Parameter(v0,  requires_grad=False))
        self.register_buffer('state_u', nn.Parameter(state_u, requires_grad=False))
        self.register_buffer('state_z', nn.Parameter(state_z, requires_grad=False))

    def initialize_f0(
            self, sampling='random',
            f0_min=49, f0_max=220, f0_diff_max=50, f0_mod_max=0.02, f0_fixed=20,
        ):
        ''' f0       : The fundamental frequency for simulation input.
                       This value can be different from the  `target_f0`,
                       in case of doing f0 pre-correction (of the detune
                       that can be caused by the string stiffness).
            target_f0: The fundamental frequency for simulated output.
                       This value should be close to intended fundamental
                       frequency of the simulated sound.
        '''
        if sampling=='random':
            f0_con = control.constant(ms.random_uniform(f0_min, f0_max, size=self.Bs, dtype=self.dtype), self.Nt, dtype=self.dtype)

            f0_1 = ms.random_uniform(f0_min, f0_max, size=self.Bs, dtype=self.dtype)
            f0_2 = ms.random_uniform(f0_min, f0_max, size=self.Bs, dtype=self.dtype).clamp(f0_1 - f0_diff_max, f0_1 + f0_diff_max)
            f0_lin = control.linear(f0_1, f0_2, self.Nt)

            # linearly-varying f0 (set `f0_diff_max=0` to disable this)
            ti_mask = torch.randn((self.Bs,)).ge(0.5).view(-1,1)
            f0 = f0_lin * ti_mask \
               + f0_con * ti_mask.logical_not()

            # vibrato (set `f0_mod_max=0` to disable this)
            vb_mask = torch.randn((self.Bs,)).ge(0.5).view(-1,1)
            vb = control.vibrato(f0, 1/self.sr, mf=[3.,5.], ma=f0_mod_max)
            f0 = f0 * vb_mask \
               + vb * vb_mask.logical_not()

        elif sampling=='equidist':
            f0 = control.constant(ms.equidistant(f0_min, f0_max, self.Bs), self.Nt)

        else: # fixed
            try:
                # f0_fixed given as ConfigList
                f0_batch_len = len(f0_fixed)
            except TypeError as err:
                # f0_fixed given as int or float
                if isinstance(f0_fixed, int) or isinstance(f0_fixed, float):
                    f0_batch_len = 0
                else:
                    raise err
            if f0_batch_len > 1:
                # f0_fixed given as ConfigList
                min_f0_fixed = min(f0_fixed)
                f0_fixed = torch.tensor(list(f0_fixed), dtype=self.dtype).view(-1,1)
            else:
                # f0_fixed given as int or float
                min_f0_fixed = f0_fixed

            assert min_f0_fixed >= self.f0_inf, \
            f"f0_fixed (== {min_f0_fixed}) should be >= than f0_inf (== {self.f0_inf})"
            f0 = f0_fixed * control.constant(torch.ones(size=(self.Bs,)), self.Nt, dtype=self.dtype)

        target_f0 = f0

        if self.precorrect:
            # pre-correct detune
            w0 = fdm.stiff_string_modes(0, self.kappa.view(-1,1), 1)[1][0]
            w0_max = w0.flatten().max().item()
            self.f0_inf = self.f0_inf / w0_max

            var = fdm.get_derived_vars(self.f0_inf, 0, self.k, self.theta_t, self.lambda_c, self.alpha_inf)
            Nx_t, Nx_l = var[2], var[4]   # maximum spatial resolution
            self.Nx_t = Nx_t
            self.Nx_l = Nx_l

            f0 = f0 / w0

        assert f0.min() >= self.f0_inf, f0.min()
        self.register_buffer('f0',  nn.Parameter(f0,  requires_grad=False))
        self.register_buffer('target_f0',  nn.Parameter(target_f0,  requires_grad=False))

    def initialize_kappa(self, sampling='random', kappa_min=0, kappa_max=0.08, kappa_fixed=0.08, kappa_hammer=0.):
        if sampling=='random':
            kappa_r = (kappa_max - kappa_min) * torch.rand(size=(self.Bs,)).to(self.dtype) \
                    + kappa_min
            kappa_h = kappa_hammer + kappa_r
            kappa_1 = kappa_r * self.hammer_mask.logical_not() \
                    + kappa_h * self.hammer_mask
            kappa = kappa_1
        elif sampling=='equidist':
            kappa = ms.equidistant(kappa_min, kappa_max, self.Bs)
        else: # fixed
            kappa = kappa_fixed * torch.ones(size=(self.Bs,), dtype=self.dtype)

        if kappa.gt(0.03).any():
            print(f"[WARNING] Current kappa value is large: {kappa.tolist()}\nThis can result to the detunes in the simulated outputs. (The `precorrection` is valid under small kappa values; <= 0.04)")
        self.register_buffer('kappa', nn.Parameter(kappa, requires_grad=False))

    def initialize_alpha(self, sampling='random', alpha_min=1, alpha_max=3, alpha_fixed=3.):
        ''' larger alpha gives more "boing"-like sound (smaller tension) '''
        if sampling=='random':
            alpha_1 = (alpha_max - alpha_min) * torch.rand(size=(self.Bs,)).to(self.dtype) \
                    + alpha_min
            alpha   = alpha_1
        elif sampling=='equidist':
            alpha = ms.equidistant(alpha_min, alpha_max, self.Bs)
        else: # fixed
            alpha_fixed = self.alpha_inf if alpha_fixed < self.alpha_inf else alpha_fixed # just to handle invalid case
            alpha = alpha_fixed * torch.ones(size=(self.Bs,), dtype=self.dtype)
        assert alpha.ge(self.alpha_inf).all(), "alpha should be greater than or equal to args.alpha_inf={self.alpha_inf}."
        self.register_buffer('alpha', nn.Parameter(alpha, requires_grad=False))


    def initialize_pluck_amplitude(self):
        ''' p_x: (batch_size, Nt, 1)
            p_a: (batch_size, Nt, 1)
        '''
        if self.plucked is None:
            if self.pluck_batch:
                # all batches are plucked
                batch_mask = torch.ones((self.Bs,1), dtype=self.dtype)
                time_mask  = torch.zeros((1,self.Nt), dtype=self.dtype)
                time_mask[:,0] = 1.
            elif isinstance(self.pluck_batch, bool):
                # all batches are NOT plucked
                batch_mask = torch.zeros((self.Bs,1), dtype=self.dtype)
                time_mask  = torch.zeros((1,self.Nt), dtype=self.dtype)
            else:
                batch_mask = self.pluck_mask
                time_mask  = torch.zeros((1,self.Nt), dtype=self.dtype)
                time_mask[:,0] = 1.
            self.plucked = batch_mask * time_mask

        if self.sampling_p_a=='random':
            p_a = ms.random_uniform(self.p_a_min, self.p_a_max, size=(self.Bs,self.Nt), dtype=self.dtype)
        elif self.sampling_p_a=='equidist':
            p_a = ms.equidistant(self.p_a_min, self.p_a_max, self.Bs).view(-1,1).tile(1,self.Nt)
        else: # fixed
            p_a = self.p_a_fixed * torch.ones(size=(self.Bs,self.Nt), dtype=self.dtype)

        if self.sampling_p_x=='random':
            p_x = ms.random_uniform(self.p_x_min, self.p_x_max, size=(self.Bs,self.Nt), dtype=self.dtype)
        elif self.sampling_p_x=='equidist':
            p_x = ms.equidistant(self.p_x_min, self.p_x_max, self.Bs).view(-1,1).tile(1,self.Nt)
        else: # fixed
            p_x = self.p_x_fixed * torch.ones(size=(self.Bs,self.Nt), dtype=self.dtype)

        p_a = (p_a * self.plucked).unsqueeze(2)
        p_x = (p_x * self.plucked).unsqueeze(2)
        return p_a, p_x

    def initialize_pickup_position(self, sampling='random', pos_min=0.3, pos_max=0.7, pos_fixed=0.5):
        if sampling=='random':
            pos = ms.random_uniform(pos_min, pos_max, size=self.Bs, dtype=self.dtype)
        elif sampling=='equidist':
            pos = ms.equidistant(pos_min, pos_max, self.Bs)
        else: # fix
            pos = pos_fixed * torch.ones(size=(self.Bs,), dtype=self.dtype)
        self.register_buffer('pos', nn.Parameter(pos, requires_grad=False))

    def initialize_T60(self, sampling='random', lossless=False,
            t60_min_1=10., t60_max_1=20., t60_min_2=20., t60_max_2=20.,
            t60_fixed=20.,):
        if sampling=='random':
            T60_freq_min = (1/240) * self.sr / 2; T60_freq_max = (1/4) * self.sr / 2
            T60_time_min = 5.; T60_time_max = 10.; T60_diff_max = 2.
   
            T60_freq_1 = ms.random_uniform(T60_freq_min+1000, T60_freq_max, size=self.Bs, dtype=self.dtype)     # high-freq
            T60_freq_2 = ms.random_uniform(T60_freq_min, T60_freq_1-1000, size=self.Bs, dtype=self.dtype)       # low-freq
            T60_time_1 = ms.random_uniform(T60_time_min, T60_time_max - 1., size=self.Bs, dtype=self.dtype)     # shorter T60 (high-freq)
            T60_time_2 = ms.random_uniform(T60_time_1, T60_time_1+T60_diff_max, size=self.Bs, dtype=self.dtype) # longer T60  (low-freq)
        elif sampling=='equidist':
            T60_freq_1 = 1000. * torch.ones(size=(self.Bs,), dtype=self.dtype)
            T60_freq_2 = 100.  * torch.ones(size=(self.Bs,), dtype=self.dtype)
            t1 = ms.equidistant(t60_min_1, t60_max_1, self.Bs-1, dtype=self.dtype) # high-freq
            t2 = ms.equidistant(t60_min_2, t60_max_2, self.Bs-1, dtype=self.dtype) # low-freq
            T60_time_1 = torch.cat([t1,torch.zeros_like(t1.narrow(-1,0,1))], dim=-1)
            T60_time_2 = torch.cat([t2,torch.zeros_like(t2.narrow(-1,0,1))], dim=-1)
        elif lossless: # fixed, lossless
            T60_freq_1 = 1000. * torch.ones(size=(self.Bs,), dtype=self.dtype)
            T60_freq_2 = 100.  * torch.ones(size=(self.Bs,), dtype=self.dtype)
            T60_time_1 = torch.zeros(size=(self.Bs,), dtype=self.dtype)
            T60_time_2 = torch.zeros(size=(self.Bs,), dtype=self.dtype)
        else: # fixed, lossy
            T60_freq_1 = 1000. * torch.ones(size=(self.Bs,), dtype=self.dtype)
            T60_freq_2 = 100.  * torch.ones(size=(self.Bs,), dtype=self.dtype)
            T60_time_1 = t60_fixed * torch.ones(size=(self.Bs,), dtype=self.dtype)
            T60_time_2 = t60_fixed * torch.ones(size=(self.Bs,), dtype=self.dtype)
        T60_1 = torch.cat((T60_freq_1.unsqueeze(-1), T60_time_1.unsqueeze(-1)), dim=-1)
        T60_2 = torch.cat((T60_freq_2.unsqueeze(-1), T60_time_2.unsqueeze(-1)), dim=-1)
        T60 = torch.cat((T60_1.unsqueeze(1), T60_2.unsqueeze(1)), dim=1)  # (self.Bs, 2, 2)

        self.register_buffer('T60', nn.Parameter(T60, requires_grad=False))


class Bow(nn.Module):
    def __init__(self, sr, length, batch_size, precision, **bow_kwargs):
        super().__init__()
        ''' x_b   Tensor(float) : bowing position profile with shape (batch_size, time,)
            v_b   Tensor(float) : bowing velocity profile with shape (batch_size, time,)
            F_b   Tensor(float) : bowing force profile    with shape (batch_size, time,)
            phi_0 Tensor(float) : bow friction coeff      with shape (batch_size, )
            phi_1 Tensor(float) : bow friction coeff      with shape (batch_size, )
            wid   Tensor(float) : bow width               with shape (batch_size, time,)
            pos   Tensor(float) : readout position        with shape (batch_size, )
            T60   Tensor(float) : T60 against frequency   with shape (batch_size, 2, 2)
        '''
        assert precision in ['single', 'double']
        self.dtype = torch.float64 if precision == 'double' else torch.float32
        self.length = length
        self.Nt = int(sr * length)   # duration of simulation (samples)
        self.sr = sr
        self.Bs = batch_size

        self.initialize_config(**bow_kwargs)

    def forward(self):
        return [self.x_b, self.v_b, self.F_b, self.phi_0, self.phi_1, self.wid]

    def initialize_config(self,
            x_b_min=0.2, x_b_max=0.5, x_b_maxdiff=0.2,
            v_b_min=0.3, v_b_max=0.4,
            F_b_min=80, F_b_max=100, F_b_maxdiff=10, do_pulloff=True,
            phi_0_max=6, phi_0_min=2, phi_1_max=0.5, phi_1_min=0.,
            wid_min=3, wid_max=6,
        ):
        self.initialize_position(x_b_min, x_b_max, x_b_maxdiff)
        self.initialize_velocity(v_b_min, v_b_max)
        self.initialize_force(F_b_min, F_b_max, F_b_maxdiff, do_pulloff)
        self.initialize_friction(phi_0_max, phi_0_min, phi_1_max, phi_1_min)
        self.initialize_width(wid_min, wid_max)

    def dump_parameter(self, par, val):
        val = torch.from_numpy(val).to(self.dtype) if isinstance(val, np.ndarray) else val
        val = nn.Parameter(val, requires_grad=True)
        for name, parameter in self.state_dict().items():
            if name == par:
                self.state_dict()[name].copy_(val)

    def initialize_position(self, x_b_min=0.2, x_b_max=0.5, x_b_maxdiff=0.2):
        x_1 = ms.random_uniform(x_b_min, x_b_max, size=self.Bs, dtype=self.dtype)
        x_2 = (x_1 + ms.random_uniform(-x_b_maxdiff, x_b_maxdiff, size=self.Bs, dtype=self.dtype)).clamp(x_b_min, x_b_max)
        x_b = control.linear(x_1, x_2, self.Nt)
        self.register_buffer('x_b', nn.Parameter(x_b, requires_grad=False))

    def initialize_velocity(self, v_b_min=0.3, v_b_max=0.4):
        v_1 = ms.random_uniform(v_b_min, v_b_max, size=self.Bs, dtype=self.dtype)
        v_2 = ms.random_uniform(v_b_min, v_b_max, size=self.Bs, dtype=self.dtype)
        v_b = control.linear(v_1, v_2, self.Nt)
        v_b = ms.pre_shaper(v_b, self.sr)
        self.register_buffer('v_b', nn.Parameter(v_b, requires_grad=False))

    def initialize_force(self, F_b_min=80, F_b_max=100, F_b_maxdiff=10, do_pulloff=True):
        F_1 = ms.random_uniform(F_b_min, F_b_max, size=self.Bs, dtype=self.dtype)
        F_2 = F_1 + ms.random_uniform(-F_b_maxdiff, F_b_maxdiff, size=self.Bs, dtype=self.dtype).clamp(F_b_min, F_b_max)
        F_b = control.linear(F_1, F_2, self.Nt)
        if do_pulloff:
            for b in range(F_b.size(0)):
                if torch.rand([1])[0] > 0.5:
                    pulloff = (3*self.length/4) * torch.rand([1])[0] + (self.length/4)
                    F_b[b] = ms.post_shaper(F_b[b], self.sr, pulloff)
        self.register_buffer('F_b', nn.Parameter(F_b, requires_grad=False))

    def initialize_friction(self, phi_0_max=7., phi_0_min=5, phi_1_max=0.1, phi_1_min=0):
        # bow friction coefficient
        phi_0 = (phi_0_max-phi_0_min) * torch.rand(size=(self.Bs,)).to(self.dtype) + phi_0_min
        phi_1 = (phi_1_max-phi_1_min) * torch.rand(size=(self.Bs,)).to(self.dtype) + phi_1_min
        self.register_buffer('phi_0', nn.Parameter(phi_0, requires_grad=False))
        self.register_buffer('phi_1', nn.Parameter(phi_1, requires_grad=False))

    def initialize_width(self, wid_min=4, wid_max=6):
        # number of samples to be excited by the bow.
        # this is normalized into relative position in (0, 1]
        # at `src.model.cpp.string.cpp` and `src.model.cpp.misc.cpp`
        wid = control.constant(ms.random_uniform(wid_min, wid_max, size=self.Bs, dtype=self.dtype), self.Nt, dtype=self.dtype)
        self.register_buffer('wid', nn.Parameter(wid, requires_grad=False))

class Hammer(nn.Module):
    def __init__(self, sr, length, batch_size, precision, k, **hammer_kwargs):
        super().__init__()
        ''' x_H   Tensor(float) : hammering position          with shape (batch_size,)
            v_H   Tensor(float) : hammer initial velocity     with shape (batch_size, time)
            u_H   Tensor(float) : hammer initial displacement with shape (batch_size, time)
            w_H   Tensor(float) : hammer stiffness parameter  with shape (batch_size,)
            M_r   Tensor(float) : hammer-string mass ratio    with shape (batch_size,)
            wid   Tensor(float) : hammer width                with shape (batch_size,)
            alpha Tensor(float) : stiffness parameter         with shape (batch_size,)
        '''
        # hammer excitations are conservative only for alpha == 1 or 3
        assert precision in ['single', 'double']
        self.dtype = torch.float64 if precision == 'double' else torch.float32
        self.Bs = batch_size
        self.length = length
        self.Nt = int(sr * length)   # duration of simulation (samples)
        self.sr = sr
        self.k = k
        self.M_HD = -1e-3 # should match the value of `M_HD` at `src/model/cpp/hammer.cpp`

        self.initialize_config(**hammer_kwargs)

    def forward(self, verbose=False):
        if verbose:
            x_H_str = 'x_H   || '
            v_H_str = 'v_H   || '
            w_H_str = 'w_H   || '
            M_r_str = 'M_r   || '
            alp_str = 'alpha || '
            x_H_str += '\t| '.join([f"{f:.3f}" for f in self.x_H.squeeze().tolist()])
            v_H_str += '\t| '.join([f"{f:.3f}" for f in self.v_H.narrow(1,1,1).squeeze().tolist()])
            w_H_str += '\t| '.join([f"{f:.0f}" for f in self.w_H.squeeze().tolist()])
            M_r_str += '\t| '.join([f"{f:.3f}" for f in self.M_r.squeeze().tolist()])
            alp_str += '\t| '.join([f"{f:.3f}" for f in self.alpha.squeeze().tolist()])
            print(x_H_str)
            print(v_H_str)
            print(w_H_str)
            print(M_r_str)
            print(alp_str)
        return [self.x_H, self.v_H, self.u_H, self.w_H, self.M_r, self.alpha]

    def initialize_config(self,
            x_H_min=0.1,   x_H_max=0.9,
            v_H_min=0.5,   v_H_max=5,
            M_r_min=10.0,  M_r_max=50.0,
            w_H_min=1000,  w_H_max=3000,
            alpha_fixed=None,
        ):
        self.v_H_min = v_H_min
        self.v_H_max = v_H_max
        # use dirac delta for the hammer width
        self.initialize_position(x_H_min, x_H_max)
        self.initialize_velocity(v_H_min, v_H_max)
        self.initialize_mass_ratio(M_r_min, M_r_max)
        self.initialize_stiffness(w_H_min, w_H_max, alpha_fixed)

    def dump_parameter(self, par, val):
        val = torch.from_numpy(val).to(self.dtype) if isinstance(val, np.ndarray) else val
        val = nn.Parameter(val, requires_grad=True)
        for name, parameter in self.state_dict().items():
            if name == par:
                if name == 'v_H':
                    val = val.float().view(1,-1)
                    self.initialize_velocity(profile=val)
                else:
                    self.state_dict()[name].copy_(val)

    def initialize_position(self, x_H_min=0.1, x_H_max=0.9):
        x_H = ms.random_uniform(x_H_min, x_H_max, size=self.Bs, dtype=self.dtype)
        self.register_buffer('x_H', nn.Parameter(x_H, requires_grad=False))

    def initialize_velocity(self, v_H_min=0.5, v_H_max=5, profile=None):
        # velocity in m/s : 0.5 (piano) ~ 5 (fortissimo)
        v_H = ms.random_uniform(v_H_min, v_H_max, size=self.Bs, dtype=self.dtype)
        if profile is None:
            profile = torch.zeros((1,self.Nt), dtype=self.dtype); profile[:,1] = 1.
        v_H = v_H.unsqueeze(-1) * profile

        u_H = torch.zeros_like(v_H); u_H[:,:2] += self.M_HD; # hammer maximum displacement
        u_H = u_H + self.k * v_H

        self.register_buffer('v_H', nn.Parameter(v_H, requires_grad=False))
        self.register_buffer('u_H', nn.Parameter(u_H, requires_grad=False))

    def initialize_mass_ratio(self, M_r_min=0.75, M_r_max=1.25):
        w = None if self.v_H_max == self.v_H_min else \
            1. - (self.v_H.max(-1).values - self.v_H_min) / (self.v_H_max - self.v_H_min)
        M_r = ms.random_uniform(M_r_min, M_r_max, size=self.Bs, weight=w, dtype=self.dtype)
        self.register_buffer('M_r', nn.Parameter(M_r, requires_grad=False))

    def initialize_stiffness(self, w_H_min=1000,  w_H_max=3000, alpha_fixed=None):
        # hammer excitations are conservative only for alpha == 1 or 3
        w_H   = ms.random_uniform(w_H_min,   w_H_max,   size=self.Bs, dtype=self.dtype)
        if alpha_fixed is None: # 1 or 3
            alpha = 2 * ms.random_uniform(0, 1, size=self.Bs, dtype=self.dtype).ge(0.5) + 1
        else:
            alpha = alpha_fixed * torch.ones(size=(self.Bs,), dtype=self.dtype)
        self.register_buffer('alpha', nn.Parameter(alpha, requires_grad=False))
        self.register_buffer('w_H',   nn.Parameter(w_H,   requires_grad=False))



