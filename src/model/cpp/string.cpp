using namespace std;
# define M_PI           3.14159265358979323846  /* pi */

#include <torch/extension.h>
#include <iostream>
#include <vector>

#include "misc.h"
#include "bow.h"
#include "hammer.h"

#include <ATen/ATen.h>
using namespace at;

vector<torch::Tensor> get_derived_vars(
    torch::Tensor f0,
    torch::Tensor kappa_rel,
    float k, float theta_t, float lambda_c,
    torch::Tensor alpha) {

    // Derived variables
    auto gamma = 2 * f0;                           // set parameters
    auto kappa = gamma * kappa_rel;                // stiffness parameter
    auto IHP = (M_PI * kappa / gamma).pow(2);      // inharmonicity parameter (>0); eq 7.21
    auto K = IHP.pow(0.5) * (gamma / M_PI);        // set parameters

    torch::Tensor h_1, h_2;
    // stability conditions (eq. 7.26)
    h_1 = lambda_c * (
        (gamma.pow(2) * pow(k, 2.) + pow(gamma.pow(4) * pow(k, 4.) + 16 * K.pow(2) * pow(k, 2.) * (2 * theta_t - 1),.5))
      / (2 * (2 * theta_t - 1))
    ).pow(.5);
    auto N_t = torch::floor(1 / h_1).to(kappa_rel.dtype());
    auto h_t = 1 / N_t;

    // stability conditions (eq. 8.28)
    h_2 = lambda_c * gamma * alpha * k;
    auto N_l = torch::floor(1 / h_2).to(kappa_rel.dtype());
    auto h_l = 1 / N_l;

    return { gamma, K, N_t, h_t, N_l, h_l };
}

vector<torch::Tensor> string_step(
    torch::Tensor uout,                   // pickup transverse displacement
    torch::Tensor zout,                   // pickup longitudinal displacement
    torch::Tensor state_u,                // transverse displacement state
    torch::Tensor state_z,                // longitudinal displacement state
    torch::Tensor v_r_out,                // relative velocity
    torch::Tensor F_H_out,                // hammering force profile
    vector<torch::Tensor> string_params,  // string parameters
    vector<torch::Tensor> bow_params,     // bow excitation parameters
    vector<torch::Tensor> hammer_params,  // hammer excitation parameters
    torch::Tensor bow_mask,               // bow excitation mask
    torch::Tensor hammer_mask,            // hammer excitation mask
    vector<float> constant,               // global constants
    int global_step,                      // global simulation time step
    int local_step,                       // local simulation time step (just for TBPTT)
    float relative_error,                 // discretization error relative to the spatial grid size
    bool surface_integral) {              // pickup output wave by surface integration
    int batch_size = uout.size(0);

    //============================== 
    // Setup variables
    //============================== 
    // string parameters
    auto kappa_rel = string_params[0]; auto alpha = string_params[1];
    auto u0 = string_params[2]; auto v0 = string_params[3];
    auto f0 = string_params[4]; auto rp = string_params[5]; auto T60 = string_params[6];

    // bow control parameters
    auto x_bow = bow_params[0]; auto v_bow = bow_params[1]; auto F_bow = bow_params[2];
    auto phi_0 = bow_params[3]; auto phi_1 = bow_params[4]; auto wid_b = bow_params[5];

    // hammer control parameters
    auto x_H = hammer_params[0]; auto v_H = hammer_params[1]; auto u_H_out = hammer_params[2];
    auto w_H = hammer_params[3]; auto M_r = hammer_params[4]; auto alpha_H = hammer_params[5];

    // constants
    float k = constant[0]; float theta_t = constant[1]; float lambda_c = constant[2];

    // derived variables
    auto vars = get_derived_vars(f0.select(1,global_step), kappa_rel, k, theta_t, lambda_c, alpha);
    auto gamma = vars[0]; auto K = vars[1];
    auto N_t = vars[2]; auto h_t = vars[3];   // transverse (u)
    auto N_l = vars[4]; auto h_l = vars[5];   // longitudinal (zeta)

    auto bow_wid_length = wid_b.select(1,global_step) * h_t;
    auto tol_t = h_t.pow(relative_error);
    auto tol_l = h_l.pow(relative_error);

    //============================== 
    // Simulation step
    //============================== 

    // Scheme loss parameters; eq 7.29
    torch::Tensor zeta1;
    torch::Tensor zeta2;

    zeta1 = torch::where(K.gt(0),
        - gamma.pow(2) + (gamma.pow(4) + 4 * K.pow(2) * (2 * M_PI * T60.select(2,0).select(1,0)).pow(2)).pow(.5),  // if is stiff string
        T60.select(2,0).select(1,0).pow(2) / gamma.pow(2)  // otherwise
    );
    zeta2 = torch::where(K.gt(0),
        - gamma.pow(2) + (gamma.pow(4) + 4 * K.pow(2) * (2 * M_PI * T60.select(2,0).select(1,1)).pow(2)).pow(.5),  // if is stiff string

        T60.select(2,0).select(1,1).pow(2) / gamma.pow(2) // otherwise
    );

    auto T60_mask = T60.prod(2).prod(1).ne(0);
    auto sig0 = torch::where(T60_mask,
        - zeta2 / T60.select(2,1).select(1,0) + zeta1 / T60.select(2,1).select(1,1), // lossy string
        T60_mask // lossless string
    );
    auto sig1 = torch::where(T60_mask,
        1 / T60.select(2,1).select(1,0) - 1 / T60.select(2,1).select(1,1), // lossy string
        T60_mask // lossless string
    );
    sig0 = (6 * log(10) * sig0 / (zeta1 - zeta2)).view({-1,1,1});   // freq-independent loss term
    sig1 = (6 * log(10) * sig1 / (zeta1 - zeta2)).view({-1,1,1});   // freq-dependent loss term

    // setup displacements
    int N_t_max = state_u.size(-1);
    int N_l_max = state_z.size(-1);
    auto u1 = state_u.narrow(1,local_step-1,1).transpose(2,1);  // (batch_size, N_t_max, 1)
    auto u2 = state_u.narrow(1,local_step-2,1).transpose(2,1);  // (batch_size, N_t_max, 1)
    auto z1 = state_z.narrow(1,local_step-1,1).transpose(2,1);  // (batch_size, N_l_max, 1)
    auto z2 = state_z.narrow(1,local_step-2,1).transpose(2,1);  // (batch_size, N_l_max, 1)
    u1 = mask_1d(u1, N_t+1, N_t_max);
    u2 = mask_1d(u2, N_t+1, N_t_max);
    z1 = mask_1d(z1, N_l+1, N_l_max);
    z2 = mask_1d(z2, N_l+1, N_l_max);

    auto w1 = torch::cat({u1, z1}, 1);
    auto w2 = torch::cat({u2, z2}, 1);

    // setup operators
    auto Id_tt    =     I(N_t+1,   0); auto Id_ll    =     I(N_l+1,   0);
    auto Dxf_tt   =   Dxf(N_t+1, h_t); auto Dxf_ll   =   Dxf(N_l+1, h_l);
    auto Dxb_tt   =   Dxb(N_t+1, h_t);// auto Dxb_ll   =   Dxb(N_l+1, h_l);
    auto Dxx_tt   =   Dxx(N_t+1, h_t); auto Dxx_ll   =   Dxx(N_l+1, h_l);
    //auto Dxxxx_tt = Dxxxx(N_t+1, h_t);// auto Dxxxx_ll = Dxxxx(N_l+1, h_l);
    auto Dxxxx_tt = Dxxxx_clamped(N_t+1, h_t);
    auto Int_tl = batched_interpolator(N_l+1, N_t+1);
    auto Int_lt = batched_interpolator(N_t+1, N_l+1);
    auto Mxc_tt = Mxc(N_t+1);

    auto Theta_tt = theta_t * Id_tt + (1-theta_t) * Mxc_tt;

    // setup recursion
    auto gamma_k = gamma.pow(2).view({-1,1,1}) * pow(k, 2.);
    auto phi_pow = gamma_k * (alpha.pow(2).view({-1,1,1}) - 1) / 4;
    auto Lam = batched_diag(torch::matmul(Dxb_tt, u1.narrow(1,0,Dxb_tt.size(-1))));
    auto Qp_tt = Theta_tt + 2 * sig0 * k * Id_tt - 2 * sig1 * k * Dxx_tt;
    auto Qm_tt = Theta_tt - 2 * sig0 * k * Id_tt + 2 * sig1 * k * Dxx_tt;
    auto Qp_ll = (1 + 2 * sig0 * k) * Id_ll - 2 * sig1 * k * Dxx_ll;
    auto Qm_ll = (1 - 2 * sig0 * k) * Id_ll + 2 * sig1 * k * Dxx_ll;
    auto K_tl = - phi_pow * torch::matmul(Dxf_tt, torch::matmul(Lam,    torch::matmul(Dxb_tt, Int_tl)));
    auto K_lt = - phi_pow * torch::matmul(Dxf_ll, torch::matmul(Int_lt, torch::matmul(Lam,    Dxb_tt)));
    auto V_tt = - phi_pow * torch::matmul(Dxf_tt, torch::matmul(Lam.pow(2), Dxb_tt));

    auto B_1 = -2 * Theta_tt - gamma_k * Dxx_tt + K.pow(2).view({-1,1,1}) * pow(k,2.) * Dxxxx_tt;
    auto B_2 =  2 * K_tl;
    auto B_3 = torch::zeros_like(B_2).transpose(1,2);
    auto B_4 = -2 * Id_ll - gamma_k * alpha.pow(2).view({-1,1,1}) * Dxx_ll;

    /* A @ w^{n+1} + B @ w^{n} + C @ w^{n-1} = 0 */
    // matrices with size (batch, N_t+N_l, N_t+N_l)
    auto A_1 = Qp_tt + V_tt; auto A_2 = K_tl; auto A_3 = K_lt; auto A_4 = Qp_ll;
    auto C_1 = Qm_tt + V_tt; auto C_2 = K_tl; auto C_3 = K_lt; auto C_4 = Qm_ll;

    // inverse A before it gets zero-padded
    int t_wid = A_1.size(-1); int l_wid = A_2.size(-1);
    auto A_b = block_matrices({ { A_1, A_2 }, { A_3, A_4 } });
    auto A_p = torch::linalg::inv(A_b);

    // zero-pad to maximal size (batch, N_t_max+N_l_max, N_t_max+N_l_max)
    auto A   = sparse_blocks({A_1, A_2, A_3, A_4}, N_t_max, N_l_max);
    auto B   = sparse_blocks({B_1, B_2, B_3, B_4}, N_t_max, N_l_max);
    auto C   = sparse_blocks({C_1, C_2, C_3, C_4}, N_t_max, N_l_max);
    auto A_P = sparse_blocks(split_blocks(A_p, t_wid, l_wid), N_t_max, N_l_max);

    auto u_H1 = u_H_out.narrow(1,global_step-1,1).view(-1);
    auto u_H2 = u_H_out.narrow(1,global_step-2,1).view(-1);

    // iterate for implicit scheme
    int iter = 0;
    bool not_converged_t = true;
    bool not_converged_l = true;
    torch::Tensor u = state_u.narrow(1,local_step-1,1).transpose(2,1);  // initialize by u1
    torch::Tensor z = state_z.narrow(1,local_step-1,1).transpose(2,1);  // initialize by z1;
    torch::Tensor u_H;
    torch::Tensor F_H;
    torch::Tensor d_H;
    torch::Tensor v_rel;
    while (not_converged_t or not_converged_l) {
        /* Bow excitation */
        auto Bow = bow_term_rhs(
            N_t, h_t, k, u, u1, u2,
            x_bow.select(1,global_step),
            v_bow.select(1,global_step),
            F_bow.select(1,global_step),
            bow_wid_length, phi_0, phi_1,
            iter);
        auto G_B = Bow[0]; v_rel = Bow[1];

        /* Hammer excitation */
        auto Hammer = hammer_term_rhs(
            N_t, h_t, k, u, u1, u2,
            x_H, u_H1, u_H2, w_H, M_r,
            alpha_H, tol_t, hammer_mask.view(-1));
        auto G_H = Hammer[0]; F_H = Hammer[1]; u_H = Hammer[2]; d_H = Hammer[3];

        G_B = expand(G_B, 1, N_t_max+N_l_max);
        G_H = expand(G_H, 1, N_t_max+N_l_max);

        // solve
        auto LHS = A;
        auto RHS = torch::matmul(B, w1)
                 + torch::matmul(C, w2)
                 +    bow_mask * G_B.nan_to_num()
                 + hammer_mask * G_H.nan_to_num();

        RHS = mask_1d(RHS, N_t+N_l+2, N_t_max+N_l_max);

        //auto w = lstsq(LHS, - RHS, A_P, 1e-8);
        //auto w = get<0>(torch::linalg::lstsq(LHS, - RHS, at::nullopt, at::nullopt));
        //auto w = torch::linalg::solve(LHS, - RHS);
        auto w = torch::matmul(A_P, - RHS);

        auto new_u = w.narrow(1,0,N_t_max);
        auto new_z = w.narrow(1,N_t_max,N_l_max);
        new_u = mask_1d(new_u, N_t+1, N_t_max);
        new_z = mask_1d(new_z, N_l+1, N_l_max);

        new_u = dirichlet_boundary(new_u, N_t, N_t_max);
        new_z = dirichlet_boundary(new_z, N_l, N_l_max);

        torch::Tensor residual_u = u - new_u;
        torch::Tensor residual_z = z - new_z;
        auto res_u = get<0>(residual_u.flatten(1).abs().max(1)); // \ell_\infty norm (values, index)
        auto res_z = get<0>(residual_z.flatten(1).abs().max(1)); // \ell_\infty norm (values, index)
        not_converged_t = res_u.gt(tol_t).any().item<bool>();
        not_converged_l = res_z.gt(tol_l).any().item<bool>();

        u = new_u;
        z = new_z;
    }

    u = u.squeeze(2);
    z = z.squeeze(2);

    // save and readout
    state_u = add_in(state_u, u, local_step, 1);
    state_z = add_in(state_z, z, local_step, 1);
    auto u_rp_int  = 1 + torch::floor(N_t * rp).view({-1,1}).to(torch::kLong);   // rounded grid index for readout
    auto u_rp_frac = 1 + rp.view({-1,1}) / h_t.view({-1,1}) - u_rp_int;          // fractional part of readout location
    auto z_rp_int  = 1 + torch::floor(N_l * rp).view({-1,1}).to(torch::kLong);   // rounded grid index for readout
    auto z_rp_frac = 1 + rp.view({-1,1}) / h_l.view({-1,1}) - z_rp_int;          // fractional part of readout location

    torch::Tensor u_out;
    torch::Tensor z_out;
    if (surface_integral) { // using surface integral of velocities
        auto r_w = 0.5 * h_t.view({-1,1,1});
        auto r_H = r_w;
        auto r_B = r_w;
        u_rp_frac = u_rp_frac.unsqueeze(2); // (B, 1, 1)
        z_rp_frac = z_rp_frac.unsqueeze(2); // (B, 1, 1)
        u_out = (u - state_u.narrow(1,local_step-1,1).squeeze(1));
        z_out = (z - state_z.narrow(1,local_step-1,1).squeeze(1));

        // Naive weighting. TODO: use distance-based weighting
        auto w_u = r_w * torch::ones_like(u_rp_frac) // (B, 1, 1)
                 + r_H * hammer_mask                 // (B, 1, 1)
                 + r_B * bow_mask;                   // (B, 1, 1)
        auto w_z = r_w * torch::ones_like(z_rp_frac) // (B, 1, 1)
                 + r_H * hammer_mask                 // (B, 1, 1)
                 + r_B * bow_mask;                   // (B, 1, 1)

        u_out = (u_out * w_u.squeeze(1) / k).sum(-1); // (B, )
        z_out = (z_out * w_z.squeeze(1) / k).sum(-1); // (B, )
    }
    else { // using interpolated pickup point
        u_out = (1 - u_rp_frac) * u.gather(1, u_rp_int  ).view({-1,1})
              +      u_rp_frac  * u.gather(1, u_rp_int+1).view({-1,1});
        z_out = (1 - z_rp_frac) * z.gather(1, z_rp_int  ).view({-1,1})
              +      z_rp_frac  * z.gather(1, z_rp_int+1).view({-1,1});
    }
    uout = assign(uout, u_out.view(-1), global_step, 1);
    zout = assign(zout, z_out.view(-1), global_step, 1);
    v_r_out = assign(v_r_out, v_rel.view(-1), global_step, /*dim*/1);
    F_H_out = assign(F_H_out,   F_H.view(-1), global_step, /*dim*/1);
    u_H_out = add_in(u_H_out,   u_H.view(-1), global_step, /*dim*/1);

    return { uout, zout, state_u, state_z, v_r_out, F_H_out, u_H_out, sig0, sig1 };
}


