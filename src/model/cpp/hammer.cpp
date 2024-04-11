using namespace std;
# define M_PI           3.14159265358979323846  /* pi */
const float M_HD = -0.01; /* max hammer displacement */

#include <torch/extension.h>
#include <iostream>
#include <vector>

#include "misc.h"

vector<torch::Tensor> hammer_loop(
    torch::Tensor u_H1,    // hammer displacement, curr. time (batch_size,)
    torch::Tensor u_H2,    // hammer displacement, prev. time (batch_size,)
    torch::Tensor eta_1,   // relative hammer displacement, curr. time (batch_size,)
    torch::Tensor eta_2,   // relative hammer displacement, prev. time (batch_size,)
    torch::Tensor alpha,   // nonlinear exponent (batch_size,)
    torch::Tensor w_H,     // hammer stiffness parameter (batch_size,)
    torch::Tensor M_r,     // hammer / string mass ratio (batch_size,)
    torch::Tensor eps_u,   // hammering point string displacement, future time (batch_size,)
    float k,
    torch::Tensor threshold,
    torch::Tensor mask) {
    int iter = 0;
    bool not_converged = true;
    torch::Tensor F_H;
    torch::Tensor u_H;
    torch::Tensor eta;
    torch::Tensor eta_estimate;

    int batch_size = u_H1.size(0);
    eta = eta_1 * mask;
    eta_estimate = eta_1 * mask;
    while (not_converged) {
        eta = eta_estimate;

        // hammering force
        auto f_H = w_H.pow(1+alpha)
            * torch::relu(eta_1).pow(alpha-1)
            * (eta + eta_2) / 2;
        F_H = torch::where(eta_1.gt(0), f_H, torch::zeros_like(f_H));

        // u_tt = - F_H
        // (u_H - 2u_H1 + u_H2) / k^2 = - F_H
        u_H = 2*u_H1 - u_H2 - pow(k, 2.) * F_H; // hammering point string displacement, future time
        u_H = torch::relu(u_H - M_HD) + M_HD;

        eta_estimate = (u_H - eps_u) * mask;

        torch::Tensor residual = (eta - eta_estimate).abs(); // (batch,)

        not_converged = residual.gt(threshold).any().item<bool>();
    }
    return { F_H, u_H };
}

vector<torch::Tensor> hammer_term_rhs(
    torch::Tensor N,
    torch::Tensor h,
    float k,
    torch::Tensor u,
    torch::Tensor u1,
    torch::Tensor u2,
    torch::Tensor x_H,       // hammer position
    torch::Tensor u_H1,      // hammer displacement, curr. time (batch_size,)
    torch::Tensor u_H2,      // hammer displacement, prev. time (batch_size,)
    torch::Tensor w_H,       // hammer stiffness parameter (batch_size,)
    torch::Tensor M_r,       // hammer / string mass ratio (batch_size,)
    torch::Tensor alpha,     // nonlinear exponent (batch_size,)
    torch::Tensor threshold, // threshold (batch_size,)
    torch::Tensor mask) {    // zero-mask updates on batches that are not hammer-excited (batch_size,)
    auto eps = floor_dirac_delta(N-1, x_H, u1.size(1)).transpose(1,2);
    auto eps_u  = torch::matmul(eps, u).view(-1);         // (batch_size,)
    auto eta_1 = u_H1 - torch::matmul(eps, u1).view(-1);  // (batch_size,)
    auto eta_2 = u_H2 - torch::matmul(eps, u2).view(-1);  // (batch_size,)

    auto loop_out = hammer_loop(
        u_H1, u_H2, eta_1, eta_2,
        alpha, w_H, M_r, eps_u, k, threshold, mask);
    auto F_H = loop_out[0].view({-1,1,1});
    auto u_H = loop_out[1].view({-1,1,1});
    auto Gamma = eps.transpose(1,2) * M_r.view({-1,1,1}) * F_H;
    auto d_H = eps.transpose(1,2) * torch::relu(u_H - eps_u.view({-1,1,1}));

    return { - pow(k, 2.) * Gamma, F_H, u_H, d_H };
}


