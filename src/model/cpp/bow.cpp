using namespace std;
# define M_PI           3.14159265358979323846  /* pi */

#include <torch/extension.h>
#include <iostream>
#include <vector>

#include "misc.h"

torch::Tensor hard_bow(torch::Tensor v_rel, torch::Tensor a, torch::Tensor eps) {
    return torch::sign(v_rel) * (eps + (1-eps) * torch::exp(-a * v_rel.abs()));
}
torch::Tensor soft_bow(torch::Tensor v_rel, torch::Tensor a, torch::Tensor eps) {
    return (2*a).pow(.5) * v_rel * torch::exp(-a * v_rel.pow(2) + 0.5);
}

vector<torch::Tensor> bow_term_rhs(
    torch::Tensor N,
    torch::Tensor h,
    float k,
    torch::Tensor u,
    torch::Tensor u1,
    torch::Tensor u2,
    torch::Tensor x_B,
    torch::Tensor v_B,
    torch::Tensor F_B,
    torch::Tensor wid,
    torch::Tensor phi_0,
    torch::Tensor phi_1,
    int iter) {

    auto rc = raised_cosine(N-1, x_B, wid, u1.size(1));    // (batch_size, max(N), 1)
    auto I = rc;
    auto J = rc / h.view({-1,1,1});

    torch::Tensor v_rel;
    if (iter == 0) { v_rel = torch::matmul(I.transpose(1,2), ((u1 - u2) / k - v_B.view({-1,1,1}))); }
    else { v_rel = torch::matmul(I.transpose(1,2), ((u - u1) / k - v_B.view({-1,1,1}))); }
    auto Gamma = J * F_B.view({-1,1,1}) * hard_bow(v_rel, phi_0.view({-1,1,1}), phi_1.view({-1,1,1}));
    return { - pow(k, 2.) * Gamma, v_rel };  // {(batch_size, 1, 1), (batch_size, 1, 1)}
}

