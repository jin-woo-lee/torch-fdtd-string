using namespace std;
# define M_PI           3.14159265358979323846  /* pi */
const float M_HD = -0.01; /* max hammer displacement */

#include <torch/extension.h>
#include <iostream>
#include <vector>

#include "misc.h"
#include "string.h"

namespace F = torch::nn::functional;

vector<torch::Tensor> forward_fn(
    torch::Tensor state_u,                //   transverse displacement state
    torch::Tensor state_z,                // longitudinal displacement state
    vector<torch::Tensor> string_params,  // string parameters
    vector<torch::Tensor> bow_params,     // bow excitation parameters
    vector<torch::Tensor> hammer_params,  // hammer excitation parameters
    torch::Tensor bow_mask,               // bow excitation mask
    torch::Tensor hammer_mask,            // hammer excitation mask
    vector<float> constant,               // global constants
    float relative_error,                 // order of the discretization error
    bool surface_integral,                // pickup configuration
    int Nt) {                             // number of simulation samples

    int batch_size = state_u.size(0);
    float k = constant[0];

    auto uout = torch::zeros({batch_size,Nt}, state_u.dtype()).to(device()); // pickup displacement for output
    auto zout = torch::zeros({batch_size,Nt}, state_u.dtype()).to(device()); // pickup displacement for output
    auto v_b = torch::zeros({batch_size,Nt}, state_u.dtype()).to(device()); // relative velocity at the bowing point
    auto F_H = torch::zeros({batch_size,Nt}, state_u.dtype()).to(device()); // hammering force profile
    auto u_H = hammer_params[2];              // hammer displacement
    torch::Tensor sig0;   // freq-independent loss term
    torch::Tensor sig1;   // freq-dependent loss term

    for (int n=2; n < Nt; n++) {
        auto results = string_step(
            uout, zout, state_u, state_z, v_b, F_H,
            string_params, bow_params, hammer_params,
            bow_mask, hammer_mask,
            constant, n, n, relative_error, surface_integral);
        uout = results[0];
        zout = results[1];
        state_u = results[2];
        state_z = results[3];
        v_b = results[4];
        F_H = results[5];
        u_H = results[6];
        sig0 = results[7];
        sig1 = results[8];
    }
    u_H = u_H / k;
    return { uout, zout, state_u, state_z, v_b, F_H, u_H, sig0, sig1 };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_fn", &forward_fn, "string-bow forward iteration");
}

