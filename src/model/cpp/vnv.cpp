using namespace std;

# define M_PI           3.14159265358979323846  /* pi */

#include <torch/extension.h>
#include <iostream>
#include <vector>

#include "misc.h"

torch::Tensor manufactured_solution_forcing_term(
    torch::Tensor gamma,
    torch::Tensor sig0,
    torch::Tensor K,
    torch::Tensor p_a,
    torch::Tensor x,
    double t
) {
    /* returns the forcing term for the manufactured solution
     * sigma == sig0
     * omega == gamma
     * mu == pi
     */
    auto sigma = sig0;
    auto omega = gamma;
    auto mu = M_PI;
    auto mu_sq = pow(M_PI,2);

    auto coeff_1 = (sigma.pow(2) - omega.pow(2) - 2*sig0*sigma)  * torch::cos(mu * x).pow(2);
    auto coeff_2 = (2*mu_sq * (4*K.pow(2)*mu_sq + gamma.pow(2))) * torch::cos(2*mu * x);
    auto coeff_3 = 2*omega*(sigma - sig0) * torch::cos(mu*x).pow(2);

    auto cos_term = (coeff_1 + coeff_2) * torch::cos(omega*t);
    auto sin_term = coeff_3 * torch::sin(omega*t);

    return p_a * (cos_term + sin_term) * torch::exp(-1 * sigma * t);
}

