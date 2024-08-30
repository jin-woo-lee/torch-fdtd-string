#ifndef VNV_H
#define VNV_H

torch::Tensor manufactured_solution_forcing_term(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double
);

#endif
