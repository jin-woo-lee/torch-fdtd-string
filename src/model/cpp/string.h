#ifndef FDM_H
#define FDM_H

vector<torch::Tensor> get_derived_vars(
    torch::Tensor, torch::Tensor, float, float, torch::Tensor
);
vector<torch::Tensor> string_step(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, // uout, zout, state_u, state_z
    torch::Tensor, torch::Tensor, // v_b, F_H
    vector<torch::Tensor>,
    vector<torch::Tensor>,
    vector<torch::Tensor>,
    torch::Tensor,
    torch::Tensor,
    vector<float>, int, int, float,
    bool, bool
);

#endif
