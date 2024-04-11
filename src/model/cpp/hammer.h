#ifndef HAMMER_H
#define HAMMER_H

vector<torch::Tensor> hammer_loop(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
    float, torch::Tensor, torch::Tensor
);
vector<torch::Tensor> hammer_term_rhs(
    torch::Tensor, torch::Tensor, float, // N, h, k
    torch::Tensor, torch::Tensor, torch::Tensor, // u, u1, u2
    torch::Tensor, torch::Tensor, torch::Tensor, // x_H, u_H1, u_H2, w_H
    torch::Tensor, torch::Tensor, torch::Tensor, // w_H, M_r, alpha
    torch::Tensor, torch::Tensor // threshold, hammer_mask
);

#endif
