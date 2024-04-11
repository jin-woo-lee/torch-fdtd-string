#ifndef BOW_H
#define BOW_H

torch::Tensor hard_bow(torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor soft_bow(torch::Tensor, torch::Tensor, torch::Tensor);
vector<torch::Tensor> bow_term_rhs(
    torch::Tensor, torch::Tensor,
    float,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    int
);

#endif
