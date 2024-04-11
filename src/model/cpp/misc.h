#ifndef MISC_H
#define MISC_H

torch::Device device();
torch::Device ntopt();
torch::Tensor raised_cosine(torch::Tensor, torch::Tensor, torch::Tensor, int);
torch::Tensor floor_dirac_delta(torch::Tensor, torch::Tensor, int);
torch::Tensor triangular(int, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor expand(torch::Tensor, int, int);
torch::Tensor Interpolator(int, int);
torch::Tensor batched_interpolator(torch::Tensor, torch::Tensor);
torch::Tensor batched_diag(torch::Tensor);
torch::Tensor I(torch::Tensor, int);
torch::Tensor Dxb(torch::Tensor, torch::Tensor);
torch::Tensor Dxf(torch::Tensor, torch::Tensor);
torch::Tensor Dxx(torch::Tensor, torch::Tensor);
torch::Tensor Dxxxx(torch::Tensor, torch::Tensor);
torch::Tensor Dxxxx_clamped(torch::Tensor, torch::Tensor);
torch::Tensor Mxc(torch::Tensor);
torch::Tensor block_matrices(vector< vector<torch::Tensor> >);
torch::Tensor mask_1d(torch::Tensor, torch::Tensor, int);
torch::Tensor mask_2d(torch::Tensor, torch::Tensor, int);
torch::Tensor dirichlet_boundary(torch::Tensor, torch::Tensor, int);
torch::Tensor inverse_like(torch::Tensor);
vector<torch::Tensor> split_blocks(torch::Tensor, int, int);
torch::Tensor sparse_blocks(vector<torch::Tensor>, int, int);
torch::Tensor tridiagonal_inverse(torch::Tensor, torch::Tensor);
torch::Tensor assign(torch::Tensor, torch::Tensor, int, int);
torch::Tensor add_in(torch::Tensor, torch::Tensor, int, int);
torch::Tensor lstsq(torch::Tensor, torch::Tensor, torch::Tensor, float);

#endif
