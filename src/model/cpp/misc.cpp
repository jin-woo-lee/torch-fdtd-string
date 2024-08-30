using namespace std;
# define M_PI           3.14159265358979323846  /* pi */

#include <torch/extension.h>
#include <iostream>
#include <vector>

#include <ATen/ATen.h>
using namespace at;

namespace F = torch::nn::functional;

torch::Device device() {
    return torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
}
torch::TensorOptions ntopt() { // new tensor options
    return torch::TensorOptions().device(device());
}

torch::Tensor raised_cosine(
    torch::Tensor n,        // number of samples in space
    torch::Tensor ctr,      // center point of raised cosine curve in (0, 1]
    torch::Tensor wid,      // width of raised cosine curve in (0, 1]
    int N) {

    float h = 1. / N;
    auto xax = torch::linspace(h, 1, N, ntopt()).view({1,-1,1}).to(n.dtype()); // (1, N, 1)
    ctr = (ctr * n / N).view({-1,1,1});  // abs portion -> rel portion (batch_size, 1, 1)
    wid = (wid * n / N).view({-1,1,1});  // abs portion -> rel portion (batch_size, 1, 1)
    auto ind = torch::sign(torch::relu(-(xax - ctr - wid / 2) * (xax - ctr + wid / 2)));
    auto out = 0.5 * ind * (1 + torch::cos(2 * M_PI * (xax - ctr) / wid));
    out = out / out.abs().sum(1, /*keepdim*/true);
    return out; // (batch_size, N, 1)
}

torch::Tensor floor_dirac_delta(
    torch::Tensor n,        // number of samples in space
    torch::Tensor ctr,      // center point of raised cosine curve
    int N) {
    auto xax = torch::ones_like(ctr).view({-1,1,1}).repeat({1,N,1}).cumsum(1) - 1;
    auto idx = torch::floor(ctr * n).view({-1,1,1});
    return torch::floor(xax).eq(idx).to(n.dtype());  // (batch_size, N, 1)
}

torch::Tensor domain_x(int N, torch::Tensor n) {
    /*  N    (int): number of maximal samples in space
     *  n    (B, 1, 1): number of actual samples in space
     */
    auto v = 2 / n;
    v = (v * torch::ones_like(v).repeat({1,1,N})).cumsum(2) - v;
    return (v.clamp(0,2).transpose(2,1) - 1)/2;
}

torch::Tensor triangular(
    int N,
    torch::Tensor n,
    torch::Tensor p_x,
    torch::Tensor p_a) {
    /*  N    (int): number of maximal samples in space
     *  n    (B,  1, 1): number of actual samples in space
     *  p_x  (B, Nt, 1): peak position
     *  p_a  (B, Nt, 1): peak amplitude
     */
    auto vel_l = p_a / p_x / n;
    auto vel_r = p_a / (1-p_x) / n;
    vel_l =  (vel_l * torch::ones_like(vel_l).repeat({1,1,N})).cumsum(2) - vel_l;
    vel_r = ((vel_r * torch::ones_like(vel_r).repeat({1,1,N})).cumsum(2) - vel_r * (N-n+1)).clamp(/*min*/0).flip(2);
    return torch::minimum(vel_l, vel_r);
}

torch::Tensor expand(torch::Tensor X, int N_w, int N_h) {
    int n_h = X.size(-2); int n_w = X.size(-1);
    auto kwargs = F::PadFuncOptions({0, N_w-n_w, 0, N_h-n_h}).mode(torch::kConstant);
    return F::pad(X.unsqueeze(1), kwargs).squeeze(1);
}

/* Interpolator */
torch::Tensor Interpolator(int dim_i, int dim_o) {
    /* dim_i (int) : input dimension
     * dim_o (int) : output dimension
     * Returns a tensor with shape (dim_o, dim_i)
     * Be sure to match the right dtype, e.g., Interpolator(...).to(foo.dtype())
     */
    auto diagonal = torch::diag_embed(torch::ones(dim_i, ntopt())).view({1,dim_i,dim_i});
    auto kwargs = F::InterpolateFuncOptions().size(vector<int64_t>({dim_o}))
                                             .mode(torch::kLinear).align_corners(true);
    return F::interpolate(diagonal, kwargs).transpose(1,2);
}

/* Interpolator operator */
torch::Tensor batched_interpolator(torch::Tensor N_i, torch::Tensor N_o) {
    /* N_i (batch_size,) : input dimension
     * N_o (batch_size,) : output dimension
     * returns a Interpolator tensor with shape (batch_size, max_o, max_i)
     */
    int batch_size = N_i.size(0);
    int max_i = torch::max(N_i).item<int>();
    int max_o = torch::max(N_o).item<int>();
    auto out = torch::zeros({batch_size, max_o, max_i}, ntopt()).to(N_i.dtype());
    for (int b=0; b < batch_size; b++) {
        int dim_i = N_i[b].item<int>();  int dim_o = N_o[b].item<int>();
        out[b] = expand(Interpolator(dim_i, dim_o).to(N_i.dtype()), max_i, max_o).squeeze(0);
    }
    return out;
}

/* Diagonalizing operator */
torch::Tensor batched_diag(torch::Tensor lam) {
    // lam  : (batch_size, N, 1) diagonal entries
    // return (batch_size, N, N) tensor with each diagonal element specified by lam
    auto maps = lam * torch::ones_like(lam).repeat({1,1,lam.size(1)});
    auto cr = torch::ones_like(lam).cumsum(1).repeat({1,1,lam.size(1)});  // (batch_size, [N], N)
    auto rc = torch::ones_like(lam).repeat({1,1,lam.size(1)}).cumsum(2);  // (batch_size,  N, [N])
    auto mask = cr.eq(rc);
    return torch::where(mask, maps, torch::zeros_like(maps));
}

/* Identity operator */
torch::Tensor I(torch::Tensor n, int diagonal=0) {
    // n (batch_size,) : width of identity matrix
    // return identity matrices of maximum width
    int l;
    if (diagonal == 0) { l = torch::max(n).item<int>(); }
    else { l = torch::max(n).item<int>() - abs(diagonal); }
    auto i = torch::ones(l, ntopt()).to(n.dtype());
    return torch::diag(i, diagonal).unsqueeze(0).repeat({n.size(0),1,1});
}

/* Difference operator */
torch::Tensor Dxx(torch::Tensor n, torch::Tensor h) {
    auto Dx = I(n, +1) - 2*I(n) + I(n, -1);
    return Dx / h.pow(2).view({-1,1,1});
}
torch::Tensor Dxf(torch::Tensor n, torch::Tensor h) {
    auto Dx = I(n, +1) - I(n);
    return Dx / h.view({-1,1,1});
}
torch::Tensor Dxb(torch::Tensor n, torch::Tensor h) {
    auto Dx = I(n) - I(n, -1);
    return Dx / h.view({-1,1,1});
}
torch::Tensor Dxxxx(torch::Tensor n, torch::Tensor h) {
    auto Dx = I(n, +2) - 4*I(n, +1) + 6*I(n) - 4*I(n, -1) + I(n, -2);
    return Dx / h.pow(4).view({-1,1,1});
}
torch::Tensor Dxxxx_clamped(torch::Tensor n, torch::Tensor h) {
    // Fourth-order difference operator for the boundary condition u_{-1} == u_{1}.
    /* [[[ 6., -4.,  1.,  0.,    ,    ],
         [-4.,  7., -4.,  1.,  0.,    ],
         [ 1., -4.,  6., -4.,  1.,  0.],
         [ 0.,  1., -4.,  6., -4.,  1.],
         [   ,  0.,  1., -4.,  7., -4.],
         [   ,    ,  0.,  1., -4.,  6.]]] / h^4 */
    int n_max = torch::max(n).item<int>();
    auto ones = torch::ones(n_max, ntopt()).to(n.dtype()).view({1,n_max,1});
    auto maps = ones.cumsum(1)-1;
    auto rpos = (n-2).view({-1,1,1}).repeat({1,n_max,1}); // (batch_size, n_max, 1) : filled with (n-2)
    auto mask_l = maps.eq(ones);        // true only at index 1
    auto mask_r = maps.eq(rpos);        // true only at index n-2
    auto SM = I(n) * mask_l.logical_or(mask_r);
    auto Dx = I(n, +2) - 4*I(n, +1) + 6*I(n) - 4*I(n, -1) + I(n, -2);
    return (Dx + SM) / h.pow(4).view({-1,1,1});
}
torch::Tensor Mxc(torch::Tensor n) {
    return (I(n, +1) + I(n, -1)) / 2;
}

torch::Tensor block_matrices(vector< vector<torch::Tensor> > X) {
    int n_rows = X.size(); int n_cols = X[0].size();
    torch::Tensor out;
    for (int i=0; i < n_rows; i++) {
        torch::Tensor row = X[i][0];
        for (int j=1; j < n_cols; j++) {
            row = torch::cat({row, X[i][j]}, 2);
        }
        if (i==0) { out = row; }
        else { out = torch::cat({out, row}, 1); }
    }
    return out;
}

torch::Tensor mask_1d(torch::Tensor u, torch::Tensor N, int N_max) {
    // u : (batch_size, N_max, 1)
    auto maps = torch::ones_like(u).cumsum(1);         // (batch_size, N_max, 1) : arange for N_max
    auto cons = N.view({-1,1,1}).repeat({1,N_max,1});  // (batch_size, N_max, 1) : filled with N
    auto mask = maps.le(cons);                         // (batch_size, N_max, 1) : boolean mask
    // mask to actual length
    return u * mask;
}
torch::Tensor mask_2d(torch::Tensor X, torch::Tensor N, int N_max) {
    // X : (batch_size, N_max, N_max)
    auto maps = torch::ones_like(X).cumsum(1).cumsum(2);   // (batch_size, N_max, N_max)
    auto cons = N.view({-1,1,1}).repeat({1,N_max,N_max});  // (batch_size, N_max, N_max)
    auto mask = maps.le(cons);                             // (batch_size, N_max, N_max)
    // mask to actual length
    return X * mask;
}
torch::Tensor dirichlet_boundary(torch::Tensor u, torch::Tensor N, int N_max) {
    // u : (batch_size, N_max, 1)
    // zero-out u at position index 0 and N
    auto maps = torch::ones_like(u).cumsum(1)-1;      // (batch_size, N_max, 1) : arange for N_max
    auto zero = torch::zeros_like(u);                 // (batch_size, N_max, 1) : filled with 0
    auto rpos = N.view({-1,1,1}).repeat({1,N_max,1}); // (batch_size, N_max, 1) : filled with N
    auto mask_l = maps.eq(zero).logical_not();        // false only at index 0
    auto mask_r = maps.eq(rpos).logical_not();        // false only at index N
    return u * mask_l * mask_r;
}

torch::Tensor inverse_like(torch::Tensor A) {
    // Compute the Moore-Penrose pseudo-inverse
    c10::optional<at::Tensor> atol; c10::optional<at::Tensor> rtol;
    return at::linalg_pinv(A, atol, rtol, /*hermitian*/true);
}
vector<torch::Tensor> split_blocks(torch::Tensor X, int N_t, int N_l) {
    auto X_split = X.split({ N_t, N_l }, /*dim*/-2);
    auto X_01    = X_split[0].split({ N_t, N_l }, /*dim*/-1);
    auto X_23    = X_split[1].split({ N_t, N_l }, /*dim*/-1);
    return { X_01[0], X_01[1], X_23[0], X_23[1] };
}
torch::Tensor sparse_blocks(vector<torch::Tensor> X, int N_t_max, int N_l_max) {
    auto X_0 = expand(X[0], /*width*/N_t_max, /*height*/N_t_max);
    auto X_1 = expand(X[1], /*width*/N_l_max, /*height*/N_t_max);
    auto X_2 = expand(X[2], /*width*/N_t_max, /*height*/N_l_max);
    auto X_3 = expand(X[3], /*width*/N_l_max, /*height*/N_l_max);
    return block_matrices({ { X_0, X_1 }, { X_2, X_3 } }); // N_t_max+N_l_max, N_t_max+N_l_max
}

torch::Tensor tridiagonal_inverse(torch::Tensor X, torch::Tensor N) {
    // X (batch_size, n, n) : tridiagonal matrix to invert
    // N (batch_size, )     : actual width + 1 of the matrix
    int batch_size = X.size(0); int n = X.size(1);
    auto k = 1 + torch::arange(n, ntopt()).to(X.dtype());              // (batch_size, )
    auto jk = torch::outer(k,k).unsqueeze(0).repeat({batch_size,1,1}); // (batch_size, n, n)
    auto kb = k.unsqueeze(0).repeat({batch_size,1});                   // (batch_size, n)

    auto a = X.select(2,0).select(1,1).unsqueeze(1);                   // (batch_size, 1)
    auto b = X.select(2,0).select(1,0).unsqueeze(1);                   // (batch_size, 1)
    auto c = X.select(2,1).select(1,0).unsqueeze(1);                   // (batch_size, 1)
    auto Nb = N.unsqueeze(1);                                          // (batch_size, 1)

    auto lam = b + (a+c) * torch::cos(kb * M_PI / Nb);                 // (batch_size, 1)
    auto Lid = torch::diag(torch::ones_like(k)).unsqueeze(0);          // (1, n, n)
    auto Lam = 1 / lam.unsqueeze(1);                                   // (batch_size, 1, 1)
    auto L = Lid * Lam;                                                // (batch_size, n, n)
    auto V = (2. / Nb.unsqueeze(-1)).pow(0.5)
           * torch::sin(jk * M_PI / Nb.unsqueeze(-1));                 // (batch_size, n, n)

    // apply mask
    L = mask_2d(L, N, n);

    return torch::matmul(V, torch::matmul(L, V.transpose(-1,-2)));
}

torch::Tensor assign(
    torch::Tensor x,
    torch::Tensor y,
    int index, int dim) {
    x = x.transpose(0,dim);    // transpose target dim with dim 0
    x[index] = y;              // assign
    return x.transpose(0,dim); // revert by transpose
}

torch::Tensor add_in(
    torch::Tensor x,
    torch::Tensor y,
    int index, int dim) {
    x = x.transpose(0,dim);    // transpose target dim with dim 0
    x[index] += y;             // assign
    return x.transpose(0,dim); // revert by transpose
}
torch::Tensor lstsq(
    torch::Tensor LHS,
    torch::Tensor RHS,
    torch::Tensor pseudo_inverse,
    float threshold=1e-4) {
    torch::Tensor solution;

    // Compute the solution
    solution = torch::matmul(pseudo_inverse, RHS);
    torch::Tensor residual = torch::matmul(LHS, solution) - RHS;
    float err = residual.norm().item<float>();
    int iter = 0;
    int max_iter = 100;
    while ((err > threshold) and (iter++ < max_iter)) {
        torch::Tensor update = torch::matmul(pseudo_inverse, residual);
        solution -= update;
        residual = torch::matmul(LHS, solution) - RHS;
        err = residual.norm().item<float>();
    }
    return solution;
}

