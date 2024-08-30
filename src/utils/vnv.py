import torch

def RDE(f_1, f_exact, dim=None):
    num = f_1 - f_exact
    den = f_exact
    num = num.sum(dim) if dim is not None else num
    den = den.sum(dim) if dim is not None else den
    return 100 * num / den

