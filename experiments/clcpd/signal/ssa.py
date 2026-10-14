"""Singular spectrum analysis (SSA) denoising, batched in PyTorch.

The recursive SSA (RSSA) of the authors' earlier work updates the decomposition sample by
sample; for a complete segment its output equals batch SSA, which is what is computed here.
The rank is chosen per series as the smallest number of eigen-triples whose cumulative energy
reaches ``energy`` (capped at ``max_rank``). Eigen-decomposition of the W x W lag-covariance
matrix is used because it is much faster than an SVD of the trajectory matrix for small W.
"""
from __future__ import annotations

import torch


def ssa_denoise(x: torch.Tensor, window: int = 25, energy: float = 0.98, max_rank: int = 10) -> torch.Tensor:
    """x: (N, T) tensor -> denoised (N, T) tensor (same dtype/device). Always computed in float32."""
    with torch.autocast(device_type=x.device.type, enabled=False):   # eigh does not support bf16/fp16
        return _ssa_fp32(x, window, energy, max_rank)


def _ssa_fp32(x, window, energy, max_rank):
    orig_dtype = x.dtype
    x = x.float()
    N, T = x.shape
    W = int(window)
    if T <= W:
        return x.to(orig_dtype)
    K = T - W + 1
    dev = x.device
    idx = torch.arange(W, device=dev)[:, None] + torch.arange(K, device=dev)[None, :]
    X = x[:, idx]                                            # (N, W, K) trajectory matrices
    C = X @ X.transpose(1, 2)                                # (N, W, W) lag covariance
    C = C + 1e-6 * torch.eye(W, device=dev)
    evals, evecs = torch.linalg.eigh(C)                      # ascending order
    evals = evals.flip(-1).clamp_min(0.0)
    evecs = evecs.flip(-1)
    frac = evals.cumsum(-1) / evals.sum(-1, keepdim=True).clamp_min(1e-12)
    rank = ((frac < energy).sum(-1) + 1).clamp(1, int(max_rank))
    keep = (torch.arange(W, device=dev)[None, :] < rank[:, None]).float()
    U = evecs * keep[:, None, :]
    Xr = U @ (U.transpose(1, 2) @ X)                         # rank-r reconstruction
    out = torch.zeros(N, T, device=dev)
    out.index_add_(1, idx.reshape(-1), Xr.reshape(N, -1))     # diagonal averaging
    cnt = torch.zeros(T, device=dev)
    cnt.index_add_(0, idx.reshape(-1), torch.ones(W * K, device=dev))
    return (out / cnt).to(orig_dtype)
