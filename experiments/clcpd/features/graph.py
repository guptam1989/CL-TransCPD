"""Cross-lead graph stream F_graph (in-degree, directed clustering, asymmetry): 12 + 12 + 66 = 90 dims."""
from __future__ import annotations

import torch


@torch.no_grad()
def lead_graph(attn_l_layers, pvalid: torch.Tensor) -> torch.Tensor:
    """Average cross-lead attention over layers, heads and valid patch positions.

    attn_l_layers: list of (B, S, H, L, L) (S includes the CLS position 0); returns (B, L, L)."""
    w = pvalid.float()                                          # (B, P)
    acc = None
    for A in attn_l_layers:
        A = A.float()[:, 1:]                                    # drop CLS position -> (B, P, H, L, L)
        m = (A.mean(2) * w[:, :, None, None]).sum(1) / w.sum(1).clamp_min(1.0)[:, None, None]
        acc = m if acc is None else acc + m
    return acc / len(attn_l_layers)


@torch.no_grad()
def graph_features(G: torch.Tensor) -> torch.Tensor:
    """G: (B, L, L) row-stochastic cross-lead attention (G[i, j] = weight lead i puts on lead j)."""
    B, L, _ = G.shape
    eye = torch.eye(L, device=G.device, dtype=G.dtype)
    A0 = G * (1 - eye)
    in_deg = A0.sum(1) / (L - 1)                                # column sums: attention received
    cyc = torch.diagonal(A0 @ A0 @ A0, dim1=1, dim2=2)          # sum_jk A(i,j) A(j,k) A(k,i)
    s = A0.sum(2)
    denom = (s ** 2 - (A0 ** 2).sum(2)).clamp_min(1e-8)
    clust = cyc / denom
    iu = torch.triu_indices(L, L, offset=1, device=G.device)
    asym = (G - G.transpose(1, 2))[:, iu[0], iu[1]]
    return torch.cat([in_deg, clust, asym], dim=1)
