"""Attention-statistics stream F_att (entropy, head diversity, peak position), batched in torch.

Inputs are the temporal attention weights of every layer, shaped (B, L, H, S, S) with S = P + 1
(position 0 is the CLS token). Only valid (non-padded) patch queries are used; keys include CLS.
Per layer and lead the stream contains H entropies, 1 head-diversity value and H peak positions,
so dim(F_att) = N_layers * L * (2H + 1).
"""
from __future__ import annotations

import torch

EPS = 1e-8


def _jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + EPS) - torch.log(m + EPS))).sum(-1)
    kl_qm = (q * (torch.log(q + EPS) - torch.log(m + EPS))).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def attention_stats(attn_layers, pvalid: torch.Tensor) -> torch.Tensor:
    """attn_layers: list (len N_L) of (B, L, H, S, S); pvalid: (B, P) bool. Returns (B, N_L*L*(2H+1))."""
    feats = []
    B, P = pvalid.shape
    qmask = pvalid.float()                                   # valid patch queries (exclude CLS row)
    nq = qmask.sum(-1).clamp_min(1.0)                         # (B,)
    for A in attn_layers:
        A = A.float()
        _, L, H, S, _ = A.shape
        Aq = A[:, :, :, 1:, :]                                # queries = patches, keys = CLS + patches
        w = qmask[:, None, None, :, None]
        # normalised entropy over valid queries
        ent_rows = -(Aq * torch.log(Aq + EPS)).sum(-1)        # (B, L, H, P)
        nk = (nq + 1.0)                                       # valid keys incl. CLS
        ent = (ent_rows * qmask[:, None, None, :]).sum(-1) / (nq[:, None, None] * torch.log(nk)[:, None, None])
        # query-averaged key distribution per head (over patch keys only, renormalised)
        abar = (Aq * w).sum(3) / nq[:, None, None, None]      # (B, L, H, S)
        abar_p = abar[..., 1:]
        abar_p = abar_p / abar_p.sum(-1, keepdim=True).clamp_min(EPS)
        div = torch.zeros(B, L, device=A.device)
        n_pairs = 0
        for h in range(H):
            for g in range(h + 1, H):
                div = div + _jsd(abar_p[:, :, h], abar_p[:, :, g])
                n_pairs += 1
        div = div / max(1, n_pairs)
        peak = abar_p.argmax(-1).float() / nq[:, None, None]  # (B, L, H) in [0, 1)
        feats.append(torch.cat([ent, div[..., None], peak], dim=-1).reshape(B, -1))
    return torch.cat(feats, dim=1)
