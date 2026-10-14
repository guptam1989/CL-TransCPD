"""Contrastive change-point pretraining (CCPP) losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def triplet_loss(za, zp, zn, margin: float = 1.0) -> torch.Tensor:
    za, zp, zn = F.normalize(za, dim=-1), F.normalize(zp, dim=-1), F.normalize(zn, dim=-1)
    return F.relu((za - zp).norm(dim=-1) - (za - zn).norm(dim=-1) + margin).mean()


def info_nce(za, zp, zn, tau: float = 0.1) -> torch.Tensor:
    """InfoNCE with the anchor's positive against all transition negatives in the batch."""
    za, zp, zn = F.normalize(za, dim=-1), F.normalize(zp, dim=-1), F.normalize(zn, dim=-1)
    pos = (za * zp).sum(-1, keepdim=True)
    neg = za @ zn.t()
    logits = torch.cat([pos, neg], dim=1) / tau
    return F.cross_entropy(logits, torch.zeros(len(za), dtype=torch.long, device=za.device))


def contrastive_loss(kind: str, za, zp, zn, margin: float, tau: float) -> torch.Tensor:
    if kind == "triplet":
        return triplet_loss(za, zp, zn, margin)
    if kind == "infonce":
        return info_nce(za, zp, zn, tau)
    if kind == "none":
        return za.new_zeros(())
    raise ValueError(kind)


def masked_mse(recon, target, mask) -> torch.Tensor:
    m = mask[:, None, :].to(recon.dtype)
    return (((recon - target) ** 2) * m).sum() / (m.sum() * recon.shape[1]).clamp_min(1.0)
