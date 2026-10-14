"""Cross-Lead Transformer (CLT) and the full CL-TransCPD front-end.

Tokens are non-overlapping patches of p samples per lead. Every layer applies
(i) temporal self-attention within each lead (over the P patches plus a CLS position) and
(ii) cross-lead self-attention across the L leads at each patch position, followed by a
feed-forward network (pre-LayerNorm residual blocks). The per-lead CLS outputs are averaged
into the global context vector; patch outputs are mean-pooled into z (Eq. pool). A per-lead
bottleneck decoder reconstructs the SSA-denoised input for the reconstruction term.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..signal.ssa import ssa_denoise
from .htfp import HTFP


def sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MHA(nn.Module):
    """Multi-head self-attention that can return the attention weights alpha = softmax(QK^T/sqrt(d_k) + M)."""

    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.p = h, d // h, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, key_pad=None, need_weights=False):
        N, S, d = x.shape
        q, k, v = self.qkv(x).reshape(N, S, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        if need_weights:
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
            if key_pad is not None:
                logits = logits.masked_fill(key_pad[:, None, None, :], float("-inf"))
            w = logits.float().softmax(-1).to(v.dtype)
            out = F.dropout(w, self.p, self.training) @ v
        else:
            mask = None if key_pad is None else (~key_pad)[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                 dropout_p=self.p if self.training else 0.0)
            w = None
        return self.proj(out.transpose(1, 2).reshape(N, S, d)), w


class DualAxisBlock(nn.Module):
    def __init__(self, d, h, d_ff, dropout, temporal=True, cross_lead=True):
        super().__init__()
        self.temporal, self.cross = temporal, cross_lead
        if temporal:
            self.ln_t, self.att_t = nn.LayerNorm(d), MHA(d, h, dropout)
        if cross_lead:
            self.ln_l, self.att_l = nn.LayerNorm(d), MHA(d, h, dropout)
        self.ln_f = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad, need_weights=False):
        B, L, S, d = x.shape
        wt = wl = None
        if self.temporal:
            hid = self.ln_t(x).reshape(B * L, S, d)
            kp = key_pad[:, None, :].expand(B, L, S).reshape(B * L, S)
            o, wt = self.att_t(hid, kp, need_weights)
            x = x + self.drop(o.reshape(B, L, S, d))
            if wt is not None:
                wt = wt.reshape(B, L, -1, S, S)                       # (B, L, H, S, S)
        if self.cross:
            hid = self.ln_l(x).transpose(1, 2).reshape(B * S, L, d)
            o, wl = self.att_l(hid, None, need_weights)
            x = x + self.drop(o.reshape(B, S, L, d).transpose(1, 2))
            if wl is not None:
                wl = wl.reshape(B, S, -1, L, L)                       # (B, S, H, L, L)
        x = x + self.drop(self.ff(self.ln_f(x)))
        return x, wt, wl


class CLT(nn.Module):
    def __init__(self, c, n_leads: int, ref_len: int, n_classes: int = 5):
        super().__init__()
        d, p = int(c.d_model), int(c.patch)
        assert ref_len % p == 0, "ref_len must be a multiple of the patch size"
        self.L, self.p, self.P, self.T = n_leads, p, ref_len // p, ref_len
        self.patch_embed = nn.Linear(p, d)
        self.lead_embed = nn.Parameter(torch.zeros(n_leads, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, 1, d))
        nn.init.trunc_normal_(self.lead_embed, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.register_buffer("pos", sinusoidal(self.P + 1, d), persistent=False)
        self.blocks = nn.ModuleList([
            DualAxisBlock(d, int(c.n_heads), int(c.d_ff), float(c.dropout),
                          temporal=bool(c.get("temporal", True)), cross_lead=bool(c.cross_lead))
            for _ in range(int(c.n_layers))])
        self.ln = nn.LayerNorm(d)
        self.decoder = nn.Sequential(nn.Linear(d, int(c.dec_bottleneck)), nn.GELU(),
                                     nn.Linear(int(c.dec_bottleneck), int(c.dec_hidden)), nn.GELU(),
                                     nn.Linear(int(c.dec_hidden), ref_len))
        self.cls_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, int(c.cls_hidden)))
        self.cls_out = nn.Linear(int(c.cls_hidden), n_classes)

    def forward(self, x, mask, need_weights=False):
        B, L, T = x.shape
        tok = self.patch_embed(x.reshape(B, L, self.P, self.p))
        pvalid = mask.reshape(B, self.P, self.p).any(-1)
        h = torch.cat([self.cls.expand(B, L, 1, -1), tok], dim=2)
        h = h + self.pos[None, None].to(h.dtype) + self.lead_embed[None, :, None, :]
        key_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~pvalid], dim=1)
        attn_t, attn_l = [], []
        for blk in self.blocks:
            h, wt, wl = blk(h, key_pad, need_weights)
            attn_t.append(wt)
            attn_l.append(wl)
        h = self.ln(h)
        cls_out = h[:, :, 0].mean(1)
        w = pvalid[:, None, :, None].to(h.dtype)
        lead_rep = (h[:, :, 1:] * w).sum(2) / w.sum(2).clamp_min(1.0)       # (B, L, d)
        z = lead_rep.mean(1)
        f_cls = self.cls_mlp(cls_out)
        return {"z": z, "recon": self.decoder(lead_rep), "f_cls": f_cls,
                "logits": self.cls_out(F.gelu(f_cls)), "attn_t": attn_t, "attn_l": attn_l,
                "pvalid": pvalid}


class CLTransCPD(nn.Module):
    """HTFP -> RSSA (straight-through) -> CLT. ``forward`` also returns the reconstruction target."""

    def __init__(self, cfg, n_classes: int = 5):
        super().__init__()
        self.cfg = cfg
        self.htfp = HTFP(cfg.htfp.n_fft, cfg.htfp.hop, cfg.htfp.channels) if cfg.htfp.enabled else None
        self.use_rssa = bool(cfg.rssa.enabled)
        self.clt = CLT(cfg.clt, int(cfg.data.n_leads), int(cfg.segmentation.ref_len), n_classes)

    def _ssa(self, x):
        B, L, T = x.shape
        r = self.cfg.rssa
        return ssa_denoise(x.reshape(B * L, T), r.window, r.energy, r.max_rank).reshape(B, L, T)

    def preprocess(self, x, mask):
        m = mask[:, None, :].to(x.dtype)
        y = x
        if self.htfp is not None:
            y, _ = self.htfp(y, mask)
        if self.use_rssa:
            with torch.no_grad():
                s = self._ssa(y.detach())
            y = y + (s - y).detach()          # straight-through gradient for HTFP
        return y * m

    @torch.no_grad()
    def target(self, x, mask):
        t = self._ssa(x) if self.use_rssa else x
        return t * mask[:, None, :].to(x.dtype)

    def forward(self, x, mask, need_weights=False, with_target=True):
        out = self.clt(self.preprocess(x, mask), mask, need_weights)
        if with_target:
            out["target"] = self.target(x, mask)
        return out


def recon_error(recon, target, mask):
    """Mean absolute reconstruction error over valid samples and leads, per segment -> (B,)."""
    m = mask[:, None, :].to(recon.dtype)
    return ((recon - target).abs() * m).sum((1, 2)) / (m.sum((1, 2)) * recon.shape[1]).clamp_min(1.0)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
