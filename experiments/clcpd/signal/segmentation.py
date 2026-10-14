"""Instantaneous Heart-Rate-Adaptive Segmentation (IHRAS) and the fixed-window ablation."""
from __future__ import annotations

import numpy as np


def window_bounds(seg_cfg, fs: float):
    wmin = int(np.floor(seg_cfg.n_cycles * 60.0 / seg_cfg.hr_max * fs))
    wmax = int(np.floor(seg_cfg.n_cycles * 60.0 / seg_cfg.hr_min * fs))
    return wmin, wmax


def segment_bounds(n: int, fs: float, rpeaks, seg_cfg, mode: str | None = None,
                   causal: bool = True):
    """Return a list of (start, length) segments covering a signal of n samples.

    IHRAS: the window spans ``n_cycles`` times the median of the last ``n_rr_median`` RR
    intervals (Eq. 2 of the manuscript), clipped to [w_min, w_max], with 50 % overlap.
    In causal mode only R peaks confirmed before the segment start are used
    (peak time + confirmation delay <= start), which is what an online monitor can do.
    """
    mode = mode or seg_cfg.mode
    hop_frac = float(seg_cfg.hop_frac)
    if mode == "fixed":
        w = int(seg_cfg.fixed_len)
        hop = max(1, int(np.floor(hop_frac * w)))
        return [(i, w) for i in range(0, n - w + 1, hop)]
    if mode != "ihras":
        raise ValueError(f"unknown segmentation mode {mode}")
    wmin, wmax = window_bounds(seg_cfg, fs)
    n_rr = int(seg_cfg.n_rr_median)
    conf = int(round(seg_cfg.confirm_delay_s * fs))
    r = np.asarray(rpeaks, dtype=np.int64)
    default_rr = 60.0 / 72.0 * fs
    segs, i = [], 0
    while True:
        if causal:
            avail = r[r + conf <= i]
            sel = avail[-(n_rr + 1):]
        else:
            j = int(np.searchsorted(r, i))
            lo = max(0, j - (n_rr + 1) // 2)
            hi = min(len(r), lo + n_rr + 1)
            lo = max(0, hi - n_rr - 1)
            sel = r[lo:hi]
        rr_hat = float(np.median(np.diff(sel))) if len(sel) >= 2 else default_rr
        w = int(np.clip(np.floor(seg_cfg.n_cycles * rr_hat), wmin, wmax))
        if i + w > n:
            break
        segs.append((i, w))
        i += max(1, int(np.floor(hop_frac * w)))
    return segs


def extract_segments(sig: np.ndarray, bounds, ref_len: int):
    """sig: (T, L). Returns X (K, L, ref_len) float32 (zero-padded / centre-cropped), mask (K, ref_len)."""
    T, L = sig.shape
    K = len(bounds)
    X = np.zeros((K, L, ref_len), dtype=np.float32)
    M = np.zeros((K, ref_len), dtype=bool)
    for k, (s, w) in enumerate(bounds):
        seg = sig[s:s + w]
        if w > ref_len:
            off = (w - ref_len) // 2
            seg = seg[off:off + ref_len]
        n = len(seg)
        X[k, :, :n] = seg.T
        M[k, :n] = True
    return X, M
