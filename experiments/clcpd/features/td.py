"""Time-domain features F_td (9) and per-lead QRS peak-to-peak voltage (QRS-V, 12).

Heart-rate-variability descriptors need several beats, so F_td is computed on a context of
the last ``td_context_s`` seconds (the full 10-s record for PTB-XL) rather than on a single
two-cycle segment. Fiducial points are estimated with a lightweight multi-lead delineator
(spatial-velocity thresholds); all intervals are in seconds and amplitudes in mV.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

TD_NAMES = ["rr_mean_s", "sdnn_s", "rmssd_s", "pnn50_pct", "qrs_dur_s", "qt_s",
            "st_dev_mV", "hr_bpm", "spectral_entropy"]
N_TD = len(TD_NAMES)


def _delineate(sig: np.ndarray, rpeaks: np.ndarray, fs: float):
    T, L = sig.shape
    vel = np.sum(np.abs(np.diff(sig, axis=0, prepend=sig[:1])), axis=1)
    rr_med = float(np.median(np.diff(rpeaks))) if len(rpeaks) > 1 else 0.8 * fs
    beats = []
    for r in rpeaks:
        r = int(r)
        if r - int(0.22 * fs) < 0 or r + int(0.5 * fs) >= T:
            continue
        a, b = r - int(0.06 * fs), r + int(0.06 * fs)
        thr = 0.15 * vel[a:b + 1].max()
        on = r
        while on > r - int(0.12 * fs) and vel[on] > thr:
            on -= 1
        off = r
        while off < r + int(0.14 * fs) and vel[off] > thr:
            off += 1
        b0, b1 = on - int(0.08 * fs), on - int(0.02 * fs)
        if b0 < 0 or b1 <= b0:
            continue
        base = np.median(sig[b0:b1], axis=0)
        st = sig[min(T - 1, off + int(0.06 * fs))] - base
        t_lo = off + int(0.10 * fs)
        t_hi = min(T - 1, r + int(min(0.65 * rr_med, 0.6 * fs)))
        if t_hi - t_lo < 3:
            continue
        dev = np.abs(sig[t_lo:t_hi] - base).sum(axis=1)
        tp = t_lo + int(np.argmax(dev))
        tpk = dev[tp - t_lo]
        te = tp
        while te < t_hi and np.abs(sig[te] - base).sum() > 0.15 * tpk:
            te += 1
        qrs = sig[on:off + 1]
        beats.append((on, off, te, st, qrs.max(axis=0) - qrs.min(axis=0)))
    return beats


def td_features(sig: np.ndarray, rpeaks: np.ndarray, fs: float):
    """sig: (T, L) band-passed ECG in mV. Returns (F_td[9], QRS-V[L]); NaN where undefined."""
    T, L = sig.shape
    ftd = np.full(N_TD, np.nan, dtype=np.float64)
    qrsv = np.full(L, np.nan, dtype=np.float64)
    rp = np.asarray(rpeaks, dtype=np.int64)
    if len(rp) >= 2:
        rr = np.diff(rp) / fs
        ftd[0] = rr.mean()
        ftd[1] = rr.std()
        ftd[7] = 60.0 / np.median(rr)
        if len(rr) >= 2:
            d = np.diff(rr)
            ftd[2] = np.sqrt(np.mean(d ** 2))
            ftd[3] = 100.0 * np.mean(np.abs(d) > 0.05)
    beats = _delineate(sig, rp, fs)
    if beats:
        on = np.array([b[0] for b in beats])
        off = np.array([b[1] for b in beats])
        te = np.array([b[2] for b in beats])
        st = np.stack([b[3] for b in beats])           # (nb, L)
        qv = np.stack([b[4] for b in beats])           # (nb, L)
        ftd[4] = np.median(off - on) / fs
        ftd[5] = np.median(te - on) / fs
        st_med = np.median(st, axis=0)
        ftd[6] = st_med[int(np.argmax(np.abs(st_med)))]
        qrsv = np.median(qv, axis=0)
    lead = 1 if L > 1 else 0
    nper = int(min(256, T))
    if nper >= 16:
        _, pxx = welch(sig[:, lead], fs=fs, nperseg=nper)
        p = pxx / (pxx.sum() + 1e-12)
        ftd[8] = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    return ftd.astype(np.float32), qrsv.astype(np.float32)
