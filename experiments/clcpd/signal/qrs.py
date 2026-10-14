"""Multi-lead Pan-Tompkins-style R-peak detector.

Each lead is band-passed (5-15 Hz), differentiated, squared and integrated over 150 ms
(Pan & Tompkins, 1985). The integrated energies of all leads are normalised and summed, peaks
are accepted when they exceed a fraction of the local (4 s) maximum envelope, and a 250 ms
refractory period is enforced.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt


def _pt_energy(x: np.ndarray, fs: float):
    hi = min(15.0, 0.45 * fs)
    sos = butter(2, [5.0 / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    y = sosfiltfilt(sos, x)
    d = np.gradient(y) * fs
    w = max(1, int(round(0.15 * fs)))
    e = np.convolve(d * d, np.ones(w) / w, mode="same")
    return y, e


def detect_rpeaks(sig: np.ndarray, fs: float, leads=None, rel_thr: float = 0.3,
                  min_rr_s: float = 0.25, env_s: float = 4.0) -> np.ndarray:
    """sig: (T, L) array. Returns sorted int64 sample indices of detected QRS complexes."""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[:, None]
    T, L = sig.shape
    if T < int(fs):
        return np.zeros(0, dtype=np.int64)
    leads = list(range(L)) if leads is None else list(leads)
    energy = np.zeros(T)
    filtered = []
    for ld in leads:
        y, e = _pt_energy(sig[:, ld], fs)
        energy += e / (np.percentile(e, 99) + 1e-12)
        filtered.append(np.abs(y))
    if not np.any(energy > 0):
        return np.zeros(0, dtype=np.int64)
    env = maximum_filter1d(energy, size=max(3, int(env_s * fs)), mode="nearest")
    refractory = max(1, int(min_rr_s * fs))
    cand, _ = find_peaks(energy, distance=refractory)
    cand = cand[energy[cand] > rel_thr * env[cand]]
    ysum = np.sum(np.stack(filtered, axis=1), axis=1)
    half = max(1, int(0.08 * fs))
    refined = []
    for p in cand:
        a, b = max(0, p - half), min(T, p + half + 1)
        refined.append(a + int(np.argmax(ysum[a:b])))
    refined = np.unique(np.asarray(refined, dtype=np.int64))
    if len(refined) < 2:
        return refined
    keep = [refined[0]]
    for p in refined[1:]:
        if p - keep[-1] >= refractory:
            keep.append(p)
    return np.asarray(keep, dtype=np.int64)


def median_hr(rpeaks: np.ndarray, fs: float) -> float:
    """Median heart rate (bpm) from R-peak indices; NaN when fewer than two beats."""
    if len(rpeaks) < 2:
        return float("nan")
    return float(60.0 * fs / np.median(np.diff(rpeaks)))


def mid_rr_points(rpeaks: np.ndarray) -> np.ndarray:
    """Mid-points between consecutive R peaks (approximately in the isoelectric TP segment)."""
    if len(rpeaks) < 2:
        return np.zeros(0, dtype=np.int64)
    return ((rpeaks[:-1] + rpeaks[1:]) // 2).astype(np.int64)
