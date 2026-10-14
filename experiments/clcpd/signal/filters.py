"""Band-pass filtering and rational resampling."""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt


def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0, order: int = 2,
             causal: bool = False, axis: int = 0) -> np.ndarray:
    hi = min(hi, 0.45 * fs)
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    if causal:
        return sosfilt(sos, x, axis=axis).astype(np.float32)
    return sosfiltfilt(sos, x, axis=axis).astype(np.float32)


def resample(x: np.ndarray, fs_in: int, fs_out: int, axis: int = 0) -> np.ndarray:
    fs_in, fs_out = int(round(fs_in)), int(round(fs_out))
    if fs_in == fs_out:
        return x.astype(np.float32)
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g, axis=axis).astype(np.float32)
