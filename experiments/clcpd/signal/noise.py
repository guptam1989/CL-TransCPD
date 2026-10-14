"""Noise injection at a prescribed SNR, using MIT-BIH NSTDB excerpts or synthetic surrogates."""
from __future__ import annotations

import numpy as np

NOISE_TYPES = ("bw", "em", "ma")


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """x, noise: (T, L). Noise is scaled per lead so that P_signal / P_noise = 10^(snr/10)."""
    ps = np.mean(x.astype(np.float64) ** 2, axis=0)
    pn = np.mean(noise.astype(np.float64) ** 2, axis=0) + 1e-12
    k = np.sqrt(ps / (pn * 10.0 ** (snr_db / 10.0)))
    return (x + noise * k[None, :]).astype(np.float32)


def sample_noise(source: np.ndarray, T: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an independent excerpt (random channel and offset) of an NSTDB noise record per lead."""
    n_src, n_ch = source.shape
    out = np.zeros((T, L), dtype=np.float32)
    for ld in range(L):
        ch = rng.integers(n_ch)
        off = rng.integers(0, max(1, n_src - T))
        seg = source[off:off + T, ch]
        if len(seg) < T:
            seg = np.resize(seg, T)
        out[:, ld] = seg - seg.mean()
    return out


def synthetic_noise(T: int, L: int, fs: float, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Surrogate noise used only when the NSTDB records are not available."""
    t = np.arange(T) / fs
    out = np.zeros((T, L), dtype=np.float32)
    for ld in range(L):
        if kind == "bw":
            n = sum(rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * t + rng.uniform(0, 2 * np.pi))
                    for _ in range(3))
        elif kind == "ma":
            w = rng.standard_normal(T + 64)
            spec = np.fft.rfft(w)
            f = np.fft.rfftfreq(T + 64, 1.0 / fs)
            spec[(f < 15) | (f > 0.45 * fs)] = 0
            n = np.fft.irfft(spec, n=T + 64)[:T]
        elif kind == "em":
            n = np.zeros(T)
            for _ in range(max(1, int(T / fs / 3))):
                c = rng.integers(T)
                width = rng.uniform(0.1, 0.6) * fs
                n += rng.normal(0, 1) * np.exp(-0.5 * ((np.arange(T) - c) / width) ** 2)
            n += 0.2 * np.cumsum(rng.standard_normal(T)) / np.sqrt(T)
        else:
            raise ValueError(kind)
        out[:, ld] = n - np.mean(n)
    return out
