"""Training datasets: CCPP triplets and HTFP denoising pairs.

CCPP triplet for an anchor segment S_k of record a (length w set by IHRAS):
  positive  S_k^+ : a neighbouring segment (k +- 1 or 2) of the same record;
  negative  S_k^- : a *transition* segment of length w whose first part comes from record a and
                    whose second part comes from a record b with a different label set, joined at
                    a mid-RR point inside [0.2w, 0.8w] (Eq. 10);
  nuisance  S_k^u : the same construction with b drawn from the same label set but a different
                    patient; it is used as a second positive so that a change of patient alone is
                    not rewarded as a change point.
Splices are harmonised (TP-segment cut, DC alignment, raised-cosine cross-fade) unless
``ccpp.harmonise_splice`` is false (shortcut ablation).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from ..signal.noise import add_noise, sample_noise, synthetic_noise, NOISE_TYPES
from ..signal.qrs import mid_rr_points
from ..signal.segmentation import segment_bounds


class _SegmentBase(Dataset):
    def __init__(self, signals, rpeaks, rows, cfg, norm, epoch_size, seed):
        self.sig = signals
        self.cfg = cfg
        self.fs = float(cfg.data.fs)
        self.mean = np.asarray(norm[0], dtype=np.float32)
        self.std = np.asarray(norm[1], dtype=np.float32)
        self.ref_len = int(cfg.segmentation.ref_len)
        self.segs, self.mids = {}, {}
        T = signals.shape[1]
        for r in rows:
            b = segment_bounds(T, self.fs, rpeaks[r], cfg.segmentation, causal=False)
            if b:
                self.segs[int(r)] = b
                self.mids[int(r)] = mid_rr_points(np.asarray(rpeaks[r]))
        self.anchors = [(r, k) for r, b in self.segs.items() for k in range(len(b))]
        self.epoch_size = int(epoch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.epoch_size

    def _rng(self, i):
        return np.random.default_rng([self.seed, self.epoch, int(i)])

    def _pad(self, x):
        """x: (w, L) in mV -> standardised (L, ref_len) float32 and bool mask."""
        x = (x - self.mean) / self.std
        w = min(len(x), self.ref_len)
        out = np.zeros((x.shape[1], self.ref_len), dtype=np.float32)
        out[:, :w] = x[:w].T
        m = np.zeros(self.ref_len, dtype=bool)
        m[:w] = True
        return out, m


class CCPPDataset(_SegmentBase):
    def __init__(self, signals, rpeaks, meta, rows, cfg, norm, epoch_size, seed, augment=True):
        super().__init__(signals, rpeaks, rows, cfg, norm, epoch_size, seed)
        self.keys = meta["label_key"].values
        self.patients = meta["patient_id"].values
        self.Y = meta[[c for c in meta.columns if c.startswith("y_")]].values.astype(np.float32)
        self.by_key = defaultdict(list)
        for r in self.segs:
            self.by_key[self.keys[r]].append(r)
        self.key_list = list(self.by_key)
        w = np.array([len(self.by_key[k]) for k in self.key_list], dtype=float)
        self.key_p = w / w.sum()
        self.augment = augment
        self.harmonise = bool(cfg.ccpp.harmonise_splice)
        self.xfade = max(1, int(round(cfg.streams.crossfade_ms * self.fs / 1000.0))) if self.harmonise else 0
        self.nuisance = bool(cfg.ccpp.nuisance_positive)
        self.pure_negative = bool(cfg.ccpp.get("pure_negative", False))

    def _other_key_record(self, rng, key):
        for _ in range(50):
            k = self.key_list[rng.choice(len(self.key_list), p=self.key_p)]
            if k != key:
                lst = self.by_key[k]
                return lst[rng.integers(len(lst))]
        return None

    def _same_key_other_patient(self, rng, key, patient):
        lst = self.by_key[key]
        for _ in range(20):
            r = lst[rng.integers(len(lst))]
            if self.patients[r] != patient:
                return r
        return None

    def _splice(self, rng, ra, sa, w, rb):
        c_off = int(rng.uniform(0.3, 0.7) * w)
        c = sa + c_off
        mids = self.mids[ra]
        if self.harmonise and len(mids):
            inwin = mids[(mids >= sa + int(0.2 * w)) & (mids <= sa + int(0.8 * w))]
            if len(inwin):
                c = int(inwin[np.argmin(np.abs(inwin - c))])
        left = self.sig[ra, sa:c]
        need = w - (c - sa)
        n = self.xfade if len(left) > self.xfade else 0
        TB = self.sig.shape[1]
        mb = self.mids.get(rb, np.zeros(0, dtype=np.int64))
        cand = mb[mb + need + n <= TB] if self.harmonise else np.zeros(0)
        jb = int(cand[rng.integers(len(cand))]) if len(cand) else int(rng.integers(0, TB - need - n + 1))
        right = self.sig[rb, jb:jb + need + n].astype(np.float32).copy()
        if self.harmonise:
            k = 5
            right += np.median(left[-k:], axis=0) - np.median(right[:k], axis=0)
        if n:
            r = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n)))[:, None].astype(np.float32)
            blend = left[-n:] * (1.0 - r) + right[:n] * r
            return np.concatenate([left[:-n], blend, right[n:]], axis=0)
        return np.concatenate([left, right[:need]], axis=0)

    def _aug(self, rng, x):
        if not self.augment:
            return x
        x = x * rng.uniform(0.9, 1.1, size=(x.shape[0], 1)).astype(np.float32)
        return x + rng.normal(0.0, 0.02, size=x.shape).astype(np.float32)

    def __getitem__(self, i):
        rng = self._rng(i)
        ra, k = self.anchors[rng.integers(len(self.anchors))]
        sa, w = self.segs[ra][k]
        xa = self.sig[ra, sa:sa + w]
        nb = len(self.segs[ra])
        cands = [j for j in (k - 2, k - 1, k + 1, k + 2) if 0 <= j < nb]
        if cands:
            sp, wp = self.segs[ra][cands[rng.integers(len(cands))]]
            xp = self.sig[ra, sp:sp + wp]
        else:
            xp = xa
        rb = self._other_key_record(rng, self.keys[ra])
        xn = self._splice(rng, ra, sa, w, rb) if rb is not None else xa
        ru = self._same_key_other_patient(rng, self.keys[ra], self.patients[ra]) if self.nuisance else None
        xu = self._splice(rng, ra, sa, w, ru) if ru is not None else xp
        views = [("a", xa), ("p", xp), ("n", xn), ("u", xu)]
        if self.pure_negative:
            # a complete segment of the post-change recording b (a state, not a transition)
            if rb is not None and rb in self.segs:
                sb, wb = self.segs[rb][rng.integers(len(self.segs[rb]))]
                views.append(("b", self.sig[rb, sb:sb + wb]))
            else:
                views.append(("b", xn))
        out = {}
        for name, x in views:
            v, m = self._pad(x)
            out[f"x{name}"] = self._aug(rng, v)
            out[f"m{name}"] = m
        out["y"] = self.Y[ra]
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}


class DenoiseDataset(_SegmentBase):
    """(noisy, clean) segment pairs for HTFP pretraining; noise from NSTDB (train half) or surrogate."""

    def __init__(self, signals, rpeaks, rows, cfg, norm, epoch_size, seed, noise_sources=None):
        super().__init__(signals, rpeaks, rows, cfg, norm, epoch_size, seed)
        self.noise = noise_sources or {}
        self.snr = cfg.htfp.pretrain_snr_db

    def __getitem__(self, i):
        rng = self._rng(i)
        r, k = self.anchors[rng.integers(len(self.anchors))]
        s, w = self.segs[r][k]
        clean = self.sig[r, s:s + w].astype(np.float32)
        kind = NOISE_TYPES[rng.integers(len(NOISE_TYPES))]
        if kind in self.noise:
            nz = sample_noise(self.noise[kind], w, clean.shape[1], rng)
        else:
            nz = synthetic_noise(w, clean.shape[1], self.fs, kind, rng)
        noisy = add_noise(clean, nz, float(rng.uniform(*self.snr)))
        xn, m = self._pad(noisy)
        xc, _ = self._pad(clean)
        return {"x": torch.from_numpy(xn), "y": torch.from_numpy(xc), "m": torch.from_numpy(m)}
