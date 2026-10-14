"""PTB-XL-CPD benchmark: simulated monitoring streams with known change points.

A stream concatenates 3-6 complete PTB-XL records of one fold. At each junction the next
record either carries a different superclass label set (a *pathological change point*, the
detection target) or the same label set from a different patient (a *nuisance junction*: the
morphology changes because the patient changes, but the diagnosis does not). Every record is
used at most once per stream set. Junctions are placed at mid-RR points (TP segment), the DC
level of the next record is aligned to the previous one and a 40-ms raised-cosine cross-fade is
applied, so that splice discontinuities cannot be used as a shortcut.

Two further sets quantify confounds:
  * control streams: a record concatenated with itself; the junction contains only the splice
    (no patient or diagnosis change) -> any alarm there is an artefact false alarm;
  * same-patient streams: all records of a patient in chronological order (real longitudinal
    changes of one person, no patient change at the junction).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..signal.qrs import median_hr, mid_rr_points


def harmonised_concat(parts, fs: float, crossfade_ms: float = 40.0, harmonise: bool = True,
                      offset_decay_s: float | None = None):
    """Concatenate (T_i, L) arrays. Returns (signal, junction sample indices).

    The DC level of each appended part is aligned to the preceding one at the junction. With
    offset_decay_s = None the alignment offset is kept for the whole part (a sustained baseline step);
    with offset_decay_s = tau it decays as exp(-t / tau), as a monitor's high-pass filter would restore
    the baseline (tau = 1 / (2 pi 0.5 Hz) = 0.32 s), so no persistent level shift marks the junction."""
    n = max(1, int(round(crossfade_ms * fs / 1000.0))) if harmonise else 0
    k = max(1, int(round(0.05 * fs)))
    out = np.asarray(parts[0], dtype=np.float32).copy()
    junctions = []
    for p in parts[1:]:
        p = np.asarray(p, dtype=np.float32).copy()
        if harmonise:
            off = np.median(out[-k:], axis=0) - np.median(p[:k], axis=0)
            if offset_decay_s:
                p += off[None, :] * np.exp(-np.arange(len(p), dtype=np.float32) / (offset_decay_s * fs))[:, None]
            else:
                p += off
        if n > 0 and len(out) > n and len(p) > n:
            r = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n)))[:, None].astype(np.float32)
            blend = out[-n:] * (1.0 - r) + p[:n] * r
            junctions.append(len(out) - n // 2)
            out = np.concatenate([out[:-n], blend, p[n:]], axis=0)
        else:
            junctions.append(len(out))
            out = np.concatenate([out, p], axis=0)
    return out, junctions


def _trim(sig, rpeaks, harmonise):
    if not harmonise:
        return sig
    mids = mid_rr_points(np.asarray(rpeaks))
    if len(mids) >= 2:
        return sig[mids[0]:mids[-1]]
    return sig


def assemble(rows, signals, rpeaks, keys, fs, harmonise=True, crossfade_ms=40.0, kind="mixed", offset_decay_s=None):
    parts = [_trim(signals[r], rpeaks[r], harmonise) for r in rows]
    sig, junctions = harmonised_concat(parts, fs, crossfade_ms, harmonise, offset_decay_s)
    cps, nuis, post_key, post_hr, pre_key = [], [], [], [], []
    for i, j in enumerate(junctions, start=1):
        if keys[rows[i]] != keys[rows[i - 1]]:
            cps.append(j / fs)
            pre_key.append(keys[rows[i - 1]])
            post_key.append(keys[rows[i]])
            post_hr.append(median_hr(np.asarray(rpeaks[rows[i]]), fs))
        else:
            nuis.append(j / fs)
    return {"signal": sig, "cps": cps, "nuisance": nuis, "artifact": [], "rows": list(map(int, rows)),
            "pre_key": pre_key, "post_key": post_key, "post_hr": post_hr,
            "boundaries": [0] + list(junctions) + [len(sig)], "kind": kind}


def _weighted_pop(pool, rng, cond=None):
    keys = [k for k, v in pool.items() if v and (cond is None or cond(k))]
    if not keys:
        return None
    w = np.array([len(pool[k]) for k in keys], dtype=float)
    k = keys[rng.choice(len(keys), p=w / w.sum())]
    return pool[k].pop()


def _pop_same(pool, key, patient, patients):
    lst = pool.get(key, [])
    for i in range(len(lst) - 1, -1, -1):
        if patients[lst[i]] != patient:
            return lst.pop(i)
    return None


def build_mixed_streams(meta, signals, rpeaks, rows, cfg_streams, fs, seed, harmonise=True, constant_offset=False):
    """constant_offset=True keeps the alignment offset for the whole appended record (no decay): the same records
    and junctions, but with the sustained baseline step of the preliminary benchmark (confound control)."""
    rng = np.random.default_rng(seed)
    keys = meta["label_key"].values
    patients = meta["patient_id"].values
    pool = defaultdict(list)
    for r in rng.permutation(np.asarray(rows)):
        pool[keys[r]].append(int(r))
    lo, hi = cfg_streams.records_per_stream
    streams = []
    while True:
        first = _weighted_pop(pool, rng)
        if first is None:
            break
        seq = [first]
        for _ in range(int(rng.integers(lo, hi + 1)) - 1):
            cur = seq[-1]
            if rng.random() < cfg_streams.p_change:
                nxt = _weighted_pop(pool, rng, cond=lambda k, ck=keys[cur]: k != ck)
            else:
                nxt = _pop_same(pool, keys[cur], patients[cur], patients)
            if nxt is None:
                nxt = _weighted_pop(pool, rng)
            if nxt is None:
                break
            seq.append(nxt)
        if len(seq) >= 2:
            decay = None if constant_offset or not harmonise else cfg_streams.get("offset_decay_s")
            streams.append(assemble(seq, signals, rpeaks, keys, fs, harmonise, cfg_streams.crossfade_ms,
                                    offset_decay_s=decay))
    return streams


def junction_steps(streams, std, fs, window_s=2.0, gap_s=0.1):
    """Baseline level step at every junction (change points and nuisance junctions): per lead, the absolute
    difference between the signal medians in the window_s seconds after and before the junction (leaving gap_s on
    either side, which covers the cross-fade), in units of the per-lead training-set SD; median over leads."""
    w, g = int(round(window_s * fs)), int(round(gap_s * fs))
    std = np.asarray(std, dtype=np.float64)
    out = []
    for st in streams:
        x = st["signal"]
        for j in st["boundaries"][1:-1]:
            a, b = x[max(0, j - g - w):max(0, j - g)], x[j + g:j + g + w]
            if len(a) < w or len(b) < w:
                continue
            out.append(float(np.median(np.abs(np.median(b, axis=0) - np.median(a, axis=0)) / std)))
    return np.asarray(out)


def build_control_streams(meta, signals, rpeaks, rows, n, fs, seed, harmonise=True, crossfade_ms=40.0,
                          offset_decay_s=None):
    rng = np.random.default_rng(seed + 7919)
    rows = np.asarray(rows)
    pick = rng.choice(rows, size=min(n, len(rows)), replace=False)
    keys = meta["label_key"].values
    out = []
    for r in pick:
        s = assemble([int(r), int(r)], signals, rpeaks, keys, fs, harmonise, crossfade_ms, kind="control",
                     offset_decay_s=offset_decay_s)
        s["artifact"], s["nuisance"] = s["nuisance"], []
        out.append(s)
    return out


def build_same_patient_streams(meta, signals, rpeaks, rows, fs, harmonise=True, crossfade_ms=40.0,
                               offset_decay_s=None):
    sub = meta.iloc[np.asarray(rows)].copy()
    sub["_row"] = np.asarray(rows)
    keys = meta["label_key"].values
    out = []
    date_col = "recording_date" if "recording_date" in sub.columns else None
    for _, g in sub.groupby("patient_id"):
        if len(g) < 2 or g["label_key"].nunique() < 2:
            continue
        g = g.sort_values(date_col) if date_col else g
        s = assemble(g["_row"].tolist(), signals, rpeaks, keys, fs, harmonise, crossfade_ms, kind="same_patient",
                     offset_decay_s=offset_decay_s)
        if s["cps"]:
            out.append(s)
    return out
