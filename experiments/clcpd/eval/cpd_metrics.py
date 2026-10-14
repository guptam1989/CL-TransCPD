"""Event-based change-point metrics for online detectors.

An alarm is raised when a segment is complete, so its time stamp is the segment end. A true
change point c is detected if an unmatched alarm falls in [c, c + tol] (one-to-one matching).
  precision = TP / (TP + FP), recall = TP / (TP + FN), F1;
  delay     = alarm time - c for detected change points (seconds);
  FA/h      = false alarms per hour of monitored signal;
  nuisance / artefact false-alarm rate = fraction of nuisance / control junctions followed by an
  unmatched alarm within tol;
  AUPRC     = area under the interpolated precision-recall curve obtained by sweeping the
              detection threshold (the online loop, including resets, is re-run per threshold).
"""
from __future__ import annotations

import numpy as np


def match(cps, alarms, tol):
    alarms = sorted(alarms)
    used = np.zeros(len(alarms), dtype=bool)
    per_cp = []
    for c in sorted(cps):
        hit = None
        for i, a in enumerate(alarms):
            if used[i] or a < c:
                continue
            if a <= c + tol:
                used[i] = True
                hit = a - c
            break
        per_cp.append(hit)
    fps = [a for i, a in enumerate(alarms) if not used[i]]
    return per_cp, fps


def stream_counts(stream, alarms, tol, fs):
    per_cp, fps = match(stream["cps"], alarms, tol)
    dur_h = len(stream["signal"]) / fs / 3600.0
    near = lambda t, js: any(j <= t <= j + tol for j in js)
    return {
        "tp": sum(d is not None for d in per_cp),
        "fn": sum(d is None for d in per_cp),
        "fp": len(fps),
        "delays": [d for d in per_cp if d is not None],
        "per_cp": per_cp,
        "hours": dur_h,
        "nuis_n": len(stream.get("nuisance", [])),
        "nuis_fa": sum(any(j <= a <= j + tol for a in fps) for j in stream.get("nuisance", [])),
        "art_n": len(stream.get("artifact", [])),
        "art_fa": sum(any(j <= a <= j + tol for a in fps) for j in stream.get("artifact", [])),
        "fp_outside": sum(not near(a, stream.get("nuisance", []) + stream.get("artifact", [])) for a in fps),
    }


def aggregate(counts):
    tp = sum(c["tp"] for c in counts)
    fn = sum(c["fn"] for c in counts)
    fp = sum(c["fp"] for c in counts)
    hours = sum(c["hours"] for c in counts)
    delays = [d for c in counts for d in c["delays"]]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    nn, na = sum(c["nuis_n"] for c in counts), sum(c["art_n"] for c in counts)
    return {
        "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fn": fn, "fp": fp,
        "delay_mean": float(np.mean(delays)) if delays else float("nan"),
        "delay_median": float(np.median(delays)) if delays else float("nan"),
        "fa_per_hour": fp / hours if hours else float("nan"),
        "nuisance_fa_rate": sum(c["nuis_fa"] for c in counts) / nn if nn else float("nan"),
        "artifact_fa_rate": sum(c["art_fa"] for c in counts) / na if na else float("nan"),
        "hours": hours,
    }


def interpolated_ap(points):
    """points: list of (recall, precision). Area under the monotone (interpolated) PR envelope."""
    pts = sorted(set((float(r), float(p)) for r, p in points))
    if not pts:
        return float("nan")
    rec = np.array([r for r, _ in pts])
    prec = np.array([p for _, p in pts])
    env = np.maximum.accumulate(prec[::-1])[::-1]
    r_prev, ap = 0.0, 0.0
    for r, p in zip(rec, env):
        ap += max(0.0, r - r_prev) * p
        r_prev = max(r_prev, r)
    return float(ap)
