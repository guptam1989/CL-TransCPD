"""Online detection loop shared by CL-TransCPD and the baselines.

Every detector consumes one segment at a time (``step``) and returns a change score; when the
score exceeds the threshold gamma an alarm is issued at the segment end time and the detector
is reset (it re-learns the new state from the following warm-up segments). gamma is the
Lambda-th percentile of the scores of change-free validation segments, with Lambda chosen
from a grid to maximise validation F1; nothing is tuned on the test streams.
"""
from __future__ import annotations

import numpy as np

from .eval.cpd_metrics import aggregate, interpolated_ap, stream_counts


class CLTScoreDetector:
    """S_k = (1 - mu) * z(R_k) + mu * z(||z_k - zbar||) with zbar an EMA of past latents (Eq. 13)."""

    def __init__(self, mu, alpha, warm, stats):
        self.mu, self.alpha, self.warm = float(mu), float(alpha), int(warm)
        self.mR, self.sR, self.mD, self.sD = [float(v) for v in stats]
        self.reset()

    def reset(self):
        self.buf, self.zbar = [], None

    def step(self, x):
        z, R = x
        if self.zbar is None:
            self.buf.append(z)
            if len(self.buf) >= self.warm:
                self.zbar = np.mean(self.buf, 0)
                self.buf = []
            return np.nan
        D = float(np.linalg.norm(z - self.zbar))
        s = (1.0 - self.mu) * (R - self.mR) / self.sR + self.mu * (D - self.mD) / self.sD
        self.zbar = (1.0 - self.alpha) * self.zbar + self.alpha * z
        return s


def run(det, X, ends, gamma):
    det.reset()
    alarms, scores = [], np.full(len(ends), np.nan)
    for k in range(len(ends)):
        s = det.step(X[k])
        scores[k] = s
        if np.isfinite(s) and s > gamma:
            alarms.append(float(ends[k]))
            det.reset()
    return alarms, scores


def stable_scores(det, items, tol):
    """Scores of segments that neither contain nor follow (within tol) any junction, gamma = inf."""
    out = []
    for st, X, starts, ends in items:
        _, s = run(det, X, ends, np.inf)
        js = list(st["cps"]) + list(st.get("nuisance", [])) + list(st.get("artifact", []))
        for k in range(len(ends)):
            if np.isfinite(s[k]) and not any(starts[k] <= j <= ends[k] + tol for j in js):
                out.append(s[k])
    return np.asarray(out)


def evaluate(det, items, gamma, fs, tol):
    counts = []
    for st, X, starts, ends in items:
        alarms, _ = run(det, X, ends, gamma)
        c = stream_counts(st, alarms, tol, fs)
        c["alarms"] = alarms
        counts.append(c)
    return aggregate(counts), counts


def calibrate(det, val_items, fs, tol, grid):
    """Pick Lambda (percentile of stable validation scores) maximising validation F1."""
    ref = stable_scores(det, val_items, tol)
    if len(ref) == 0:
        return {"gamma": np.inf, "lambda": None, "val": None}
    best = None
    for lam in grid:
        g = float(np.percentile(ref, lam))
        res, _ = evaluate(det, val_items, g, fs, tol)
        if best is None or res["f1"] > best["val"]["f1"]:
            best = {"gamma": g, "lambda": lam, "val": res}
    best["ref_scores_quantiles"] = np.percentile(ref, [50, 90, 99]).tolist()
    return best


def pr_curve(det, items, fs, tol, n_thr=60, q_extra=()):
    """Sweep gamma over quantiles of all (gamma = inf) scores; returns points and interpolated AP.
    q_extra: additional (e.g. extreme) quantiles, used to reach the low false-alarm regime of long recordings."""
    allscores = []
    for st, X, starts, ends in items:
        _, s = run(det, X, ends, np.inf)
        allscores.append(s[np.isfinite(s)])
    allscores = np.concatenate(allscores) if allscores else np.zeros(0)
    if len(allscores) == 0:
        return [], float("nan")
    qs = np.unique(np.quantile(allscores, np.concatenate([np.linspace(0.0, 0.99, n_thr - 6),
                                                          [0.993, 0.995, 0.997, 0.998, 0.999, 0.9995],
                                                          np.asarray(q_extra, dtype=float)])))
    pts = []
    for g in qs:
        res, _ = evaluate(det, items, float(g), fs, tol)
        pts.append((res["recall"], res["precision"], float(g), res["fa_per_hour"]))
    return pts, interpolated_ap([(p[0], p[1]) for p in pts])


def recall_at_fah(pts, max_fah):
    """Highest recall among the swept thresholds whose false-alarm rate is <= max_fah (FROC-type point)."""
    ok = [p[0] for p in pts if len(p) > 3 and np.isfinite(p[3]) and p[3] <= max_fah]
    return float(max(ok)) if ok else 0.0


def bootstrap_streams(counts, B=1000, seed=0):
    """95 % CIs for F1, recall, precision, mean delay and FA/h by resampling streams."""
    rng = np.random.default_rng(seed)
    n = len(counts)
    keys = ["f1", "recall", "precision", "delay_mean", "fa_per_hour"]
    vals = {k: [] for k in keys}
    for _ in range(B):
        res = aggregate([counts[i] for i in rng.integers(0, n, size=n)])
        for k in keys:
            vals[k].append(res[k])
    return {k: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))] for k, v in vals.items()}


def hr_stratified(items, counts, bins):
    """Recall and mean delay per post-change heart-rate bin (real heart rates of the records)."""
    rows = []
    for (st, *_), c in zip(items, counts):
        for hr, d in zip(st.get("post_hr", []), c["per_cp"]):
            rows.append((hr, d))
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [d for hr, d in rows if np.isfinite(hr) and lo <= hr < hi]
        det = [d for d in sel if d is not None]
        out.append({"lo": lo, "hi": hi, "n": len(sel),
                    "recall": len(det) / len(sel) if sel else float("nan"),
                    "delay_mean": float(np.mean(det)) if det else float("nan"),
                    "delay_sd": float(np.std(det)) if det else float("nan")})
    return out
