"""Uncertainty and significance: cluster (patient / stream) bootstrap and Holm correction."""
from __future__ import annotations

import numpy as np


def _cluster_weights(groups, rng):
    uniq, inv = np.unique(groups, return_inverse=True)
    draw = rng.integers(0, len(uniq), size=len(uniq))
    counts = np.bincount(draw, minlength=len(uniq))
    return counts[inv]


def bootstrap_ci(metric_fn, arrays, groups, B=1000, seed=0, alpha=0.05):
    """metric_fn(*arrays_subset) -> float. Resamples clusters (e.g. patients) with replacement."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    vals = []
    for _ in range(B):
        w = _cluster_weights(groups, rng)
        idx = np.repeat(np.arange(n), w)
        try:
            vals.append(metric_fn(*[a[idx] for a in arrays]))
        except ValueError:
            continue
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap(metric_fn, y, pa, pb, groups, B=1000, seed=0):
    """Two-sided paired test of metric(y, pa) - metric(y, pb) by cluster bootstrap.

    Returns (observed difference, 95% CI of the difference, p-value)."""
    rng = np.random.default_rng(seed)
    obs = metric_fn(y, pa) - metric_fn(y, pb)
    n = len(groups)
    diffs = []
    for _ in range(B):
        idx = np.repeat(np.arange(n), _cluster_weights(groups, rng))
        try:
            diffs.append(metric_fn(y[idx], pa[idx]) - metric_fn(y[idx], pb[idx]))
        except ValueError:
            continue
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return float(obs), (float("nan"), float("nan")), float("nan")
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(obs), (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))), float(min(1.0, p))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
