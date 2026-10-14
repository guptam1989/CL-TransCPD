"""Classification metrics.

Definitions (stated identically in the manuscript):
  * multi-label (PTB-XL benchmark): macro-AUC, macro-AUPRC and macro-F1 over the 5 superclasses;
    F1 thresholds are tuned per class on the validation fold.
  * single-label subset: ACC = overall accuracy; SEN/SPE = macro-averaged one-vs-rest recall and
    specificity; G-mean = sqrt(SEN * SPE); F1 = macro-F1; AUC/AUPRC = macro one-vs-rest;
    MCC and kappa = multi-class (Gorodkin 2004; Cohen 1960); ECE = top-label, 15 equal-width bins;
    DOR = geometric mean of the per-class one-vs-rest DORs with a 0.5 continuity correction.
  * per-class table: one-vs-rest AUPRC, class-wise ECE, DOR, binary MCC and binary kappa;
    the "macro" row of that table is the arithmetic mean of the per-class values.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)


def _safe_auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def _safe_ap(y, p):
    return average_precision_score(y, p) if y.sum() > 0 else np.nan


def tune_thresholds(Y, P, grid=np.linspace(0.05, 0.95, 19)):
    th = []
    for c in range(Y.shape[1]):
        f = [f1_score(Y[:, c], P[:, c] >= t, zero_division=0) for t in grid]
        th.append(float(grid[int(np.argmax(f))]))
    return np.asarray(th)


def multilabel_metrics(Y, P, thresholds=None):
    Y = np.asarray(Y).astype(int)
    P = np.asarray(P, dtype=float)
    C = Y.shape[1]
    th = np.full(C, 0.5) if thresholds is None else np.asarray(thresholds)
    auc = np.array([_safe_auc(Y[:, c], P[:, c]) for c in range(C)])
    ap = np.array([_safe_ap(Y[:, c], P[:, c]) for c in range(C)])
    f1 = np.array([f1_score(Y[:, c], P[:, c] >= th[c], zero_division=0) for c in range(C)])
    return {"macro_auc": float(np.nanmean(auc)), "macro_auprc": float(np.nanmean(ap)),
            "macro_f1": float(np.mean(f1)), "auc_per_class": auc.tolist(), "auprc_per_class": ap.tolist(),
            "f1_per_class": f1.tolist()}


def ece_top_label(y, P, n_bins=15):
    conf = P.max(1)
    pred = P.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def ece_classwise(yb, p, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(yb[m].mean() - p[m].mean())
    return float(ece)


def _dor(tp, tn, fp, fn):
    return (tp + 0.5) * (tn + 0.5) / ((fp + 0.5) * (fn + 0.5))


def multiclass_metrics(y, P, n_classes=5, n_bins=15):
    y = np.asarray(y).astype(int)
    P = np.asarray(P, dtype=float)
    pred = P.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    sen = tp / np.maximum(tp + fn, 1)
    spe = tn / np.maximum(tn + fp, 1)
    onehot = np.eye(n_classes)[y]
    auc = np.array([_safe_auc(onehot[:, c], P[:, c]) for c in range(n_classes)])
    ap = np.array([_safe_ap(onehot[:, c], P[:, c]) for c in range(n_classes)])
    dor = np.array([_dor(tp[c], tn[c], fp[c], fn[c]) for c in range(n_classes)])
    per_class = {
        "auprc": ap.tolist(),
        "auc": auc.tolist(),
        "ece": [ece_classwise(onehot[:, c], P[:, c], n_bins) for c in range(n_classes)],
        "dor": dor.tolist(),
        "mcc": [float(matthews_corrcoef(onehot[:, c], pred == c)) for c in range(n_classes)],
        "kappa": [float(cohen_kappa_score(onehot[:, c], (pred == c).astype(int))) for c in range(n_classes)],
        "sen": sen.tolist(),
        "spe": spe.tolist(),
        "support": cm.sum(1).tolist(),
    }
    return {
        "acc": float(tp.sum() / cm.sum()),
        "sen": float(sen.mean()),
        "spe": float(spe.mean()),
        "gmean": float(np.sqrt(sen.mean() * spe.mean())),
        "f1": float(f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0)),
        "auc": float(np.nanmean(auc)),
        "auprc": float(np.nanmean(ap)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred)),
        "ece": ece_top_label(y, P, n_bins),
        "dor_gmean": float(np.exp(np.mean(np.log(dor)))),
        "per_class": per_class,
        "confusion": cm.tolist(),
    }
