"""End-to-end deep classification baselines on PTB-XL (and fine-tuned transfer to PTB Diagnostic).

  python scripts/train_cls_baselines.py --model resnet1d_wang --task ml --seed 0
  python scripts/train_cls_baselines.py --task-index 4      # PBS array over tasks_cls_baselines()
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

from _common import config_from, parser, pick_task
from clcpd.data.ptbxl import SUPERCLASSES, label_matrix
from clcpd.eval.cls_metrics import multiclass_metrics, multilabel_metrics, tune_thresholds
from clcpd.models.cls_baselines import build_classifier, fit_classifier, predict_proba, pretrain_clocs
from clcpd.pipeline import PTBXL, load_pickle
from clcpd.utils import get_device, log, pbs_array_index, save_json, set_seed


def tasks_cls_baselines(cfg):
    return [(m, t, s) for m in cfg.cls_baselines.models for t in ("ml", "sl") for s in cfg.eval.seeds]


def to_input(P, rows):
    return np.ascontiguousarray(P.standardise(np.asarray(P.signals[rows])).transpose(0, 2, 1)).astype(np.float32)


def main():
    ap = parser(__doc__)
    ap.add_argument("--model", default="resnet1d_wang")
    ap.add_argument("--task", default="ml", choices=["ml", "sl"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-index", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from(args)
    idx = args.task_index if args.task_index is not None else pbs_array_index(None)
    name, task, seed = (args.model, args.task, args.seed) if idx is None else pick_task(tasks_cls_baselines(cfg), idx)
    out = Path(cfg.paths.work_dir) / "results" / "cls" / name / task
    if (out / f"seed{seed}.json").exists():
        log(f"{name}/{task}/seed{seed} done - skipping")
        return
    set_seed(seed)
    device = get_device()
    P = PTBXL(cfg)
    meta = P.meta
    lab = meta["n_labels"].values > 0 if task == "ml" else meta["n_labels"].values == 1
    fold = meta["fold"].values
    sel = lambda folds: np.where(np.isin(fold, folds) & lab)[0]
    tr, va, te = sel(cfg.data.train_folds), sel(cfg.data.val_folds), sel(cfg.data.test_folds)
    Y = label_matrix(meta).astype(np.float32) if task == "ml" else meta["label_single"].values.astype(np.int64)
    Xtr, Xva, Xte = to_input(P, tr), to_input(P, va), to_input(P, te)
    model = build_classifier(name, int(cfg.data.n_leads), len(SUPERCLASSES), Xtr.shape[-1])
    c = cfg.cls_baselines
    if name == "clocs":
        pretrain_clocs(model, Xtr, device, epochs=int(c.clocs_pretrain_epochs), lr=float(c.lr), seed=seed, log=log)
    model, best = fit_classifier(model, Xtr, Y[tr], Xva, Y[va], task, device, epochs=int(c.epochs),
                                 bs=int(c.batch_size), lr=float(c.lr), wd=float(c.weight_decay), seed=seed, log=log)
    pva, pte = predict_proba(model, Xva, task, device), predict_proba(model, Xte, task, device)
    if task == "ml":
        th = tune_thresholds(Y[va], pva)
        m = multilabel_metrics(Y[te], pte, th)
        m["thresholds"] = th.tolist()
    else:
        m = multiclass_metrics(Y[te], pte, len(SUPERCLASSES), int(cfg.eval.ece_bins))
    m["best_val"] = best
    m["params"] = int(sum(p.numel() for p in model.parameters()))
    out.mkdir(parents=True, exist_ok=True)
    save_json(m, out / f"seed{seed}.json")
    np.savez(out / f"seed{seed}_pred.npz", y=Y[te], p=pte, groups=meta["patient_id"].values[te], rows=te)
    ck = Path(cfg.paths.work_dir) / "runs" / "cls_baselines" / name / task / f"seed{seed}"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck / "model.pt")
    log(f"{name} [{task}] seed {seed}: " + (f"macro-AUC={m['macro_auc']:.4f}" if task == "ml" else
                                           f"ACC={m['acc']:.4f} AUC={m['auc']:.4f}"))
    if task == "sl":
        transfer_ptbdiag(cfg, P, model, name, seed, device)


def transfer_ptbdiag(cfg, P, base_model, name, seed, device):
    d = Path(cfg.paths.work_dir) / "cache" / "ptbdiag"
    if not (d / "crops.npy").exists():
        return
    crops = np.load(d / "crops.npy")
    aux = np.load(d / "aux.npz")
    info = load_pickle(d / "records.pkl")
    rec_idx = aux["rec_idx"]
    labels = np.array([info[i]["label"] for i in rec_idx], dtype=object)
    keep = np.array([lab is not None for lab in labels])
    classes = [c for c in SUPERCLASSES if c in set(labels[keep])]
    X = np.ascontiguousarray(P.standardise(crops[keep]).transpose(0, 2, 1)).astype(np.float32)
    rec_idx, labels = rec_idx[keep], labels[keep]
    y = np.array([classes.index(c) for c in labels])
    groups = np.array([info[i]["patient"] for i in rec_idx])
    skf = StratifiedGroupKFold(n_splits=int(cfg.data.ptbdiag_cv_folds), shuffle=True, random_state=seed)
    fold_metrics, ys, ps, gs = [], [], [], []
    for k, (tr, te) in enumerate(skf.split(X, y, groups)):
        model = copy.deepcopy(base_model)
        head = model.head if hasattr(model, "head") else None
        if head is None:
            return
        model.head = nn.Linear(head.in_features, len(classes))
        # inner split by patient for early stopping
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X[tr], y[tr], groups[tr])
        itr, iva = next(inner)
        model, _ = fit_classifier(model, X[tr][itr], y[tr][itr], X[tr][iva], y[tr][iva], "sl", device,
                                  epochs=30, bs=64, lr=3e-4, wd=1e-2, seed=seed, patience=8)
        p = predict_proba(model, X[te], "sl", device)
        recs = rec_idx[te]
        ur = np.unique(recs)
        pr = np.stack([p[recs == r].mean(0) for r in ur])
        yr = np.array([y[te][recs == r][0] for r in ur])
        fold_metrics.append(multiclass_metrics(yr, pr, len(classes), int(cfg.eval.ece_bins)))
        ys.append(yr), ps.append(pr), gs.append(np.array([groups[te][recs == r][0] for r in ur]))
    keys = ("acc", "sen", "spe", "gmean", "f1", "auc", "auprc", "mcc", "kappa", "ece")
    yy, pp = np.concatenate(ys), np.concatenate(ps)
    res = {"classes": classes, "pooled": multiclass_metrics(yy, pp, len(classes), int(cfg.eval.ece_bins)),
           "fold_mean": {k: float(np.mean([f[k] for f in fold_metrics])) for k in keys},
           "fold_sd": {k: float(np.std([f[k] for f in fold_metrics])) for k in keys},
           "n_records": int(len(yy))}
    od = Path(cfg.paths.work_dir) / "results" / "ptbdiag" / name
    od.mkdir(parents=True, exist_ok=True)
    save_json(res, od / f"seed{seed}.json")
    np.savez(od / f"seed{seed}_pred.npz", y=yy, p=pp, groups=np.concatenate(gs))
    log(f"{name} PTB Diagnostic transfer seed {seed}: ACC={res['fold_mean']['acc']:.4f}")


if __name__ == "__main__":
    main()
