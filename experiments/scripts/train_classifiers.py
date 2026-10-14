"""Feature-based classifiers: SkEnsem on CL-TransCPD streams, classical and prior-work baselines.

PTB-XL: multi-label superdiagnostic task ('ml', all labelled records) and single-label subset
('sl'); train folds 1-8, validation fold 9 (meta-learner and F1 thresholds), test fold 10.
PTB Diagnostic: patient-grouped stratified 5-fold cross-validation on 10-s crops, predictions
averaged per record.

  python scripts/train_classifiers.py --method clt --variant full --seed 0
  python scripts/train_classifiers.py --method clt --variant full --seed 0 --streams td cls
  python scripts/train_classifiers.py --method clt --variant full --seed 0 --qrsv
  python scripts/train_classifiers.py --method lightgbm --seed 0
  python scripts/train_classifiers.py --task-index 5         # PBS array over tasks_classifiers()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

from _common import config_from, parser, pick_task
from clcpd.classify.skensem import SkEnsem
from clcpd.data.ptbxl import SUPERCLASSES, label_matrix
from clcpd.eval.cls_metrics import multiclass_metrics, multilabel_metrics, tune_thresholds
from clcpd.models.cpd_baselines import fixed_bounds, ocae_latent, windows_of
from clcpd.pipeline import (MAIN_ABLATIONS, PTBXL, denoise_windows, get_deep_cpd_model, load_pickle, run_dir,
                            save_pickle, unpack_ragged)
from clcpd.utils import get_device, log, pbs_array_index, save_json, threads

DEFAULT_STREAMS = ("td", "cls", "att", "graph")
STREAM_SUBSETS = [("td",), ("cls",), ("att",), ("graph",), ("td", "cls"), ("td", "cls", "att"),
                  ("td", "cls", "graph"), ("cls", "att", "graph")]


def tasks_classifiers(cfg):
    t = []
    for s in cfg.eval.seeds:
        t.append(dict(method="clt", variant="full", seed=s, streams=DEFAULT_STREAMS, qrsv=False))
        t.append(dict(method="clt", variant="full", seed=s, streams=DEFAULT_STREAMS, qrsv=True))
        for m in ("adaboost", "lightgbm", "prior"):
            t.append(dict(method=m, variant=None, seed=s, streams=None, qrsv=False))
    # ablation grid: one LightGBM probe on the same features for every row (bounded cost; the
    # comparison isolates the representation from the ensemble)
    for s in cfg.eval.ablation_seeds:
        for sub in [DEFAULT_STREAMS] + STREAM_SUBSETS:
            t.append(dict(method="clt", variant="full", seed=s, streams=sub, qrsv=False, clf="lgbm"))
        for v in MAIN_ABLATIONS:
            if v != "full":
                t.append(dict(method="clt", variant=v, seed=s, streams=DEFAULT_STREAMS, qrsv=False, clf="lgbm"))
    return t


def method_name(method, variant, streams, qrsv, clf="skensem"):
    if method != "clt":
        return {"prior": "prior_skensem"}.get(method, method)
    name = f"clt_{variant}"
    if tuple(streams) != DEFAULT_STREAMS:
        name += "_s-" + "+".join(streams)
    return name + ("_qrsv" if qrsv else "") + ("_lgbm" if clf == "lgbm" else "")


def task_state(cfg, t):
    """'done' | 'ready' | 'wait' (CL-TransCPD features not extracted yet) - used by run_stage --wait-jobs."""
    import time
    clf = t.get("clf", "skensem")
    name = method_name(t["method"], t["variant"], t["streams"] or DEFAULT_STREAMS, t["qrsv"], clf)
    if (Path(cfg.paths.work_dir) / "results" / "cls" / name / "sl" / f"seed{t['seed']}.json").exists():
        return "done"
    if t["method"] != "clt":
        return "ready"
    rd = run_dir(cfg, t["variant"], t["seed"])
    need = ["features_ptbxl.npz"] + (["features_ptbdiag.npz"] if t["variant"] == "full" and clf != "lgbm" else [])
    ok = all((rd / f).exists() and time.time() - (rd / f).stat().st_mtime > 60 for f in need)
    return "ready" if ok else "wait"


class ClassicalAdapter:
    def __init__(self, kind, task, seed):
        self.kind, self.task, self.seed = kind, task, seed

    def _est(self):
        if self.kind == "adaboost":
            est = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=0.5,
                                     random_state=self.seed)
        else:
            from lightgbm import LGBMClassifier
            est = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
                                 colsample_bytree=0.9, random_state=self.seed, verbose=-1, n_jobs=threads())
        return OneVsRestClassifier(est) if self.task == "ml" else est

    def fit(self, X, Y, groups=None, Xval=None, Yval=None, score_fn=None):
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        self.est = self._est().fit(self.imp.transform(X), Y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(self.imp.transform(X))


def make_estimator(method, task, seed, cfg, n_classes=5, clf="skensem"):
    if method in ("adaboost", "lightgbm"):
        return ClassicalAdapter(method, task, seed)
    if clf == "lgbm":
        return ClassicalAdapter("lightgbm", task, seed)
    return SkEnsem(task, n_classes=n_classes, n_folds=int(cfg.skensem.n_folds), seed=seed, cfg=cfg.skensem)


def _ocae_record_latent(cfg, seed, P, device, signals_mv):
    model = get_deep_cpd_model(cfg, "ocae", seed, P, device, log=log)
    b = fixed_bounds(signals_mv[0].shape[0], P.fs, cfg.cpd_baselines.window_s, cfg.cpd_baselines.hop_s)
    out = []
    for i in range(0, len(signals_mv), 512):
        W = np.stack([windows_of(P.standardise(np.asarray(s)), b) for s in signals_mv[i:i + 512]])
        n, k = W.shape[:2]
        flat = denoise_windows(W.reshape(n * k, *W.shape[2:]), cfg, device)   # same RSSA as in OCAE training
        lat = ocae_latent(model, flat, device)
        out.append(lat.reshape(n, k, -1).mean(1))
    return np.concatenate(out).astype(np.float32)


def ptbxl_matrix(cfg, P, method, variant, seed, streams, qrsv, device):
    cache = Path(cfg.paths.work_dir) / "cache" / "ptbxl"
    td = np.load(cache / "td.npz")
    if method == "clt":
        f = np.load(run_dir(cfg, variant, seed) / "features_ptbxl.npz")
        rows = f["rows"]
        mats = {"td": td["td"][rows], "cls": f["f_cls"], "att": f["att"], "graph": f["graph"]}
        parts = [mats[s] for s in streams] + ([td["qrsv"][rows]] if qrsv else [])
        return rows, np.concatenate(parts, 1)
    rows = np.where(P.meta["n_labels"].values > 0)[0]
    if method in ("adaboost", "lightgbm"):
        return rows, np.concatenate([td["td"][rows], td["qrsv"][rows]], 1)
    lat = _ocae_record_latent(cfg, seed, P, device, [P.signals[r] for r in rows])
    return rows, np.concatenate([td["td"][rows], lat], 1)


def run_ptbxl(cfg, P, X, rows, method, seed, name, out_root, save_model_dir=None, clf="skensem"):
    meta = P.meta.iloc[rows]
    fold = meta["fold"].values
    pid = meta["patient_id"].values
    results = {}
    for task in ("ml", "sl"):
        keep = np.ones(len(rows), bool) if task == "ml" else (meta["n_labels"].values == 1)
        Y = label_matrix(meta).astype(int) if task == "ml" else meta["label_single"].values.astype(int)
        tr = np.isin(fold, cfg.data.train_folds) & keep
        va = np.isin(fold, cfg.data.val_folds) & keep
        te = np.isin(fold, cfg.data.test_folds) & keep
        score = (lambda y, p: multilabel_metrics(y, p)["macro_auc"]) if task == "ml" else \
            (lambda y, p: -float(np.mean(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)))))
        est = make_estimator(method, task, seed, cfg, clf=clf)
        est.fit(X[tr], Y[tr], groups=pid[tr], Xval=X[va], Yval=Y[va], score_fn=score)
        pva, pte = est.predict_proba(X[va]), est.predict_proba(X[te])
        if task == "ml":
            th = tune_thresholds(Y[va], pva)
            m = multilabel_metrics(Y[te], pte, th)
            m["thresholds"] = th.tolist()
        else:
            m = multiclass_metrics(Y[te], pte, len(SUPERCLASSES), int(cfg.eval.ece_bins))
        m["n_train"], m["n_val"], m["n_test"] = int(tr.sum()), int(va.sum()), int(te.sum())
        if isinstance(est, SkEnsem):
            m["selected_fraction"] = est.selected_fraction()
            m["meta_choice"] = list(est.meta_choice)
            m["n_features"] = int(X.shape[1])
        d = out_root / "cls" / name / task
        d.mkdir(parents=True, exist_ok=True)
        save_json(m, d / f"seed{seed}.json")
        np.savez(d / f"seed{seed}_pred.npz", y=Y[te], p=pte, groups=pid[te], rows=rows[te])
        if save_model_dir is not None:
            save_pickle(est, Path(save_model_dir) / f"skensem_{task}.pkl")
        results[task] = m
        log(f"{name} [{task}] seed {seed}: " + (f"macro-AUC={m['macro_auc']:.4f}" if task == "ml"
                                               else f"ACC={m['acc']:.4f} AUC={m['auc']:.4f} F1={m['f1']:.4f}"))
    return results


def run_ptbdiag(cfg, P, method, variant, seed, streams, qrsv, name, out_root, device):
    d = Path(cfg.paths.work_dir) / "cache" / "ptbdiag"
    if not (d / "crops.npy").exists():
        return None
    aux = np.load(d / "aux.npz")
    info = load_pickle(d / "records.pkl")
    rec_idx = aux["rec_idx"]
    labels = np.array([info[i]["label"] for i in rec_idx], dtype=object)
    keep = np.array([lab is not None for lab in labels])
    if method == "clt":
        f = np.load(run_dir(cfg, variant, seed) / "features_ptbdiag.npz")
        mats = {"td": aux["td"], "cls": f["f_cls"], "att": f["att"], "graph": f["graph"]}
        X = np.concatenate([mats[s] for s in streams] + ([aux["qrsv"]] if qrsv else []), 1)
    elif method in ("adaboost", "lightgbm"):
        X = np.concatenate([aux["td"], aux["qrsv"]], 1)
    else:
        crops = np.load(d / "crops.npy", mmap_mode="r")
        X = np.concatenate([aux["td"], _ocae_record_latent(cfg, seed, P, device, [crops[i] for i in range(len(crops))])], 1)
    classes = [c for c in SUPERCLASSES if c in set(labels[keep])]
    X, rec_idx, labels = X[keep], rec_idx[keep], labels[keep]
    y = np.array([classes.index(c) for c in labels])
    groups = np.array([info[i]["patient"] for i in rec_idx])
    skf = StratifiedGroupKFold(n_splits=int(cfg.data.ptbdiag_cv_folds), shuffle=True, random_state=seed)
    fold_metrics, all_y, all_p, all_g = [], [], [], []
    for k, (tr, te) in enumerate(skf.split(X, y, groups)):
        est = make_estimator(method, "sl", seed, cfg, n_classes=len(classes))
        est.fit(X[tr], y[tr], groups=groups[tr])
        p = est.predict_proba(X[te])
        recs = rec_idx[te]
        ur = np.unique(recs)
        pr = np.stack([p[recs == r].mean(0) for r in ur])
        yr = np.array([y[te][recs == r][0] for r in ur])
        gr = np.array([groups[te][recs == r][0] for r in ur])
        fold_metrics.append(multiclass_metrics(yr, pr, len(classes), int(cfg.eval.ece_bins)))
        all_y.append(yr), all_p.append(pr), all_g.append(gr)
    yy, pp, gg = np.concatenate(all_y), np.concatenate(all_p), np.concatenate(all_g)
    pooled = multiclass_metrics(yy, pp, len(classes), int(cfg.eval.ece_bins))
    res = {"classes": classes, "pooled": pooled,
           "fold_mean": {k: float(np.mean([fm[k] for fm in fold_metrics]))
                         for k in ("acc", "sen", "spe", "gmean", "f1", "auc", "auprc", "mcc", "kappa", "ece")},
           "fold_sd": {k: float(np.std([fm[k] for fm in fold_metrics]))
                       for k in ("acc", "sen", "spe", "gmean", "f1", "auc", "auprc", "mcc", "kappa", "ece")},
           "n_records": int(len(yy)), "n_patients": int(len(np.unique(gg)))}
    od = out_root / "ptbdiag" / name
    od.mkdir(parents=True, exist_ok=True)
    save_json(res, od / f"seed{seed}.json")
    np.savez(od / f"seed{seed}_pred.npz", y=yy, p=pp, groups=gg)
    log(f"{name} PTB Diagnostic seed {seed}: ACC={res['fold_mean']['acc']:.4f} AUC={res['fold_mean']['auc']:.4f}")
    return res


def main():
    ap = parser(__doc__)
    ap.add_argument("--method", default="clt", choices=["clt", "adaboost", "lightgbm", "prior"])
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--streams", default=",".join(DEFAULT_STREAMS), help="comma-separated, e.g. td,cls")
    ap.add_argument("--qrsv", action="store_true")
    ap.add_argument("--clf", default="skensem", choices=["skensem", "lgbm"],
                    help="classifier on the CL-TransCPD features (lgbm = ablation probe)")
    ap.add_argument("--no-ptbdiag", action="store_true")
    ap.add_argument("--task-index", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from(args)
    idx = args.task_index if args.task_index is not None else pbs_array_index(None)
    if idx is not None:
        t = pick_task(tasks_classifiers(cfg), idx)
        method, variant, seed, streams, qrsv = t["method"], t["variant"], t["seed"], t["streams"], t["qrsv"]
        clf = t.get("clf", "skensem")
    else:
        method, variant, seed, qrsv, clf = args.method, args.variant, args.seed, args.qrsv, args.clf
        streams = tuple(s for s in args.streams.split(",") if s)
    streams = tuple(streams) if streams else DEFAULT_STREAMS
    name = method_name(method, variant, streams, qrsv, clf)
    out_root = Path(cfg.paths.work_dir) / "results"
    if (out_root / "cls" / name / "sl" / f"seed{seed}.json").exists():
        log(f"{name} seed {seed} already done - skipping")
        return
    device = get_device()
    P = PTBXL(cfg)
    rows, X = ptbxl_matrix(cfg, P, method, variant, seed, streams, qrsv, device)
    save_dir = run_dir(cfg, variant, seed) if (method == "clt" and name == f"clt_{variant}") else None
    run_ptbxl(cfg, P, X, rows, method, seed, name, out_root, save_dir, clf)
    if not args.no_ptbdiag and clf != "lgbm" and (method != "clt" or variant == "full"):
        run_ptbdiag(cfg, P, method, variant, seed, streams, qrsv, name, out_root, device)


if __name__ == "__main__":
    main()
