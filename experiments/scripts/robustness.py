"""Robustness to recording noise (MIT-BIH NSTDB, test halves) and to missing leads.

CPD: test stream set 0 with noise added at each SNR (thresholds from the clean validation
calibration, i.e. no re-tuning). Classification: PTB-XL test fold with the same noise.

  python scripts/robustness.py --method clt --variant full --seed 0
  python scripts/robustness.py --method bocpd --seed 0
  python scripts/robustness.py --task-index 2
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from _common import config_from, parser, pick_task
from clcpd.eval.cls_metrics import multilabel_metrics
from clcpd.data.ptbxl import label_matrix
from clcpd.models.cls_baselines import build_classifier, predict_proba
from clcpd.online import CLTScoreDetector, evaluate
from clcpd.pipeline import PTBXL, load_model, load_pickle, run_dir, signal_features
from clcpd.signal.noise import NOISE_TYPES, add_noise, sample_noise, synthetic_noise
from clcpd.signal.qrs import detect_rpeaks
from clcpd.features.td import td_features
from clcpd.utils import get_device, load_json, log, pbs_array_index, save_json, set_seed
from run_cpd import BaselineInputs, CLTInputs, load_streams

DEFAULT_METHODS = [("clt", "full"), ("clt", "no_htfp"), ("clt", "temporal_only"), ("cusum", None),
                   ("bocpd", None), ("tscp2", None), ("ocae", None)]


def tasks_robustness(cfg):
    det = ("cusum", "bocpd", "rulsif")
    return [(m, v, s) for m, v in DEFAULT_METHODS
            for s in (cfg.eval.ablation_seeds[:1] if m in det else cfg.eval.ablation_seeds)]


def noise_source(cfg):
    p = Path(cfg.paths.work_dir) / "cache" / "nstdb" / "noise.pkl"
    return {k: v["test"] for k, v in load_pickle(p).items()} if p.exists() else {}


def corrupt(x, kind, snr, src, rng, fs):
    T, L = x.shape
    nz = sample_noise(src[kind], T, L, rng) if kind in src else synthetic_noise(T, L, fs, kind, rng)
    return add_noise(x, nz, snr)


def drop_leads(x, k, rng):
    if k <= 0:
        return x
    y = x.copy()
    y[:, rng.choice(x.shape[1], size=k, replace=False)] = 0.0
    return y


def main():
    ap = parser(__doc__)
    ap.add_argument("--method", default="clt")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-index", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from(args)
    idx = args.task_index if args.task_index is not None else pbs_array_index(None)
    method, variant, seed = (args.method, args.variant, args.seed) if idx is None else pick_task(tasks_robustness(cfg), idx)
    name = f"clt_{variant}" if method == "clt" else method
    res_path = Path(cfg.paths.work_dir) / "results" / "cpd" / name / f"seed{seed}.json"
    if not res_path.exists():
        log(f"missing {res_path}; run run_cpd.py first")
        return
    out = Path(cfg.paths.work_dir) / "results" / "robustness" / name / f"seed{seed}.json"
    if out.exists():
        log("done - skipping")
        return
    set_seed(seed)
    device = get_device()
    P = PTBXL(cfg)
    clean = load_json(res_path)
    gamma = clean["calibration"]["gamma"]
    streams = load_streams(cfg)["test_sets"][0]
    tol = float(cfg.streams.tolerance_s)
    if method == "clt":
        model, vcfg, _ = load_model(cfg, variant, seed, device)
        inputs = CLTInputs(model, vcfg, P, device, latent=clean.get("latent_selected", "z"))
        s = vcfg.score
        det = CLTScoreDetector(s.mu, s.ema_alpha, s.warmup_segments, clean["score_stats"])
    else:
        inputs = BaselineInputs(method, cfg, P, device, seed)
        det = inputs.detector()
    src = noise_source(cfg)
    rng = np.random.default_rng(seed)
    res = {"method": name, "gamma": gamma, "noise_source": "NSTDB" if src else "synthetic", "noise": {}, "dropout": {}}
    for kind in NOISE_TYPES:
        res["noise"][kind] = {}
        for snr in cfg.robustness.snr_db:
            noisy = [dict(st, signal=corrupt(st["signal"], kind, snr, src, rng, P.fs)) for st in streams]
            agg, _ = evaluate(det, inputs.items(noisy), gamma, P.fs, tol)
            res["noise"][kind][str(snr)] = agg
            log(f"{name} {kind} {snr} dB: F1={agg['f1']:.3f} FA/h={agg['fa_per_hour']:.2f}")
    for k in cfg.robustness.lead_dropout:
        dropped = [dict(st, signal=drop_leads(st["signal"], int(k), rng)) for st in streams]
        agg, _ = evaluate(det, inputs.items(dropped), gamma, P.fs, tol)
        res["dropout"][str(k)] = agg
    if method == "clt" and variant == "full":
        res["classification"] = classification_robustness(cfg, P, model, vcfg, device, seed, src, rng)
    save_json(res, out)
    log(f"saved {out}")


def classification_robustness(cfg, P, model, vcfg, device, seed, src, rng):
    pk = run_dir(cfg, "full", seed) / "skensem_ml.pkl"
    rows = P.test
    Y = label_matrix(P.meta.iloc[rows]).astype(int)
    ck = Path(cfg.paths.work_dir) / "runs" / "cls_baselines" / "resnet1d_wang" / "ml" / f"seed{seed}" / "model.pt"
    resnet = None
    if ck.exists():
        resnet = build_classifier("resnet1d_wang")
        resnet.load_state_dict(torch.load(ck, map_location=device))
        resnet.to(device)
    est = load_pickle(pk) if pk.exists() else None
    out = {}
    levels = [("clean", None, None)] + [(f"{k}_{s}", k, s) for k in NOISE_TYPES for s in cfg.robustness.snr_db]
    for tag, kind, snr in levels:
        sigs = [np.asarray(P.signals[r]) for r in rows]
        if kind is not None:
            sigs = [corrupt(x, kind, snr, src, rng, P.fs) for x in sigs]
        entry = {}
        if est is not None:
            rps = [detect_rpeaks(x, P.fs) for x in sigs]
            feats, _ = signal_features(model, sigs, rps, vcfg, P.standardise, device)
            td = np.stack([td_features(x, r, P.fs)[0] for x, r in zip(sigs, rps)])
            Xb = np.concatenate([td, feats["f_cls"], feats["att"], feats["graph"]], 1)
            entry["clt_full"] = multilabel_metrics(Y, est.predict_proba(Xb))["macro_auc"]
        if resnet is not None:
            X = np.ascontiguousarray(P.standardise(np.stack(sigs)).transpose(0, 2, 1)).astype(np.float32)
            entry["resnet1d_wang"] = multilabel_metrics(Y, predict_proba(resnet, X, "ml", device))["macro_auc"]
        out[tag] = entry
        log(f"classification {tag}: {entry}")
    return out


if __name__ == "__main__":
    main()
