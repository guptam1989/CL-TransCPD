"""Model selection for the online change score on VALIDATION streams only (fold 9).

For a trained encoder, evaluates every combination of latent type (z / cls / both), latent weight
mu and reference update rate eta: threshold calibrated on validation, validation F1/precision/
recall/FA-h/NFA/AFA and validation AUPRC. No test data is touched.

  python scripts/design_study.py --variant full --seed 0
  python scripts/design_study.py --run-name pureneg --variant full --seed 0 ccpp.pure_negative=true
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from _common import config_from, parser
from clcpd.online import CLTScoreDetector, calibrate, pr_curve, stable_scores
from clcpd.pipeline import PTBXL, load_model
from clcpd.utils import get_device, log, save_json
from run_cpd import CLTInputs, latent_of, load_streams


def main():
    ap = parser(__doc__)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--latents", default="z,cls,both,prob")
    args = ap.parse_args()
    cfg = config_from(args)
    device = get_device()
    P = PTBXL(cfg)
    model, vcfg, rd = load_model(cfg, args.variant, args.seed, device)
    st = load_streams(cfg)
    inputs = CLTInputs(model, vcfg, P, device)
    t0 = time.time()
    emb = inputs.embed(st["val"]) + inputs.embed(st["control_val"])
    log(f"embedded validation streams in {time.time() - t0:.0f}s")
    tol, fs, warm = float(cfg.streams.tolerance_s), P.fs, int(cfg.score.warmup_segments)
    rows = []
    for lat in args.latents.split(","):
        items = [(s, list(zip(latent_of(e, lat), e["R"])), a, b) for s, e, a, b in emb]
        for eta in (0.05, 0.1, 0.3, 1.0):
            D = stable_scores(CLTScoreDetector(1.0, eta, warm, (0, 1, 0, 1)), items, tol)
            R = stable_scores(CLTScoreDetector(0.0, eta, warm, (0, 1, 0, 1)), items, tol)
            stats = (R.mean(), R.std() + 1e-8, D.mean(), D.std() + 1e-8)
            for mu in (0.0, 0.25, 0.5, 0.75, 1.0):
                det = CLTScoreDetector(mu, eta, warm, stats)
                cal = calibrate(det, items, fs, tol, cfg.score.percentile_grid)
                _, ap_ = pr_curve(det, items, fs, tol, 30)
                v = cal["val"] or {}
                rows.append({"latent": lat, "eta": eta, "mu": mu, "lambda": cal["lambda"], "val_auprc": ap_,
                             **{k: v.get(k) for k in ("f1", "precision", "recall", "fa_per_hour", "delay_mean",
                                                      "nuisance_fa_rate", "artifact_fa_rate")}})
                log(f"latent={lat} eta={eta} mu={mu}: AUPRC={ap_:.3f} F1={v.get('f1', 0):.3f} "
                    f"NFA={v.get('nuisance_fa_rate', float('nan')):.3f} AFA={v.get('artifact_fa_rate', float('nan')):.3f}")
    rows.sort(key=lambda r: (-(r["val_auprc"] or 0), -(r["f1"] or 0)))
    out = Path(cfg.paths.work_dir) / "results" / "design" / f"{args.variant}_seed{args.seed}{args.tag}.json"
    save_json({"variant": args.variant, "seed": args.seed, "offset_decay_s": cfg.streams.get("offset_decay_s"),
               "rows": rows}, out)                  # stream version: results without it predate the corrected streams
    log("best by validation AUPRC:", rows[0])
    log("best by validation F1:", max(rows, key=lambda r: r["f1"] or 0))


if __name__ == "__main__":
    main()
