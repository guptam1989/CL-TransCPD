"""Online change-point detection: calibration on validation streams, evaluation on test streams.

  python scripts/run_cpd.py --method clt --variant full --seed 0
  python scripts/run_cpd.py --method bocpd --seed 0
  python scripts/run_cpd.py --task-index 12          # PBS array over tasks_cpd()
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from _common import config_from, parser, pick_task
from clcpd.data.ptbxl import SUPERCLASSES
from clcpd.eval.cls_metrics import multilabel_metrics
from clcpd.features.td import td_features
from clcpd.models.cpd_baselines import (CUSUMDetector, FeatureProjector, deep_outputs, fixed_bounds, make_detector,
                                        morph_features, supervised_outputs, windows_of)
from clcpd.online import (CLTScoreDetector, bootstrap_streams, calibrate, evaluate, hr_stratified, pr_curve,
                          recall_at_fah, run, stable_scores)
from clcpd.pipeline import (DEEP_CPD, MAIN_ABLATIONS, PTBXL, SUPERVISED_CPD, cpd_baseline_dir, denoise_windows,
                            embed_segments, get_deep_cpd_model, load_model, load_pickle, run_dir, signal_features,
                            stream_segments, train_signals_std)
from clcpd.signal.qrs import detect_rpeaks
from clcpd.utils import get_device, load_json, log, pbs_array_index, save_json, set_seed

CLASSICAL = ("cusum", "bocpd", "rulsif")
CONST_OFFSET = "test_set0_constoffset.pkl"         # test set 0 with a sustained junction step (prepare_data junctions)


def tasks_cpd(cfg):
    t = [("clt", "full", s) for s in cfg.eval.seeds]
    t += [("clt", v, s) for v in MAIN_ABLATIONS if v != "full" for s in cfg.eval.ablation_seeds]
    for m in cfg.cpd_baselines.methods:              # classical detectors are deterministic: one run suffices
        seeds = cfg.eval.seeds[:1] if m in CLASSICAL else cfg.eval.seeds
        t += [(m, None, s) for s in seeds]
    return t


def load_streams(cfg):
    d = Path(cfg.paths.work_dir) / "cache" / "streams"
    out = {"val": load_pickle(d / "val_set0.pkl"), "control_val": load_pickle(d / "control_val.pkl"),
           "test_sets": [], "control_test": load_pickle(d / "control_test.pkl"),
           "same_patient": load_pickle(d / "same_patient_test.pkl"),
           "naive": load_pickle(d / "test_set0_naive.pkl") if (d / "test_set0_naive.pkl").exists() else [],
           "constoffset": load_pickle(d / CONST_OFFSET) if (d / CONST_OFFSET).exists() else []}
    s = 0
    while (d / f"test_set{s}.pkl").exists():
        out["test_sets"].append(load_pickle(d / f"test_set{s}.pkl"))
        s += 1
    st3 = Path(cfg.paths.work_dir) / "cache" / "staff3" / "streams.pkl"
    out["staff3"] = load_pickle(st3) if st3.exists() else None
    return out


# ----------------------------------------------------------------------------- CL-TransCPD inputs
def latent_of(e, kind):
    """Representation used by the change score: pooled patch latent z, CLS context, both (each l2-normalised),
    or the diagnostic-head probability vector (prob)."""
    nz = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    if kind == "cls":
        return nz(e["f_cls"])
    if kind == "both":
        return nz(np.concatenate([nz(e["z"]), nz(e["f_cls"])], axis=1))
    if kind == "prob":
        return np.asarray(e["logits"], dtype=np.float64)
    return e["z"]


class CLTInputs:
    def __init__(self, model, vcfg, P, device, latent=None):
        self.model, self.vcfg, self.P, self.device = model, vcfg, P, device
        self.latent = latent or vcfg.score.get("latent", "z")

    def embed(self, streams, chunk=256):
        """Per stream: (stream, embeddings dict, starts, ends)."""
        out = []
        ref = int(self.vcfg.segmentation.ref_len)
        for c0 in range(0, len(streams), chunk):
            part = streams[c0:c0 + chunk]
            segs = [stream_segments(st["signal"], self.P.fs, self.vcfg.segmentation, self.P.standardise, ref)
                    for st in part]
            X = np.concatenate([s[0] for s in segs]) if segs else np.zeros((0,))
            M = np.concatenate([s[1] for s in segs]) if segs else np.zeros((0,))
            e = embed_segments(self.model, X, M, self.device, need_weights=False) if len(X) else None
            k = 0
            for st, (Xs, Ms, starts, ends, rp) in zip(part, segs):
                n = len(Xs)
                sub = {key: (v[k:k + n] if v is not None else None) for key, v in e.items()} if e else \
                    {"z": np.zeros((0, 1)), "R": np.zeros(0), "f_cls": np.zeros((0, 1))}
                out.append((st, sub, starts, ends))
                k += n
        return out

    def items(self, streams, chunk=256):
        return [(st, list(zip(latent_of(e, self.latent), e["R"])), s, t) for st, e, s, t in self.embed(streams, chunk)]


def clt_stats(val_items, vcfg, tol):
    s = vcfg.score
    D = stable_scores(CLTScoreDetector(1.0, s.ema_alpha, s.warmup_segments, (0, 1, 0, 1)), val_items, tol)
    R = stable_scores(CLTScoreDetector(0.0, s.ema_alpha, s.warmup_segments, (0, 1, 0, 1)), val_items, tol)
    return (float(R.mean()), float(R.std() + 1e-8), float(D.mean()), float(D.std() + 1e-8))


def with_latent(emb, lat):
    return [(st, list(zip(latent_of(e, lat), e["R"])), a, b) for st, e, a, b in emb]


def select_latent(val_emb, vcfg, fs, tol):
    """Representation for the change score, chosen by validation AUPRC among score.latent_candidates."""
    s = vcfg.score
    res = {}
    for lat in s.latent_candidates:
        items = with_latent(val_emb, lat)
        det = CLTScoreDetector(s.mu, s.ema_alpha, s.warmup_segments, clt_stats(items, vcfg, tol))
        res[lat] = pr_curve(det, items, fs, tol, 30)[1]
    best = max(res, key=lambda k: res[k] if np.isfinite(res[k]) else -1)
    log(f"latent selection on validation (AUPRC): {res} -> {best}")
    return best, res


# ----------------------------------------------------------------------------- baseline inputs
class BaselineInputs:
    def __init__(self, name, cfg, P, device, seed):
        self.name, self.cfg, self.P, self.device = name, cfg, P, device
        self.fs = P.fs
        cb = cfg.cpd_baselines
        self.bounds_for = lambda T: fixed_bounds(T, self.fs, cb.window_s, cb.hop_s)
        d = cpd_baseline_dir(cfg, name, seed)
        if name in CLASSICAL:
            fp = d / "fitted.pkl"
            if fp.exists():
                self.fitted = load_pickle(fp)
            else:
                sig = train_signals_std(P)
                b = self.bounds_for(sig.shape[1])
                per_rec = [morph_features(denoise_windows(windows_of(s, b), cfg, device)) for s in sig]
                proj = FeatureProjector(int(cb.pca_dims)).fit(np.concatenate(per_rec))
                per_rec_p = [proj.transform(f) for f in per_rec]
                inv, kappa = CUSUMDetector.fit(per_rec_p, warm=int(cfg.score.warmup_segments))
                self.fitted = {"proj": proj, "inv_cov": inv, "kappa": kappa, "dim": proj.pca.n_components_}
                with open(fp, "wb") as fh:
                    pickle.dump(self.fitted, fh)
            self.model = None
        else:
            self.fitted = {}
            self.model = get_deep_cpd_model(cfg, name, seed, P, device, log=log)
        self.rep = "prob"                    # supervised reference: probabilities or features (chosen on validation)

    def detector(self):
        if self.name in SUPERVISED_CPD:     # the change score of CL-TransCPD (mu = 1, same EMA rate and warm-up)
            s = self.cfg.score
            return CLTScoreDetector(1.0, s.ema_alpha, s.warmup_segments, (0.0, 1.0, 0.0, 1.0))
        return make_detector(self.name, self.fitted, self.cfg)

    def items(self, streams):
        out = []
        for st in streams:
            b = self.bounds_for(len(st["signal"]))
            W = denoise_windows(windows_of(self.P.standardise(st["signal"]), b), self.cfg, self.device)
            if self.name in CLASSICAL:
                X = list(self.fitted["proj"].transform(morph_features(W))) if len(W) else []
            elif self.name in SUPERVISED_CPD:
                X = [(v, 0.0) for v in supervised_outputs(self.model, W, self.device)[self.rep]] if len(W) else []
            else:
                X = list(deep_outputs(self.name, self.model, W, self.device)) if len(W) else []
            starts = np.array([s for s, _ in b], dtype=float) / self.fs
            ends = np.array([s + w for s, w in b], dtype=float) / self.fs
            out.append((st, X, starts, ends))
        return out


# ----------------------------------------------------------------------------- evaluation driver
def staff3_eval(det, inputs, st3, gamma, cfg, fs):
    """Zero-shot STAFF III evaluation with the PTB-XL-calibrated threshold."""
    t3 = float(cfg.streams.staff3_tolerance_s)
    cp_items = inputs.items(st3["cp"])
    cp_res, cp_counts = evaluate(det, cp_items, gamma, fs, t3)
    null_items = inputs.items(st3["null"])
    null_res, _ = evaluate(det, null_items, gamma, fs, t3)
    cp_res["delays"] = [d for c in cp_counts for d in c["delays"]]
    cp_res["kinds"] = [s.get("kind") for s in st3["cp"]]
    cp_res["detected"] = [c["tp"] > 0 for c in cp_counts]
    cp_res["n_streams"] = len(st3["cp"])
    cp_res["n_patients"] = len({s.get("patient") for s in st3["cp"]})
    traces = []
    within = [it for it in cp_items if it[0].get("kind") == "within"]     # no splice: cleanest examples
    for st_, X_, s_, e_ in (within or cp_items)[:3]:
        alarms, sc = run(det, X_, e_, gamma)
        traces.append({"patient": st_.get("patient"), "artery": st_.get("artery"), "kind": st_.get("kind"),
                       "record": st_.get("record"), "cp": st_["cps"][0], "ends": list(map(float, e_)),
                       "scores": [float(v) for v in sc], "alarms": alarms, "gamma": float(gamma)})
    out = {"cp": cp_res, "null": null_res, "traces": traces,
           "fa_per_hour_total": (cp_res["fp"] + null_res["fp"]) / max(1e-9, cp_res["hours"] + null_res["hours"])}
    # threshold-free operating points (the PTB-XL threshold need not transfer) and the recall that a Poisson
    # alarm process with the same false-alarm rate would reach by chance within the tolerance
    pts, ap = pr_curve(det, cp_items + null_items, fs, t3, 30, q_extra=(0.9998, 0.9999, 0.99995, 1.0))
    out["pr_points"], out["auprc"] = pts, ap
    out["recall_at_fah5"] = recall_at_fah(pts, 5.0)
    out["recall_at_fah10"] = recall_at_fah(pts, 10.0)
    out["chance_recall"] = float(1.0 - np.exp(-out["fa_per_hour_total"] * t3 / 3600.0))
    # within-recording onsets (no splice) versus baseline + inflation recordings joined by a splice
    for kind in ("within", "spliced"):
        sub = [it for it in cp_items if it[0].get("kind") == kind]
        if sub:
            r, c = evaluate(det, sub, gamma, fs, t3)
            r["delays"] = [d for cc in c for d in cc["delays"]]
            r["n_streams"] = len(sub)
            out[kind] = r
    w = out.get("within", {})
    log(f"  STAFF III: recall={cp_res['recall']:.3f} median delay={cp_res['delay_median']:.1f}s "
        f"FA/h={out['fa_per_hour_total']:.2f} (chance recall {out['chance_recall']:.3f}; {cp_res['n_streams']} streams); "
        f"within-record: recall={w.get('recall', float('nan')):.3f} median delay={w.get('delay_median', float('nan')):.1f}s; "
        f"REC@10={out['recall_at_fah10']:.3f} REC@5={out['recall_at_fah5']:.3f}")
    return out


def evaluate_all(det, inputs, streams, cfg, fs, name, extra=None, val_items=None):
    tol = float(cfg.streams.tolerance_s)
    t0 = time.time()
    if val_items is None:
        val_items = inputs.items(streams["val"]) + inputs.items(streams["control_val"])
    calib = calibrate(det, val_items, fs, tol, cfg.score.percentile_grid)
    gamma = calib["gamma"]
    log(f"{name}: gamma={gamma:.4f} (Lambda={calib['lambda']}) val F1={calib['val']['f1']:.4f}")
    res = {"method": name, "calibration": calib, "test_sets": [], "auprc": []}
    pooled_items, pooled_counts = [], []
    for s, ts in enumerate(streams["test_sets"]):
        items = inputs.items(ts)
        agg, counts = evaluate(det, items, gamma, fs, tol)
        pts, ap = pr_curve(det, items, fs, tol, int(cfg.score.n_pr_thresholds))
        agg["auprc"] = ap
        agg["recall_at_fah10"] = recall_at_fah(pts, 10.0)
        agg["recall_at_fah60"] = recall_at_fah(pts, 60.0)
        res["test_sets"].append(agg)
        res["auprc"].append(ap)
        if s == 0:
            res["pr_points"] = pts
            res["ci_set0"] = bootstrap_streams(counts, int(cfg.eval.bootstrap), seed=0)
            res["per_stream_set0"] = [[c["tp"], c["fn"], c["fp"], c["hours"]] for c in counts]
            res["tolerance"] = {str(t): evaluate(det, items, gamma, fs, float(t))[0] for t in cfg.streams.tolerance_grid_s}
            set0_items, set0_counts = items, counts
        pooled_items += items
        pooled_counts += counts
        log(f"  test set {s}: F1={agg['f1']:.4f} P={agg['precision']:.4f} R={agg['recall']:.4f} "
            f"delay={agg['delay_mean']:.2f}s FA/h={agg['fa_per_hour']:.2f} AUPRC={ap:.4f}")
    keys = ["f1", "precision", "recall", "delay_mean", "delay_median", "fa_per_hour", "nuisance_fa_rate", "auprc",
            "recall_at_fah10", "recall_at_fah60"]
    res["mean"] = {k: float(np.nanmean([r[k] for r in res["test_sets"]])) for k in keys}
    res["sd"] = {k: float(np.nanstd([r[k] for r in res["test_sets"]])) for k in keys}
    res["delays"] = [d for c in pooled_counts for d in c["delays"]]
    res["hr"] = hr_stratified(pooled_items, pooled_counts, cfg.robustness.hr_bins)
    ctrl, _ = evaluate(det, inputs.items(streams["control_test"]), gamma, fs, tol)
    res["control"] = ctrl
    if streams["same_patient"]:
        res["same_patient"], _ = evaluate(det, inputs.items(streams["same_patient"]), gamma, fs, tol)
    if streams["naive"]:
        res["naive_splice"], _ = evaluate(det, inputs.items(streams["naive"]), gamma, fs, tol)
    if streams.get("constoffset"):
        res["const_offset"], _ = evaluate(det, inputs.items(streams["constoffset"]), gamma, fs, tol)
    if streams.get("staff3"):
        res["staff3"] = staff3_eval(det, inputs, streams["staff3"], gamma, cfg, fs)
    res["seconds"] = time.time() - t0
    if extra:
        res.update(extra(det, inputs, streams, gamma, set0_items, set0_counts))
    return res


def restore_detector(cfg, method, variant, seed, res, device, P):
    """Detector and inputs of an existing result (its selected representation and score statistics) for the
    partial passes, which reuse the calibrated threshold res['calibration']['gamma']."""
    if method == "clt":
        model, vcfg, _ = load_model(cfg, variant, seed, device)
        inputs = CLTInputs(model, vcfg, P, device, latent=res.get("latent_selected", "z"))
        s = vcfg.score
        return CLTScoreDetector(s.mu, s.ema_alpha, s.warmup_segments, res["score_stats"]), inputs, vcfg
    inputs = BaselineInputs(method, cfg, P, device, seed)
    inputs.rep = res.get("rep_selected", "prob")
    return inputs.detector(), inputs, cfg


def select_supervised_rep(inputs, streams, fs, tol):
    """Representation of the supervised reference (probabilities or penultimate features), chosen like the CL-TransCPD
    score representation: by change-point AUPRC on the validation streams."""
    res = {}
    for rep in ("prob", "feat"):
        inputs.rep = rep
        vi = inputs.items(streams["val"]) + inputs.items(streams["control_val"])
        res[rep] = pr_curve(inputs.detector(), vi, fs, tol, 30)[1]
    inputs.rep = max(res, key=lambda k: res[k] if np.isfinite(res[k]) else -1)
    log(f"supervised reference: representation by validation AUPRC {res} -> {inputs.rep}")
    return {"rep_selected": inputs.rep, "rep_val_auprc": res}


def triggered_diagnosis(cfg, model, vcfg, P, device, seed, set0_items, set0_counts):
    """Classify the post-alarm buffer for every detected change point (SkEnsem trained on PTB-XL)."""
    pk = run_dir(cfg, "full", seed) / "skensem_ml.pkl"
    if not pk.exists():
        return {"note": "skensem_ml.pkl missing - run train_classifiers.py first"}
    est = load_pickle(pk)
    fs = P.fs
    bufs, targets = [], []
    for (st, X, starts, ends), c in zip(set0_items, set0_counts):
        bounds = st["boundaries"]
        for cp, post_key, d in zip(st["cps"], st["post_key"], c["per_cp"]):
            if d is None:
                continue
            a = int(round((cp + d) * fs))
            nxt = min([b for b in bounds if b > a] + [len(st["signal"])])
            e = min(nxt, a + int(cfg.streams.post_buffer_s * fs))
            if e - a < 3 * fs:
                continue
            bufs.append(np.asarray(st["signal"][a:e]))
            targets.append([int(k in post_key.split("+")) for k in SUPERCLASSES])
    if not bufs:
        return {"triggered": {"n": 0}}
    rps = [detect_rpeaks(b, fs) for b in bufs]
    feats, _ = signal_features(model, bufs, rps, vcfg, P.standardise, device)
    td = np.stack([td_features(b, r, fs)[0] for b, r in zip(bufs, rps)])
    Xb = np.concatenate([td, feats["f_cls"], feats["att"], feats["graph"]], 1)
    prob = est.predict_proba(Xb)
    Y = np.asarray(targets)
    m = multilabel_metrics(Y, prob)
    single = Y.sum(1) == 1
    acc = float((prob[single].argmax(1) == Y[single].argmax(1)).mean()) if single.any() else float("nan")
    return {"triggered": {"n": int(len(Y)), "macro_auc": m["macro_auc"], "macro_auprc": m["macro_auprc"],
                          "top1_acc_single": acc, "n_single": int(single.sum())}}


def main():
    ap = parser(__doc__)
    ap.add_argument("--method", default="clt")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--staff3-only", action="store_true",
                    help="(re)compute only the STAFF III section of an existing result with its calibrated threshold")
    ap.add_argument("--triggered-only", action="store_true",
                    help="add the triggered diagnosis to an existing clt_full result (needs skensem_ml.pkl)")
    ap.add_argument("--const-offset-only", action="store_true",
                    help="add the constant-offset control (test set 0 with a sustained junction step) to an existing "
                         "result, with its calibrated threshold")
    args = ap.parse_args()
    cfg = config_from(args)
    idx = args.task_index if args.task_index is not None else pbs_array_index(None)
    method, variant, seed = (args.method, args.variant, args.seed) if idx is None else pick_task(tasks_cpd(cfg), idx)
    name = f"clt_{variant}" if method == "clt" else method
    out = Path(cfg.paths.work_dir) / "results" / "cpd" / name / f"seed{seed}.json"
    if args.triggered_only:
        if method != "clt" or variant != "full" or not out.exists():
            return
        res = load_json(out)
        if "triggered" in res:
            log(f"{name} seed {seed}: triggered diagnosis present - skipping")
            return
        device = get_device()
        P = PTBXL(cfg)
        model, vcfg, _ = load_model(cfg, variant, seed, device)
        inputs = CLTInputs(model, vcfg, P, device, latent=res.get("latent_selected", "z"))
        s = vcfg.score
        det = CLTScoreDetector(s.mu, s.ema_alpha, s.warmup_segments, res["score_stats"])
        items = inputs.items(load_streams(cfg)["test_sets"][0])
        _, counts = evaluate(det, items, res["calibration"]["gamma"], P.fs, float(cfg.streams.tolerance_s))
        tg = triggered_diagnosis(cfg, model, vcfg, P, device, seed, items, counts)
        if "triggered" in tg:
            res.pop("note", None)
            res.update(tg)
            save_json(res, out)
        log(f"{name} seed {seed}: {tg}")
        return
    if args.staff3_only:
        st3p = Path(cfg.paths.work_dir) / "cache" / "staff3" / "streams.pkl"
        if not out.exists() or not st3p.exists():
            log(f"{name} seed {seed}: result or STAFF III cache missing - skipping")
            return
        res = load_json(out)
        device = get_device()
        P = PTBXL(cfg)
        det, inputs, vcfg = restore_detector(cfg, method, variant, seed, res, device, P)
        res["staff3"] = staff3_eval(det, inputs, load_pickle(st3p), res["calibration"]["gamma"], vcfg, P.fs)
        save_json(res, out)
        return
    if args.const_offset_only:
        cop = Path(cfg.paths.work_dir) / "cache" / "streams" / CONST_OFFSET
        if not out.exists() or not cop.exists():
            log(f"{name} seed {seed}: result or constant-offset streams missing - skipping")
            return
        res = load_json(out)
        if "const_offset" in res:
            log(f"{name} seed {seed}: constant-offset control present - skipping")
            return
        device = get_device()
        P = PTBXL(cfg)
        det, inputs, vcfg = restore_detector(cfg, method, variant, seed, res, device, P)
        r, _ = evaluate(det, inputs.items(load_pickle(cop)), res["calibration"]["gamma"], P.fs,
                        float(vcfg.streams.tolerance_s))
        res["const_offset"] = r
        save_json(res, out)
        log(f"{name} seed {seed}: constant offset F1={r['f1']:.4f} FA/h={r['fa_per_hour']:.2f} "
            f"NFA={r['nuisance_fa_rate']:.3f}")
        return
    if out.exists():
        log(f"{name} seed {seed} done - skipping")
        return
    set_seed(seed)
    device = get_device()
    P = PTBXL(cfg)
    streams = load_streams(cfg)
    tol = float(cfg.streams.tolerance_s)
    if method == "clt":
        model, vcfg, rd = load_model(cfg, variant, seed, device)
        inputs = CLTInputs(model, vcfg, P, device, latent="z")
        val_emb = inputs.embed(streams["val"]) + inputs.embed(streams["control_val"])
        lat, lat_sel = vcfg.score.latent, None
        if lat == "auto":
            lat, lat_sel = select_latent(val_emb, vcfg, P.fs, tol)
        inputs.latent = lat
        val_items = with_latent(val_emb, lat)
        stats = clt_stats(val_items, vcfg, tol)
        s = vcfg.score
        det = CLTScoreDetector(s.mu, s.ema_alpha, s.warmup_segments, stats)

        def extra(det, inputs, streams, gamma, set0_items, set0_counts):
            ex = {"score_stats": stats, "latent_selected": lat, "latent_val_auprc": lat_sel}
            if variant == "full":
                sens = {}
                test0 = inputs.items(streams["test_sets"][0])
                for mu in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                    d2 = CLTScoreDetector(mu, s.ema_alpha, s.warmup_segments, stats)
                    cal = calibrate(d2, val_items, P.fs, tol, cfg.score.percentile_grid)
                    r, _ = evaluate(d2, test0, cal["gamma"], P.fs, tol)
                    r["auprc"] = pr_curve(d2, test0, P.fs, tol, 30)[1]
                    sens[str(mu)] = r
                ex["mu_sensitivity"] = sens
                ex.update(triggered_diagnosis(cfg, model, vcfg, P, device, seed, set0_items, set0_counts))
            return ex

        res = evaluate_all(det, inputs, streams, vcfg, P.fs, name, extra, val_items=val_items)
    else:
        inputs = BaselineInputs(method, cfg, P, device, seed)
        sel = select_supervised_rep(inputs, streams, P.fs, tol) if method in SUPERVISED_CPD else {}
        res = evaluate_all(inputs.detector(), inputs, streams, cfg, P.fs, name)
        res.update(sel)
        info = cpd_baseline_dir(cfg, method, seed) / "train_info.json"
        if info.exists():
            res["train_info"] = load_json(info)
    res["seed"] = seed
    save_json(res, out)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
