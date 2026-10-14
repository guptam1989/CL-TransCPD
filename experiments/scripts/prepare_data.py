"""Build all caches used by the experiments.

  python scripts/prepare_data.py                 # every dataset that is present under DATA_ROOT
  python scripts/prepare_data.py --only ptbxl streams
  python scripts/prepare_data.py --only junctions     # constant-offset copy of test set 0 + junction step sizes
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from _common import config_from, parser
from clcpd.config import dataset_dir
from clcpd.data import ptbdiag as ptbdiag_mod
from clcpd.data import ptbxl as ptbxl_mod
from clcpd.data import staff3 as staff3_mod
from clcpd.data.nstdb import load_nstdb
from clcpd.data.streams import (build_control_streams, build_mixed_streams, build_same_patient_streams,
                                junction_steps)
from clcpd.features.td import td_features
from clcpd.pipeline import PTBXL, load_pickle, pack_ragged, save_pickle
from clcpd.signal.filters import bandpass
from clcpd.signal.qrs import detect_rpeaks, median_hr
from clcpd.utils import log, save_json


def _rpeaks_chunk(args):
    sigs, fs = args
    return [detect_rpeaks(s, fs) for s in sigs]


def detect_many(signals, fs, n_jobs=8, chunk=256):
    jobs = [(np.asarray(signals[i:i + chunk]), fs) for i in range(0, len(signals), chunk)]
    if n_jobs <= 1:
        res = [_rpeaks_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(n_jobs) as ex:
            res = list(ex.map(_rpeaks_chunk, jobs))
    return [r for part in res for r in part]


def prep_ptbxl(cfg, n_jobs):
    root = dataset_dir(cfg, "ptbxl")
    if not (root / "ptbxl_database.csv").exists():
        log(f"PTB-XL not found at {root} - skipped")
        return False
    out = Path(cfg.paths.work_dir) / "cache" / "ptbxl"
    out.mkdir(parents=True, exist_ok=True)
    fs = int(cfg.data.fs)
    meta = ptbxl_mod.load_metadata(root)
    if cfg.data.max_records:
        meta = meta.iloc[:int(cfg.data.max_records)]
    meta = meta.reset_index()
    log(f"PTB-XL: {len(meta)} records, {meta['patient_id'].nunique()} patients; loading signals")
    sig = ptbxl_mod.load_signals(root, meta, fs)
    lo, hi = cfg.data.bandpass
    for i in range(0, len(sig), 2000):
        sig[i:i + 2000] = bandpass(sig[i:i + 2000], fs, lo, hi, axis=1)
    np.save(out / "signals.npy", sig)
    log("R-peak detection")
    rp = detect_many(sig, fs, n_jobs)
    meta["hr"] = [median_hr(r, fs) for r in rp]
    meta["n_beats"] = [len(r) for r in rp]
    flat, off = pack_ragged([np.asarray(r, dtype=np.int64) for r in rp])
    np.savez(out / "rpeaks.npz", flat=flat, off=off)
    tr = np.isin(meta["fold"].values, cfg.data.train_folds)
    mean = sig[tr].mean(axis=(0, 1))
    std = sig[tr].std(axis=(0, 1)) + 1e-6
    save_json({"mean": mean, "std": std}, out / "norm.json")
    log("time-domain features")
    td = np.zeros((len(sig), 9), np.float32)
    qv = np.zeros((len(sig), 12), np.float32)
    for i in range(len(sig)):
        td[i], qv[i] = td_features(sig[i], rp[i], fs)
    np.savez(out / "td.npz", td=td, qrsv=qv)
    meta.to_pickle(out / "meta.pkl")
    counts = {c: int(meta[f"y_{c}"].sum()) for c in ptbxl_mod.SUPERCLASSES}
    save_json({"n_records": len(meta), "n_patients": int(meta["patient_id"].nunique()),
               "n_labelled": int((meta["n_labels"] > 0).sum()), "n_single": int((meta["n_labels"] == 1).sum()),
               "label_counts": counts,
               "single_counts": {c: int((meta["label_single"] == k).sum()) for k, c in enumerate(ptbxl_mod.SUPERCLASSES)},
               "hr_quantiles": np.nanpercentile(meta["hr"], [1, 5, 25, 50, 75, 95, 99]).tolist()},
              Path(cfg.paths.work_dir) / "results" / "data" / "ptbxl_summary.json")
    return True


def prep_streams(cfg):
    P = PTBXL(cfg)
    out = Path(cfg.paths.work_dir) / "cache" / "streams"
    out.mkdir(parents=True, exist_ok=True)
    fs, sc = P.fs, cfg.streams
    decay = sc.get("offset_decay_s")
    sig = np.asarray(P.signals)
    summary = {"offset_decay_s": decay}
    val = build_mixed_streams(P.meta, sig, P.rpeaks, P.val, sc, fs, seed=1000, harmonise=sc.harmonise)
    save_pickle(val, out / "val_set0.pkl")
    save_pickle(build_control_streams(P.meta, sig, P.rpeaks, P.val, sc.n_control, fs, 1000, sc.harmonise, sc.crossfade_ms,
                                      decay), out / "control_val.pkl")
    for s in range(int(sc.n_sets)):
        test = build_mixed_streams(P.meta, sig, P.rpeaks, P.test, sc, fs, seed=2000 + s, harmonise=sc.harmonise)
        save_pickle(test, out / f"test_set{s}.pkl")
        summary[f"test_set{s}"] = _summ(test, fs)
    save_pickle(build_control_streams(P.meta, sig, P.rpeaks, P.test, sc.n_control, fs, 2000, sc.harmonise, sc.crossfade_ms,
                                      decay), out / "control_test.pkl")
    sp = build_same_patient_streams(P.meta, sig, P.rpeaks, P.test, fs, sc.harmonise, sc.crossfade_ms, decay)
    save_pickle(sp, out / "same_patient_test.pkl")
    summary["val_set0"] = _summ(val, fs)
    summary["same_patient_test"] = _summ(sp, fs)
    # naive (non-harmonised) splicing of the same test records: shortcut analysis
    naive = build_mixed_streams(P.meta, sig, P.rpeaks, P.test, sc, fs, seed=2000, harmonise=False)
    save_pickle(naive, out / "test_set0_naive.pkl")
    save_json(summary, Path(cfg.paths.work_dir) / "results" / "data" / "streams_summary.json")
    log(f"streams: {summary}")


def prep_junctions(cfg, window_s=2.0, gap_s=0.1):
    """Constant-offset copy of test set 0 (the same records and junctions, but the alignment offset is kept for the
    whole appended record, as in the preliminary benchmark) and the baseline level step at the junctions of the
    main, constant-offset and naive copies of test set 0."""
    P = PTBXL(cfg)
    d = Path(cfg.paths.work_dir) / "cache" / "streams"
    const = build_mixed_streams(P.meta, np.asarray(P.signals), P.rpeaks, P.test, cfg.streams, P.fs, seed=2000,
                                harmonise=True, constant_offset=True)
    tmp = d / "test_set0_constoffset.pkl.tmp"
    save_pickle(const, tmp)
    os.replace(tmp, d / "test_set0_constoffset.pkl")      # CPD tasks may be reading the cache directory
    out = {"offset_decay_s": cfg.streams.get("offset_decay_s"), "window_s": window_s, "gap_s": gap_s}
    for k, st in (("main", load_pickle(d / "test_set0.pkl")), ("constant", const),
                  ("naive", load_pickle(d / "test_set0_naive.pkl"))):
        v = junction_steps(st, P.std, P.fs, window_s, gap_s)
        out[k] = {"n": int(len(v)), "median": float(np.median(v)) if len(v) else None,
                  "p90": float(np.percentile(v, 90)) if len(v) else None}
    save_json(out, Path(cfg.paths.work_dir) / "results" / "data" / "junction_steps.json")
    log(f"junction steps: {out}")


def _summ(streams, fs):
    return {"n_streams": len(streams), "n_cps": int(sum(len(s["cps"]) for s in streams)),
            "n_nuisance": int(sum(len(s["nuisance"]) for s in streams)),
            "hours": float(sum(len(s["signal"]) for s in streams) / fs / 3600.0)}


def prep_ptbdiag(cfg, n_jobs):
    root = dataset_dir(cfg, "ptbdiag")
    if not (root / "RECORDS").exists():
        log(f"PTB Diagnostic not found at {root} - skipped")
        return
    fs = int(cfg.data.fs)
    recs = ptbdiag_mod.load_ptbdiag(root, cfg.data.ptbdiag_mapping, fs, tuple(cfg.data.bandpass),
                                    max_records=int(cfg.data.max_records))
    crop = int(cfg.data.ptbdiag_crop_s * fs)
    crops, rec_idx = [], []
    for i, r in enumerate(recs):
        for s in range(0, len(r["signal"]) - crop + 1, crop):
            crops.append(r["signal"][s:s + crop])
            rec_idx.append(i)
    crops = np.stack(crops).astype(np.float32)
    rp = detect_many(crops, fs, n_jobs)
    td = np.zeros((len(crops), 9), np.float32)
    qv = np.zeros((len(crops), 12), np.float32)
    for i in range(len(crops)):
        td[i], qv[i] = td_features(crops[i], rp[i], fs)
    flat, off = pack_ragged([np.asarray(r, dtype=np.int64) for r in rp])
    info = [{k: v for k, v in r.items() if k != "signal"} for r in recs]
    out = Path(cfg.paths.work_dir) / "cache" / "ptbdiag"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "crops.npy", crops)
    np.savez(out / "aux.npz", rec_idx=np.asarray(rec_idx), flat=flat, off=off, td=td, qrsv=qv)
    save_pickle(info, out / "records.pkl")
    from collections import Counter
    save_json({"n_records": len(recs), "n_patients": len({r["patient"] for r in info}),
               "reasons": Counter(r["reason"] for r in info),
               "mapped_records": Counter(str(r["label"]) for r in info),
               "mapped_patients": Counter(str(lab) for _, lab in {(r["patient"], r["label"]) for r in info}),
               "n_crops": len(crops), "mapping": cfg.data.ptbdiag_mapping},
              Path(cfg.paths.work_dir) / "results" / "data" / "ptbdiag_summary.json")
    log(f"PTB Diagnostic: {len(recs)} records, {len(crops)} crops")


def prep_staff3(cfg):
    root = dataset_dir(cfg, "staff3")
    if not (root / "STAFF-III-Database-Annotations.xlsx").exists():
        log(f"STAFF III not found at {root} - skipped")
        return
    sc = cfg.streams
    cps, nulls = staff3_mod.build_streams(root, cfg.data.staff3_exclude_patients, sc.staff3_pre_s, sc.staff3_min_pre_s,
                                          int(cfg.data.fs), tuple(cfg.data.bandpass),
                                          offset_decay_s=sc.get("offset_decay_s"))
    save_pickle({"cp": cps, "null": nulls}, Path(cfg.paths.work_dir) / "cache" / "staff3" / "streams.pkl")
    save_json({"n_cp_streams": len(cps), "n_within": sum(s["kind"] == "within" for s in cps),
               "n_spliced": sum(s["kind"] == "spliced" for s in cps), "n_null": len(nulls),
               "n_patients": len({s["patient"] for s in cps}),
               "hours_cp": sum(len(s["signal"]) for s in cps) / cfg.data.fs / 3600,
               "hours_null": sum(len(s["signal"]) for s in nulls) / cfg.data.fs / 3600},
              Path(cfg.paths.work_dir) / "results" / "data" / "staff3_summary.json")
    log(f"STAFF III: {len(cps)} change-point streams, {len(nulls)} null streams")


def prep_nstdb(cfg):
    root = dataset_dir(cfg, "nstdb")
    noise = load_nstdb(root, int(cfg.data.fs)) if root.exists() else {}
    if not noise:
        log(f"NSTDB not found at {root} - synthetic surrogate noise will be used")
        return
    save_pickle(noise, Path(cfg.paths.work_dir) / "cache" / "nstdb" / "noise.pkl")
    log(f"NSTDB: {sorted(noise)}")


def main():
    ap = parser(__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    cfg = config_from(args)
    steps = args.only or ["ptbxl", "streams", "junctions", "ptbdiag", "staff3", "nstdb"]
    ok = True
    if "ptbxl" in steps:
        ok = prep_ptbxl(cfg, args.jobs)
    if "streams" in steps and ok:
        prep_streams(cfg)
    if "junctions" in steps and ok:
        prep_junctions(cfg)
    if "ptbdiag" in steps:
        prep_ptbdiag(cfg, args.jobs)
    if "staff3" in steps:
        prep_staff3(cfg)
    if "nstdb" in steps:
        prep_nstdb(cfg)


if __name__ == "__main__":
    main()
