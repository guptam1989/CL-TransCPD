"""Generate a tiny synthetic dataset in PTB-XL / PTB Diagnostic / STAFF III / NSTDB layout.

Only for smoke-testing the pipeline end to end (no scientific value):
  python scripts/make_synthetic_data.py --out ./data_synth --n 240
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASS_CODES = {"NORM": "NORM", "MI": "IMI", "STTC": "NDT", "CD": "CLBBB", "HYP": "LVH"}


def beat_train(T, fs, hr, cls, rng, L=12):
    t = np.arange(T) / fs
    x = np.zeros((T, L))
    gains = rng.uniform(0.4, 1.2, L) * np.where(np.arange(L) == 3, -1, 1)
    rr = 60.0 / hr
    b = rng.uniform(0.1, 0.5)
    while b < T / fs:
        qrs_w = 0.012 if cls != "CD" else 0.03
        st = 0.15 if cls == "MI" else 0.0
        tamp = -0.25 if cls == "STTC" else 0.3
        ramp = 2.2 if cls == "HYP" else 1.0
        comps = [(0.1, -0.16, 0.02), (ramp, 0.0, qrs_w), (-0.2 * ramp, 0.04, 0.012), (tamp, 0.28, 0.045)]
        for amp, mu, sd in comps:
            x += (amp * np.exp(-0.5 * ((t - b - mu) / sd) ** 2))[:, None] * gains[None, :]
        if st:
            x += (st * ((t > b + 0.06) & (t < b + 0.22)))[:, None] * gains[None, :]
        b += rr * rng.uniform(0.95, 1.05)
    return x + 0.02 * rng.standard_normal(x.shape)


def write_wfdb(path: Path, sig, fs, names, comments=None):
    import wfdb

    path.parent.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(path.name, fs=fs, units=["mV"] * sig.shape[1], sig_name=names, p_signal=sig,
                fmt=["16"] * sig.shape[1], write_dir=str(path.parent), comments=comments or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data_synth")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--scp", default=None, help="optional real scp_statements.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    root = Path(args.out)
    px = root / "ptb-xl" / "1.0.3"
    classes = list(CLASS_CODES)
    rows = []
    for i in range(args.n):
        ecg_id = i + 1
        # records i and i+10 share fold and (for some blocks) patient -> same-patient transitions exist
        pid = 5000 + (i // 20) * 10 + (i % 10) if (i // 10) % 4 in (0, 1) else 2000 + i
        cls = classes[(i // 10) % len(classes)]
        codes = {CLASS_CODES[cls]: 100.0}
        if i % 11 == 0 and cls != "NORM":
            codes["NDT"] = 50.0
        hr = float(rng.uniform(50, 120))
        sig = beat_train(1000, 100, hr, cls, rng)
        rel = f"records100/00000/{ecg_id:05d}_lr"
        write_wfdb(px / rel, sig, 100, LEADS)
        rows.append({"ecg_id": ecg_id, "patient_id": float(pid), "age": 60, "sex": i % 2, "scp_codes": str(codes),
                     "strat_fold": (i % 10) + 1, "filename_lr": rel, "filename_hr": rel,
                     "recording_date": f"199{i % 10}-01-01 00:00:00", "baseline_drift": np.nan,
                     "static_noise": np.nan, "burst_noise": np.nan, "electrodes_problems": np.nan})
    pd.DataFrame(rows).to_csv(px / "ptbxl_database.csv", index=False)
    if args.scp:
        import shutil
        shutil.copy(args.scp, px / "scp_statements.csv")
    else:
        scp = pd.DataFrame({"description": ["normal", "inferior MI", "non-diag T", "LBBB", "LVH"],
                            "diagnostic": [1.0] * 5, "form": [np.nan] * 5, "rhythm": [np.nan] * 5,
                            "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP"],
                            "diagnostic_subclass": ["NORM", "IMI", "STTC", "CLBBB", "LVH"]},
                           index=["NORM", "IMI", "NDT", "CLBBB", "LVH"])
        scp.to_csv(px / "scp_statements.csv")
    # PTB Diagnostic
    pdb = root / "ptbdb" / "1.0.0"
    reasons = {"NORM": "Healthy control", "MI": "Myocardial infarction", "CD": "Bundle branch block",
               "HYP": "Hypertrophy"}
    recs = []
    for p in range(24):
        cls = list(reasons)[p % 4]
        for r in range(1 + p % 2):
            sig = beat_train(32000, 1000, rng.uniform(55, 100), cls, rng, L=15)
            name = f"patient{p + 1:03d}/s{p * 3 + r:04d}lre"
            write_wfdb(pdb / name, sig, 1000, ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6",
                                                 "vx", "vy", "vz"],
                       comments=["age: 60", f"Reason for admission: {reasons[cls]}"])
            recs.append(name)
    (pdb / "RECORDS").write_text("\n".join(recs) + "\n")
    # STAFF III (2 patients)
    st = root / "staffiii" / "1.0.0"
    import wfdb

    names9 = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "II", "III"]
    sheet = [["D0"] + [None] * 28 for _ in range(9)]
    sheet[8] = ["Patient", "Age", "Sex", "Baseline room (BR)", "Baseline cathlab (BC)"] + [None] * 24
    hdr = ["#", None, None, "BR", "BC1", "BC2", "BI1", "BI1:Occluded artery", "BI1:D0;D1;D2", "BI1:Injection time(s)"]
    hdr += ["BI2", "BI2:Occluded artery", "BI2:D0;D1;D2", "BI2:Injection time(s)", "BI3", "BI3:Occluded artery",
            "BI3:D0;D1;D2", "BI3:Injection time(s)", "BI4", "BI4:Occluded artery", "BI4:D0;D1;D2", "BI5",
            "BI5:Occluded artery", "BI5:D0;D1;D2", "PC1", "PC2", "PR1", "PR2", None]
    sheet.append(hdr)
    for p in (2, 3):
        base = beat_train(60000, 1000, 70, "NORM", rng, L=9)
        write_wfdb(st / "data" / f"{p:03d}a", base, 1000, names9)
        pre = beat_train(60000, 1000, 72, "NORM", rng, L=9)
        post = beat_train(120000, 1000, 80, "MI", rng, L=9)
        sig = np.concatenate([pre, post])
        write_wfdb(st / "data" / f"{p:03d}c", sig, 1000, names9)
        wfdb.wrann(f"{p:03d}c", "event", np.array([60000, 170000]), symbol=['"', '"'],
                   aux_note=["balloon inflation", "balloon deflation"], fs=1000, write_dir=str(st / "data"))
        row = [p, 60, "m", f"{p}a", None, None, f"{p}c", "mid RCA", "60;110;10", None] + [None] * 19
        sheet.append(row)
    pd.DataFrame(sheet).to_excel(st / "STAFF-III-Database-Annotations.xlsx", header=False, index=False)
    # NSTDB
    ns = root / "nstdb" / "1.0.0"
    for kind in ("bw", "em", "ma"):
        noise = np.cumsum(rng.standard_normal((36000, 2)), 0) * 0.01 if kind == "bw" else rng.standard_normal((36000, 2)) * 0.1
        write_wfdb(ns / kind, noise, 360, ["noise1", "noise2"])
    print(f"synthetic data written to {root.resolve()}")


if __name__ == "__main__":
    main()
