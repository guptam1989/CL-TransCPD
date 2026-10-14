"""PTB-XL loader (Wagner et al., 2020) with the standard superdiagnostic label aggregation.

Labels follow the official example code / Strodthoff et al. (2021): every diagnostic SCP
statement of a record is mapped to its diagnostic superclass, giving a multi-label target.
The single-label subset contains the records with exactly one superclass.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NOISE_COLS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]
# diagnostic subclasses used for the interpretability analysis
SUBGROUPS = ["IMI", "AMI", "LMI", "CLBBB", "CRBBB", "LAFB/LPFB", "LVH", "RVH", "ISCA", "ISCI"]


def load_metadata(root) -> pd.DataFrame:
    root = Path(root)
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag = scp[scp["diagnostic"] == 1]

    def sup(codes):
        return sorted({diag.at[c, "diagnostic_class"] for c in codes if c in diag.index})

    def sub(codes):
        return sorted({str(diag.at[c, "diagnostic_subclass"]) for c in codes if c in diag.index})

    df["super"] = df["scp_codes"].apply(sup)
    df["sub"] = df["scp_codes"].apply(sub)
    for c in SUPERCLASSES:
        df[f"y_{c}"] = df["super"].apply(lambda s, c=c: int(c in s))
    df["n_labels"] = df["super"].apply(len)
    df["label_single"] = df["super"].apply(lambda s: SUPERCLASSES.index(s[0]) if len(s) == 1 else -1)
    df["label_key"] = df["super"].apply(lambda s: "+".join(s) if s else "NONE")
    for g in SUBGROUPS:
        df[f"sub_{g}"] = df["sub"].apply(lambda s, g=g: int(g in s))
    present = [c for c in NOISE_COLS if c in df.columns]
    df["clean"] = df[present].isna().all(axis=1).astype(int) if present else 1
    df["fold"] = df["strat_fold"].astype(int)
    return df


def load_signals(root, df: pd.DataFrame, fs: int = 100) -> np.ndarray:
    import wfdb

    root = Path(root)
    col = "filename_lr" if fs == 100 else "filename_hr"
    n = 10 * fs
    out = np.zeros((len(df), n, 12), dtype=np.float32)
    for i, f in enumerate(df[col].values):
        sig, _ = wfdb.rdsamp(str(root / f))
        sig = np.nan_to_num(sig[:n].astype(np.float32))
        out[i, :len(sig)] = sig
    return out


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"y_{c}" for c in SUPERCLASSES]].values.astype(np.float32)
