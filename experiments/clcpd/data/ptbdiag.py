"""PTB Diagnostic ECG Database (Bousseljot et al., 1995): 549 records, 290 subjects, 1 kHz.

The database has no superclass labels; the "Reason for admission" field of each header is
mapped to the PTB-XL superclasses. The mapping is reported in the manuscript (Table 4):
  strict   : Healthy control->NORM, Myocardial infarction->MI, Bundle branch block->CD,
             Hypertrophy->HYP (other admission reasons are excluded)
  extended : strict + Myocarditis/Stable angina/Unstable angina->STTC, Dysrhythmia->CD,
             Cardiomyopathy->HYP
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..signal.filters import bandpass, resample

MAP_STRICT = {
    "healthy control": "NORM",
    "myocardial infarction": "MI",
    "bundle branch block": "CD",
    "hypertrophy": "HYP",
}
MAP_EXTENDED = dict(MAP_STRICT, **{
    "myocarditis": "STTC",
    "stable angina": "STTC",
    "unstable angina": "STTC",
    "dysrhythmia": "CD",
    "cardiomyopathy": "HYP",
})
LEAD_ORDER = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]


def reason_for_admission(comments) -> str:
    for line in comments:
        if "reason for admission" in line.lower():
            return line.split(":", 1)[1].strip()
    return "n/a"


def load_ptbdiag(root, mapping: str = "strict", fs_out: int = 100, band=(0.5, 40.0), max_records: int = 0):
    """Returns a list of dicts: record, patient, reason, label (superclass or None), signal (T, 12) mV."""
    import wfdb

    root = Path(root)
    table = MAP_EXTENDED if mapping == "extended" else MAP_STRICT
    names = [r.strip() for r in (root / "RECORDS").read_text().splitlines() if r.strip()]
    if max_records:
        names = names[:max_records]
    out = []
    for name in names:
        rec = wfdb.rdrecord(str(root / name))
        reason = reason_for_admission(rec.comments)
        label = table.get(reason.lower())
        if label is None:
            for key, val in table.items():           # e.g. "Heart failure (NYHA 2)" never matches strict
                if reason.lower().startswith(key):
                    label = val
                    break
        idx = [ [s.lower() for s in rec.sig_name].index(ld) for ld in LEAD_ORDER ]
        sig = np.nan_to_num(rec.p_signal[:, idx].astype(np.float32))
        sig = resample(sig, rec.fs, fs_out)
        sig = bandpass(sig, fs_out, band[0], band[1])
        out.append({"record": name, "patient": name.split("/")[0], "reason": reason,
                    "label": label, "signal": sig})
    return out
