"""STAFF III (Martinez et al., 2017): 12-lead ECG during prolonged PTCA balloon inflation.

Each inflation recording has a ``.event`` annotation with aux notes 'balloon inflation' and
'balloon deflation'. The curators' spreadsheet lists, per patient, the baseline recordings
(BR: room, BC: cath lab), the inflation recordings (BI1..BI5) with D0;D1;D2 = seconds from file
onset to inflation; inflation duration; seconds after deflation to end of file, and post-
inflation recordings. Nine leads are recorded (V1-V6, I, II, III); aVR, aVL and aVF are derived.

Change-point streams (ground truth = balloon inflation onset):
  * within  : inflation file with >= min_pre_s of pre-inflation signal; the stream starts
              min(D0, pre_s) seconds before the inflation and ends at deflation.
  * spliced : inflation starts at file onset (D0 < min_pre_s); the last pre_s seconds of the
              same patient's cath-lab (or room) baseline recording are prepended.
Baseline-room recordings without any intervention are used as change-free streams (FA/h).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..signal.filters import bandpass, resample

RECORDED = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "II", "III"]


def _rec_name(code) -> str | None:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    code = str(code).strip()
    if not code or code.lower() == "nan":
        return None
    num = "".join(ch for ch in code if ch.isdigit())
    let = "".join(ch for ch in code if ch.isalpha())
    return f"{int(num):03d}{let.lower()}" if num and let else None


def parse_sheet(root) -> pd.DataFrame:
    root = Path(root)
    path = root / "STAFF-III-Database-Annotations.xlsx"
    raw = pd.read_excel(path, header=None)
    hdr_row = None
    for i in range(min(20, len(raw))):
        if str(raw.iat[i, 0]).strip() == "#":
            hdr_row = i
            break
    if hdr_row is None:
        raise RuntimeError("could not find the header row ('#') in the STAFF III spreadsheet")
    cols = [str(c).strip() for c in raw.iloc[hdr_row].tolist()]
    cols[1], cols[2], cols[-1] = "Age", "Sex", "PriorMI"
    df = raw.iloc[hdr_row + 1:].copy()
    df.columns = cols
    df = df[pd.to_numeric(df["#"], errors="coerce").notna()]
    df["#"] = df["#"].astype(int)
    return df.reset_index(drop=True)


def _derive_12(sig9: np.ndarray, names) -> np.ndarray:
    idx = {n.upper(): i for i, n in enumerate(names)}
    g = lambda n: sig9[:, idx[n]]
    I, II, III = g("I"), g("II"), g("III")
    out = np.stack([I, II, III, -(I + II) / 2.0, I - II / 2.0, II - I / 2.0,
                    g("V1"), g("V2"), g("V3"), g("V4"), g("V5"), g("V6")], axis=1)
    return out.astype(np.float32)


def load_record(root, name: str, fs_out: int = 100, band=(0.5, 40.0)):
    import wfdb

    rec = wfdb.rdrecord(str(Path(root) / "data" / name))
    sig = _derive_12(np.nan_to_num(rec.p_signal), rec.sig_name)
    sig = resample(sig, rec.fs, fs_out)
    return bandpass(sig, fs_out, band[0], band[1]), rec.fs


def inflation_events(root, name: str):
    """List of (inflation_s, deflation_s) from the .event annotation (seconds from file onset)."""
    import wfdb

    path = Path(root) / "data" / name
    if not Path(str(path) + ".event").exists():
        return []
    ann = wfdb.rdann(str(path), "event")
    fs = ann.fs or 1000
    infl = [s / fs for s, a in zip(ann.sample, ann.aux_note) if "inflation" in a.lower()]
    defl = [s / fs for s, a in zip(ann.sample, ann.aux_note) if "deflation" in a.lower()]
    pairs = []
    for t in infl:
        later = [d for d in defl if d > t]
        pairs.append((t, later[0] if later else None))
    return pairs


def build_streams(root, exclude=(), pre_s=120.0, min_pre_s=30.0, fs_out=100, band=(0.5, 40.0),
                  max_patients: int = 0, offset_decay_s=None):
    """Returns (cp_streams, null_streams); each stream is a dict with signal (T,12), cps [s], meta.
    offset_decay_s: decay of the DC-alignment offset of spliced streams (see streams.harmonised_concat)."""
    from ..signal.qrs import mid_rr_points, detect_rpeaks
    from .streams import harmonised_concat

    root = Path(root)
    sheet = parse_sheet(root)
    cp_streams, null_streams = [], []
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = load_record(root, name, fs_out, band)[0]
        return cache[name]

    def available(name):
        return all((root / "data" / f"{name}.{ext}").exists() for ext in ("hea", "dat"))

    patients = [r for _, r in sheet.iterrows() if int(r["#"]) not in set(exclude)]
    if max_patients:
        patients = patients[:max_patients]
    used = set()
    for row in patients:
        pid = int(row["#"])
        baselines = [n for n in (_rec_name(row.get(c)) for c in ("BC2", "BC1", "BR")) if n]
        for b in range(1, 6):
            name = _rec_name(row.get(f"BI{b}"))
            if not name or name in used or not available(name):
                continue
            used.add(name)
            events = inflation_events(root, name)
            if not events:
                continue
            t_in, t_out = events[0]
            try:
                x = get(name)
            except Exception:
                continue
            end = int(round((t_out if t_out else len(x) / fs_out) * fs_out))
            i0 = int(round(t_in * fs_out))
            meta = {"patient": pid, "record": name, "artery": str(row.get(f"BI{b}:Occluded artery")),
                    "prior_mi": str(row.get("PriorMI"))}
            if t_in >= min_pre_s:
                s0 = max(0, i0 - int(pre_s * fs_out))
                sig = x[s0:end]
                cp_streams.append(dict(meta, kind="within", signal=sig, cps=[(i0 - s0) / fs_out]))
            else:
                base = next((n for n in baselines if available(n)), None)
                if base is None:
                    continue
                try:
                    pre = get(base)[-int(pre_s * fs_out):]
                except Exception:
                    continue
                post = x[i0:end]
                sig, junction = harmonised_concat([pre, post], fs_out, crossfade_ms=40, harmonise=True,
                                                  offset_decay_s=offset_decay_s)
                cp_streams.append(dict(meta, kind="spliced", baseline=base, signal=sig,
                                       cps=[junction[0] / fs_out]))
        br = _rec_name(row.get("BR"))
        if br and available(br):
            try:
                null_streams.append({"patient": pid, "record": br, "kind": "null", "signal": get(br), "cps": []})
            except Exception:
                pass
    return cp_streams, null_streams
