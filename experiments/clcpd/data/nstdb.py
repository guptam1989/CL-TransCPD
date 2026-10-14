"""MIT-BIH Noise Stress Test Database (Moody et al., 1984): baseline wander (bw), electrode
motion (em) and muscle artefact (ma) records, 360 Hz, 2 channels, 30 min each.

The first half of each record is used to train the HTFP denoiser and the second half for the
robustness evaluation, so no noise excerpt is seen in both.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..signal.filters import resample


def load_nstdb(root, fs_out: int = 100):
    import wfdb

    out = {}
    for kind in ("bw", "em", "ma"):
        path = Path(root) / kind
        if not Path(str(path) + ".hea").exists():
            continue
        rec = wfdb.rdrecord(str(path))
        x = resample(np.nan_to_num(rec.p_signal.astype(np.float32)), rec.fs, fs_out)
        half = len(x) // 2
        out[kind] = {"train": x[:half], "test": x[half:]}
    return out
