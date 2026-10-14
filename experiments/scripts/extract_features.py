"""Four-stream feature extraction with a trained CL-TransCPD encoder (PTB-XL records, PTB Diagnostic crops).

  python scripts/extract_features.py --variant full --seed 0
  python scripts/extract_features.py --task-index 3      # same task list as pretrain_clt.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import config_from, parser, pick_task
from clcpd.pipeline import PTBXL, load_model, run_dir, signal_features, unpack_ragged
from clcpd.utils import Timer, get_device, log, pbs_array_index
from pretrain_clt import tasks_pretrain


def main():
    ap = parser(__doc__)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-index", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from(args)
    idx = args.task_index if args.task_index is not None else pbs_array_index(None)
    variant, seed = (args.variant, args.seed) if idx is None else pick_task(tasks_pretrain(cfg), idx)
    rd0 = run_dir(cfg, variant, seed)
    if not (rd0 / "summary.json").exists():
        log(f"{variant} seed {seed}: training not finished (or running elsewhere) - no features")
        return
    have = [rd0 / "features_ptbxl.npz"] + ([rd0 / "features_ptbdiag.npz"]
                                          if (Path(cfg.paths.work_dir) / "cache" / "ptbdiag" / "crops.npy").exists() else [])
    if all(p.exists() and p.stat().st_mtime >= (rd0 / "model.pt").stat().st_mtime for p in have):
        log(f"{variant} seed {seed}: features up to date - skipping")
        return
    device = get_device()
    model, vcfg, rd = load_model(cfg, variant, seed, device)
    P = PTBXL(vcfg)
    rows = np.where(P.meta["n_labels"].values > 0)[0]
    with Timer(f"PTB-XL features {variant}/seed{seed}"):
        feats, G = signal_features(model, [P.signals[r] for r in rows], [P.rpeaks[r] for r in rows],
                                   vcfg, P.standardise, device)
    save_npz_atomic(rd / "features_ptbxl.npz", rows=rows, G=G if G is not None else np.zeros(0), **feats)
    d = Path(cfg.paths.work_dir) / "cache" / "ptbdiag"
    if (d / "crops.npy").exists():
        crops = np.load(d / "crops.npy", mmap_mode="r")
        aux = np.load(d / "aux.npz")
        rp = unpack_ragged(aux["flat"], aux["off"])
        with Timer("PTB Diagnostic features"):
            f2, G2 = signal_features(model, [crops[i] for i in range(len(crops))], rp, vcfg, P.standardise, device)
        save_npz_atomic(rd / "features_ptbdiag.npz", G=G2 if G2 is not None else np.zeros(0), **f2)
    log("features saved to", rd)


def save_npz_atomic(path: Path, **arrays):
    """Readers (the classifier stage) never see a partially written file, even if two jobs extract at once."""
    import os

    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
