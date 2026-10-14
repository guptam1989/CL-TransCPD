"""Shared pipeline utilities used by the scripts (caches, variants, model I/O, segment embedding)."""
from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import apply_overrides, load_config, work_path
from .features.attention import attention_stats
from .features.graph import graph_features, lead_graph
from .models.clt import CLTransCPD, recon_error
from .signal.qrs import detect_rpeaks
from .signal.segmentation import extract_segments, segment_bounds
from .utils import load_json, save_json

# ablation variants -> configuration overrides (the score-weight mu is varied without retraining)
VARIANTS = {
    "full": [],
    "fixed_window": ["segmentation.mode=fixed"],
    "no_htfp": ["htfp.enabled=false"],
    "no_rssa": ["rssa.enabled=false"],
    "temporal_only": ["clt.cross_lead=false"],
    "lead_only": ["clt.temporal=false"],
    "rec_only": ["ccpp.loss=none"],
    "con_only": ["ccpp.lambda_rec=0", "score.mu=1.0"],
    "infonce": ["ccpp.loss=infonce"],
    "no_nuisance": ["ccpp.nuisance_positive=false"],
    "naive_splice": ["ccpp.harmonise_splice=false"],
    "no_cls_loss": ["ccpp.lambda_cls=0"],
    "no_pureneg": ["ccpp.pure_negative=false"],
    # model-selection candidates (validation only, scripts/design_study.py)
    "cand_pureneg": ["ccpp.pure_negative=true"],
    "cand_nopureneg": ["ccpp.pure_negative=false"],
}
MAIN_ABLATIONS = ["full", "fixed_window", "no_htfp", "no_rssa", "temporal_only", "rec_only",
                  "con_only", "infonce", "no_nuisance", "naive_splice", "no_cls_loss", "no_pureneg"]


def variant_config(cfg, variant):
    c = copy.deepcopy(cfg)
    return apply_overrides(c, VARIANTS[variant])


def run_dir(cfg, variant, seed) -> Path:
    p = Path(cfg.paths.work_dir) / "runs" / "clt" / variant / f"seed{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------- caches
def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=4)


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def pack_ragged(arrays):
    lens = np.array([len(a) for a in arrays], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    flat = np.concatenate(arrays) if len(arrays) and lens.sum() else np.zeros(0, dtype=np.int64)
    return flat, off


def unpack_ragged(flat, off):
    return [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]


class PTBXL:
    """Cached PTB-XL: signals (N, 1000, 12) in mV (band-passed), meta, R peaks, per-lead norm."""

    def __init__(self, cfg):
        d = Path(cfg.paths.work_dir) / "cache" / "ptbxl"
        self.signals = np.load(d / "signals.npy", mmap_mode="r")
        self.meta = pd.read_pickle(d / "meta.pkl")
        z = np.load(d / "rpeaks.npz")
        self.rpeaks = unpack_ragged(z["flat"], z["off"])
        nrm = load_json(d / "norm.json")
        self.mean = np.asarray(nrm["mean"], dtype=np.float32)
        self.std = np.asarray(nrm["std"], dtype=np.float32)
        self.fs = int(cfg.data.fs)
        f = self.meta["fold"].values
        lab = self.meta["n_labels"].values > 0
        self.train = np.where(np.isin(f, cfg.data.train_folds) & lab)[0]
        self.val = np.where(np.isin(f, cfg.data.val_folds) & lab)[0]
        self.test = np.where(np.isin(f, cfg.data.test_folds) & lab)[0]

    def standardise(self, x):
        return (x - self.mean) / self.std


# ----------------------------------------------------------------------------- model I/O
def load_model(cfg, variant, seed, device):
    rd = run_dir(cfg, variant, seed)
    vcfg = variant_config(cfg, variant)
    model = CLTransCPD(vcfg).to(device)
    state = torch.load(rd / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, vcfg, rd


# ----------------------------------------------------------------------------- embedding
@torch.no_grad()
def embed_segments(model, X, M, device, need_weights=False, bs=512):
    """X: (K, L, T) standardised, M: (K, T). Returns dict of numpy arrays per segment."""
    out = {"z": [], "R": [], "f_cls": [], "logits": [], "att": [], "graph": [], "G": []}
    for i in range(0, len(X), bs):
        x = torch.as_tensor(X[i:i + bs], device=device)
        m = torch.as_tensor(M[i:i + bs], device=device)
        o = model(x, m, need_weights=need_weights)
        out["z"].append(F.normalize(o["z"].float(), dim=-1).cpu().numpy())
        out["R"].append(recon_error(o["recon"].float(), o["target"].float(), m).cpu().numpy())
        out["f_cls"].append(o["f_cls"].float().cpu().numpy())
        out["logits"].append(torch.sigmoid(o["logits"].float()).cpu().numpy())
        if need_weights:
            if o["attn_t"][0] is not None:
                out["att"].append(attention_stats(o["attn_t"], o["pvalid"]).cpu().numpy())
            if o["attn_l"][0] is not None:
                G = lead_graph(o["attn_l"], o["pvalid"])
                out["graph"].append(graph_features(G).cpu().numpy())
                out["G"].append(G.cpu().numpy())
    res = {}
    for k, v in out.items():
        res[k] = np.concatenate(v) if v else None
    return res


def stream_segments(signal_mv, fs, seg_cfg, std_fn, ref_len):
    """Causal R-peak-driven segmentation of a stream -> (X, M, starts_s, ends_s, rpeaks)."""
    rp = detect_rpeaks(signal_mv, fs)
    bounds = segment_bounds(len(signal_mv), fs, rp, seg_cfg, causal=True)
    X, M = extract_segments(std_fn(signal_mv), bounds, ref_len)
    starts = np.array([s for s, _ in bounds], dtype=float) / fs
    ends = np.array([s + w for s, w in bounds], dtype=float) / fs
    return X, M, starts, ends, rp


DEEP_CPD = ("tire", "lstm_ae", "tscp2", "ocae")
SUPERVISED_CPD = ("sup_resnet",)        # trained with the diagnostic labels (reference for CL-TransCPD)


def denoise_torch(x, cfg):
    """Shared RSSA preprocessing for baseline windows; x: (B, L, w) tensor."""
    if not cfg.rssa.enabled or x.numel() == 0:
        return x
    from .signal.ssa import ssa_denoise

    B, L, w = x.shape
    return ssa_denoise(x.reshape(B * L, w), cfg.rssa.window, cfg.rssa.energy, cfg.rssa.max_rank).reshape(B, L, w)


@torch.no_grad()
def denoise_windows(W, cfg, device, bs=4096):
    """Numpy (K, L, w) -> RSSA-denoised numpy array (identity when RSSA is disabled)."""
    if not cfg.rssa.enabled or len(W) == 0:
        return W
    out = np.empty_like(W)
    for i in range(0, len(W), bs):
        out[i:i + bs] = denoise_torch(torch.as_tensor(W[i:i + bs], device=device), cfg).cpu().numpy()
    return out


def cpd_baseline_dir(cfg, name, seed) -> Path:
    p = Path(cfg.paths.work_dir) / "runs" / "cpd_baselines" / name / f"seed{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def train_signals_std(P):
    return P.standardise(np.asarray(P.signals[P.train])).astype(np.float32)


def get_deep_cpd_model(cfg, name, seed, P, device, log=None, train_std=None):
    """Load a trained deep CPD baseline, or train it on consecutive training-fold windows."""
    from .models.cpd_baselines import PairSampler, build_deep, train_deep_baseline

    if name in SUPERVISED_CPD:
        return get_supervised_cpd_model(cfg, name, seed, P, device, log=log, train_std=train_std)
    d = cpd_baseline_dir(cfg, name, seed)
    w = int(round(cfg.cpd_baselines.window_s * P.fs))
    if (d / "model.pt").exists():
        model = build_deep(name, int(cfg.data.n_leads), w).to(device)
        model.load_state_dict(torch.load(d / "model.pt", map_location=device))
        return model.eval()
    sig = train_std if train_std is not None else train_signals_std(P)
    sampler = PairSampler(sig, np.arange(len(sig)), P.fs, cfg.cpd_baselines.window_s, cfg.cpd_baselines.hop_s)
    model = train_deep_baseline(name, sampler, int(cfg.data.n_leads), device, epochs=int(cfg.cpd_baselines.epochs),
                                seed=seed, log=log, transform=lambda x: denoise_torch(x, cfg))
    torch.save(model.state_dict(), d / "model.pt")
    return model.eval()


def get_supervised_cpd_model(cfg, name, seed, P, device, log=None, train_std=None):
    """Load or train the supervised reference detector: ResNet1d-Wang on random crops of the training fold with the
    superclass labels, selected by macro-AUC on the fold-9 crops (window and hop of the CPD baselines)."""
    from .data.ptbxl import label_matrix
    from .models.cls_baselines import ResNet1dWang
    from .models.cpd_baselines import CropSampler, fixed_bounds, train_supervised_baseline, windows_of

    d = cpd_baseline_dir(cfg, name, seed)
    n_leads, n_cls = int(cfg.data.n_leads), 5
    if (d / "model.pt").exists():
        model = ResNet1dWang(n_leads, n_cls).to(device)
        model.load_state_dict(torch.load(d / "model.pt", map_location=device))
        return model.eval()
    cb = cfg.cpd_baselines
    sig = train_std if train_std is not None else train_signals_std(P)
    ytr = label_matrix(P.meta.iloc[P.train])
    vsig = P.standardise(np.asarray(P.signals[P.val])).astype(np.float32)
    b = fixed_bounds(vsig.shape[1], P.fs, cb.window_s, cb.hop_s)
    Xva = denoise_windows(np.concatenate([windows_of(s, b) for s in vsig]), cfg, device)
    yva = np.repeat(label_matrix(P.meta.iloc[P.val]), len(b), axis=0)
    model, info = train_supervised_baseline(CropSampler(sig, ytr, P.fs, cb.window_s), Xva, yva, n_leads, device,
                                            epochs=int(cb.epochs), seed=seed, log=log,
                                            transform=lambda x: denoise_torch(x, cfg))
    torch.save(model.state_dict(), d / "model.pt")
    save_json(info, d / "train_info.json")
    return model.eval()


FEATURE_KEYS = ("f_cls", "att", "graph", "z", "logits")


@torch.no_grad()
def signal_features(model, signals_mv, rpeaks_list, vcfg, std_fn, device, bs=512, chunk=512):
    """Per-signal (record / crop / buffer) features: segment features averaged over segments.

    Returns (dict of (N, d) arrays for FEATURE_KEYS, lead graphs (N, L, L) or None)."""
    fs, ref = float(vcfg.data.fs), int(vcfg.segmentation.ref_len)
    out = {k: [] for k in FEATURE_KEYS}
    graphs = []
    N = len(signals_mv)
    for c0 in range(0, N, chunk):
        idx = range(c0, min(N, c0 + chunk))
        Xs, Ms, owner = [], [], []
        for j, i in enumerate(idx):
            sig = np.asarray(signals_mv[i], dtype=np.float32)
            b = segment_bounds(len(sig), fs, rpeaks_list[i], vcfg.segmentation, causal=False)
            if not b:
                b = [(0, min(len(sig), ref))]
            X, M = extract_segments(std_fn(sig), b, ref)
            Xs.append(X)
            Ms.append(M)
            owner += [j] * len(b)
        e = embed_segments(model, np.concatenate(Xs), np.concatenate(Ms), device, need_weights=True, bs=bs)
        owner = np.asarray(owner)
        n = len(idx)
        cnt = np.bincount(owner, minlength=n).astype(np.float64)[:, None]
        for k in FEATURE_KEYS:
            if e[k] is None:
                out[k].append(np.zeros((n, 0), np.float32))
                continue
            agg = np.zeros((n, e[k].shape[1]))
            np.add.at(agg, owner, e[k])
            out[k].append((agg / cnt).astype(np.float32))
        if e["G"] is not None:
            g = np.zeros((n,) + e["G"].shape[1:])
            np.add.at(g, owner, e["G"])
            graphs.append((g / cnt[:, :, None]).astype(np.float32))
    feats = {k: np.concatenate(v) for k, v in out.items()}
    return feats, (np.concatenate(graphs) if graphs else None)
