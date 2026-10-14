"""Parameters, FLOPs and per-segment latency of every pipeline stage (CPU single thread and GPU).

Latency does not depend on the trained weights; the trained seed-0 model is used when present.

  python scripts/benchmark_latency.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from _common import config_from, parser
from clcpd.features.attention import attention_stats
from clcpd.features.graph import graph_features, lead_graph
from clcpd.features.td import td_features
from clcpd.models.clt import CLTransCPD, count_parameters
from clcpd.models.cls_baselines import build_classifier
from clcpd.models.cpd_baselines import (BOCPDDetector, CUSUMDetector, FeatureProjector, RuLSIFDetector, build_deep,
                                        morph_features)
from clcpd.pipeline import load_pickle, run_dir
from clcpd.signal.qrs import detect_rpeaks
from clcpd.signal.segmentation import segment_bounds
from clcpd.signal.ssa import ssa_denoise
from clcpd.utils import hardware_string, log, save_json


def timeit(fn, n=300, warm=30):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000.0)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95))}


def flops(model, *inputs):
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                model(*inputs)
        return int(fc.get_total_flops())
    except Exception as exc:  # pragma: no cover
        log(f"FLOP counting failed: {exc}")
        return None


def synthetic_ecg(T, fs, hr=72.0, L=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fs
    x = np.zeros((T, L), np.float32)
    for b in np.arange(0.3, T / fs, 60.0 / hr):
        for amp, mu, sd in ((0.1, -0.2, 0.025), (1.0, 0.0, 0.01), (-0.2, 0.03, 0.01), (0.3, 0.3, 0.04)):
            x += (amp * np.exp(-0.5 * ((t - b - mu) / sd) ** 2))[:, None] * rng.uniform(0.5, 1.0, L)[None, :]
    return x + 0.01 * rng.standard_normal(x.shape).astype(np.float32)


def main():
    ap = parser(__doc__)
    args = ap.parse_args()
    cfg = config_from(args)
    fs = int(cfg.data.fs)
    ref = int(cfg.segmentation.ref_len)
    res = {"hardware": hardware_string(), "torch": torch.__version__}
    model = CLTransCPD(cfg)
    rd = run_dir(cfg, "full", 0)
    if (rd / "model.pt").exists():
        model.load_state_dict(torch.load(rd / "model.pt", map_location="cpu"))
        res["weights"] = "trained (full, seed 0)"
    model.eval()
    res["params"] = {"total": count_parameters(model), "htfp": count_parameters(model.htfp) if model.htfp else 0,
                     "clt_total": count_parameters(model.clt), "decoder": count_parameters(model.clt.decoder),
                     "cls_head": count_parameters(model.clt.cls_mlp) + count_parameters(model.clt.cls_out)}
    res["params"]["encoder"] = res["params"]["clt_total"] - res["params"]["decoder"] - res["params"]["cls_head"]
    x = torch.randn(1, 12, ref)
    m = torch.zeros(1, ref, dtype=torch.bool)
    m[:, :200] = True
    res["flops_clt_per_segment"] = flops(model.clt, x, m)
    buf = synthetic_ecg(10 * fs, fs)
    torch.set_num_threads(1)
    with torch.no_grad():
        cpu = {
            "rpeak_10s_buffer": timeit(lambda: detect_rpeaks(buf, fs), n=100),
            "ihras_window": timeit(lambda: segment_bounds(len(buf), fs, detect_rpeaks(buf, fs), cfg.segmentation), n=50),
            "htfp": timeit(lambda: model.htfp(x, m)) if model.htfp else None,
            "rssa": timeit(lambda: ssa_denoise(x.reshape(12, ref), cfg.rssa.window, cfg.rssa.energy, cfg.rssa.max_rank)),
            "clt_forward": timeit(lambda: model.clt(x, m)),
            "full_forward_with_target": timeit(lambda: model(x, m)),
        }

        def triggered():
            o = model(x, m, need_weights=True)
            attention_stats(o["attn_t"], o["pvalid"])
            graph_features(lead_graph(o["attn_l"], o["pvalid"]))
            td_features(buf, detect_rpeaks(buf, fs), fs)

        cpu["triggered_features"] = timeit(triggered, n=100)
        pk = rd / "skensem_ml.pkl"
        if pk.exists():
            est = load_pickle(pk)
            nfeat = len(est.imp.statistics_)
            xf = np.random.randn(1, nfeat)
            cpu["skensem_predict"] = timeit(lambda: est.predict_proba(xf), n=100)
    cpu["online_path_ms"] = sum(v["median_ms"] for k, v in cpu.items()
                                if v and k in ("htfp", "rssa", "full_forward_with_target"))
    res["cpu_single_thread"] = cpu
    # baselines (per 2-s window)
    w = int(round(cfg.cpd_baselines.window_s * fs))
    W = np.random.randn(1, 12, w).astype(np.float32)
    proj = FeatureProjector(int(cfg.cpd_baselines.pca_dims)).fit(np.random.randn(500, 60))
    f = proj.transform(morph_features(W))[0]
    det_c = CUSUMDetector(np.eye(len(f)), 1.0)
    det_b = BOCPDDetector(len(f))
    det_r = RuLSIFDetector(n=int(cfg.cpd_baselines.rulsif_n))
    for _ in range(50):
        det_c.step(f), det_b.step(f), det_r.step(f)
    base = {
        "cusum": timeit(lambda: det_c.step(proj.transform(morph_features(W))[0])),
        "bocpd": timeit(lambda: det_b.step(proj.transform(morph_features(W))[0])),
        "rulsif": timeit(lambda: det_r.step(proj.transform(morph_features(W))[0])),
    }
    params = {}
    with torch.no_grad():
        for name in ("tire", "lstm_ae", "tscp2", "ocae"):
            net = build_deep(name, 12, w).eval()
            params[name] = count_parameters(net)
            xt = torch.from_numpy(W)
            if name == "tire":
                fn = lambda: net(torch.nn.functional.avg_pool1d(xt, 4, 4).flatten(1))
            elif name == "lstm_ae":
                fn = lambda: net(torch.nn.functional.avg_pool1d(xt, 4, 4).transpose(1, 2))
            else:
                fn = lambda: net(xt)
            base[name] = timeit(fn)
        for name in cfg.cls_baselines.models:
            net = build_classifier(name).eval()
            params[f"cls_{name}"] = count_parameters(net)
            xr = torch.randn(1, 12, 10 * fs)
            base[f"cls_{name}_10s"] = timeit(lambda: net(xr), n=50)
    res["baselines_cpu_single_thread"] = base
    res["baseline_params"] = params
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        mg = model.to(dev)
        xg, mgk = x.to(dev), m.to(dev)
        xb, mb = torch.randn(256, 12, ref, device=dev), mgk.repeat(256, 1)

        def sync(fn):
            def g():
                fn()
                torch.cuda.synchronize()
            return g

        with torch.no_grad():
            res["gpu"] = {"device": torch.cuda.get_device_name(0),
                          "forward_batch1": timeit(sync(lambda: mg(xg, mgk))),
                          "forward_batch256": timeit(sync(lambda: mg(xb, mb)), n=50)}
    out = Path(cfg.paths.work_dir) / "results" / "latency" / "latency.json"
    save_json(res, out)
    log(f"saved {out}: params={res['params']} online path={cpu['online_path_ms']:.2f} ms")


if __name__ == "__main__":
    main()
