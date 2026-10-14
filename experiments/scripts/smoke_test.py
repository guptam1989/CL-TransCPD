"""End-to-end smoke test on synthetic data with tiny settings (minutes on a CPU).

  python scripts/smoke_test.py --root /tmp/clcpd_smoke
Writes data, caches, results and paper assets ONLY below --root.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

TINY = [
    "ccpp.epochs=2", "ccpp.anchors_per_epoch=192", "ccpp.batch_size=32", "ccpp.num_workers=0", "ccpp.amp=false",
    "htfp.pretrain_epochs=1", "streams.n_sets=2", "streams.records_per_stream=[2,3]", "streams.n_control=8",
    "eval.seeds=[0]", "eval.ablation_seeds=[0]", "eval.bootstrap=20", "score.n_pr_thresholds=8",
    "cls_baselines.epochs=2", "cls_baselines.clocs_pretrain_epochs=1", "cls_baselines.batch_size=32",
    "cpd_baselines.epochs=1", "skensem.n_estimators=20", "skensem.cnn_epochs=2", "skensem.n_folds=2",
    "data.ptbdiag_cv_folds=2", "robustness.snr_db=[12,0]", "robustness.lead_dropout=[0,6]",
]


def run(script, *args, env):
    cmd = [sys.executable, str(HERE / script), *args]
    print("\n>>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--quick", action="store_true", help="skip most ablations/baselines")
    args = ap.parse_args()
    root = Path(os.path.abspath(args.root))          # abspath keeps junctions/short paths intact
    env = dict(os.environ, DATA_ROOT=str(root / "data"), WORK_DIR=str(root / "work"), PAPER_DIR=str(root / "paper"))
    run("make_synthetic_data.py", "--out", str(root / "data"), "--n", str(args.n), env=env)
    ov = TINY
    run("prepare_data.py", "--jobs", "1", *ov, env=env)
    variants = ["full"] if args.quick else ["full", "fixed_window", "temporal_only"]
    for v in variants:
        run("pretrain_clt.py", "--variant", v, "--seed", "0", *ov, env=env)
        run("extract_features.py", "--variant", v, "--seed", "0", *ov, env=env)
        run("train_classifiers.py", "--method", "clt", "--variant", v, "--seed", "0", *ov, env=env)
    run("train_classifiers.py", "--method", "clt", "--variant", "full", "--seed", "0", "--streams", "td,cls", *ov, env=env)
    run("train_classifiers.py", "--method", "clt", "--variant", "full", "--seed", "0", "--streams", "td,cls", "--clf",
        "lgbm", *ov, env=env)
    for m in (["lightgbm"] if args.quick else ["lightgbm", "adaboost", "prior"]):
        run("train_classifiers.py", "--method", m, "--seed", "0", *ov, env=env)
    for m in (["resnet1d_wang"] if args.quick else ["resnet1d_wang", "inception1d", "bilstm", "transformer", "patchtst", "clocs"]):
        for t in ("ml", "sl"):
            run("train_cls_baselines.py", "--model", m, "--task", t, "--seed", "0", *ov, env=env)
    for v in variants:
        run("run_cpd.py", "--method", "clt", "--variant", v, "--seed", "0", *ov, env=env)
    for m in (["cusum", "ocae", "sup_resnet"] if args.quick else
              ["cusum", "bocpd", "rulsif", "tire", "lstm_ae", "tscp2", "ocae", "sup_resnet"]):
        run("run_cpd.py", "--method", m, "--seed", "0", *ov, env=env)
    sup = json.loads((root / "work" / "results" / "cpd" / "sup_resnet" / "seed0.json").read_text(encoding="utf-8"))
    assert sup.get("rep_selected") in ("prob", "feat") and "train_info" in sup, "supervised reference incomplete"
    run("run_cpd.py", "--method", "sup_resnet", "--seed", "0", "--staff3-only", *ov, env=env)
    # partial passes used on the cluster: refresh STAFF III, add a missing triggered diagnosis
    run("run_cpd.py", "--method", "cusum", "--seed", "0", "--staff3-only", *ov, env=env)
    res = root / "work" / "results" / "cpd" / "clt_full" / "seed0.json"
    d = json.loads(res.read_text(encoding="utf-8"))
    d.pop("triggered", None)
    res.write_text(json.dumps(d), encoding="utf-8")
    run("run_cpd.py", "--method", "clt", "--variant", "full", "--seed", "0", "--triggered-only", *ov, env=env)
    assert "triggered" in json.loads(res.read_text(encoding="utf-8")), "triggered-only pass did not write its result"
    # constant-offset control: written by run_cpd itself, and added afterwards by the partial pass
    assert (root / "work" / "results" / "data" / "junction_steps.json").exists(), "junction steps not written"
    for args_ in (("--method", "clt", "--variant", "full"), ("--method", "cusum")):
        p = root / "work" / "results" / "cpd" / ("clt_full" if "clt" in args_ else "cusum") / "seed0.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        assert "const_offset" in d, f"run_cpd did not evaluate the constant-offset control ({p.parent.name})"
        d.pop("const_offset")
        p.write_text(json.dumps(d), encoding="utf-8")
        run("run_cpd.py", *args_, "--seed", "0", "--const-offset-only", *ov, env=env)
        assert "const_offset" in json.loads(p.read_text(encoding="utf-8")), "const-offset-only pass wrote nothing"
    run("robustness.py", "--method", "clt", "--variant", "full", "--seed", "0", *ov, env=env)
    run("robustness.py", "--method", "cusum", "--seed", "0", *ov, env=env)
    run("benchmark_latency.py", *ov, env=env)
    run("make_schematics.py", "--out", str(root / "paper" / "figures" / "generated"), env=env)
    run("make_paper_assets.py", *ov, env=env)
    print("\nSMOKE TEST PASSED; outputs under", root)


if __name__ == "__main__":
    main()
