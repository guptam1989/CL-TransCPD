"""Plain-text summary of every result file (for writing the narrative and sanity checks).

  python scripts/summarize_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _common import config_from, parser


def load(p):
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ms(v, nd=3):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "  -  "
    return f"{np.mean(v):.{nd}f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.{nd}f}(n={len(v)})"


def main():
    args = parser(__doc__).parse_args()
    cfg = config_from(args)
    R = Path(cfg.paths.work_dir) / "results"
    print("== classification (PTB-XL)")
    for d in sorted((R / "cls").glob("*")):
        ml = [load(f)["macro_auc"] for f in sorted((d / "ml").glob("seed*.json"))]
        sl = [load(f) for f in sorted((d / "sl").glob("seed*.json"))]
        print(f"{d.name:28s} ML-AUC {ms(ml)}  SL-ACC {ms([s['acc'] for s in sl])}  SL-F1 {ms([s['f1'] for s in sl])} "
              f" SL-AUC {ms([s['auc'] for s in sl])}  MCC {ms([s['mcc'] for s in sl])}  ECE {ms([s['ece'] for s in sl])}")
    print("== PTB Diagnostic")
    for d in sorted((R / "ptbdiag").glob("*")):
        fm = [load(f)["fold_mean"] for f in sorted(d.glob("seed*.json"))]
        print(f"{d.name:28s} ACC {ms([x['acc'] for x in fm])}  F1 {ms([x['f1'] for x in fm])}  AUC {ms([x['auc'] for x in fm])}")
    print("== change-point detection (PTB-XL-CPD test sets)")
    for d in sorted((R / "cpd").glob("*")):
        rs = [load(f) for f in sorted(d.glob("seed*.json"))]
        g = lambda k: [t.get(k) for r in rs for t in r.get("test_sets", [])]
        lat = [r.get("latent_selected") for r in rs if r.get("latent_selected")]
        st = [r["staff3"]["cp"] for r in rs if r.get("staff3")]
        print(f"{d.name:22s} F1 {ms(g('f1'))} P {ms(g('precision'))} R {ms(g('recall'))} AUPRC {ms(g('auprc'))} "
              f"R@10 {ms(g('recall_at_fah10'))} Td {ms(g('delay_mean'), 2)} FA/h {ms(g('fa_per_hour'), 1)} "
              f"NFA {ms(g('nuisance_fa_rate'))} AFA {ms([r['control']['artifact_fa_rate'] for r in rs])} "
              f"sameP-F1 {ms([r['same_patient']['f1'] for r in rs if r.get('same_patient')])} "
              f"naive-F1 {ms([r['naive_splice']['f1'] for r in rs if r.get('naive_splice')])} "
              f"STAFF-R {ms([s['recall'] for s in st])} STAFF-Td {ms([s['delay_median'] for s in st], 1)} "
              f"STAFF-FA/h {ms([r['staff3']['fa_per_hour_total'] for r in rs if r.get('staff3')], 2)} latent {lat}")
    trig = [load(f).get("triggered") for f in sorted((R / "cpd" / "clt_full").glob("seed*.json"))]
    trig = [t for t in trig if t]
    if trig:
        print("triggered diagnosis:", "AUC", ms([t.get("macro_auc") for t in trig]), "top1", ms([t.get("top1_acc_single") for t in trig]))
    print("== robustness (F1 by noise/SNR)")
    for d in sorted((R / "robustness").glob("*")):
        rs = [load(f) for f in sorted(d.glob("seed*.json"))]
        cells = []
        for kind in ("bw", "em", "ma"):
            for snr in ("24", "12", "0"):
                cells.append(f"{kind}{snr}:{ms([r['noise'][kind][snr]['f1'] for r in rs if snr in r['noise'].get(kind, {})], 2)}")
        print(f"{d.name:22s}", " ".join(cells))
    L = R / "latency" / "latency.json"
    if L.exists():
        lat = load(L)
        print("== latency", lat.get("params"), "online ms", lat.get("cpu_single_thread", {}).get("online_path_ms"))


if __name__ == "__main__":
    main()
