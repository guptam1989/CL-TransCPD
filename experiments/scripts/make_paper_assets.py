"""Turn experiment results into manuscript assets.

Writes  <paper>/results/tables/*.tex   (tabular bodies, \\input by Main.tex)
        <paper>/results/macros.tex     (every number quoted in the text)
        <paper>/figures/generated/*.pdf

Missing results are rendered as \\tbd (a red dash), so the manuscript always compiles.
  python scripts/make_paper_assets.py
  python scripts/make_paper_assets.py --placeholders     # structure only, no results needed
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _common import config_from, parser

SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
CLS_THIS = ["resnet1d_wang", "inception1d", "bilstm", "transformer", "patchtst", "clocs",
            "adaboost", "lightgbm", "prior_skensem"]
OURS_CLS = ["clt_full", "clt_full_qrsv"]
CPD_METHODS = ["cusum", "bocpd", "rulsif", "tire", "lstm_ae", "tscp2", "ocae", "sup_resnet", "clt_full"]
SUPERVISED = ["sup_resnet"]                      # trained with the diagnostic labels, like CL-TransCPD
UNSUP_CPD = [n for n in CPD_METHODS if n != "clt_full" and n not in SUPERVISED]
# rec_only and con_only keep the auxiliary diagnostic loss (lambda_c = 0.5); only no_cls_loss removes it
ABLATIONS = [("full", "Full model"), ("fixed_window", "Fixed window (IHRAS removed)"), ("no_htfp", "w/o HTFP"),
             ("no_rssa", "w/o RSSA"), ("temporal_only", "Temporal attention only (no cross-lead axis)"),
             ("rec_only", "w/o contrastive loss"),
             ("con_only", r"w/o reconstruction loss ($\lambda_r{=}0$)"),
             ("infonce", "InfoNCE instead of triplet loss"), ("no_nuisance", "w/o nuisance positives"),
             ("naive_splice", "Naive (non-harmonised) training splices"),
             ("no_cls_loss", r"w/o auxiliary diagnostic loss ($\lambda_c{=}0$)"),
             ("no_pureneg", "w/o post-change negative")]
STREAM_SETS = [("td",), ("cls",), ("att",), ("graph",), ("td", "cls"), ("td", "cls", "att"), ("td", "cls", "graph"),
               ("cls", "att", "graph"), ("td", "cls", "att", "graph")]
REPORTED_ML = [("resnet1d\\_wang", 0.930, 0.006), ("xresnet1d101", 0.928, 0.006), ("inception1d", 0.921, 0.006),
               ("lstm", 0.927, 0.005), ("fcn\\_wang", 0.925, 0.006), ("ensemble", 0.934, 0.005)]
DISPLAY = {
    "clt_full": r"CL-TransCPD", "clt_full_qrsv": r"CL-TransCPD + QRS-V",
    "prior_skensem": r"SkEnsem~\cite{gupta2025deep}", "adaboost": r"AdaBoost~\cite{freund1997}",
    "lightgbm": r"LightGBM~\cite{ke2017}", "resnet1d_wang": r"ResNet1d-Wang~\cite{wang2017time}",
    "inception1d": r"Inception1d~\cite{fawaz2020inceptiontime}", "bilstm": r"BiLSTM~\cite{strodthoff2020}",
    "transformer": r"Transformer encoder~\cite{vaswani2017}", "patchtst": r"PatchTST~\cite{patchtst2023}",
    "clocs": r"CLOCS-CMSC~\cite{kiyasseh2021}", "cusum": r"CUSUM~\cite{page1954,crosier1988}",
    "bocpd": r"BOCPD~\cite{adams2007}", "rulsif": r"RuLSIF~\cite{liu2013}", "tire": r"TIRE~\cite{deryck2021}",
    "lstm_ae": r"LSTM-AE~\cite{hou2020}", "tscp2": r"TS-CP$^2$~\cite{deldari2021tscp2}",
    "ocae": r"OCAE-CPD~\cite{gupta2025deep}", "sup_resnet": r"ResNet-CPD$^{\ast}$~\cite{wang2017time}",
}
SHORT = {"clt_full": "CL-TransCPD", "cusum": "CUSUM", "bocpd": "BOCPD", "rulsif": "RuLSIF", "tire": "TIRE",
         "lstm_ae": "LSTM-AE", "tscp2": "TS-CP²", "ocae": "OCAE-CPD", "sup_resnet": "ResNet-CPD",
         "prior_skensem": "SkEnsem",
         "resnet1d_wang": "ResNet1d-Wang", "inception1d": "Inception1d", "transformer": "Transformer",
         "patchtst": "PatchTST", "clocs": "CLOCS", "lightgbm": "LightGBM", "adaboost": "AdaBoost", "bilstm": "BiLSTM"}
TBD = r"\tbd"


# ----------------------------------------------------------------------------- I/O helpers
class Results:
    def __init__(self, root: Path, placeholders: bool):
        self.root, self.ph = root, placeholders

    def seeds(self, *parts):
        if self.ph:
            return []
        d = self.root.joinpath(*parts)
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("seed*.json")):
            with open(p, "r", encoding="utf-8") as fh:
                out.append(json.load(fh))
        return out

    def one(self, *parts):
        if self.ph:
            return None
        p = self.root.joinpath(*parts)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)


def finite(vals):
    return [float(v) for v in vals if v is not None and np.isfinite(v)]


def mean_of(vals):
    v = finite(vals)
    return float(np.mean(v)) if v else None


def pm(vals, nd=3, scale=1.0):
    v = [x * scale for x in finite(vals)]
    if not v:
        return TBD
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{np.mean(v):.{nd}f}\\,$\\pm$\\,{np.std(v, ddof=1):.{nd}f}"


def num(v, nd=3, scale=1.0):
    return TBD if v is None or not np.isfinite(v) else f"{v * scale:.{nd}f}"


def mark(cells, means, higher=True):
    """Bold the best and underline the second-best entry of a column (by mean)."""
    idx = [i for i, m in enumerate(means) if m is not None]
    if len(idx) < 2:
        return cells
    order = sorted(idx, key=lambda i: means[i], reverse=higher)
    out = list(cells)
    out[order[0]] = r"\textbf{" + out[order[0]] + "}"
    if len(order) > 1:
        out[order[1]] = r"\sbb{" + out[order[1]] + "}"
    return out


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tabular(spec, header, rows, midrules=()):
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\midrule")
        lines.append(r if isinstance(r, str) else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def columns_marked(names, getters, higher_flags, fmt_nd):
    """names: row keys; getters: list of fn(name)->list of values. Returns rows of formatted cells."""
    cols = []
    for g, hi, nd in zip(getters, higher_flags, fmt_nd):
        vals = [g(n) for n in names]
        cells = [pm(v, nd) for v in vals]
        cols.append(mark(cells, [mean_of(v) for v in vals], hi) if hi is not None else cells)
    return [[c[i] for c in cols] for i in range(len(names))]


# ----------------------------------------------------------------------------- tables
def t_ptbxl_ml(R, out):
    names = CLS_THIS + OURS_CLS
    g = lambda key: (lambda n: [s.get(key) for s in R.seeds("cls", n, "ml")])
    params = lambda n: [s.get("params") for s in R.seeds("cls", n, "ml")]
    body = columns_marked(names, [g("macro_auc"), g("macro_auprc"), g("macro_f1")], [True, True, True], [3, 3, 3])
    rows = []
    lat = R.one("latency", "latency.json") or {}
    for (lab, v, ci) in REPORTED_ML:
        rows.append([lab + r"~\cite{strodthoff2020}$^{\dagger}$", f"{v:.3f}\\,($\\pm$\\,{ci:.3f})", "--", "--", "--"])
    for n, cells in zip(names, body):
        pr = mean_of(params(n)) if not n.startswith("clt_") else lat.get("params", {}).get("total")
        rows.append([DISPLAY[n]] + cells + [f"{pr / 1e6:.2f}\\,M" if pr else "--"])
    write(out / "tab_ptbxl_ml.tex", tabular("@{}lcccr@{}", ["Model", "Macro-AUC", "Macro-AUPRC", "Macro-F1", "Params"],
                                            rows, midrules=(len(REPORTED_ML), len(REPORTED_ML) + len(CLS_THIS))))


def t_ptbxl_sl(R, out):
    names = CLS_THIS + OURS_CLS
    g = lambda key: (lambda n: [s.get(key) for s in R.seeds("cls", n, "sl")])
    keys = ["acc", "sen", "spe", "gmean", "f1", "auc"]
    body = columns_marked(names, [g(k) for k in keys], [True] * 6, [3] * 6)
    rows = [[DISPLAY[n]] + c for n, c in zip(names, body)]
    write(out / "tab_ptbxl_sl.tex", tabular("@{}lcccccc@{}", ["Model", "ACC", "SEN", "SPE", r"$G_\mu$", "F1", "AUC"],
                                            rows, midrules=(len(CLS_THIS),)))


def t_adv(R, out):
    names = CLS_THIS + OURS_CLS
    g = lambda key: (lambda n: [s.get(key) for s in R.seeds("cls", n, "sl")])
    body = columns_marked(names, [g("mcc"), g("kappa"), g("auprc"), g("ece"), g("dor_gmean")],
                          [True, True, True, False, True], [3, 3, 3, 3, 1])
    rows = [[DISPLAY[n]] + c for n, c in zip(names, body)]
    write(out / "tab_adv.tex", tabular("@{}lccccc@{}", ["Model", "MCC", r"$\kappa$", "AUPRC", r"ECE$\downarrow$",
                                                        r"DOR$_{g}$"], rows, midrules=(len(CLS_THIS),)))


def t_perclass(R, out):
    ours, ref = R.seeds("cls", "clt_full", "sl"), R.seeds("cls", "prior_skensem", "sl")
    keys = [("auprc", 3), ("ece", 3), ("dor", 1), ("mcc", 3), ("kappa", 3)]

    def cell(c, k, nd):
        a = mean_of([s["per_class"][k][c] for s in ours]) if ours else None
        b = mean_of([s["per_class"][k][c] for s in ref]) if ref else None
        return f"{num(a, nd)}\\,({num(b, nd)})"

    rows = [[cl] + [cell(i, k, nd) for k, nd in keys] for i, cl in enumerate(SUPER)]
    macro = []
    for k, nd in keys:
        a = mean_of([np.mean(s["per_class"][k]) for s in ours]) if ours else None
        b = mean_of([np.mean(s["per_class"][k]) for s in ref]) if ref else None
        macro.append(f"{num(a, nd)}\\,({num(b, nd)})")
    rows.append([r"Mean of classes"] + macro)
    write(out / "tab_perclass.tex", tabular("@{}lccccc@{}", ["Class", "AUPRC", r"ECE$\downarrow$", "DOR", "MCC",
                                                             r"$\kappa$"], rows, midrules=(len(SUPER),)))


def t_ptbdiag(R, out):
    names = CLS_THIS + OURS_CLS
    g = lambda key: (lambda n: [s["fold_mean"].get(key) for s in R.seeds("ptbdiag", n)])
    keys = ["acc", "f1", "auc", "mcc", "kappa"]
    body = columns_marked(names, [g(k) for k in keys], [True] * 5, [3] * 5)
    rows = [[DISPLAY[n]] + c for n, c in zip(names, body)]
    write(out / "tab_ptbdiag.tex", tabular("@{}lccccc@{}", ["Model", "ACC", "Macro-F1", "AUC", "MCC", r"$\kappa$"],
                                           rows, midrules=(len(names) - 2,)))


def _cpd_vals(R, name, key, where="test_sets", seeds=None):
    vals = []
    for s in R.seeds("cpd", name):
        if seeds is not None and s.get("seed") not in seeds:
            continue
        if where == "test_sets":
            vals += [t.get(key) for t in s.get("test_sets", [])]
        elif where in s and s[where] is not None:
            vals.append(s[where].get(key))
    return vals


def t_cpd(R, out):
    names = CPD_METHODS
    spec = [("precision", True, 3, "test_sets"), ("recall", True, 3, "test_sets"), ("f1", True, 3, "test_sets"),
            ("auprc", True, 3, "test_sets"), ("recall_at_fah10", True, 3, "test_sets"),
            ("delay_mean", False, 2, "test_sets"), ("fa_per_hour", False, 1, "test_sets"),
            ("nuisance_fa_rate", False, 3, "test_sets"), ("artifact_fa_rate", False, 3, "control")]
    getters = [(lambda k, w: (lambda n: _cpd_vals(R, n, k, w)))(k, w) for k, _, _, w in spec]
    body = columns_marked(names, getters, [h for _, h, _, _ in spec], [nd for _, _, nd, _ in spec])
    rows = [[DISPLAY[n]] + c for n, c in zip(names, body)]
    write(out / "tab_cpd.tex", tabular("@{}lccccccccc@{}", ["Method", "PREC", "REC", "F1", "AUPRC",
                                                            r"REC@10", r"$T_d$ (s)$\downarrow$", r"FA/h$\downarrow$",
                                                            r"NFA$\downarrow$", r"AFA$\downarrow$"],
                                       rows, midrules=(len(names) - 1,)))


def t_cpd_real(R, out):
    names = CPD_METHODS
    sp = lambda k: (lambda n: [s.get("same_patient", {}).get(k) for s in R.seeds("cpd", n) if s.get("same_patient")])
    st = lambda k: (lambda n: [s.get("staff3", {}).get("cp", {}).get(k) for s in R.seeds("cpd", n) if s.get("staff3")])
    sfa = lambda n: [s.get("staff3", {}).get("fa_per_hour_total") for s in R.seeds("cpd", n) if s.get("staff3")]
    sw = lambda n: [s.get("staff3", {}).get("within", {}).get("recall") for s in R.seeds("cpd", n) if s.get("staff3")]
    s3 = lambda k: (lambda n: [s.get("staff3", {}).get(k) for s in R.seeds("cpd", n) if s.get("staff3")])
    # STAFF III delay and FA/h are not ranked: a detector that rarely alarms has a low FA/h and, when an alarm falls
    # in the window by chance, a short delay, so neither is a merit on its own (recall and REC@10 are ranked)
    body = columns_marked(names, [sp("f1"), sp("delay_mean"), st("recall"), sw, st("delay_median"), sfa,
                                  s3("chance_recall"), s3("recall_at_fah10")],
                          [True, False, True, True, None, None, None, True], [3, 2, 3, 3, 1, 2, 3, 3])
    for n, c in zip(names, body):   # a delay is undefined (not missing) when the detector found no change point
        has = {1: any(s.get("same_patient") for s in R.seeds("cpd", n)),
               4: any(s.get("staff3") for s in R.seeds("cpd", n))}
        for j, ok in has.items():
            if ok and c[j] == TBD:
                c[j] = "--"
    rows = [[DISPLAY[n]] + c for n, c in zip(names, body)]
    header = (r" & \multicolumn{2}{c}{Same patient (PTB-XL)} & \multicolumn{6}{c}{STAFF III, zero-shot} \\"
              "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-9}" "\n"
              r"Method & F1 & $T_d$ (s)$\downarrow$ & REC & REC$_{\mathrm{within}}$ & Median $T_d$ (s)$\downarrow$ & "
              r"FA/h$\downarrow$ & REC$_{\mathrm{chance}}$ & REC@10")
    write(out / "tab_cpd_real.tex", tabular("@{}lcccccccc@{}", [header], rows, midrules=(len(names) - 1,)))


def t_confounds(R, out):
    """Sensitivity to the non-pathological changes created by splicing recordings: patient change (NFA), the splice
    itself (AFA) and the junction treatment, compared on test set 0 at the calibrated thresholds: main (harmonised,
    decaying alignment offset), step (the same junctions with a constant offset, i.e. a sustained baseline step, as
    in the preliminary benchmark) and naive (non-harmonised splices of the same records)."""
    names = CPD_METHODS
    h = lambda k: (lambda n: [s["test_sets"][0].get(k) for s in R.seeds("cpd", n) if s.get("test_sets")])
    v = lambda sec, k: (lambda n: [s[sec].get(k) for s in R.seeds("cpd", n) if s.get(sec)])
    body = columns_marked(names, [lambda n: _cpd_vals(R, n, "nuisance_fa_rate"),
                                  lambda n: _cpd_vals(R, n, "artifact_fa_rate", "control"),
                                  h("f1"), v("const_offset", "f1"), v("naive_splice", "f1"),
                                  h("fa_per_hour"), v("const_offset", "fa_per_hour"), v("naive_splice", "fa_per_hour")],
                          [False, False, True, None, None, None, None, None], [3, 3, 3, 3, 3, 1, 1, 1])
    rows = [[DISPLAY[n]] + c for n, c in zip(names, body)]
    header = (r" & & & \multicolumn{3}{c}{F1 (test set 0)} & \multicolumn{3}{c}{FA/h (test set 0)} \\" "\n"
              r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}" "\n"
              r"Method & NFA$\downarrow$ & AFA$\downarrow$ & main & step & naive & main & step & naive")
    write(out / "tab_confounds.tex", tabular("@{}lcccccccc@{}", [header], rows, midrules=(len(names) - 1,)))


def _ablation_seeds(R):
    """Seeds of the ablation runs; the full model is compared on the same seeds."""
    s = {r.get("seed") for v, _ in ABLATIONS if v != "full" for r in R.seeds("cpd", f"clt_{v}")}
    return s or None


def t_ablation(R, out):
    rows, means = [], {}
    sd = _ablation_seeds(R)
    cv = lambda v, k, w="test_sets": _cpd_vals(R, f"clt_{v}", k, w, seeds=sd if v == "full" else None)
    getters = [
        lambda v: cv(v, "f1"), lambda v: cv(v, "auprc"), lambda v: cv(v, "delay_mean"), lambda v: cv(v, "fa_per_hour"),
        lambda v: cv(v, "nuisance_fa_rate"), lambda v: cv(v, "artifact_fa_rate", "control"),
        lambda v: [s.get("macro_auc") for s in R.seeds("cls", f"clt_{v}_lgbm", "ml")],     # LightGBM probe
        lambda v: [s.get("acc") for s in R.seeds("cls", f"clt_{v}_lgbm", "sl")],
    ]
    nds = [3, 3, 2, 1, 3, 3, 3, 3]
    for v, lab in ABLATIONS:
        rows.append([lab] + [pm(g(v), nd) for g, nd in zip(getters, nds)])
    write(out / "tab_ablation.tex", tabular("@{}lcccccccc@{}", ["Variant", "F1", "AUPRC", r"$T_d$ (s)", "FA/h", "NFA",
                                                                 "AFA", r"AUC$_{\mathrm{ML}}$", r"ACC$_{\mathrm{SL}}$"],
                                            rows, midrules=(1,)))


def t_streams(R, out):
    rows = []
    for st in STREAM_SETS:
        name = ("clt_full" if st == ("td", "cls", "att", "graph") else "clt_full_s-" + "+".join(st)) + "_lgbm"
        lab = " + ".join({"td": r"$F_{td}$", "cls": r"$F_{cls}$", "att": r"$F_{att}$", "graph": r"$F_{graph}$"}[s] for s in st)
        ml = [s.get("macro_auc") for s in R.seeds("cls", name, "ml")]
        sl = R.seeds("cls", name, "sl")
        rows.append([lab, pm(ml), pm([s.get("acc") for s in sl]), pm([s.get("f1") for s in sl]),
                     pm([s.get("mcc") for s in sl])])
    write(out / "tab_streams.tex", tabular("@{}lcccc@{}", ["Feature streams", r"Macro-AUC (ML)", "ACC (SL)",
                                                           "Macro-F1 (SL)", "MCC (SL)"], rows,
                                           midrules=(4, len(STREAM_SETS) - 1)))


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def attention_groups(meta):
    return [("NORM", (meta["label_key"] == "NORM").values),
            ("Inferior MI", ((meta["label_key"] == "MI") & (meta["sub_IMI"] == 1) & (meta["sub_AMI"] == 0)).values),
            ("Anterior MI", ((meta["label_key"] == "MI") & (meta["sub_AMI"] == 1) & (meta["sub_IMI"] == 0)).values),
            ("CLBBB", ((meta["label_key"] == "CD") & (meta["sub_CLBBB"] == 1)).values),
            ("CRBBB", ((meta["label_key"] == "CD") & (meta["sub_CRBBB"] == 1)).values),
            ("LVH", ((meta["label_key"] == "HYP") & (meta["sub_LVH"] == 1)).values)]


def t_attention(R, cfg, out, B=1000):
    """Attention received by each key lead (cross-lead attention averaged over query leads), difference of each
    diagnostic group from NORM on the test fold, with 95 % bootstrap CIs over records (full model, seed 0)."""
    fpath = Path(cfg.paths.work_dir) / "runs" / "clt" / "full" / "seed0" / "features_ptbxl.npz"
    mpath = Path(cfg.paths.work_dir) / "cache" / "ptbxl" / "meta.pkl"
    rows, full = [], {}
    if not R.ph and fpath.exists() and mpath.exists():
        import pandas as pd

        f = np.load(fpath)
        meta = pd.read_pickle(mpath).iloc[f["rows"]]
        recv = f["G"].mean(1)                                    # (records, key lead)
        te = np.isin(meta["fold"].values, cfg.data.test_folds)
        groups = attention_groups(meta)
        nidx = np.where(te & groups[0][1])[0]
        ref = recv[nidx].mean(0)
        rng = np.random.default_rng(0)
        for g, msk in groups[1:]:
            idx = np.where(te & msk)[0]
            if len(idx) < 10:
                continue
            d = recv[idx].mean(0) - ref
            bs = np.stack([recv[rng.choice(idx, len(idx))].mean(0) - recv[rng.choice(nidx, len(nidx))].mean(0)
                           for _ in range(B)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            full[g] = {"n": int(len(idx)), "leads": LEADS, "diff": d.tolist(), "ci_low": lo.tolist(),
                       "ci_high": hi.tolist(), "norm_value": ref.tolist()}

            def cell(k):
                return f"{LEADS[k]} ${d[k]:+.3f}$ (${100 * d[k] / ref[k]:+.0f}$\\%)"
            up = [k for k in np.argsort(-d) if lo[k] > 0][:3]
            dn = [k for k in np.argsort(d) if hi[k] < 0][:2]
            rows.append([f"{g} ({len(idx)})", "; ".join(cell(k) for k in up) or "--",
                         "; ".join(cell(k) for k in dn) or "--"])
        write(out.parent / "attention_changes.json", json.dumps(full, indent=1))
    if not rows:
        rows = [[TBD, TBD, TBD]]
    write(out / "tab_attention.tex", tabular("@{}lll@{}", ["Group ($n$)", "Largest increases", "Largest decreases"],
                                             rows))


def t_mu(R, out):
    rows = []
    seeds = R.seeds("cpd", "clt_full")
    for mu in ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"):
        g = lambda k: [s.get("mu_sensitivity", {}).get(mu, {}).get(k) for s in seeds]
        rows.append([f"{float(mu):.1f}", pm(g("f1")), pm(g("auprc")), pm(g("delay_mean"), 2), pm(g("fa_per_hour"), 1)])
    write(out / "tab_mu.tex", tabular("@{}lcccc@{}", [r"$\mu$", "F1", "AUPRC", r"$T_d$ (s)", "FA/h"], rows))


def t_compute(R, out):
    L = R.one("latency", "latency.json")
    c = (L or {}).get("cpu_single_thread", {})
    b = (L or {}).get("baselines_cpu_single_thread", {})
    p = (L or {}).get("params", {})
    bp = (L or {}).get("baseline_params", {})
    ms = lambda d: TBD if not d else f"{d['median_ms']:.2f} ({d['p95_ms']:.2f})"
    k = lambda v: TBD if v is None else (f"{v / 1e3:.1f}\\,K" if v < 1e6 else f"{v / 1e6:.2f}\\,M")
    rows = [
        ["R-peak detection (10-s buffer) + IHRAS", "--", ms(c.get("rpeak_10s_buffer"))],
        ["HTFP spectral mask", k(p.get("htfp")), ms(c.get("htfp"))],
        ["RSSA denoising (12 leads)", "--", ms(c.get("rssa"))],
        ["CLT encoder + decoder + change score", k(p.get("clt_total")), ms(c.get("full_forward_with_target"))],
        [r"\emph{Online path per segment (sum)}", k(p.get("total")),
         TBD if not c.get("online_path_ms") else f"{c['online_path_ms']:.2f}"],
        ["Triggered feature extraction (4 streams)", "--", ms(c.get("triggered_features"))],
        ["SkEnsem prediction", "--", ms(c.get("skensem_predict"))],
        ["CUSUM / BOCPD / RuLSIF (per window)", "--",
         " / ".join(ms(b.get(x)) for x in ("cusum", "bocpd", "rulsif")) if b else TBD],
        ["TIRE / LSTM-AE / TS-CP$^2$ / OCAE-CPD", " / ".join(k(bp.get(x)) for x in ("tire", "lstm_ae", "tscp2", "ocae")) if bp else TBD,
         " / ".join(ms(b.get(x)) for x in ("tire", "lstm_ae", "tscp2", "ocae")) if b else TBD],
    ]
    write(out / "tab_compute.tex", tabular("@{}p{6.2cm}cc@{}", ["Stage / method", "Params",
                                                                 "CPU latency, ms: median (p95)"], rows, midrules=(4, 7)))


def t_robust(R, out):
    methods = ["clt_full", "clt_no_htfp", "clt_temporal_only", "cusum", "bocpd", "tscp2", "ocae"]
    disp = dict(DISPLAY, clt_no_htfp=r"CL-TransCPD w/o HTFP", clt_temporal_only=r"CL-TransCPD temporal-only")
    rows = []
    for m in methods:
        seeds = R.seeds("robustness", m)
        cells = []
        for kind in ("bw", "em", "ma"):
            for snr in ("12", "0"):
                cells.append(pm([s["noise"][kind][snr]["f1"] for s in seeds if kind in s.get("noise", {})
                                 and snr in s["noise"][kind]]))
        rows.append([disp[m]] + cells)
    write(out / "tab_robust.tex", tabular("@{}lcccccc@{}", ["Method", "BW 12\\,dB", "BW 0\\,dB", "EM 12\\,dB",
                                                            "EM 0\\,dB", "MA 12\\,dB", "MA 0\\,dB"], rows))


def t_data(R, out):
    px = R.one("data", "ptbxl_summary.json") or {}
    pd_ = R.one("data", "ptbdiag_summary.json") or {}
    s3 = R.one("data", "staff3_summary.json") or {}
    ss = R.one("data", "streams_summary.json") or {}
    g = lambda d, *ks: (lambda v: TBD if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.1f}"))(
        _dig(d, ks))
    rows = [
        [r"PTB-XL~\cite{ptbxl2020}", g(px, "n_records"), g(px, "n_patients"),
         "; ".join(f"{c} {g(px, 'label_counts', c)}" for c in SUPER) if px else TBD],
        [r"PTB-XL single-label subset", g(px, "n_single"), "--",
         "; ".join(f"{c} {g(px, 'single_counts', c)}" for c in SUPER) if px else TBD],
        [r"PTB Diagnostic~\cite{bousseljot1995,goldberger2000}", g(pd_, "n_records"), g(pd_, "n_patients"),
         "; ".join(f"{'excluded' if k == 'None' else k} {v}" for k, v in
                   sorted((pd_.get("mapped_records") or {}).items(),
                          key=lambda kv: SUPER.index(kv[0]) if kv[0] in SUPER else len(SUPER))) or TBD],
        [r"STAFF III~\cite{martinez2017staff}", g(s3, "n_cp_streams"), g(s3, "n_patients"),
         f"{g(s3, 'n_within')} within-record, {g(s3, 'n_spliced')} spliced; {g(s3, 'hours_cp')}\\,h" if s3 else TBD],
        [r"PTB-XL-CPD test stream sets (each)", g(ss, "test_set0", "n_streams"), "--",
         f"{g(ss, 'test_set0', 'n_cps')} change points, {g(ss, 'test_set0', 'n_nuisance')} nuisance junctions, "
         f"{g(ss, 'test_set0', 'hours')}\\,h" if ss else TBD],
    ]
    write(out / "tab_data.tex", tabular("@{}p{3.6cm}rrp{7.4cm}@{}", ["Dataset", "Records", "Patients", "Composition"], rows))


def _design(R, v):
    """Design-study result on the current validation streams; files without the stream version predate the
    corrected (decaying-offset) streams and are not used."""
    d = R.one("design", f"{v}_seed0.json")
    return d if d and "offset_decay_s" in d else None


def t_design(R, out):
    """Validation design study (fold 9 only): best (mu, eta) per CCPP variant and score representation."""
    rows = []
    lab_v = {"cand_nopureneg": "transition + nuisance", "cand_pureneg": "+ post-change negative"}
    lab_l = {"prob": "diagnostic probabilities", "cls": "CLS context", "z": "pooled latent $\\mathbf{z}$",
             "both": "$\\mathbf{z}$ + CLS"}
    for v in ("cand_nopureneg", "cand_pureneg"):
        d = _design(R, v)
        for lat in ("prob", "cls", "z", "both"):
            cand = [r for r in (d or {}).get("rows", []) if r["latent"] == lat]
            if not cand:
                rows.append([lab_v[v], lab_l[lat]] + [TBD] * 7)
                continue
            b = max(cand, key=lambda r: (r["val_auprc"] or 0))
            rows.append([lab_v[v], lab_l[lat], f"{b['mu']:.2f}", f"{b['eta']:.2f}", num(b["val_auprc"]), num(b["f1"]),
                         num(b["fa_per_hour"], 1), num(b["nuisance_fa_rate"]), num(b["artifact_fa_rate"])])
    write(out / "tab_design.tex", tabular("@{}llccccccc@{}", ["CCPP negatives", "Score representation", r"$\mu$",
                                                             r"$\eta$", "AUPRC", "F1", "FA/h", "NFA", "AFA"],
                                          rows, midrules=(4,)))


def _dig(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _seed_avg_preds(R, name, task):
    """Seed-averaged test probabilities (aligned on the test rows of the first seed)."""
    if R.ph:
        return None
    d = R.root / "cls" / name / task
    files = sorted(d.glob("seed*_pred.npz"))
    if not files:
        return None
    zs = [np.load(f, allow_pickle=True) for f in files]
    rows = zs[0]["rows"]
    ps = [z["p"] for z in zs if np.array_equal(z["rows"], rows)]
    return {"rows": rows, "y": zs[0]["y"], "p": np.mean(ps, axis=0), "groups": zs[0]["groups"]}


def _align(a, b):
    common, ia, ib = np.intersect1d(a["rows"], b["rows"], return_indices=True)
    return a["y"][ia], a["p"][ia], b["p"][ib], a["groups"][ia]


def _per_stream(R, name):
    arrs = [np.asarray(s["per_stream_set0"], dtype=float) for s in R.seeds("cpd", name) if s.get("per_stream_set0")]
    if not arrs:
        return None
    n = min(len(a) for a in arrs)
    return np.sum([a[:n] for a in arrs], axis=0)


def _f1(_, C):
    tp, fn, fp = C[:, 0].sum(), C[:, 1].sum(), C[:, 2].sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _fah(_, C):
    return C[:, 2].sum() / max(C[:, 3].sum(), 1e-9)


def t_significance(R, cfg, out):
    from sklearn.metrics import roc_auc_score
    from clcpd.eval.stats import holm, paired_bootstrap

    B = int(cfg.eval.bootstrap)
    auc = lambda y, p: roc_auc_score(y, p, average="macro")
    acc = lambda y, p: float((p.argmax(1) == y).mean())
    fmt = lambda r: TBD if r is None else f"{r[0]:+.3f} [{r[1][0]:+.3f}, {r[1][1]:+.3f}]"

    def block(names, tests):
        res = {n: [None] * len(tests) for n in names}
        for j, test in enumerate(tests):
            for n in names:
                res[n][j] = test(n)
            pv = [res[n][j][2] for n in names if res[n][j] is not None]
            adj = iter(holm(pv)) if pv else iter([])
            for n in names:
                if res[n][j] is not None:
                    res[n][j] = (*res[n][j][:2], next(adj))
        return res

    def cls_test(task, metric):
        def f(n):
            a, b = _seed_avg_preds(R, "clt_full", task), _seed_avg_preds(R, n, task)
            if a is None or b is None:
                return None
            y, pa, pb, g = _align(a, b)
            return paired_bootstrap(metric, y, pa, pb, g, B=B, seed=0)
        return f

    def cpd_test(metric):
        def f(n):
            a, b = _per_stream(R, "clt_full"), _per_stream(R, n)
            if a is None or b is None or len(a) != len(b):
                return None
            return paired_bootstrap(metric, np.zeros(len(a)), a, b, np.arange(len(a)), B=B, seed=0)
        return f

    cls_names = [n for n in CLS_THIS]
    cpd_names = [n for n in CPD_METHODS if n != "clt_full"]
    rc = block(cls_names, [cls_test("ml", auc), cls_test("sl", acc)])
    rd = block(cpd_names, [cpd_test(_f1), cpd_test(_fah)])
    pfmt = lambda r: TBD if r is None else (r"$<$0.001" if r[2] < 1e-3 else f"{r[2]:.3f}")
    rows = [r"\multicolumn{5}{@{}l}{\emph{Classification (PTB-XL test fold): CL-TransCPD minus baseline}}"]
    rows[0] += r" \\"
    rows.append(r" & \multicolumn{2}{c}{Macro-AUC (multi-label)} & \multicolumn{2}{c}{ACC (single-label)} \\")
    for n in cls_names:
        rows.append([DISPLAY[n], fmt(rc[n][0]), pfmt(rc[n][0]), fmt(rc[n][1]), pfmt(rc[n][1])])
    rows.append(r"\midrule \multicolumn{5}{@{}l}{\emph{Change-point detection (test stream set 0): CL-TransCPD minus baseline}} \\")
    rows.append(r" & \multicolumn{2}{c}{F1} & \multicolumn{2}{c}{FA/h} \\")
    for n in cpd_names:
        rows.append([DISPLAY[n], fmt(rd[n][0]), pfmt(rd[n][0]), fmt(rd[n][1]), pfmt(rd[n][1])])
    js = lambda r: None if r is None else {"delta": float(r[0]), "ci": [float(r[1][0]), float(r[1][1])],
                                           "p_holm": float(r[2])}
    write(out.parent / "significance.json",                # read by macros() for the text
          json.dumps({"cls": {n: {"ml_auc": js(rc[n][0]), "sl_acc": js(rc[n][1])} for n in cls_names},
                      "cpd": {n: {"f1": js(rd[n][0]), "fah": js(rd[n][1])} for n in cpd_names}}, indent=1))
    write(out / "tab_significance.tex",
          tabular("@{}lcccc@{}", ["Baseline", r"$\Delta$ [95\% CI]", r"$p_{\mathrm{Holm}}$", r"$\Delta$ [95\% CI]",
                                  r"$p_{\mathrm{Holm}}$"], rows))


# ----------------------------------------------------------------------------- macros
def _camel(v):
    return "".join(w.capitalize() for w in v.split("_"))


def _names(ns):
    """'A', 'A and B', 'A, B and C' from result names (display short names)."""
    ns = [SHORT.get(n, n).replace("²", "$^2$") for n in ns]
    return TBD if not ns else ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]


def _text_macros(R, cfg, m, path=None):
    """Further numbers quoted in the Results and Discussion (added to m)."""
    fm = lambda v, nd=3: num(v, nd)
    # cross-lead attention: largest increase (absolute and relative to NORM) with a CI excluding zero
    ap = path.parent / "attention_changes.json" if path is not None else None
    att = json.loads(ap.read_text(encoding="utf-8")) if ap is not None and ap.exists() and not R.ph else {}
    inc = [(g["diff"][k], 100 * g["diff"][k] / g["norm_value"][k]) for g in att.values() for k in range(len(g["diff"]))
           if g["ci_low"][k] > 0 and g["norm_value"][k] > 0]
    m["ResAttnMaxDiff"] = fm(max(d for d, _ in inc)) if inc else TBD
    m["ResAttnMaxRelPct"] = f"{max(r for _, r in inc):.0f}" if inc else TBD
    pct = lambda v: TBD if v is None or not np.isfinite(v) else f"{100 * v:.1f}\\%"
    base = [n for n in CPD_METHODS if n != "clt_full"]
    sd = _ablation_seeds(R)
    # ablation: every variant on the ablation seeds (the full model restricted to the same seeds)
    for v, _ in ABLATIONS:
        name, k = f"clt_{v}", _camel(v)
        cv = lambda key: mean_of(_cpd_vals(R, name, key, seeds=sd if v == "full" else None))
        m[f"ResAbl{k}FOne"], m[f"ResAbl{k}Nfa"] = fm(cv("f1")), fm(cv("nuisance_fa_rate"))
        m[f"ResAbl{k}Fah"], m[f"ResAbl{k}Delay"] = fm(cv("fa_per_hour"), 1), fm(cv("delay_mean"), 2)
        m[f"ResAbl{k}Auc"] = fm(mean_of([s.get("macro_auc") for s in R.seeds("cls", f"{name}_lgbm", "ml")]))
    # strongest unsupervised CPD baseline (by F1): alarm burden and nuisance alarms (as ResCPDbestBaseName)
    f1 = {n: mean_of(_cpd_vals(R, n, "f1")) for n in UNSUP_CPD}
    f1 = {n: v for n, v in f1.items() if v is not None}
    bn = max(f1, key=f1.get) if f1 else None
    m["ResCPDfahBestBase"] = fm(mean_of(_cpd_vals(R, bn, "fa_per_hour")), 1) if bn else TBD
    m["ResCPDnfaBestBase"] = pct(mean_of(_cpd_vals(R, bn, "nuisance_fa_rate"))) if bn else TBD
    m["ResCPDauprcBestBase"] = fm(mean_of(_cpd_vals(R, bn, "auprc"))) if bn else TBD

    # test set 0 under the three junction treatments (main, constant offset, naive splices)
    def t0(n, sec, key):
        return mean_of([(s["test_sets"][0] if sec == "main" else s[sec]).get(key) for s in R.seeds("cpd", n)
                        if s.get("test_sets") and (sec == "main" or s.get(sec))])
    m["ResStepOursMain"], m["ResStepOursStep"] = fm(t0("clt_full", "main", "f1")), fm(t0("clt_full", "const_offset", "f1"))
    m["ResNaiveOurs"] = fm(t0("clt_full", "naive_splice", "f1"))
    m["ResSupMainFOne"], m["ResSupStepFOne"] = fm(t0("sup_resnet", "main", "f1")), fm(t0("sup_resnet", "const_offset", "f1"))
    m["ResSupNaiveFOne"] = fm(t0("sup_resnet", "naive_splice", "f1"))
    for sec, key in (("const_offset", "Step"), ("naive_splice", "Naive")):
        gain = {n: t0(n, sec, "f1") - t0(n, "main", "f1") for n in base
                if t0(n, sec, "f1") is not None and t0(n, "main", "f1") is not None}
        m[f"Res{key}GainNames"] = _names([n for n in base if gain.get(n, 0.0) > 0.02]) if gain else TBD
        top = max(gain, key=gain.get) if gain else None
        m[f"Res{key}MaxName"] = _names([top]) if top else TBD
        m[f"Res{key}MaxMain"] = fm(t0(top, "main", "f1")) if top else TBD
        m[f"Res{key}MaxCopy"] = fm(t0(top, sec, "f1")) if top else TBD
    # STAFF III: recall that random alarms reach at 10 FA/h within the tolerance, and detectors above it
    t3 = float(cfg.streams.staff3_tolerance_s) if cfg is not None else 120.0
    chance10 = 1.0 - np.exp(-10.0 * t3 / 3600.0)
    m["ResStaffChanceTen"] = pct(chance10)
    s3 = lambda n, *k: mean_of([_dig(s.get("staff3", {}), list(k)) for s in R.seeds("cpd", n) if s.get("staff3")])
    have = [n for n in base if s3(n, "recall_at_fah10") is not None]
    above = [n for n in have if s3(n, "recall_at_fah10") > chance10]
    m["ResStaffAboveChanceNames"] = TBD if not have else (_names(above) if above else "none of the baselines")
    m["ResStaffSplicedRecall"] = pct(s3("clt_full", "spliced", "recall"))
    m["ResStaffRecFiveOurs"] = pct(s3("clt_full", "recall_at_fah5"))
    # same-patient streams
    sp = lambda n, k: mean_of([s.get("same_patient", {}).get(k) for s in R.seeds("cpd", n) if s.get("same_patient")])
    m["ResSameFOneOurs"], m["ResSameRecOurs"] = fm(sp("clt_full", "f1")), fm(sp("clt_full", "recall"))
    m["ResSameFahOurs"] = fm(sp("clt_full", "fa_per_hour"), 1)
    sb = [(sp(n, "f1"), n) for n in base if sp(n, "f1") is not None]
    m["ResSameFOneBestBase"] = fm(max(sb)[0]) if sb else TBD
    m["ResSameBestBaseName"] = _names([max(sb)[1]]) if sb else TBD

    # heart-rate strata: lowest and highest bins, IHRAS versus fixed windows (same seeds)
    def hr(n, seeds=None):
        ss = [s for s in R.seeds("cpd", n) if s.get("hr") and (seeds is None or s.get("seed") in seeds)]
        if not ss:
            return None, None
        k = len(ss[0]["hr"])
        return [float(np.nanmean([s["hr"][i]["recall"] for s in ss])) for i in range(k)], [b["n"] for b in ss[0]["hr"]]
    for n, key, seeds in (("clt_full", "Ours", sd), ("clt_fixed_window", "Fixed", None)):
        rec, cnt = hr(n, seeds)
        m[f"ResHrRec{key}Low"], m[f"ResHrRec{key}High"] = (fm(rec[0]), fm(rec[-1])) if rec else (TBD, TBD)
        if key == "Ours":
            m["ResHrNLow"], m["ResHrNHigh"] = (str(cnt[0]), str(cnt[-1])) if cnt else (TBD, TBD)
    # tolerance, mu, triggered diagnosis, selected threshold percentile
    st = R.seeds("cpd", "clt_full")
    tol = lambda t: mean_of([v.get("f1") for s in st for k, v in (s.get("tolerance") or {}).items() if float(k) == t])
    m["ResTolTwoFOne"], m["ResTolFiveFOne"] = fm(tol(2.0)), fm(tol(5.0))
    m["ResMuZeroFOne"] = fm(mean_of([_dig(s, ["mu_sensitivity", "0.0", "f1"]) for s in st]))
    tn = mean_of([_dig(s, ["triggered", "n"]) for s in st])
    m["ResTrigN"] = f"{tn:.0f}" if tn is not None else TBD
    lam = sorted({float(s["calibration"]["lambda"]) for s in st if s.get("calibration")})
    m["ResLambdaSel"] = TBD if not lam else (f"{lam[0]:g}" if len(lam) == 1 else f"{lam[0]:g}--{lam[-1]:g}")
    # feature streams (LightGBM probe)
    sa = lambda nm: mean_of([s.get("macro_auc") for s in R.seeds("cls", nm, "ml")])
    m["ResStreamTdAuc"], m["ResStreamClsAuc"] = fm(sa("clt_full_s-td_lgbm")), fm(sa("clt_full_s-cls_lgbm"))
    m["ResStreamAllAuc"] = fm(sa("clt_full_lgbm"))
    # PTB-XL: the generic Transformer baselines and the best single-label accuracy among the baselines
    mlv = lambda n: mean_of([s.get("macro_auc") for s in R.seeds("cls", n, "ml")])
    m["ResMLaucTransformer"], m["ResMLaucPatchtst"] = fm(mlv("transformer")), fm(mlv("patchtst"))
    sa_ = [(mean_of([s.get("acc") for s in R.seeds("cls", n, "sl")]), n) for n in CLS_THIS]
    sa_ = [x for x in sa_ if x[0] is not None]
    m["ResSLaccBestBase"] = fm(max(sa_)[0]) if sa_ else TBD
    m["ResSLaccBestBaseName"] = _names([max(sa_)[1]]) if sa_ else TBD
    # PTB Diagnostic
    pdv = lambda n, k: mean_of([s["fold_mean"].get(k) for s in R.seeds("ptbdiag", n)])
    m["ResPtbdAccOurs"], m["ResPtbdFOneOurs"] = fm(pdv("clt_full", "acc")), fm(pdv("clt_full", "f1"))
    m["ResPtbdAucOurs"], m["ResPtbdMccOurs"] = fm(pdv("clt_full", "auc")), fm(pdv("clt_full", "mcc"))
    pbase = CLS_THIS
    for k, key in (("auc", ""), ("acc", "Acc"), ("mcc", "Mcc")):
        pb = [(pdv(n, k), n) for n in pbase if pdv(n, k) is not None]
        m[f"ResPtbd{key or 'Auc'}BestBase"] = fm(max(pb)[0]) if pb else TBD
        m[f"ResPtbd{key}BestBaseName"] = _names([max(pb)[1]]) if pb else TBD
    # per-class AUPRC on the single-label subset (hardest and easiest class)
    pc = [s["per_class"]["auprc"] for s in R.seeds("cls", "clt_full", "sl") if s.get("per_class")]
    a = np.mean(pc, axis=0) if pc else None
    m["ResPerclassMinName"] = SUPER[int(np.argmin(a))] if pc else TBD
    m["ResPerclassMinAuprc"] = fm(float(np.min(a))) if pc else TBD
    m["ResPerclassMaxName"] = SUPER[int(np.argmax(a))] if pc else TBD
    m["ResPerclassMaxAuprc"] = fm(float(np.max(a))) if pc else TBD
    # validation design study (re-run on the current validation streams by the assets job)
    lab = {"prob": "diagnostic probabilities", "cls": "CLS context", "z": "pooled latent",
           "both": "pooled latent and CLS context"}
    for v, key in (("cand_pureneg", "Pureneg"), ("cand_nopureneg", "NoPureneg")):
        rows = (_design(R, v) or {}).get("rows", [])
        ok = [r for r in rows if r.get("val_auprc") is not None and np.isfinite(r["val_auprc"])]
        b = max(ok, key=lambda r: r["val_auprc"]) if ok else None
        m[f"ResDesign{key}BestAuprc"] = fm(b["val_auprc"]) if b else TBD
        if key == "Pureneg":
            m["ResDesignBestConfig"] = (f"{lab.get(b['latent'], b['latent'])}, $\\mu={b['mu']:g}$, "
                                        f"$\\eta={b['eta']:g}$") if b else TBD
            fx = [r for r in ok if abs(r["mu"] - 1.0) < 1e-9 and abs(r["eta"] - 0.1) < 1e-9]
            fx = max(fx, key=lambda r: r["val_auprc"]) if fx else None
            m["ResDesignFixedAuprc"] = fm(fx["val_auprc"]) if fx else TBD
            m["ResDesignFixedRank"] = (f"{1 + sum(r['val_auprc'] > fx['val_auprc'] for r in ok)} of {len(ok)}"
                                       if fx else TBD)
    # robustness: CPD F1 without noise (no lead dropped), at 0 dB (mean of the noise types) and with 9 leads dropped
    rb = lambda n: R.seeds("robustness", n)
    rz = lambda n: mean_of([np.mean([s["noise"][k]["0"]["f1"] for k in s["noise"] if "0" in s["noise"][k]])
                            for s in rb(n) if s.get("noise")])
    rd = lambda n, k: mean_of([_dig(s, ["dropout", k, "f1"]) for s in rb(n)])
    m["ResRobCleanOurs"], m["ResRobZeroOurs"] = fm(rd("clt_full", "0")), fm(rz("clt_full"))
    m["ResRobZeroNoHtfp"], m["ResRobCleanNoHtfp"] = fm(rz("clt_no_htfp")), fm(rd("clt_no_htfp", "0"))
    rbb = [(rz(n), n) for n in base if rz(n) is not None]
    m["ResRobZeroBestBase"] = fm(max(rbb)[0]) if rbb else TBD
    m["ResRobZeroBestBaseName"] = _names([max(rbb)[1]]) if rbb else TBD
    m["ResDropThreeOurs"], m["ResDropNineOurs"] = fm(rd("clt_full", "3")), fm(rd("clt_full", "9"))
    m["ResDropThreeTemporal"] = fm(rd("clt_temporal_only", "3"))
    db = [(rd(n, "3"), n) for n in base if rd(n, "3") is not None]
    m["ResDropThreeBestBase"] = fm(max(db)[0]) if db else TBD
    m["ResDropThreeBestBaseName"] = _names([max(db)[1]]) if db else TBD
    cr = [s["classification"] for s in rb("clt_full") if s.get("classification")]
    crz = lambda who: mean_of([np.mean([c[t][who] for t in c if t.endswith("_0") and who in c[t]]) for c in cr
                               if any(t.endswith("_0") and who in c[t] for t in c)])
    m["ResRobClsCleanOurs"] = fm(mean_of([_dig(c, ["clean", "clt_full"]) for c in cr]))
    m["ResRobClsZeroOurs"] = fm(crz("clt_full"))
    m["ResRobClsCleanResnet"] = fm(mean_of([_dig(c, ["clean", "resnet1d_wang"]) for c in cr]))
    m["ResRobClsZeroResnet"] = fm(crz("resnet1d_wang"))


def macros(R, path: Path, cfg=None):
    m = {}
    ml = lambda n, k: mean_of([s.get(k) for s in R.seeds("cls", n, "ml")])
    sl = lambda n, k: mean_of([s.get(k) for s in R.seeds("cls", n, "sl")])
    cpd = lambda n, k: mean_of(_cpd_vals(R, n, k))
    fm = lambda v, nd=3: num(v, nd)
    m["ResMLaucOurs"] = fm(ml("clt_full", "macro_auc"))
    m["ResMLaucOursQRSV"] = fm(ml("clt_full_qrsv", "macro_auc"))
    best_ml = max(((ml(n, "macro_auc"), n) for n in CLS_THIS if ml(n, "macro_auc") is not None), default=(None, None))
    m["ResMLaucBestBase"] = fm(best_ml[0])
    m["ResMLbestBaseName"] = SHORT.get(best_ml[1], TBD) if best_ml[1] else TBD
    sg = path.parent / "significance.json"                 # written by t_significance (paired bootstrap, Holm)
    sig = json.loads(sg.read_text(encoding="utf-8")) if sg.exists() and not R.ph else {}

    def sigtxt(r, nd=3):
        if not r or not np.isfinite(r["delta"]) or not np.isfinite(r["p_holm"]):
            return TBD
        p = "<0.001" if r["p_holm"] < 1e-3 else f"={r['p_holm']:.3f}"
        return (f"$\\Delta={r['delta']:+.{nd}f}$, 95\\% CI ${r['ci'][0]:+.{nd}f}$ to ${r['ci'][1]:+.{nd}f}$, "
                f"$p_{{\\mathrm{{Holm}}}}{p}$")
    m["ResSigMLBest"] = sigtxt(_dig(sig, ["cls", best_ml[1], "ml_auc"])) if best_ml[1] else TBD
    m["ResSLaccOurs"] = fm(sl("clt_full", "acc"))
    m["ResSLfOneOurs"] = fm(sl("clt_full", "f1"))
    m["ResSLaucOurs"] = fm(sl("clt_full", "auc"))
    m["ResSLmccOurs"] = fm(sl("clt_full", "mcc"))
    m["ResSLeceOurs"] = fm(sl("clt_full", "ece"))
    m["ResSLaccPrior"] = fm(sl("prior_skensem", "acc"))
    for k, nd in (("f1", 3), ("precision", 3), ("recall", 3), ("auprc", 3), ("delay_mean", 2), ("fa_per_hour", 1)):
        key = {"f1": "fOne", "precision": "Prec", "recall": "Rec", "auprc": "Auprc", "delay_mean": "Delay",
               "fa_per_hour": "Fah"}[k]
        m[f"ResCPD{key}Ours"] = fm(cpd("clt_full", k), nd)
        m[f"ResCPD{key}Prior"] = fm(cpd("ocae", k), nd)
    base = [(cpd(n, "f1"), n) for n in UNSUP_CPD if cpd(n, "f1") is not None]     # strongest unsupervised baseline
    if base:
        bf, bn = max(base)
        m["ResCPDfOneBestBase"], m["ResCPDbestBaseName"] = fm(bf), SHORT[bn].replace("²", "$^2$")
        m["ResCPDdelayBestBase"] = fm(cpd(bn, "delay_mean"), 2)
        m["ResSigCPDBest"] = sigtxt(_dig(sig, ["cpd", bn, "f1"]))
    else:
        m["ResCPDfOneBestBase"] = m["ResCPDbestBaseName"] = m["ResCPDdelayBestBase"] = TBD
        m["ResSigCPDBest"] = TBD
    d_o, d_p = cpd("clt_full", "delay_mean"), cpd("ocae", "delay_mean")
    f_o, f_p = cpd("clt_full", "fa_per_hour"), cpd("ocae", "fa_per_hour")
    m["ResCPDdelayRedPct"] = fm(100 * (d_p - d_o) / d_p, 1) if d_o is not None and d_p else TBD
    m["ResCPDfahRedPct"] = fm(100 * (f_p - f_o) / f_p, 1) if f_o is not None and f_p else TBD
    pct = lambda v: TBD if v is None or not np.isfinite(v) else f"{100 * v:.1f}\\%"
    m["ResCPDafaOurs"] = pct(mean_of(_cpd_vals(R, "clt_full", "artifact_fa_rate", "control")))
    m["ResCPDnfaOurs"] = pct(cpd("clt_full", "nuisance_fa_rate"))
    st = R.seeds("cpd", "clt_full")
    m["ResStaffRecall"] = pct(mean_of([s.get("staff3", {}).get("cp", {}).get("recall") for s in st if s.get("staff3")]))
    m["ResStaffDelay"] = fm(mean_of([s.get("staff3", {}).get("cp", {}).get("delay_median") for s in st if s.get("staff3")]), 1)
    m["ResStaffFah"] = fm(mean_of([s.get("staff3", {}).get("fa_per_hour_total") for s in st if s.get("staff3")]), 2)
    m["ResStaffWithinRecall"] = pct(mean_of([s.get("staff3", {}).get("within", {}).get("recall") for s in st
                                             if s.get("staff3")]))
    m["ResStaffWithinDelay"] = fm(mean_of([s.get("staff3", {}).get("within", {}).get("delay_median") for s in st
                                           if s.get("staff3")]), 1)
    s3r = lambda n, *k: mean_of([_dig(s.get("staff3", {}), list(k)) for s in R.seeds("cpd", n) if s.get("staff3")])
    sb = [(s3r(n, "cp", "recall"), n) for n in CPD_METHODS if n != "clt_full" and s3r(n, "cp", "recall") is not None]
    if sb:
        _, bn = max(sb)
        m["ResStaffBestBaseName"] = SHORT[bn].replace("²", "$^2$")
        m["ResStaffBestBaseRecall"] = pct(s3r(bn, "cp", "recall"))
        m["ResStaffBestBaseWithin"] = pct(s3r(bn, "within", "recall"))
        m["ResStaffBestBaseFah"] = fm(s3r(bn, "fa_per_hour_total"), 2)
    else:
        m["ResStaffBestBaseName"] = m["ResStaffBestBaseRecall"] = m["ResStaffBestBaseWithin"] = TBD
        m["ResStaffBestBaseFah"] = TBD
    m["ResStaffRecTenOurs"] = pct(mean_of([s.get("staff3", {}).get("recall_at_fah10") for s in st if s.get("staff3")]))
    m["ResStaffChanceOurs"] = pct(mean_of([s.get("staff3", {}).get("chance_recall") for s in st if s.get("staff3")]))
    if sb:
        m["ResStaffBestBaseChance"] = pct(s3r(bn, "chance_recall"))
        m["ResStaffBestBaseRecTen"] = pct(s3r(bn, "recall_at_fah10"))
    else:
        m["ResStaffBestBaseChance"] = m["ResStaffBestBaseRecTen"] = TBD
    rt3 = [(s3r(n, "recall_at_fah10"), n) for n in CPD_METHODS if n != "clt_full" and s3r(n, "recall_at_fah10") is not None]
    m["ResStaffRecTenBestBase"] = pct(max(rt3)[0]) if rt3 else TBD
    m["ResStaffRecTenBestBaseName"] = SHORT[max(rt3)[1]].replace("²", "$^2$") if rt3 else TBD
    m["ResCPDRecTenOurs"] = fm(cpd("clt_full", "recall_at_fah10"))
    rt = [(cpd(n, "recall_at_fah10"), n) for n in CPD_METHODS if n != "clt_full" and cpd(n, "recall_at_fah10") is not None]
    m["ResCPDRecTenBestBase"] = fm(max(rt)[0]) if rt else TBD
    m["ResCPDRecTenBestBaseName"] = SHORT[max(rt)[1]].replace("²", "$^2$") if rt else TBD
    # supervised reference detector (ResNet-CPD), trained with the diagnostic labels like CL-TransCPD
    sv, svs = "sup_resnet", R.seeds("cpd", "sup_resnet")
    for k, nd, key in (("f1", 3, "FOne"), ("precision", 3, "Prec"), ("recall", 3, "Rec"), ("auprc", 3, "Auprc"),
                       ("delay_mean", 2, "Delay"), ("fa_per_hour", 1, "Fah"), ("recall_at_fah10", 3, "RecTen")):
        m[f"ResSup{key}"] = fm(cpd(sv, k), nd)
    m["ResSupNfa"] = pct(cpd(sv, "nuisance_fa_rate"))
    m["ResSupAfa"] = pct(mean_of(_cpd_vals(R, sv, "artifact_fa_rate", "control")))
    m["ResSigSup"] = sigtxt(_dig(sig, ["cpd", sv, "f1"]))
    m["ResSigSupFah"] = sigtxt(_dig(sig, ["cpd", sv, "fah"]), 1)
    m["ResSupStaffRecall"] = pct(s3r(sv, "cp", "recall"))
    m["ResSupStaffFah"] = fm(s3r(sv, "fa_per_hour_total"), 2)
    m["ResSupStaffRecTen"] = pct(s3r(sv, "recall_at_fah10"))
    m["ResSupSameFOne"] = fm(mean_of([s.get("same_patient", {}).get("f1") for s in svs if s.get("same_patient")]))
    m["ResSupValAuc"] = fm(mean_of([_dig(s, ["train_info", "val_macro_auc_crops"]) for s in svs]))
    rep = [s.get("rep_selected") for s in svs if s.get("rep_selected")]
    rn = {"prob": "the superclass probabilities", "feat": "the penultimate features"}
    m["ResSupRepSel"] = TBD if not rep else (f"{rn[rep[0]]} in all {len(rep)} seeds" if len(set(rep)) == 1 else
                                             " and ".join(f"{rn[k]} in {rep.count(k)}" for k in sorted(set(rep)))
                                             + f" of {len(rep)} seeds")
    lat = [s.get("latent_selected") for s in st if s.get("latent_selected")]
    nm = {"prob": "the diagnostic probabilities", "cls": "the CLS context", "z": "the pooled latent",
          "both": "the pooled latent with the CLS context"}
    if not lat:
        m["ResLatentSel"] = TBD
    elif len(set(lat)) == 1:
        m["ResLatentSel"] = f"{nm.get(lat[0], lat[0])} in all {len(lat)} seeds"
    else:
        m["ResLatentSel"] = " and ".join(f"{nm.get(k, k)} in {lat.count(k)}" for k in sorted(set(lat), key=lat.count,
                                                                                            reverse=True)) + \
            f" of {len(lat)} seeds"
    m["ResTrigAuc"] = fm(mean_of([s.get("triggered", {}).get("macro_auc") for s in st]))
    m["ResTrigAcc"] = fm(mean_of([s.get("triggered", {}).get("top1_acc_single") for s in st]))
    L = R.one("latency", "latency.json") or {}
    p = L.get("params", {})
    m["ResParamsTotal"] = TBD if not p else f"{p['total'] / 1e3:.0f}K"
    m["ResParamsEncoder"] = TBD if not p else f"{p['encoder'] / 1e3:.0f}K"
    fl = L.get("flops_clt_per_segment")
    m["ResFlops"] = TBD if not fl else f"{fl / 1e6:.1f}"
    c = L.get("cpu_single_thread", {})
    m["ResLatOnline"] = TBD if not c.get("online_path_ms") else f"{c['online_path_ms']:.1f}"
    m["ResHardware"] = L.get("hardware", TBD).replace("_", r"\_") if L else TBD
    ss = R.one("data", "streams_summary.json") or {}
    m["ResNstreams"] = str(_dig(ss, ["test_set0", "n_streams"])) if ss else TBD
    m["ResNcps"] = str(_dig(ss, ["test_set0", "n_cps"])) if ss else TBD
    m["ResHours"] = f"{_dig(ss, ['test_set0', 'hours']):.1f}" if ss else TBD
    s3 = R.one("data", "staff3_summary.json") or {}
    m["ResStaffStreams"] = str(s3.get("n_cp_streams", TBD)) if s3 else TBD
    m["ResStaffPatients"] = str(s3.get("n_patients", TBD)) if s3 else TBD
    js = R.one("data", "junction_steps.json") or {}      # baseline step at the junctions of test set 0 (SD units)
    for k, key in (("main", "Decay"), ("constant", "Const"), ("naive", "Naive")):
        m[f"ResStep{key}Med"] = fm(_dig(js, [k, "median"]), 2)
        m[f"ResStep{key}Pninety"] = fm(_dig(js, [k, "p90"]), 2)
    _text_macros(R, cfg, m, path)
    unknown = sorted(set(m) - set(MACRO_NAMES))     # every macro needs a placeholder in macros_default.tex
    assert not unknown, f"add to MACRO_NAMES: {unknown}"
    lines = ["% generated by experiments/scripts/make_paper_assets.py - do not edit by hand"]
    lines += [f"\\renewcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    write(path, "\n".join(lines) + "\n")
    return m


def default_macros(path: Path, names):
    lines = ["% placeholder values; overwritten by results/macros.tex"]
    lines += [f"\\providecommand{{\\{k}}}{{\\tbd}}" for k in sorted(names)]
    write(path, "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- figures
def figures(R, cfg, out: Path):
    from clcpd.report import style
    import matplotlib.pyplot as plt

    style.apply()
    out.mkdir(parents=True, exist_ok=True)
    made = []
    # PR curves (test set 0, seed 0 of each method)
    data = {n: (R.seeds("cpd", n) or [None])[0] for n in CPD_METHODS}
    if any(d and d.get("pr_points") for d in data.values()):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.6))
        for n in CPD_METHODS:
            d = data[n]
            if not d or not d.get("pr_points"):
                continue
            pts = sorted((q[0], q[1]) for q in d["pr_points"])
            r = np.array([a for a, _ in pts])
            p = np.maximum.accumulate(np.array([b for _, b in pts])[::-1])[::-1]
            kw = dict(color=style.ENTITY[n], ls=style.LINESTYLE[n], lw=2.0 if n == "clt_full" else 1.2)
            # pr_points belong to test set 0, so the legend shows the AUPRC of that set
            ax.plot(r, p, label=f"{SHORT[n]} ({d['auprc'][0]:.3f})", drawstyle="steps-post", **kw)
            fx, fy = _froc(d["pr_points"])
            ax2.plot(fx, fy, drawstyle="steps-post", **kw)
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="(a) precision-recall")
        ax2.set_xscale("log")
        ax2.axvline(10, color=style.MUTED, lw=0.8, ls="--")
        ax2.set(xlabel="False alarms per hour", ylabel="Recall", ylim=(0, 1.02), xlim=(0.3, 1000),
                title="(b) recall vs. alarm burden")
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), title="Method (AUPRC)",
                   title_fontsize=7.5)
        fig.savefig(out / "fig_cpd_pr.pdf")
        plt.close(fig)
        made.append("fig_cpd_pr.pdf")
    # STAFF III: recall vs. alarm burden with the chance level of random alarms (120-s tolerance)
    s3 = {n: (R.seeds("cpd", n) or [None])[0] for n in CPD_METHODS}
    s3 = {n: d["staff3"]["pr_points"] for n, d in s3.items() if d and d.get("staff3", {}).get("pr_points")}
    if s3:
        fig, ax = plt.subplots(figsize=(5.2, 3.1))
        lam = np.logspace(np.log10(0.3), 3, 200)
        ax.fill_between(lam, 0, 1 - np.exp(-lam * float(cfg.streams.staff3_tolerance_s) / 3600.0),
                        color=style.GRID, lw=0, label="random alarms")
        for n, pts in s3.items():
            fx, fy = _froc(pts)
            ax.plot(fx, fy, color=style.ENTITY[n], ls=style.LINESTYLE[n], lw=2.0 if n == "clt_full" else 1.2,
                    drawstyle="steps-post", label=SHORT[n])
        ax.set_xscale("log")
        ax.axvline(10, color=style.MUTED, lw=0.8, ls="--")
        ax.set(xlabel="False alarms per hour", ylabel="Recall within 120 s", ylim=(0, 1.02), xlim=(0.3, 1000))
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.0, handlelength=2.2)   # outside the axes
        fig.tight_layout()
        fig.savefig(out / "fig_staff3_froc.pdf")
        plt.close(fig)
        made.append("fig_staff3_froc.pdf")
    # detection-delay distributions
    delays = {n: [x for s in R.seeds("cpd", n) for x in s.get("delays", [])] for n in CPD_METHODS}
    if any(delays.values()):
        names = [n for n in CPD_METHODS if delays[n]]
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        bp = ax.boxplot([delays[n] for n in names], notch=True, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": style.INK, "lw": 1.2})
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(style.ENTITY[n])
            patch.set_alpha(0.55)
            patch.set_edgecolor(style.ENTITY[n])
        ax.set_xticks(range(1, len(names) + 1), [SHORT[n] for n in names], rotation=20, ha="right",
                      rotation_mode="anchor")
        ax.set_ylabel("Detection delay $T_d$ (s)")
        ax.grid(axis="x", visible=False)
        fig.savefig(out / "fig_cpd_delay.pdf")
        plt.close(fig)
        made.append("fig_cpd_delay.pdf")
    # heart-rate strata: recall and delay (two panels, no dual axis)
    hr_series = {}
    for n in ["clt_full", "clt_fixed_window", "ocae", "tscp2", "bocpd"]:
        seeds = R.seeds("cpd", n)
        if seeds and seeds[0].get("hr"):
            bins = seeds[0]["hr"]
            hr_series[n] = (bins, [np.nanmean([s["hr"][i]["recall"] for s in seeds]) for i in range(len(bins))],
                            [np.nanmean([s["hr"][i]["delay_mean"] for s in seeds]) for i in range(len(bins))])
    if hr_series:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
        for n, (bins, rec, dl) in hr_series.items():
            labels = [f"{int(b['lo'])}–{int(b['hi'])}" if b["hi"] < 300 else f"≥{int(b['lo'])}" for b in bins]
            labels[0] = f"<{int(bins[0]['hi'])}"
            x = np.arange(len(bins))
            col = style.VARIANT_GRAY if n == "clt_fixed_window" else style.ENTITY[n]
            lab = "CL-TransCPD, fixed window" if n == "clt_fixed_window" else SHORT[n]
            ls = "--" if n == "clt_fixed_window" else style.LINESTYLE.get(n, "-")
            mk = "o" if n == "clt_fixed_window" else style.MARKER.get(n, "o")
            for ax, y in zip(axes, (rec, dl)):
                ax.plot(x, y, color=col, ls=ls, marker=mk, ms=4, lw=2.0 if n == "clt_full" else 1.2, label=lab)
                ax.set_xticks(x, labels)
        counts = [b["n"] for b in next(iter(hr_series.values()))[0]]
        axes[0].set(xlabel="Post-change heart rate (bpm)", ylabel="Recall", ylim=(0, 1.02))
        axes[1].set(xlabel="Post-change heart rate (bpm)", ylabel="Mean delay $T_d$ (s)")
        axes[0].set_title("n per bin: " + ", ".join(map(str, counts)), loc="left", color=style.INK2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
        fig.savefig(out / "fig_cpd_hr.pdf")
        plt.close(fig)
        made.append("fig_cpd_hr.pdf")
    # noise robustness
    rob = {n: R.seeds("robustness", n) for n in ["clt_full", "clt_no_htfp", "cusum", "bocpd", "tscp2", "ocae"]}
    if any(rob.values()):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)
        for ax, kind, title in zip(axes, ("bw", "em", "ma"), ("Baseline wander", "Electrode motion", "Muscle artefact")):
            for n, seeds in rob.items():
                if not seeds:
                    continue
                keys = [set(s["noise"].get(kind, {})) for s in seeds]
                snrs = sorted(set.intersection(*keys), key=float, reverse=True) if keys else []
                if not snrs:
                    continue
                y = [np.nanmean([s["noise"][kind][k]["f1"] for s in seeds]) for k in snrs]
                col = style.VARIANT_GRAY if n == "clt_no_htfp" else style.ENTITY[n]
                ax.plot([float(k) for k in snrs], y, color=col, ls="--" if n == "clt_no_htfp" else style.LINESTYLE[n],
                        marker=style.MARKER.get(n, "o"), ms=3.5, lw=2.0 if n == "clt_full" else 1.2,
                        label="CL-TransCPD w/o HTFP" if n == "clt_no_htfp" else SHORT[n])
            ax.set(title=title, xlabel="SNR (dB)")
            ax.invert_xaxis()
        axes[0].set_ylabel("CPD F1")
        hl = {}                                     # shared legend below the panels (up to six methods)
        for ax in axes:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                hl.setdefault(lab, h)
        fig.tight_layout(rect=(0, 0.2, 1, 1))
        fig.legend(list(hl.values()), list(hl), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(out / "fig_robust.pdf")
        plt.close(fig)
        made.append("fig_robust.pdf")
    # cross-lead attention graphs per diagnostic subgroup
    fpath = Path(cfg.paths.work_dir) / "runs" / "clt" / "full" / "seed0" / "features_ptbxl.npz"
    mpath = Path(cfg.paths.work_dir) / "cache" / "ptbxl" / "meta.pkl"
    if not R.ph and fpath.exists() and mpath.exists():
        import pandas as pd

        f = np.load(fpath)
        meta = pd.read_pickle(mpath).iloc[f["rows"]]
        G = f["G"]
        te = np.isin(meta["fold"].values, cfg.data.test_folds)
        groups = attention_groups(meta)
        leads = LEADS
        mats = [(g, G[te & msk].mean(0), int((te & msk).sum())) for g, msk in groups if (te & msk).sum() > 0]
        if mats:
            ref = mats[0][1]
            fig, axes = plt.subplots(2, len(mats), figsize=(7.2, 3.0))
            axes = np.atleast_2d(axes)
            allv = np.concatenate([m.ravel() for _, m, _ in mats])
            vmin, vmax = np.percentile(allv, 2), np.percentile(allv, 98)
            dmax = max(np.abs(m - ref).max() for _, m, _ in mats) or 1e-6
            for j, (g, mtx, n) in enumerate(mats):
                axes[1, j].set_xlabel("key lead", fontsize=7)
                im0 = axes[0, j].imshow(mtx, cmap=style.SEQ_BLUE, vmin=vmin, vmax=vmax)
                im1 = axes[1, j].imshow(mtx - ref, cmap=style.DIVERGING, vmin=-dmax, vmax=dmax)
                axes[0, j].set_title(f"{g}\n(n={n})", fontsize=7.5)
                for i in (0, 1):
                    axes[i, j].set_xticks(range(12), leads if i == 1 else [""] * 12, rotation=90, fontsize=6)
                    axes[i, j].set_yticks(range(12), leads if j == 0 else [""] * 12, fontsize=6)
                    axes[i, j].grid(False)
            axes[0, 0].set_ylabel("query lead", fontsize=7)
            axes[1, 0].set_ylabel("minus NORM", fontsize=7)
            for im, row, lab in ((im0, 0, "attention"), (im1, 1, "difference")):
                cb = fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.015, pad=0.01)
                cb.set_label(lab, fontsize=7)
                cb.ax.tick_params(labelsize=6.5)
            fig.savefig(out / "fig_attention.pdf")
            plt.close(fig)
            made.append("fig_attention.pdf")
    # reliability diagram (single-label subset, seed 0)
    rel = {}
    for n in ["clt_full", "prior_skensem", "resnet1d_wang"]:
        p = Path(cfg.paths.work_dir) / "results" / "cls" / n / "sl" / "seed0_pred.npz"
        if not R.ph and p.exists():
            z = np.load(p)
            rel[n] = (z["y"], z["p"])
    if rel:
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        ax.plot([0, 1], [0, 1], color=style.MUTED, lw=0.8)
        edges = np.linspace(0, 1, 11)
        for n, (y, P) in rel.items():
            conf, acc = P.max(1), (P.argmax(1) == y).astype(float)
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                msk = (conf > lo) & (conf <= hi)
                if msk.sum() >= 10:
                    xs.append(conf[msk].mean())
                    ys.append(acc[msk].mean())
            ax.plot(xs, ys, color=style.ENTITY[n], marker=style.MARKER.get(n, "o"), ms=4,
                    lw=2.0 if n == "clt_full" else 1.2, label=SHORT[n])
        ax.set(xlabel="Confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")
        fig.savefig(out / "fig_reliability.pdf")
        plt.close(fig)
        made.append("fig_reliability.pdf")
    # dataset composition (single colour per series; small multiples)
    px, pdg = R.one("data", "ptbxl_summary.json"), R.one("data", "ptbdiag_summary.json")
    if px:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
        x = np.arange(len(SUPER))
        ax = axes[0]
        ax.bar(x - 0.2, [px["label_counts"][c] for c in SUPER], 0.38, color=style.SLOTS[0], label="records with label (multi-label)")
        ax.bar(x + 0.2, [px["single_counts"][c] for c in SUPER], 0.38, color=style.SLOTS[1], label="single-label records")
        ax.set_xticks(x, SUPER)
        ax.set(title=f"PTB-XL ({px['n_records']:,} records, {px['n_patients']:,} patients)", ylabel="Records")
        ax.set_ylim(0, 1.3 * max(px["label_counts"].values()))
        ax.legend(loc="upper right", ncol=1)
        ax.grid(axis="x", visible=False)
        if pdg:
            mr = pdg.get("mapped_records", {})
            cls = [c for c in SUPER if c in mr]
            axes[1].bar(np.arange(len(cls)), [mr[c] for c in cls], 0.5, color=style.SLOTS[0])
            axes[1].set_xticks(np.arange(len(cls)), cls)
            axes[1].set(title=f"PTB Diagnostic, mapped ({sum(mr.get(c, 0) for c in cls)} of {pdg['n_records']} records)",
                        ylabel="Records")
            axes[1].grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(out / "fig_dataset.pdf")
        plt.close(fig)
        made.append("fig_dataset.pdf")
    # Figure 1 replacement: median beats before vs. after balloon inflation (real STAFF III data)
    sp = Path(cfg.paths.work_dir) / "cache" / "staff3" / "streams.pkl"
    if not R.ph and sp.exists():
        made += _fig_intro_staff3(cfg, sp, out, style, plt)
    # STAFF III example traces
    st = (R.seeds("cpd", "clt_full") or [None])[0]
    tr = (st or {}).get("staff3", {}).get("traces") if st else None
    if tr:
        fig, axes = plt.subplots(len(tr), 1, figsize=(7.0, 1.4 * len(tr)), sharex=False)
        axes = np.atleast_1d(axes)
        for ax, t in zip(axes, tr):
            ax.plot(t["ends"], t["scores"], color=style.SLOTS[0], lw=1.0)
            ax.axhline(t["gamma"], color=style.MUTED, lw=0.8, ls="--")
            ax.axvline(t["cp"], color=style.INK, lw=0.8)
            sc = np.asarray(t["scores"], dtype=float)
            sc = sc[np.isfinite(sc)]
            if len(sc):                     # keep the threshold line inside the panel, with some headroom
                lo, hi = min(sc.min(), t["gamma"]), max(sc.max(), t["gamma"])
                ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.25 * (hi - lo))
            ax.annotate("$\\gamma$", (1.0, t["gamma"]), xycoords=("axes fraction", "data"), xytext=(2, 0),
                        textcoords="offset points", va="center", fontsize=7, color=style.MUTED)
            for a in t["alarms"]:
                ax.plot([a], [t["gamma"]], marker="v", color=style.SLOTS[7], ms=5)
            ax.set_ylabel("$\\mathcal{S}_k$")
            ax.set_title(f"patient {t['patient']}, {t['artery']} occlusion", loc="left", fontsize=7)
        axes[-1].set_xlabel("Time (s); vertical line = balloon inflation, triangles = alarms")
        fig.tight_layout()
        fig.savefig(out / "fig_staff3_trace.pdf")
        plt.close(fig)
        made.append("fig_staff3_trace.pdf")
        try:
            made += _fig_graphical_abstract(R, cfg, tr, out, style, plt)
        except Exception as exc:  # the abstract figure is optional
            print(f"graphical abstract skipped: {exc}")
    return made


def _fig_graphical_abstract(R, cfg, tr, out, style, plt):
    """Graphical abstract from real results: pipeline sketch + a STAFF III onset (median beats and score)."""
    import pickle

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fs = int(cfg.data.fs)
    t = tr[0]          # first within-recording onset, chosen without regard to whether it was detected
    sp = Path(cfg.paths.work_dir) / "cache" / "staff3" / "streams.pkl"
    sig = None
    if sp.exists():
        with open(sp, "rb") as fh:
            for st in pickle.load(fh)["cp"]:
                same = (st.get("record") == t["record"]) if t.get("record") else abs(st["cps"][0] - t["cp"]) < 0.5
                if same and st.get("patient") == t.get("patient"):
                    sig = np.asarray(st["signal"])
                    break
    # designed for the 13 cm x 5 cm display size of a graphical abstract
    fig = plt.figure(figsize=(5.4, 2.16))
    axl = fig.add_axes([0.0, 0.0, 0.46, 1.0])
    axl.set_xlim(0, 100)
    axl.set_ylim(0, 100)
    axl.axis("off")
    steps = [("12-lead ECG stream", "#2F5F9E"),
             ("two-cycle windows · spectral gate · RSSA", "#2F5F9E"),
             ("Cross-Lead Transformer, nuisance-aware\ncontrastive change-point pretraining", "#B4581C"),
             ("alarm → four feature streams →\nensemble diagnosis (NORM/MI/STTC/CD/HYP)", "#2E7D32")]
    h, g = 18.5, 6.2
    ys = [97 - h - i * (h + g) for i in range(4)]
    for (txt, col), y in zip(steps, ys):
        axl.add_patch(FancyBboxPatch((4, y), 92, h, boxstyle="round,pad=0.6,rounding_size=2.5", fc="white", ec=col,
                                     lw=1.0))
        axl.text(50, y + h / 2, txt, ha="center", va="center", fontsize=6.6, color=style.INK, linespacing=1.25)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        axl.add_patch(FancyArrowPatch((50, y0 - 0.6), (50, y1 + h + 0.6), arrowstyle="-|>", mutation_scale=7,
                                      color=style.MUTED, lw=0.9, shrinkA=0, shrinkB=0))
    # right: a real onset of ischaemia (STAFF III), detected zero-shot
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    cp = float(t["cp"])
    alarm = next((a for a in t["alarms"] if cp <= a <= cp + 120), None)
    fig.text(0.5, 0.95, "Zero-shot test: real ischaemia onset (STAFF III)", fontsize=6.8, weight="bold",
             color=style.INK, ha="left", va="center")
    if sig is not None:
        c = int(cp * fs)
        a_end = int((alarm if alarm is not None else cp + 60) * fs) + 10 * fs
        pre = _median_beat(sig[max(0, c - 25 * fs):c - 2 * fs], fs)
        post = _median_beat(sig[c:min(len(sig), max(a_end, c + 20 * fs))], fs)
        if pre is not None and post is not None:
            li = int(np.argmax(np.abs(post - pre).mean(0)))
            ax1 = fig.add_axes([0.535, 0.56, 0.2, 0.29])
            tm = (np.arange(len(pre)) / fs - 0.25) * 1000
            ax1.plot(tm, pre[:, li], color=style.MUTED, lw=1.0, label="before")
            ax1.plot(tm, post[:, li], color=style.SLOTS[0], lw=1.3, label="after")
            ax1.set_title(f"median beat, lead {lead_names[li]}", fontsize=6.0, loc="left", pad=2)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.legend(fontsize=5.6, loc="best", frameon=False, handlelength=1.0, borderaxespad=0.2)
    f1 = mean_of(_cpd_vals(R, "clt_full", "f1"))
    fah = mean_of(_cpd_vals(R, "clt_full", "fa_per_hour"))
    fb = [v for v in (mean_of(_cpd_vals(R, n, "f1")) for n in CPD_METHODS if n != "clt_full") if v is not None]
    r10 = mean_of([s_.get("staff3", {}).get("recall_at_fah10") for s_ in R.seeds("cpd", "clt_full")
                   if s_.get("staff3")])
    ch10 = 1.0 - np.exp(-10.0 * float(cfg.streams.staff3_tolerance_s) / 3600.0)   # random alarms at 10 FA/h
    if f1 is not None and r10 is not None:     # key results, all read from the experiment files
        best = f"\n(best baseline {max(fb):.2f})" if fb else ""
        fig.text(0.765, 0.7, f"PTB-XL-CPD:\nF1 {f1:.2f}, {fah:.0f} FA/h{best}\n\nSTAFF III, ≤10 FA/h:\n"
                 f"recall {100 * r10:.0f}% (chance {100 * ch10:.0f}%)", fontsize=6.2, color=style.INK, ha="left",
                 va="center", linespacing=1.2)
    ax2 = fig.add_axes([0.575, 0.15, 0.405, 0.29])
    e = np.asarray(t["ends"]) - cp
    s = np.asarray(t["scores"], dtype=float)
    keep = (e >= -60) & (e <= 180)
    ax2.plot(e[keep], s[keep], color=style.SLOTS[0], lw=1.0)
    ax2.axhline(t["gamma"], color=style.MUTED, lw=0.8, ls="--")
    ax2.axvline(0, color=style.INK, lw=0.8)
    if alarm is not None:
        ax2.plot([alarm - cp], [t["gamma"]], marker="v", color=style.SLOTS[7], ms=5)
        ax2.annotate(f"alarm +{alarm - cp:.0f} s", (alarm - cp, t["gamma"]), xytext=(5, 6), textcoords="offset points",
                     fontsize=6.0, color=style.INK)
    else:
        ax2.text(0.98, 0.95, "no alarm within 120 s", transform=ax2.transAxes, ha="right", va="top", fontsize=5.6,
                 color=style.INK2)
    ax2.set_xlabel("seconds from balloon inflation", fontsize=6.0, labelpad=1)
    ax2.set_ylabel("score", fontsize=6.0, labelpad=1)
    ax2.tick_params(labelsize=5.6, pad=1)
    fig.savefig(out / "graphical_abstract.pdf")
    fig.savefig(out / "graphical_abstract.png", dpi=400)
    plt.close(fig)
    return ["graphical_abstract.pdf"]


def _froc(pts):
    """Upper envelope of recall as a function of the false-alarm rate (points: recall, precision, gamma, FA/h)."""
    q = sorted((float(p[3]), float(p[0])) for p in pts if len(p) > 3 and np.isfinite(p[3]))
    if not q:
        return np.array([]), np.array([])
    fah = np.array([a for a, _ in q])
    rec = np.maximum.accumulate(np.array([b for _, b in q]))
    fah = np.maximum(fah, 0.3)          # zero false alarms are drawn at the left edge of the log axis
    return np.append(fah, 1000.0), np.append(rec, rec[-1])


def _median_beat(x, fs):
    from clcpd.signal.qrs import detect_rpeaks

    r = detect_rpeaks(x, fs)
    a, b = int(0.25 * fs), int(0.55 * fs)
    beats = [x[i - a:i + b] for i in r if i - a >= 0 and i + b <= len(x)]
    return np.median(np.stack(beats), axis=0) if len(beats) >= 3 else None


def _fig_intro_staff3(cfg, path, out, style, plt):
    import pickle

    with open(path, "rb") as fh:
        streams = pickle.load(fh)["cp"]
    fs = int(cfg.data.fs)
    best = None
    within = [st for st in streams if st.get("kind") == "within"]   # before and after from the same recording
    for st in (within or streams):
        x, cp = st["signal"], int(st["cps"][0] * fs)
        if cp < 25 * fs or len(x) < cp + 120 * fs:      # windows exactly as stated in the caption
            continue
        post_start, post_end = cp + 90 * fs, cp + 120 * fs
        pre, post = _median_beat(x[cp - 25 * fs:cp - 2 * fs], fs), _median_beat(x[post_start:post_end], fs)
        if pre is None or post is None:
            continue
        score = float(np.max(np.abs(post - pre).mean(0)))
        if best is None or score > best[0]:
            best = (score, st, pre, post)
    if best is None:
        return []
    _, st, pre, post = best
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    t = (np.arange(len(pre)) / fs - 0.25) * 1000
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.1), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(t, pre[:, i], color=style.MUTED, lw=1.0, label="25–2 s before inflation")
        ax.plot(t, post[:, i], color=style.SLOTS[0], lw=1.4, label="90–120 s after inflation")
        ax.set_title(leads[i], fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.grid(True)
    axes[1, 0].set_xlabel("ms from R peak", fontsize=7)
    axes[0, 0].set_ylabel("mV", fontsize=7)
    axes[1, 0].set_ylabel("mV", fontsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"STAFF III patient {st.get('patient')}, {st.get('artery')} occlusion: median beats", fontsize=8.5, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=7.5, bbox_to_anchor=(0.99, 1.0))
    fig.savefig(out / "fig_intro_staff3.pdf")
    plt.close(fig)
    return ["fig_intro_staff3.pdf"]


MACRO_NAMES = ["ResMLaucOurs", "ResMLaucOursQRSV", "ResMLaucBestBase", "ResMLbestBaseName", "ResSLaccOurs",
               "ResSLfOneOurs", "ResSLaucOurs", "ResSLmccOurs", "ResSLeceOurs", "ResSLaccPrior",
               "ResCPDfOneOurs", "ResCPDPrecOurs", "ResCPDRecOurs", "ResCPDAuprcOurs", "ResCPDDelayOurs",
               "ResCPDFahOurs", "ResCPDfOnePrior", "ResCPDPrecPrior", "ResCPDRecPrior", "ResCPDAuprcPrior",
               "ResCPDDelayPrior", "ResCPDFahPrior", "ResCPDfOneBestBase", "ResCPDbestBaseName",
               "ResCPDdelayBestBase", "ResCPDdelayRedPct", "ResCPDfahRedPct", "ResCPDafaOurs", "ResCPDnfaOurs",
               "ResStaffRecall", "ResStaffDelay", "ResStaffFah", "ResTrigAuc", "ResTrigAcc", "ResParamsTotal",
               "ResParamsEncoder", "ResFlops", "ResLatOnline", "ResHardware", "ResNstreams", "ResNcps", "ResHours",
               "ResStaffStreams", "ResStaffPatients", "ResStaffWithinRecall", "ResStaffWithinDelay",
               "ResStaffBestBaseName", "ResStaffBestBaseRecall", "ResStaffBestBaseWithin", "ResStaffBestBaseFah",
               "ResCPDRecTenOurs", "ResCPDRecTenBestBase", "ResCPDRecTenBestBaseName", "ResLatentSel",
               "ResStaffRecTenOurs", "ResStaffChanceOurs", "ResStaffBestBaseChance", "ResStaffBestBaseRecTen",
               "ResStaffRecTenBestBase", "ResStaffRecTenBestBaseName", "ResStepDecayMed", "ResStepDecayPninety",
               "ResStepConstMed", "ResStepConstPninety", "ResStepNaiveMed", "ResStepNaivePninety",
               "ResSigMLBest", "ResSigCPDBest"]
MACRO_NAMES += ["ResCPDfahBestBase", "ResCPDnfaBestBase", "ResCPDauprcBestBase", "ResStepOursMain", "ResStepOursStep",
                "ResNaiveOurs", "ResStepGainNames", "ResStepMaxName", "ResStepMaxMain", "ResStepMaxCopy",
                "ResNaiveGainNames", "ResNaiveMaxName", "ResNaiveMaxMain", "ResNaiveMaxCopy", "ResStaffChanceTen",
                "ResStaffAboveChanceNames", "ResStaffSplicedRecall", "ResStaffRecFiveOurs", "ResSameFOneOurs",
                "ResSameRecOurs", "ResSameFahOurs", "ResSameFOneBestBase", "ResSameBestBaseName", "ResHrRecOursLow",
                "ResHrRecOursHigh", "ResHrRecFixedLow", "ResHrRecFixedHigh", "ResHrNLow", "ResHrNHigh",
                "ResTolTwoFOne", "ResTolFiveFOne", "ResMuZeroFOne", "ResTrigN", "ResLambdaSel", "ResStreamTdAuc",
                "ResStreamClsAuc", "ResStreamAllAuc", "ResPtbdAccOurs", "ResPtbdFOneOurs", "ResPtbdAucOurs",
                "ResPtbdAucBestBase", "ResPtbdBestBaseName", "ResPerclassMinName", "ResPerclassMinAuprc",
                "ResPerclassMaxName", "ResPerclassMaxAuprc", "ResDesignPurenegBestAuprc",
                "ResDesignNoPurenegBestAuprc", "ResDesignBestConfig", "ResDesignFixedAuprc", "ResDesignFixedRank",
                "ResRobCleanOurs", "ResRobZeroOurs", "ResRobZeroNoHtfp", "ResRobCleanNoHtfp", "ResRobZeroBestBase",
                "ResRobZeroBestBaseName", "ResDropThreeOurs", "ResDropNineOurs", "ResRobClsCleanOurs",
                "ResRobClsZeroOurs", "ResRobClsCleanResnet", "ResRobClsZeroResnet", "ResAttnMaxDiff", "ResAttnMaxRelPct",
                "ResMLaucTransformer", "ResMLaucPatchtst", "ResSLaccBestBase", "ResSLaccBestBaseName",
                "ResPtbdMccOurs", "ResPtbdAccBestBase", "ResPtbdAccBestBaseName", "ResPtbdMccBestBase",
                "ResPtbdMccBestBaseName", "ResDropThreeTemporal", "ResDropThreeBestBase", "ResDropThreeBestBaseName"]
MACRO_NAMES += [f"ResAbl{_camel(v)}{s}" for v, _ in ABLATIONS for s in ("FOne", "Nfa", "Fah", "Delay", "Auc")]
MACRO_NAMES += [f"ResSup{k}" for k in ("FOne", "Prec", "Rec", "Auprc", "Delay", "Fah", "RecTen", "Nfa", "Afa",
                                       "StaffRecall", "StaffFah", "StaffRecTen", "SameFOne", "ValAuc", "RepSel",
                                       "MainFOne", "StepFOne", "NaiveFOne")] + ["ResSigSup", "ResSigSupFah"]


def main():
    ap = parser(__doc__)
    ap.add_argument("--placeholders", action="store_true")
    args = ap.parse_args()
    cfg = config_from(args)
    paper = Path(cfg.paths.paper_dir)
    R = Results(Path(cfg.paths.work_dir) / "results", args.placeholders)
    tables = paper / "results" / "tables"
    for fn in (t_data, t_ptbxl_ml, t_ptbxl_sl, t_adv, t_perclass, t_ptbdiag, t_cpd, t_cpd_real, t_confounds, t_ablation,
               t_streams,
               t_mu, t_compute, t_robust, t_design):
        fn(R, tables)
    t_significance(R, cfg, tables)
    t_attention(R, cfg, tables)
    default_macros(paper / "results" / "macros_default.tex", MACRO_NAMES)
    if not args.placeholders:
        m = macros(R, paper / "results" / "macros.tex", cfg)
        made = figures(R, cfg, paper / "figures" / "generated")
        print(f"tables -> {tables}\nmacros -> {len(m)} values\nfigures -> {made}")
    else:
        print(f"placeholder tables written to {tables}")


if __name__ == "__main__":
    main()
