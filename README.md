# CL-TransCPD

**Contrastive Cross-Lead Transformer for Online Change-Point Detection and Change-Triggered Diagnosis in Multi-Lead ECG**

Muktesh Gupta, Pranjal Kumar and Himanshu Verma
Department of Computer Science and Engineering, Dr. B. R. Ambedkar National Institute of Technology Jalandhar, India

This repository contains the complete code of the study: data preparation, the proposed method, all baselines, the evaluation protocol and the scripts that produce every table, number and data figure of the manuscript.

---

## Overview

CL-TransCPD monitors a 12-lead ECG stream online, raises an alarm when the cardiac state changes and then diagnoses the new state.

| Phase | What it does |
|---|---|
| Signal conditioning | 0.5–40 Hz band-pass filtering at 100 Hz, heart-rate-adaptive two-cycle segmentation (IHRAS), a learned time-frequency gate (HTFP) and singular spectrum analysis denoising (RSSA) |
| Cross-Lead Transformer and change score | patch tokens per lead, attention within each lead and across the 12 leads, and an online change score that measures the deviation of each segment from a moving-average reference of the current state |
| Change-triggered diagnosis | after an alarm, four feature streams (time-domain intervals, CLS context, attention statistics, cross-lead graph) are classified by a stacked ensemble (SkEnsem) into NORM, MI, STTC, CD and HYP |

The encoder is trained with **contrastive change-point pretraining (CCPP)**, a triplet objective in which transitions between diagnostic states and post-change segments are negatives, while neighbouring segments and *nuisance* splices (a change of patient without a change of diagnosis) are positives.

The evaluation protocol is part of the contribution:

* **PTB-XL** official folds (1–8 training, 9 validation, 10 test) for classification;
* **PTB-XL-CPD**, a change-point benchmark built from concatenated PTB-XL records, with nuisance junctions, splice-only controls, same-patient streams, and modified copies of the test streams that expose junction artefacts;
* seven unsupervised change-point detectors and a supervised reference detector (ResNet-CPD) scored with the same change score;
* zero-shot detection of real ischaemia onsets during coronary balloon inflation (**STAFF III**), reported against chance-level recall;
* **PTB Diagnostic** patient-grouped cross-validation, noise-stress (**MIT-BIH NSTDB**) and lead-dropout tests, ablations, multiple seeds and paired bootstrap tests with Holm correction.

All thresholds and score settings are selected on validation data only.

---

## Repository structure

```
experiments/
├── clcpd/                 Python package
│   ├── signal/            filtering, QRS detection, IHRAS segmentation, SSA, noise
│   ├── data/              PTB-XL, PTB Diagnostic, STAFF III, NSTDB loaders; stream builder; CCPP triplets
│   ├── models/            Cross-Lead Transformer, HTFP, losses, CPD and classification baselines
│   ├── features/          time-domain, attention and cross-lead graph features
│   ├── classify/          SkEnsem stacked ensemble
│   ├── eval/              classification and change-point metrics, statistics
│   ├── online.py          online detection loop, calibration, PR sweeps
│   └── pipeline.py        shared utilities (caches, variants, model I/O)
├── configs/default.yaml   all settings (overridable as key=value on the command line)
├── scripts/               entry points (one per stage, see below)
├── pbs/                   PBS job arrays for running the full study on a cluster
└── requirements.txt
```

---

## Installation

Python 3.10 or newer (tested with 3.11).

```bash
conda create -n clcpd python=3.11 -y
conda activate clcpd
pip install torch --index-url https://download.pytorch.org/whl/cu124   # choose the build that matches your CUDA
pip install -r experiments/requirements.txt
```

A CPU-only PyTorch build is sufficient for the smoke test.

---

## Data

All four datasets are public on PhysioNet and are used in their original folder layout.

```bash
bash experiments/scripts/download_data.sh /path/to/data        # about 5 GB
export DATA_ROOT=/path/to/data/physionet.org/files
export WORK_DIR=/path/to/scratch/clcpd_work                    # caches, checkpoints, raw results (~40 GB)
```

| Dataset | Role | Folder under `$DATA_ROOT` |
|---|---|---|
| [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | training, classification, change-point benchmark | `ptb-xl/1.0.3` |
| [PTB Diagnostic ECG Database 1.0.0](https://physionet.org/content/ptbdb/1.0.0/) | external classification | `ptbdb/1.0.0` |
| [STAFF III 1.0.0](https://physionet.org/content/staffiii/1.0.0/) | real ischaemia onsets (zero-shot test) | `staffiii/1.0.0` |
| [MIT-BIH Noise Stress Test Database 1.0.0](https://physionet.org/content/nstdb/1.0.0/) | denoiser pretraining and noise robustness | `nstdb/1.0.0` |

---

## Quick check

The smoke test runs the whole pipeline on small synthetic data with tiny settings and writes only below `--root` (about 30–40 minutes on a CPU):

```bash
cd experiments
python scripts/smoke_test.py --root /tmp/clcpd_smoke --quick
```

It must end with `SMOKE TEST PASSED`.

---

## Reproducing the study

All commands are run from `experiments/`. Every script skips work that is already finished, so an interrupted run can simply be restarted.

**1. Prepare the data** (caches, benchmark streams, dataset summaries)

```bash
python scripts/prepare_data.py --jobs 16
```

**2. Run the stages.** `run_stage.py` runs all tasks of a stage with several worker processes on one machine; `list_tasks.py` shows what each task does.

```bash
python scripts/run_stage.py pretrain      --workers 3   # HTFP pretraining, CCPP training, feature extraction
python scripts/run_stage.py classifiers   --workers 4   # SkEnsem and the LightGBM probe
python scripts/run_stage.py cls_baselines --workers 3   # ResNet1d, Inception1d, BiLSTM, Transformer, PatchTST, CLOCS
python scripts/run_stage.py cpd           --workers 3   # all change-point detectors, STAFF III, controls
python scripts/run_stage.py robustness    --workers 3   # NSTDB noise and lead dropout

python scripts/list_tasks.py cpd --show
```

| Stage | Script | Tasks |
|---|---|---|
| pretrain | `pretrain_clt.py`, `extract_features.py` | 38 (5 seeds of the full model, 11 ablations × 3 seeds) |
| classifiers | `train_classifiers.py` | 85 |
| cls_baselines | `train_cls_baselines.py` | 60 (6 models × 2 tasks × 5 seeds) |
| cpd | `run_cpd.py` | 66 (classical detectors run once; deep detectors 5 seeds) |
| robustness | `robustness.py` | 17 |

**3. Validation design study** (score representation, μ and η, with and without the post-change negative; fold 9 only)

```bash
python scripts/pretrain_clt.py --variant cand_pureneg   --seed 0
python scripts/pretrain_clt.py --variant cand_nopureneg --seed 0
python scripts/design_study.py --variant cand_pureneg   --seed 0
python scripts/design_study.py --variant cand_nopureneg --seed 0
```

**4. Latency and paper assets**

```bash
python scripts/benchmark_latency.py
python scripts/make_paper_assets.py
```

`make_paper_assets.py` turns the raw results (`$WORK_DIR/results/*.json`) into `results/macros.tex`, `results/tables/*.tex` and `figures/generated/*.pdf` in the repository root (set `PAPER_DIR` to write them elsewhere). Paired significance tests are computed in the same step.

**On a PBS cluster**, edit `pbs/env.sh` (modules, conda environment, `DATA_ROOT`, `WORK_DIR`) and the queue names in `pbs/*.pbs`, then submit all stages with their dependencies:

```bash
cd experiments
bash pbs/submit_all.sh
```

Configuration values can be overridden on any command, for example `eval.seeds=[0,1,2] ccpp.epochs=40`.

The full study was run on NVIDIA H100 GPUs. The online path of the detector runs in about 19 ms per segment on a single CPU thread.

---

## Notes on the implementation

* RSSA is computed in batch form per segment, which gives the same output as the recursive update used online.
* The delineator behind the time-domain features runs at 100 Hz (10 ms resolution for QRS and QT).
* The deep change-point and classification baselines are re-implementations following the original papers, trained with the same data and budget.
* The classical detectors (CUSUM, BOCPD, RuLSIF) operate on a compact 12-lead morphology descriptor.

---

## Citation

The citation will be added when the article is published. Until then, please cite the repository:

```bibtex
@misc{gupta2026cltranscpd,
  author = {Gupta, Muktesh and Kumar, Pranjal and Verma, Himanshu},
  title  = {{CL-TransCPD}: Contrastive Cross-Lead Transformer for Online Change-Point Detection
            and Change-Triggered Diagnosis in Multi-Lead {ECG}},
  year   = {2026},
  howpublished = {\url{https://github.com/guptam1989/CL-TransCPD}}
}
```

## Licence

Released under the [MIT License](LICENSE).

## Acknowledgements

This work was supported by the Anusandhan National Research Foundation (ANRF), Government of India, under the Prime Minister Early Career Research Grant (PM-ECRG) [Grant/File No. ANRF/ECRG/2025/000038/ENS]. The authors thank the creators of the PTB-XL, PTB Diagnostic, STAFF III and MIT-BIH Noise Stress Test databases and PhysioNet for making the data openly available.
