#!/bin/bash
# ---------------------------------------------------------------------------
# Cluster environment for every PBS job. EDIT THE MARKED LINES FOR YOUR CLUSTER.
# ---------------------------------------------------------------------------
set -euo pipefail

# 1) software (module names differ between clusters - check `module avail`)
module purge >/dev/null 2>&1 || true
module load cuda/12.4 >/dev/null 2>&1 || true          # EDIT
module load anaconda3 >/dev/null 2>&1 || true          # EDIT (or miniconda / python)
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate clcpd                                     # EDIT if your env has another name

# 2) locations
export EXP_DIR="${PBS_O_WORKDIR:-$(pwd)}"                # the experiments/ folder (qsub from there)
export DATA_ROOT="${DATA_ROOT:-$HOME/data/physionet.org/files}"   # EDIT: contains ptb-xl/, ptbdb/, staffiii/, nstdb/
export WORK_DIR="${WORK_DIR:-$HOME/clcpd_work}"                   # EDIT: fast scratch space, ~40 GB
export PAPER_DIR="${PAPER_DIR:-$(dirname "$EXP_DIR")}"            # manuscript root (parent of experiments/)

# 3) threads
export OMP_NUM_THREADS="${NCPUS:-8}"
export MKL_NUM_THREADS="${NCPUS:-8}"
export PYTHONUNBUFFERED=1

cd "$EXP_DIR"
mkdir -p logs "$WORK_DIR"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) job=${PBS_JOBID:-local} task=${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-none}}"
