#!/bin/bash
# Submit the whole pipeline with job dependencies (PBS Pro syntax).
#   cd experiments && bash pbs/submit_all.sh                                   # full study
#   bash pbs/submit_all.sh eval.seeds=[0,1,2] eval.ablation_seeds=[0]          # reduced budget
# Overrides are exported as CLCPD_OVERRIDES and forwarded to every job (qsub -v), so all
# stages see the same configuration.
# Torque: use "-t" instead of "-J", and "afterokarray:" instead of "afterok:" for array dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
export CLCPD_OVERRIDES="$*"
source pbs/env.sh >/dev/null

range() { local k; k=$(python scripts/list_tasks.py "$1"); if [ "$k" -le 1 ]; then echo "0-1"; else echo "0-$((k - 1))"; fi; }
# PBS Pro arrays need >= 2 sub-jobs; a surplus index exits immediately (nothing to do).
Q="qsub -v CLCPD_OVERRIDES"

PREP=$($Q pbs/01_prepare.pbs)
echo "prepare      : $PREP"
PRE=$($Q -W depend=afterok:"$PREP" -J "$(range pretrain)" pbs/02_pretrain.pbs)
echo "pretrain+feat: $PRE"
CLB=$($Q -W depend=afterok:"$PREP" -J "$(range cls_baselines)" pbs/04_cls_baselines.pbs)
echo "cls baselines: $CLB"
CLS=$($Q -W depend=afterok:"$PRE" -J "$(range classifiers)" pbs/03_classifiers.pbs)
echo "classifiers  : $CLS"
CPD=$($Q -W depend=afterok:"$CLS" -J "$(range cpd)" pbs/05_cpd.pbs)
echo "cpd          : $CPD"
ROB=$($Q -W depend=afterok:"$CPD":"$CLB" -J "$(range robustness)" pbs/06_robustness.pbs)
echo "robustness   : $ROB"
AST=$($Q -W depend=afterany:"$ROB" pbs/07_latency_and_assets.pbs)
echo "assets       : $AST"
echo "Monitor with: qstat -u $USER ; logs are in experiments/logs/"
