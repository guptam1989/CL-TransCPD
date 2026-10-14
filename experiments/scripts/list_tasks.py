"""Print the number of array tasks of each stage (used by pbs/submit_all.sh).

  python scripts/list_tasks.py pretrain      -> e.g. 35
  python scripts/list_tasks.py --show cpd    -> lists the tasks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import config_from  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pretrain", "classifiers", "cls_baselines", "cpd", "robustness"])
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = config_from(args)
    if args.stage == "pretrain":
        from pretrain_clt import tasks_pretrain as f
    elif args.stage == "classifiers":
        from train_classifiers import tasks_classifiers as f
    elif args.stage == "cls_baselines":
        from train_cls_baselines import tasks_cls_baselines as f
    elif args.stage == "cpd":
        from run_cpd import tasks_cpd as f
    else:
        from robustness import tasks_robustness as f
    tasks = f(cfg)
    if args.show:
        for i, t in enumerate(tasks):
            print(i, t)
    else:
        print(len(tasks))


if __name__ == "__main__":
    main()
