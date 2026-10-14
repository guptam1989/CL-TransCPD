"""Bootstrap for the scripts: put experiments/ on sys.path and parse --config / overrides."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clcpd.config import load_config  # noqa: E402
from clcpd.utils import threads  # noqa: E402

try:  # respect the CPU allocation of the job / worker
    import torch  # noqa: E402

    torch.set_num_threads(threads())
except ImportError:
    pass


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    return ap


def config_from(args):
    return load_config(args.config, args.overrides)


def pick_task(tasks, idx):
    """Task for a PBS array index; None (clean exit) when the index is beyond the task list."""
    if idx is None:
        return None
    if idx >= len(tasks):
        print(f"array index {idx} >= {len(tasks)} tasks: nothing to do", flush=True)
        sys.exit(0)
    return tasks[idx]
