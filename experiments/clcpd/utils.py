"""Small shared helpers: seeding, devices, JSON I/O, logging, timing, PBS array index."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


def threads() -> int:
    """CPU threads this process may use (CLCPD_THREADS, else the PBS NCPUS allocation, else all cores)."""
    for key in ("CLCPD_THREADS", "NCPUS"):
        if os.environ.get(key, "").isdigit():
            return max(1, int(os.environ[key]))
    return os.cpu_count() or 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(*msg) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}]", *msg, flush=True)


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_builtin(obj), fh, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def pbs_array_index(default: int | None = None):
    """Index of the current PBS Pro (PBS_ARRAY_INDEX) or Torque (PBS_ARRAYID) array task."""
    for key in ("PBS_ARRAY_INDEX", "PBS_ARRAYID", "SLURM_ARRAY_TASK_ID"):
        if os.environ.get(key, "") != "":
            return int(os.environ[key])
    return default


class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        if self.name:
            log(f"{self.name}: {self.seconds:.1f}s")


def hardware_string() -> str:
    """Human-readable description of the CPU/GPU, recorded next to every latency result."""
    import platform

    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    gpu = "none"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return f"CPU: {cpu}; GPU: {gpu}; Python {sys.version.split()[0]}"
