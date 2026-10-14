"""YAML configuration with ${VAR:-default} expansion and key.sub=value overrides."""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]          # .../experiments
DEFAULT_CONFIG = EXP_ROOT / "configs" / "default.yaml"
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class Cfg(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __deepcopy__(self, memo):
        return Cfg({k: copy.deepcopy(v, memo) for k, v in self.items()})

    def to_dict(self):
        return json.loads(json.dumps(self))


def _wrap(obj):
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _expand(obj):
    if isinstance(obj, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def apply_overrides(cfg: Cfg, overrides) -> Cfg:
    """Apply ``a.b.c=value`` strings; values are parsed as YAML (numbers, lists, bools)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = _wrap(yaml.safe_load(raw))
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node:
                node[p] = Cfg()
            node = node[p]
        node[parts[-1]] = value
    return cfg


def load_config(path=None, overrides=None) -> Cfg:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _wrap(_expand(raw))
    # overrides from the environment (set by pbs/submit_all.sh) come first, command-line ones win
    env_ov = os.environ.get("CLCPD_OVERRIDES", "").split()
    cfg = apply_overrides(cfg, env_ov + list(overrides or []))
    # resolve relative paths against experiments/
    for key in ("data_root", "work_dir", "paper_dir"):
        p = Path(str(cfg.paths[key]))
        cfg.paths[key] = str(p if p.is_absolute() else (EXP_ROOT / p).resolve())
    return cfg


def dataset_dir(cfg: Cfg, name: str) -> Path:
    return Path(cfg.paths.data_root) / cfg.paths[f"{name}_dir"]


def work_path(cfg: Cfg, *parts) -> Path:
    p = Path(cfg.paths.work_dir).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
