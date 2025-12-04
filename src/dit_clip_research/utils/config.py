"""Configuration helpers for experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config_dict(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged
