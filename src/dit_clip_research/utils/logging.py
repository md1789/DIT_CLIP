"""Lightweight logging + checkpoint utilities."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class AverageMeter:
    """Tracks running average of a scalar."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_checkpoint_path(root: Path, epoch: int) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"checkpoint_epoch_{epoch:04d}.pth"


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path, epoch: int, config: Dict[str, Any]) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
    }
    torch.save(payload, path)

    # Also persist a tiny JSON sidecar with metadata for quick inspection.
    meta_path = Path(str(path) + ".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "path": str(path)}, f, indent=2)
