"""Learning rate and diffusion schedule helpers."""
from __future__ import annotations

from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

from dit_clip_research.models.diffusion import DiffusionConfig, DiffusionSchedule, create_diffusion_config


def create_lr_scheduler(optimizer: Optimizer, epochs: int, steps_per_epoch: int) -> _LRScheduler:
    total_steps = max(1, epochs * steps_per_epoch)
    return CosineAnnealingLR(optimizer, T_max=total_steps)


def build_diffusion_schedule(config: Optional[dict] = None) -> DiffusionSchedule:
    cfg = create_diffusion_config(**(config or {}))
    return DiffusionSchedule(cfg)
