"""Simple distributed helpers (safe to import when not using DDP)."""
from __future__ import annotations

import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_initialized():
        dist.barrier()
