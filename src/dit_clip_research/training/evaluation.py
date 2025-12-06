"""Sampling and evaluation utilities."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from dit_clip_research.models.clip_conditioning import ClipTextEncoder
from dit_clip_research.models.diffusion import DiffusionSchedule, p_sample_loop
from dit_clip_research.models.dit import DiT
from dit_clip_research.data.transforms import denormalize


@torch.no_grad()
def sample_images(
    model: DiT,
    schedule: DiffusionSchedule,
    prompts: Optional[List[str]] = None,
    clip_encoder: Optional[ClipTextEncoder] = None,
    num_samples: int = 4,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate samples conditioned on optional text prompts."""

    device = device or next(model.parameters()).device
    model.eval()

    model_kwargs = {}
    if prompts is not None and clip_encoder is not None:
        clip_cond = clip_encoder.encode_text(prompts, device=device)
        model_kwargs["clip_text"] = clip_cond

    samples = p_sample_loop(model, schedule, (num_samples, model.out_channels, model.img_size, model.img_size), model_kwargs=model_kwargs, device=device)
    return samples


def make_sample_grid(tensor: torch.Tensor, nrow: int = 4, denorm: bool = True) -> torch.Tensor:
    if denorm:
        tensor = denormalize(tensor)
    return make_grid(tensor.clamp(-1, 1), nrow=nrow)


def upscale_grid(grid: torch.Tensor, factor: int = 1, mode: str = "nearest") -> torch.Tensor:
    """Optionally upsample a CHW grid tensor for visualization."""

    if factor <= 1:
        return grid
    return F.interpolate(grid.unsqueeze(0), scale_factor=factor, mode=mode).squeeze(0)


# Placeholder for FID hook; left as a stub for future extension.
def compute_fid_stub() -> None:  # pragma: no cover - documentation placeholder
    raise NotImplementedError("FID computation can be plugged in with pytorch-fid or torchmetrics")
