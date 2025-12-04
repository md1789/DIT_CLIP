"""Diffusion utilities used by DiT models.

This module keeps the DDPM schedule configuration in a small dataclass so that
training and sampling can switch schedules without touching global state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch


@dataclass
class DiffusionConfig:
    """Configuration for DDPM schedules.

    Attributes:
        timesteps: Number of diffusion steps ``T``.
        beta_start: Starting beta value for the noise schedule.
        beta_end: Ending beta value for the noise schedule.
        schedule: Name of the beta schedule (``"linear"`` or ``"cosine"``).
    """

    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule: str = "linear"


def create_diffusion_config(**overrides: Any) -> DiffusionConfig:
    """Helper to create a :class:`DiffusionConfig` with partial overrides."""

    return DiffusionConfig(**overrides)


def get_beta_schedule(config: DiffusionConfig) -> torch.Tensor:
    """Return betas following the requested schedule."""

    if config.schedule.lower() == "linear":
        return torch.linspace(config.beta_start, config.beta_end, config.timesteps, dtype=torch.float32)

    if config.schedule.lower() == "cosine":
        # Cosine schedule from Nichol & Dhariwal 2021
        steps = config.timesteps + 1
        x = torch.linspace(0, config.timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / config.timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.float().clamp(max=0.999)

    raise ValueError(f"Unsupported beta schedule: {config.schedule}")


class DiffusionSchedule:
    """Pre-computed tensors for a diffusion process.

    Keeping everything in a small container avoids global variables and makes it
    easy to keep multiple schedules around for experiments.
    """

    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        betas = get_beta_schedule(config)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        for name, value in list(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract values from a 1D tensor at specific timesteps and reshape for broadcast."""

    batch_size = t.shape[0]
    out = a.to(t.device)[t].float()
    return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))


def q_sample(schedule: DiffusionSchedule, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Forward diffusion: add noise to a clean image ``x_start`` at timestep ``t``."""

    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(schedule.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_index: int,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Single reverse diffusion step ``p(x_{t-1} | x_t)``."""

    if model_kwargs is None:
        model_kwargs = {}

    betas_t = extract(schedule.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(schedule.sqrt_recip_alphas, t, x.shape)

    model_output = model(x, t, **model_kwargs)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean

    posterior_variance_t = extract(schedule.posterior_variance, t, x.shape)
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise


def p_sample_loop(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: torch.Size,
    model_kwargs: Optional[Dict[str, Any]] = None,
    noise: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    timestep_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """Full reverse diffusion loop that starts from noise and returns samples."""

    if device is None:
        device = next(model.parameters()).device

    if noise is None:
        img = torch.randn(shape, device=device)
    else:
        img = noise.to(device)

    if model_kwargs is None:
        model_kwargs = {}

    for i in reversed(range(schedule.config.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model, schedule, img, t, i, model_kwargs=model_kwargs)
        if timestep_callback is not None:
            timestep_callback(i, img)

    return img
