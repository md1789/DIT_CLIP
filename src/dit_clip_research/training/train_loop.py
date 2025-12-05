"""Reusable training loop for DiT + CLIP conditioning."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from dit_clip_research.data.datasets import Cifar10TextDataset, TextImagePairDataset, create_dataloader
from dit_clip_research.data import transforms as T
from dit_clip_research.models.clip_conditioning import ClipTextEncoder
from dit_clip_research.models.diffusion import DiffusionSchedule, create_diffusion_config, q_sample
from dit_clip_research.models.dit import DiT
from dit_clip_research.utils.logging import AverageMeter, get_checkpoint_path, save_checkpoint, set_seed


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    dataset: str = "cifar10_text"
    data_root: str = "./data"
    annotations: Optional[str] = None
    image_size: int = 32
    patch_size: int = 4
    hidden_size: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    use_clip_conditioning: bool = False
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_embed_dim: int = 512
    diffusion: Dict[str, Any] = field(default_factory=lambda: {"timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "schedule": "linear"})
    seed: int = 42
    num_workers: int = 4
    log_every: int = 100
    save_dir: str = "./checkpoints"
    save_every: int = 1


@torch.no_grad()
def maybe_encode_text(batch: Dict[str, Any], clip_encoder: Optional[ClipTextEncoder], device: torch.device) -> Optional[torch.Tensor]:
    if clip_encoder is None:
        return None
    if batch.get("text_embedding") is not None:
        return batch["text_embedding"].to(device)
    return clip_encoder.encode_text(batch["text"], device=device)


def train_one_epoch(
    model: DiT,
    schedule: DiffusionSchedule,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_encoder: Optional[ClipTextEncoder] = None,
) -> float:
    model.train()
    loss_meter = AverageMeter()

    for step, batch in enumerate(tqdm(dataloader, desc="train", leave=False)):
        images = batch["image"].to(device)
        t = torch.randint(0, schedule.config.timesteps, (images.size(0),), device=device)
        noise = torch.randn_like(images)
        x_noisy = q_sample(schedule, images, t, noise=noise)

        clip_cond = maybe_encode_text(batch, clip_encoder, device)
        pred_noise = model(x_noisy, t, clip_text=clip_cond)

        # Defensive debug: catch None/shape mismatches early.
        if pred_noise is None or noise is None:
            raise ValueError(
                f"pred_noise or noise is None (step {step}); "
                f"pred_noise={type(pred_noise)}, noise={type(noise)}, "
                f"clip_cond={None if clip_cond is None else clip_cond.shape}"
            )
        if pred_noise.shape != noise.shape:
            raise ValueError(
                f"Shape mismatch at step {step}: pred_noise {pred_noise.shape} vs noise {noise.shape}; "
                f"x_noisy {x_noisy.shape}, images {images.shape}"
            )

        loss = F.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_meter.update(loss.item(), n=images.size(0))

    return float(loss_meter.avg)


def build_dataloader(config: TrainConfig):
    if config.dataset == "cifar10_text":
        dataset = Cifar10TextDataset(root=config.data_root, split="train", download=True, transform=T.training_transforms(config.image_size))
    else:
        if config.annotations is None:
            raise ValueError("annotations must be provided for custom datasets")
        dataset = TextImagePairDataset(
            root=config.data_root,
            annotations=config.annotations,
            transform=T.training_transforms(config.image_size),
        )

    return create_dataloader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)


def train(config: TrainConfig) -> None:
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = build_dataloader(config)

    model = DiT(
        img_size=config.image_size,
        patch_size=config.patch_size,
        in_chans=3,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        use_clip_conditioning=config.use_clip_conditioning,
        clip_embed_dim=config.clip_embed_dim if config.use_clip_conditioning else None,
    ).to(device)

    diffusion_cfg = create_diffusion_config(**config.diffusion)
    schedule = DiffusionSchedule(diffusion_cfg).to(device)

    clip_encoder = None
    if config.use_clip_conditioning:
        clip_encoder = ClipTextEncoder(model_name=config.clip_model_name, pretrained=config.clip_pretrained, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    save_root = Path(config.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(model, schedule, dataloader, optimizer, device, clip_encoder=clip_encoder)
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {avg_loss:.4f}")

        if (epoch + 1) % config.save_every == 0:
            ckpt_path = get_checkpoint_path(save_root, epoch + 1)
            save_checkpoint(model, optimizer, ckpt_path, epoch + 1, config.__dict__)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training finished")
