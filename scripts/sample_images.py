"""Load a DiT checkpoint and sample images with optional text prompts."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import torch
from torchvision.utils import save_image

from dit_clip_research.models.clip_conditioning import ClipTextEncoder
from dit_clip_research.models.diffusion import DiffusionSchedule, create_diffusion_config
from dit_clip_research.models.dit import DiT
from dit_clip_research.training.evaluation import make_sample_grid, sample_images


def parse_prompts(text: Optional[str]) -> Optional[List[str]]:
    if text is None or text.strip() == "":
        return None
    return [p.strip() for p in text.split("|") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample images from a DiT checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./samples.png"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--use-clip", action="store_true")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--prompts", type=str, default=None, help="Pipe-separated list of prompts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiT(
        img_size=args.image_size,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        use_clip_conditioning=args.use_clip,
        clip_embed_dim=512 if args.use_clip else None,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    diffusion_cfg = create_diffusion_config(**checkpoint.get("config", {}).get("diffusion", {}))
    schedule = DiffusionSchedule(diffusion_cfg).to(device)

    prompts = parse_prompts(args.prompts)
    clip_encoder = None
    if args.use_clip and prompts is not None:
        clip_encoder = ClipTextEncoder(model_name=args.clip_model, pretrained=args.clip_pretrained, device=device)

    samples = sample_images(
        model,
        schedule,
        prompts=prompts,
        clip_encoder=clip_encoder,
        num_samples=args.num_samples,
        device=device,
    )

    grid = make_sample_grid(samples, nrow=int(math.sqrt(args.num_samples)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, args.output)
    print(f"Saved samples to {args.output}")


if __name__ == "__main__":
    main()
