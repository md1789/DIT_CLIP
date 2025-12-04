# Training DiT with CLIP conditioning

## Setup
1. Install dependencies and this repo in editable mode:
   ```bash
   pip install -e .
   pip install open-clip-torch
   ```
2. Download data (example):
   ```bash
   python scripts/download_data.py --dataset cifar10 --data-root ./data
   ```
3. Pick a config, e.g., `configs/cifar10_baseline.yaml`.

## Running training (Python API)
```python
from pathlib import Path
import torch
from dit_clip_research.training.train_loop import TrainConfig, train
from dit_clip_research.utils.config import load_config_dict

cfg = load_config_dict("configs/cifar10_baseline.yaml")
train_config = TrainConfig(**cfg["training"])
train_config.diffusion = cfg["diffusion"]
train(train_config)
```
- The training loop lives in `src/dit_clip_research/training/train_loop.py`.
- Conditioning is merged in `DiT.forward` (`src/dit_clip_research/models/dit.py`).
- Text embeddings come from `ClipTextEncoder` (`src/dit_clip_research/models/clip_conditioning.py`).

## Colab workflow
- Use `notebooks/02_train_dit_with_clip_colab.py` as a drop-in set of cells.
- Mount Google Drive (optional) and point `TrainConfig.save_dir` to `/content/drive/...` to persist checkpoints.
- Keep batch size small for sanity checks; increase once the pipeline runs end-to-end.

## Sampling with prompts
```python
from dit_clip_research.models.diffusion import DiffusionSchedule, create_diffusion_config
from dit_clip_research.training.evaluation import sample_images, make_sample_grid
from dit_clip_research.models.clip_conditioning import ClipTextEncoder
from dit_clip_research.models.dit import DiT
import torch

ckpt = torch.load("experiments/cifar10_baseline/checkpoints/checkpoint_epoch_0001.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiT(img_size=32, patch_size=4, hidden_size=384, depth=8, num_heads=6,
            use_clip_conditioning=True, clip_embed_dim=512).to(device)
model.load_state_dict(ckpt["model"])

schedule = DiffusionSchedule(create_diffusion_config(**ckpt["config"].get("diffusion", {}))).to(device)
clip_encoder = ClipTextEncoder(device=device)

prompts = ["a dog", "a ship"]
samples = sample_images(model, schedule, prompts=prompts, clip_encoder=clip_encoder, num_samples=len(prompts), device=device)
```

## Extending conditioning
- Project any new conditioning signal to `hidden_size` and add it to the `cond` vector in `DiT.forward`.
- For classifier-free guidance, add an unconditional branch in `p_sample` (see comments in `docs/dit_paper_summary.md`).
