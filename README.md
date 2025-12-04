# DiT + CLIP Research Playground

Research repo for experimenting with Diffusion Transformers (DiT) and CLIP-based text conditioning, inspired by the "Scalable Diffusion Models with Transformers" paper and the official [facebookresearch/DiT](https://github.com/facebookresearch/DiT) implementation.

## Quickstart
1. Install:
   ```bash
   pip install -e .
   pip install open-clip-torch
   ```
2. Download CIFAR-10 (demo dataset):
   ```bash
   python scripts/download_data.py --dataset cifar10 --data-root ./data
   ```
3. Run a small experiment:
   ```bash
   python - <<"PY"
   from dit_clip_research.training.train_loop import TrainConfig, train
   from dit_clip_research.utils.config import load_config_dict
   cfg = load_config_dict("configs/cifar10_baseline.yaml")
   config = TrainConfig(**cfg["training"])
   config.diffusion = cfg["diffusion"]
   train(config)
   PY
   ```
4. Sampling:
   ```bash
   python scripts/sample_images.py --checkpoint experiments/cifar10_baseline/checkpoints/checkpoint_epoch_0001.pth \
       --use-clip --prompts "a dog|a ship|a red car" --output samples.png
   ```
5. Colab: open `notebooks/02_train_dit_with_clip_colab.py`, paste cells into a Colab notebook, and run `pip install -e .` in `/content/dit-clip-research`.

## Repository layout
- `src/dit_clip_research/` — library code
  - `models/dit.py` — DiT backbone with adaLN-Zero
  - `models/diffusion.py` — beta schedules, q/p sampling
  - `models/clip_conditioning.py` — CLIP text encoder wrapper
  - `data/` — datasets, transforms for image–text pairs
  - `training/` — train loop, sampling helpers, schedulers
  - `utils/` — config, logging, distributed stubs
- `configs/` — YAML configs (CIFAR-10 baseline, custom dataset template)
- `notebooks/` — Colab-ready cell scripts for architecture exploration and training
- `experiments/` — example run directories, configs, and sample outputs
- `scripts/` — dataset download and sampling CLI helpers
- `docs/` — paper summary, architecture notes, and CLIP training guide

## References
- Paper: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2308.07926)
- Official repo: [facebookresearch/DiT](https://github.com/facebookresearch/DiT)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP) / [open-clip](https://github.com/mlfoundations/open_clip)
