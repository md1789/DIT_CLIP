# %% [markdown]
# # Train DiT with CLIP Conditioning (Colab template)
# This script is organized into Colab-friendly cells so you can paste it into a
# notebook. It uses the library code under `src/` via `pip install -e .`.

# %% [markdown]
# ## 1) Environment setup

# %%
# If this notebook runs in Google Colab, start in /content.
# Clone your repository (replace URL if needed) and install the package.
# !git clone https://github.com/your-username/dit-clip-research.git
# %cd /content/dit-clip-research

# Install dependencies in editable mode
# !pip install -e .
# !pip install open-clip-torch

# %% [markdown]
# ## 2) Imports and config

# %%
import torch
from pathlib import Path

from dit_clip_research.training.train_loop import TrainConfig, train
from dit_clip_research.utils.config import load_config_dict
from dit_clip_research.models.dit import DiT
from dit_clip_research.models.diffusion import DiffusionSchedule, create_diffusion_config
from dit_clip_research.training.evaluation import sample_images, make_sample_grid
from dit_clip_research.models.clip_conditioning import ClipTextEncoder

# Load a YAML config from the configs/ folder
cfg_path = Path("configs/cifar10_baseline.yaml")
if cfg_path.exists():
    cfg_dict = load_config_dict(str(cfg_path))
else:
    cfg_dict = {}

train_config = TrainConfig(**cfg_dict.get("training", {}))
train_config.diffusion = cfg_dict.get("diffusion", train_config.diffusion)

print(train_config)

# %% [markdown]
# ## 3) Train for a few epochs (sanity check)

# %%
train(train_config)

# %% [markdown]
# Optionally mount Google Drive to save checkpoints during Colab sessions.

# %%
# from google.colab import drive
# drive.mount(/content/drive)
# drive_dir = Path(/content/drive/MyDrive/dit_clip_runs)
# train_config.save_dir = str(drive_dir / checkpoints)
# train(train_config)

# %% [markdown]
# ## 4) Sampling with prompts

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ckpt = Path(train_config.save_dir) / "checkpoint_epoch_0001.pth"
checkpoint = torch.load(model_ckpt, map_location=device)

model = DiT(
    img_size=train_config.image_size,
    patch_size=train_config.patch_size,
    hidden_size=train_config.hidden_size,
    depth=train_config.depth,
    num_heads=train_config.num_heads,
    mlp_ratio=train_config.mlp_ratio,
    use_clip_conditioning=train_config.use_clip_conditioning,
    clip_embed_dim=train_config.clip_embed_dim,
).to(device)
model.load_state_dict(checkpoint["model"])

schedule = DiffusionSchedule(create_diffusion_config(**train_config.diffusion)).to(device)

prompts = [
    "a small airplane on a runway",
    "a dog running through grass",
    "a ship on the ocean at sunset",
    "a red automobile in a city street",
]

clip_encoder = None
if train_config.use_clip_conditioning:
    clip_encoder = ClipTextEncoder(device=device)

samples = sample_images(model, schedule, prompts=prompts, clip_encoder=clip_encoder, num_samples=len(prompts), device=device)

grid = make_sample_grid(samples, nrow=2)

# visualize inside colab (matplotlib or display)
# import matplotlib.pyplot as plt
# plt.imshow(grid.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
# plt.axis(off)
# plt.show()

# save to file
# from torchvision.utils import save_image
# save_image(grid, colab_samples.png)
