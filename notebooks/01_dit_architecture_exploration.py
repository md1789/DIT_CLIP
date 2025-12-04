# %% [markdown]
# # DiT Architecture Exploration
# This notebook-like script showcases how to instantiate the DiT backbone,
# inspect parameter counts, and run a single diffusion step.

# %%
import torch
from dit_clip_research.models.dit import DiT
from dit_clip_research.models.diffusion import DiffusionSchedule, create_diffusion_config, q_sample

# %% [markdown]
# ## Instantiate a tiny DiT

# %%
model = DiT(img_size=32, patch_size=4, hidden_size=256, depth=4, num_heads=4)
total_params = sum(p.numel() for p in model.parameters())
print(f"DiT parameters: {total_params/1e6:.2f}M")

# %% [markdown]
# ## Forward a noisy CIFAR-sized batch

# %%
bs = 2
x_start = torch.randn(bs, 3, 32, 32)
t = torch.randint(0, 1000, (bs,))

# Build a diffusion schedule and add noise to the clean image
schedule = DiffusionSchedule(create_diffusion_config(timesteps=50))
x_noisy = q_sample(schedule, x_start, t)

with torch.no_grad():
    pred_noise = model(x_noisy, t)

print("Input shape", x_noisy.shape, "Output shape", pred_noise.shape)

# %% [markdown]
# You can tweak `hidden_size`, `depth`, `num_heads`, and `patch_size` to mirror
# DiT-S/4, DiT-B/4, etc., and observe how parameter counts scale.
