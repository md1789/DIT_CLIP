# Architecture notes for this implementation

## What changed versus the official DiT repo
- **Single-file modules**: Core pieces live in `src/dit_clip_research/models/` instead of a monorepo-style layout. It is intentionally compact for notebooks.
- **Pixel-space by default**: The original paper trains in VAE latent space; this repo keeps pixel-space diffusion for clarity. Swap in a VAE encoder/decoder at the data boundary to match the paper exactly.
- **Minimal dependencies**: Uses PyTorch, torchvision, einops, and open-clip. Distributed training is stubbed in `utils/distributed.py` for future extension.
- **Configuration**: YAML configs under `configs/` hydrate the `TrainConfig` dataclass. Hyperparameters map 1:1 to DiT-S/4, DiT-B/4, etc. by changing `hidden_size`, `depth`, `num_heads`, and `patch_size`.

## Key components and where to tweak them
- **Patch sizing / resolution**: `PatchEmbed` and `DiT.unpatchify` in `models/dit.py` use `patch_size` and `img_size`. Adjust those to move from /8 to /2 variants.
- **Model scale**: `hidden_size`, `depth`, and `num_heads` parameters in `DiT` control width and depth. Increase together to emulate larger DiT models; keep `mlp_ratio` at 4.0 unless experimenting with wider MLPs.
- **Conditioning**: The conditioning vector is `t_embed + clip_proj(text_embed)` when CLIP is enabled. To add new conditioning (class labels, VAE latents, etc.), project them to `hidden_size` and add to the conditioning vector before it enters `DiTBlock`.
- **Diffusion schedule**: `models/diffusion.py` holds `DiffusionConfig` and `DiffusionSchedule`. Swap schedules via YAML (`schedule: cosine|linear`) or by constructing a new config.
- **Sampling**: `training/evaluation.py` wraps `p_sample_loop` and grid creation. Add classifier-free guidance by mixing conditional/unconditional predictions inside `p_sample`.

## Extending to new datasets
- Implement a loader that returns `{image, text, text_embedding?}` or reuse `TextImagePairDataset` with a CSV/JSON manifest.
- Add a config under `configs/` pointing to the new data root/annotations.
- If text conditioning is heavy, precompute embeddings with `cache_text_embeddings` and store them in your manifest, then the training loop will reuse them automatically.
