# Scalable Diffusion Models with Transformers (DiT) — working notes

## Core ideas
- **Latent diffusion**: Images are encoded with a VAE; diffusion happens in latent space, which reduces compute and enables larger transformer models. (This repo defaults to pixel-space for simplicity; swap in a VAE encoder/decoder to mirror the paper.)
- **Patchified transformer U-Net**: Instead of convolutions, the DiT treats latent feature maps as non-overlapping patches that a Vision Transformer backbone processes. See `PatchEmbed`/`unpatchify` for the tokenization and reconstruction logic.
- **adaLN / adaLN-Zero conditioning**: Timesteps (and other conditioning vectors) modulate LayerNorm scales and shifts plus residual gates, allowing conditioning to flow through every block. Implemented in `DiTBlock` and `FinalLayer`.
- **Scaling behavior**: Model families (DiT-S/B/L/XL with patch sizes like /2 or /4) trade FLOPs for quality. Larger backbones and smaller patch sizes improve FID at the cost of compute. Hidden size, depth, and patch size in `DiT` map directly to these variants.
- **Classifier-free guidance**: Guidance is applied at sampling time by mixing conditional and unconditional predictions. The current sampling loop is unconditional; plug guidance into `p_sample` to combine two model calls if needed.

## Paper sections mapped to code
- *Model architecture* (Sec. 3): `src/dit_clip_research/models/dit.py` (`PatchEmbed`, `DiTBlock`, `FinalLayer`, adaLN-Zero path).
- *Diffusion objectives* (Sec. 2): `src/dit_clip_research/models/diffusion.py` (`q_sample`, `p_sample_loop`, configurable beta schedules).
- *Conditioning* (Sec. 3.2 / 4.2): `src/dit_clip_research/models/clip_conditioning.py` and the conditioning merge in `DiT.forward`.
- *Scaling results* (Sec. 5): Adjust `hidden_size`, `depth`, `num_heads`, and `patch_size` when building `DiT` instances or YAML configs to reproduce DiT-S/4, DiT-B/4, etc.
- *Guidance & sampling* (Appendix): Extend `p_sample` with CFG by running the model twice (conditional + unconditional) and mixing outputs before computing `model_mean`.
