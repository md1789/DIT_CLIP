"""Diffusion Transformer (DiT) architecture with optional CLIP conditioning.

The code follows the design in "Scalable Diffusion Models with Transformers"
while keeping the implementation compact for research and notebook usage.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from einops import rearrange


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation.

    The adaptive layer norm (adaLN / adaLN-Zero) uses a conditioning vector to
    generate per-channel ``shift`` and ``scale`` terms. This utility keeps the
    algebra in one place.
    """

    return x * (1 + scale) + shift


class PatchEmbed(nn.Module):
    """Convert an image to a sequence of non-overlapping patches.

    Images are split into ``patch_size x patch_size`` blocks using a strided
    convolution. Each patch is projected to ``embed_dim`` so that the transformer
    can process them like tokens.
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP projection."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # adaLN-Zero produces modulation + gates for both attention and MLP.
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)

        x_mod1 = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        attn_output, _ = self.attn(x_mod1, x_mod1, x_mod1)
        x = x + gate_msa.unsqueeze(1) * attn_output

        x_mod2 = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod2)
        return x


class FinalLayer(nn.Module):
    """Project hidden states back to image patches with adaLN conditioning."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.linear(x)


class DiT(nn.Module):
    """Diffusion Transformer backbone.

    The model uses patch embeddings + learned position embeddings, then applies a
    stack of :class:`DiTBlock` layers. Conditioning is provided through the
    adaLN modulation path: diffusion timesteps are embedded with
    :class:`TimestepEmbedder`, and optional CLIP text embeddings are projected
    and added to the timestep embedding before entering the blocks.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        use_clip_conditioning: bool = False,
        clip_embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.out_channels = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_clip_conditioning = use_clip_conditioning

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, hidden_size)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.use_clip_conditioning:
            if clip_embed_dim is None:
                raise ValueError("clip_embed_dim must be provided when use_clip_conditioning=True")
            self.clip_proj = nn.Linear(clip_embed_dim, hidden_size)
        else:
            self.clip_proj = None

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from patch tokens."""

        p = self.patch_size
        c = self.out_channels
        h = w = self.img_size // p
        return rearrange(x, "n (h w) (p1 p2 c) -> n c (h p1) (w p2)", h=h, w=w, p1=p, p2=p, c=c)

    def forward(self, x: torch.Tensor, t: torch.Tensor, clip_text: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed

        cond = self.t_embedder(t)
        if self.clip_proj is not None and clip_text is not None:
            cond = cond + self.clip_proj(clip_text)
