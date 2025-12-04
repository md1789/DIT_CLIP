"""CLIP text encoder wrapper used for conditioning DiT."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch

try:
    import open_clip
except ImportError as exc:  # pragma: no cover - dependency is optional at runtime
    raise ImportError("open-clip-torch is required for CLIP conditioning") from exc


class ClipTextEncoder:
    """Thin wrapper around `open_clip` that returns normalized text embeddings."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or torch.device("cpu")

        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.eval().to(self.device)

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """Tokenize and encode a list of text prompts."""

        device = device or self.device
        tokens = self.tokenizer(list(prompts)).to(device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    def to(self, device: torch.device) -> "ClipTextEncoder":
        self.device = device
        self.model = self.model.to(device)
        return self


@torch.no_grad()
def cache_text_embeddings(
    dataset: Iterable,
    encoder: ClipTextEncoder,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Pre-compute text embeddings for a dataset.

    The dataset is expected to yield dictionaries with a ``"text"`` field.
    Returned embeddings are aligned with dataset ordering.
    """

    device = device or encoder.device
    buffer: List[str] = []
    cached: List[torch.Tensor] = []

    for sample in dataset:
        buffer.append(sample["text"])
        if len(buffer) >= batch_size:
            cached.append(encoder.encode_text(buffer, device=device).cpu())
            buffer = []

    if buffer:
        cached.append(encoder.encode_text(buffer, device=device).cpu())

    return cached
