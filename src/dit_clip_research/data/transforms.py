"""Reusable torchvision transform helpers."""
from __future__ import annotations

from typing import Tuple

import torchvision.transforms as T


IMAGENET_NORMALIZATION = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def training_transforms(image_size: int = 32, normalize_to_unit: bool = False) -> T.Compose:
    """Augmentations for training."""

    mean, std = IMAGENET_NORMALIZATION if normalize_to_unit else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def eval_transforms(image_size: int = 32, normalize_to_unit: bool = False) -> T.Compose:
    """Deterministic transforms for evaluation/sampling."""

    mean, std = IMAGENET_NORMALIZATION if normalize_to_unit else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std)])


def denormalize(tensor, mean_std: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    """Undo normalization for visualization."""

    mean, std = mean_std
    mean = tensor.new_tensor(mean).view(1, -1, 1, 1)
    std = tensor.new_tensor(std).view(1, -1, 1, 1)
    return tensor * std + mean
