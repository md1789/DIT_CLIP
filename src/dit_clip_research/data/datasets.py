"""Datasets for paired image-text training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from . import transforms as T


DEFAULT_CIFAR10_TEXT = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class TextImagePairDataset(Dataset):
    """Generic dataset that reads ``image_path`` and ``text`` pairs from metadata."""

    def __init__(
        self,
        root: str,
        annotations: str,
        image_key: str = "image_path",
        text_key: str = "text",
        embedding_key: str = "text_embedding",
        transform: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_key = image_key
        self.text_key = text_key
        self.embedding_key = embedding_key
        self.transform = transform or T.training_transforms()
        self.samples = self._load_annotations(annotations)

    def _load_annotations(self, annotations: str) -> List[Dict[str, Any]]:
        path = Path(annotations)
        if path.suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "items" in data:
                    return data["items"]
                return data
        if path.suffix in {".csv", ".tsv"}:
            df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
            return df.to_dict(orient="records")
        raise ValueError(f"Unsupported annotation format: {path.suffix}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.samples[idx]
        img_path = self.root / record[self.image_key]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        sample: Dict[str, Any] = {"image": image, "text": record.get(self.text_key, "")}
        if self.embedding_key in record:
            sample["text_embedding"] = torch.tensor(record[self.embedding_key], dtype=torch.float32)
        return sample


class Cifar10TextDataset(Dataset):
    """CIFAR-10 with synthetic captions (class name prompts)."""

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        download: bool = True,
        transform: Optional[Any] = None,
        text_prefix: str = "a photo of a",
    ) -> None:
        super().__init__()
        train = split == "train"
        self.dataset = CIFAR10(root=root, train=train, download=download, transform=transform or T.training_transforms())
        self.text_prefix = text_prefix

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, label = self.dataset[idx]
        text = f"{self.text_prefix} {DEFAULT_CIFAR10_TEXT[label]}"
        return {"image": image, "text": text, "label": label}


def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    texts = [b.get("text", "") for b in batch]

    if any("text_embedding" in b for b in batch):
        embeddings: List[torch.Tensor] = []
        for b in batch:
            emb = b.get("text_embedding")
            if emb is None:
                raise ValueError("Missing text_embedding for one of the samples when others are present.")
            embeddings.append(emb if isinstance(emb, torch.Tensor) else torch.tensor(emb, dtype=torch.float32))
        text_embeddings = torch.stack(embeddings)
    else:
        text_embeddings = None

    return {"image": images, "text": texts, "text_embedding": text_embeddings}


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        collate_fn=default_collate,
    )
