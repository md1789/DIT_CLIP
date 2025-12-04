"""Helper script to fetch datasets used in the examples."""
from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import CIFAR10


def download_cifar10(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    CIFAR10(root=str(root), train=True, download=True)
    CIFAR10(root=str(root), train=False, download=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for DiT experiments")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"], help="Dataset name")
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Output directory")
    args = parser.parse_args()

    if args.dataset == "cifar10":
        download_cifar10(args.data_root)
        print(f"CIFAR-10 downloaded to {args.data_root}")


if __name__ == "__main__":
    main()
