# CIFAR-10 Baseline Notes

- Config: see `experiments/cifar10_baseline/config.yaml`.
- Conditioning: CLIP text prompts generated from CIFAR-10 class names.
- Metrics: track MSE loss; plug in FID via `training/evaluation.py` stub when ready.
- Sampling: use `scripts/sample_images.py --checkpoint experiments/cifar10_baseline/checkpoints/checkpoint_epoch_0001.pth --use-clip --prompts "a dog|a ship"`.
