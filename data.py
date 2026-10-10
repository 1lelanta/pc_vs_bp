"""Data loading for the PC vs. Backprop study.

Both learners consume exactly the same loaders, so any difference in
results comes from the learning rule and not from the data pipeline.
"""
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import DataCfg

# name -> (torchvision class, mean, std)
_DATASETS = {
    "mnist": (datasets.MNIST, 0.1307, 0.3081),
    "fashion_mnist": (datasets.FashionMNIST, 0.2860, 0.3530),
}

NUM_CLASSES = 10


def _flatten(t: torch.Tensor) -> torch.Tensor:
    """(1, 28, 28) image -> (784,) vector. A named function (not a lambda)
    so it stays picklable if num_workers > 0."""
    return t.view(-1)


def _build_transform(mean: float, std: float) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
        transforms.Lambda(_flatten),
    ])


def get_loaders(cfg: DataCfg, seed: int = 0
                ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    The train/validation split is seeded, so the same seed always gives
    the same split. Validation is for tuning; test is touched only for
    final reporting.
    """
    if cfg.dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'. "
                         f"Choose from {list(_DATASETS)}")

    ds_cls, mean, std = _DATASETS[cfg.dataset]
    tf = _build_transform(mean, std)

    full_train = ds_cls(cfg.root, train=True, download=True, transform=tf)
    test_set = ds_cls(cfg.root, train=False, download=True, transform=tf)

    n_val = int(len(full_train) * cfg.val_fraction)
    n_train = len(full_train) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val],
                                      generator=gen)

    def make(ds, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers)

    return make(train_set, True), make(val_set, False), make(test_set, False)


def to_one_hot(y: torch.Tensor, num_classes: int = NUM_CLASSES
               ) -> torch.Tensor:
    """Integer labels (B,) -> float one-hot targets (B, num_classes)."""
    return torch.nn.functional.one_hot(y, num_classes).float()