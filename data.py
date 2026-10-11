"""Data loading for the PC vs. Backprop study.

The whole dataset is converted to normalised float tensors ONCE and kept
in memory. That removes per-batch image decoding, so epoch times measure
the learning rule and not the data pipeline. Both learners consume
exactly the same loaders.
"""
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

from config import DataCfg

# name -> (torchvision class, mean, std)
_DATASETS = {
    "mnist": (datasets.MNIST, 0.1307, 0.3081),
    "fashion_mnist": (datasets.FashionMNIST, 0.2860, 0.3530),
}

NUM_CLASSES = 10


def _load_split(ds_cls, root: str, train: bool, mean: float, std: float
                ) -> TensorDataset:
    """Download if needed, then convert the whole split to tensors once.
    Images become (N, 784) floats in the same normalisation as before."""
    ds = ds_cls(root, train=train, download=True)
    x = ds.data.float().div_(255.0).sub_(mean).div_(std).view(len(ds), -1)
    y = ds.targets.long()
    return TensorDataset(x, y)


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
    full_train = _load_split(ds_cls, cfg.root, True, mean, std)
    test_set = _load_split(ds_cls, cfg.root, False, mean, std)

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