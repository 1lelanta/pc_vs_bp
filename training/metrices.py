"""Bookkeeping utilities: weighted averages, timing, and a History
container that is saved to JSON next to the config."""
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import torch


class WeightedMean:
    """Running sample-weighted mean of named scalars.

    Weighting by batch size keeps the epoch average correct even when
    the last batch is smaller than the rest.
    """

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._count = 0

    def update(self, values: Dict[str, float], n: int) -> None:
        for key, val in values.items():
            self._sums[key] = self._sums.get(key, 0.0) + val * n
        self._count += n

    def result(self) -> Dict[str, float]:
        return {k: s / self._count for k, s in self._sums.items()}


class Timer:
    """Wall-clock context manager. Synchronises CUDA so GPU timings are
    honest (otherwise asynchronous kernels make PC look faster than it is)."""

    def __init__(self, device: str = "cpu"):
        self.use_cuda = device.startswith("cuda")
        self.elapsed = 0.0

    def __enter__(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start


@dataclass
class History:
    """Everything recorded for one (learner, seed) run."""
    name: str
    seed: int
    epochs: List[Dict[str, float]] = field(default_factory=list)
    final: Dict[str, float] = field(default_factory=dict)  # test metrics

    def log_epoch(self, **metrics: float) -> None:
        self.epochs.append(dict(metrics))

    def series(self, key: str) -> List[float]:
        """One metric across epochs, e.g. history.series("val_acc")."""
        return [e[key] for e in self.epochs]

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path) -> "History":
        return History(**json.loads(Path(path).read_text()))