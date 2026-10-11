"""Model-agnostic training loop.

The trainer only calls learner.train_step / learner.evaluate. It never
calls loss.backward(), never mentions energy or inference, and logs
whatever metric keys the learner returns, so PC's energies show up in
the history without the trainer knowing they exist.
"""
import random
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.base import LearnerBase
from training.metrics import History, Timer, WeightedMean


def seed_everything(seed: int) -> None:
    """Seed every RNG that affects initialisation or data order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loader(learner: LearnerBase, loader: DataLoader
                    ) -> Dict[str, float]:
    """Loss and accuracy over a whole loader, averaged per sample."""
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        r = learner.evaluate(x, y)
        total_loss += r["loss"] * r["n"]
        correct += r["correct"]
        n += r["n"]
    return {"loss": total_loss / n, "acc": correct / n}


class Trainer:
    def __init__(self, learner: LearnerBase, train_loader: DataLoader,
                 val_loader: DataLoader, seed: int = 0,
                 verbose: bool = True,
                 on_epoch_end: Optional[Callable[[int], None]] = None):
        self.learner = learner
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.seed = seed
        self.verbose = verbose
        self.on_epoch_end = on_epoch_end

    def fit(self, epochs: int) -> History:
        history = History(name=self.learner.name, seed=self.seed)
        device = self.learner.train_cfg.device

        for epoch in range(1, epochs + 1):
            meter = WeightedMean()

            # Timed section covers training only (not validation), so
            # epoch_time reflects the true cost of each learning rule.
            with Timer(device) as timer:
                for x, y in self.train_loader:
                    metrics = self.learner.train_step(x, y)
                    meter.update(metrics, y.size(0))

            val = evaluate_loader(self.learner, self.val_loader)

            record = {f"train_{k}": v for k, v in meter.result().items()}
            record.update(epoch=epoch, val_loss=val["loss"],
                          val_acc=val["acc"], epoch_time=timer.elapsed)
            history.log_epoch(**record)

            if self.verbose:
                print(f"[{self.learner.name} | seed {self.seed}] "
                      f"epoch {epoch:>3}/{epochs} "
                      f"train_loss {record['train_loss']:.4f} "
                      f"val_loss {val['loss']:.4f} "
                      f"val_acc {val['acc']:.4f} "
                      f"({timer.elapsed:.1f}s)")

            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch)
        return history

    def test(self, test_loader: DataLoader, history: History) -> Dict[str, float]:
        """Final evaluation. Call once, after all tuning is finished."""
        result = evaluate_loader(self.learner, test_loader)
        history.final = {"test_loss": result["loss"],
                         "test_acc": result["acc"]}
        return history.final