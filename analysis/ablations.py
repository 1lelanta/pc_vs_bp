"""Ablations over PC's two hyperparameters, scored on the VALIDATION set.

  sweep_T   : vary inference steps T      (state_lr fixed)
  sweep_lr  : vary state learning rate    (T fixed)

Each setting trains from the same initial weights and the same batch
order, so differences come only from the hyperparameter being swept. A
backprop run with the same budget is included as a reference line.

The test set is never touched here. Pick the best setting on validation,
then pass it to run_comparison via --T and --state-lr.

Run:  python -m analysis.ablations --epochs 3 --train-size 10000
"""
import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import ExperimentCfg
from data import get_loaders
from models.backprop import BackpropLearner
from models.predictive import PCLearner
from training.trainer import Trainer, seed_everything


def subsample(loader: DataLoader, n: Optional[int], seed: int) -> DataLoader:
    """Optional smaller training set so sweeps finish quickly."""
    if n is None or n >= len(loader.dataset):
        return loader
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(loader.dataset), generator=gen)[:n].tolist()
    return DataLoader(Subset(loader.dataset, idx),
                      batch_size=loader.batch_size, shuffle=True,
                      num_workers=loader.num_workers)


def summarize_run(history, learner_name: str) -> Dict[str, float]:
    train_loss = history.series("train_loss")
    diverged = not all(math.isfinite(v) for v in train_loss)
    out = {
        "diverged": diverged,
        "val_acc": history.series("val_acc")[-1],
        "val_loss": history.series("val_loss")[-1],
        "epoch_time": float(np.mean(history.series("epoch_time"))),
    }
    if learner_name == "predictive_coding":
        out["final_energy"] = history.series("train_energy")[-1]
    return out


def train_pc(cfg: ExperimentCfg, seed: int, T: int, state_lr: float,
             epochs: int, train_loader, val_loader) -> Dict[str, float]:
    seed_everything(seed)
    pc_cfg = replace(cfg.pc, inference_steps=T, state_lr=state_lr)
    pc = PCLearner(cfg.model, cfg.train, pc_cfg)
    seed_everything(seed)                       # identical batch order
    history = Trainer(pc, train_loader, val_loader, seed=seed,
                      verbose=False).fit(epochs)
    return {"T": T, "state_lr": state_lr, **summarize_run(history, pc.name)}


def train_backprop(cfg: ExperimentCfg, seed: int, epochs: int,
                   train_loader, val_loader) -> Dict[str, float]:
    seed_everything(seed)
    bp = BackpropLearner(cfg.model, cfg.train)
    seed_everything(seed)
    history = Trainer(bp, train_loader, val_loader, seed=seed,
                      verbose=False).fit(epochs)
    return summarize_run(history, bp.name)


def print_sweep(title: str, rows: List[dict], key: str, ref: dict) -> None:
    print(f"\n{title}")
    header = f"{key:>9} | {'val acc %':>10} {'val loss':>10} {'s/epoch':>9}  note"
    print(header)
    print("-" * len(header))
    for r in rows:
        note = "DIVERGED" if r["diverged"] else ""
        print(f"{r[key]:>9} | {100 * r['val_acc']:>10.2f} "
              f"{r['val_loss']:>10.4f} {r['epoch_time']:>9.2f}  {note}")
    print(f"{'backprop':>9} | {100 * ref['val_acc']:>10.2f} "
          f"{ref['val_loss']:>10.4f} {ref['epoch_time']:>9.2f}  (reference)")


def best_setting(rows: List[dict]) -> Optional[dict]:
    ok = [r for r in rows if not r["diverged"]]
    return max(ok, key=lambda r: r["val_acc"]) if ok else None


def main() -> None:
    p = argparse.ArgumentParser(description="PC hyperparameter ablations")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-size", type=int, default=None,
                   help="use a random subset of the training set")
    p.add_argument("--T-list", type=int, nargs="+",
                   default=[1, 2, 5, 10, 20, 50])
    p.add_argument("--lr-list", type=float, nargs="+",
                   default=[0.01, 0.05, 0.1, 0.2, 0.5])
    p.add_argument("--fixed-T", type=int, default=None)
    p.add_argument("--fixed-lr", type=float, default=None)
    p.add_argument("--out-dir", default="./results")
    args = p.parse_args()

    cfg = ExperimentCfg()
    fixed_T = args.fixed_T or cfg.pc.inference_steps
    fixed_lr = args.fixed_lr or cfg.pc.state_lr

    seed_everything(args.seed)
    train_loader, val_loader, _ = get_loaders(cfg.data, seed=args.seed)
    train_loader = subsample(train_loader, args.train_size, args.seed)

    ref = train_backprop(cfg, args.seed, args.epochs, train_loader, val_loader)

    sweep_T = []
    for T in args.T_list:
        print(f"sweep T: T={T}, state_lr={fixed_lr} ...")
        sweep_T.append(train_pc(cfg, args.seed, T, fixed_lr, args.epochs,
                                train_loader, val_loader))

    sweep_lr = []
    for lr in args.lr_list:
        print(f"sweep state_lr: T={fixed_T}, state_lr={lr} ...")
        sweep_lr.append(train_pc(cfg, args.seed, fixed_T, lr, args.epochs,
                                 train_loader, val_loader))

    print_sweep(f"Sweep over T (state_lr={fixed_lr})", sweep_T, "T", ref)
    print_sweep(f"Sweep over state_lr (T={fixed_T})", sweep_lr, "state_lr", ref)

    bT, bL = best_setting(sweep_T), best_setting(sweep_lr)
    if bT:
        print(f"\nBest T on validation: {bT['T']} "
              f"(val acc {100 * bT['val_acc']:.2f}%)")
    if bL:
        print(f"Best state_lr on validation: {bL['state_lr']} "
              f"(val acc {100 * bL['val_acc']:.2f}%)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablations.json").write_text(json.dumps({
        "epochs": args.epochs, "seed": args.seed,
        "train_size": args.train_size, "fixed_T": fixed_T,
        "fixed_state_lr": fixed_lr, "reference_backprop": ref,
        "sweep_T": sweep_T, "sweep_lr": sweep_lr}, indent=2))


if __name__ == "__main__":
    main()