"""One command reproduces the whole comparison.

    python -m experiments.run_comparison                  # full run
    python -m experiments.run_comparison --epochs 1 --seeds 0   # smoke test

For each seed it builds BOTH learners from identical initial weights,
trains them on identical batch orders, evaluates once on the test set,
and saves everything (histories, PC energy traces, config, summary) to
the output directory for the analysis scripts to plot.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import ExperimentCfg
from data import get_loaders, to_one_hot
from models.backprop import BackpropLearner
from models.predictive import PCLearner
from training.metrics import History
from training.trainer import Trainer, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backprop vs Predictive Coding")
    p.add_argument("--dataset", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--T", type=int, default=None, help="PC inference steps")
    p.add_argument("--state-lr", type=float, default=None)
    p.add_argument("--lr", type=float, default=None, help="weight lr")
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def build_cfg(args: argparse.Namespace) -> ExperimentCfg:
    """Defaults come from config.py; command-line flags override them."""
    cfg = ExperimentCfg()
    data, pc, train = cfg.data, cfg.pc, cfg.train
    if args.dataset: data = replace(data, dataset=args.dataset)
    if args.T: pc = replace(pc, inference_steps=args.T)
    if args.state_lr: pc = replace(pc, state_lr=args.state_lr)
    if args.epochs: train = replace(train, epochs=args.epochs)
    if args.seeds: train = replace(train, seeds=tuple(args.seeds))
    if args.lr: train = replace(train, lr=args.lr)
    if args.device: train = replace(train, device=args.device)
    if args.out_dir: train = replace(train, out_dir=args.out_dir)
    return replace(cfg, data=data, pc=pc, train=train)


def make_energy_recorder(pc: PCLearner, probe_x, probe_y, store: dict):
    """Callback that records PC's (T+1, L) layer-wise energy trace on a
    fixed probe batch. Calling it with epoch=0 captures the untrained net."""
    px = probe_x.to(pc.device)
    target = to_one_hot(probe_y).to(pc.device)

    def record(epoch: int) -> None:
        store[str(epoch)] = pc.infer(px, target).energy_trace.cpu().tolist()
    return record


def run_seed(cfg: ExperimentCfg, seed: int, out: Path) -> Dict[str, History]:
    seed_everything(seed)
    train_loader, val_loader, test_loader = get_loaders(cfg.data, seed=seed)

    # Identical starting point: build backprop, then copy its weights.
    seed_everything(seed)
    bp = BackpropLearner(cfg.model, cfg.train)
    pc = PCLearner(cfg.model, cfg.train, cfg.pc)
    pc.load_weights_from(bp)

    probe_x, probe_y = next(iter(val_loader))       # fixed, unshuffled batch
    traces: dict = {}
    record = make_energy_recorder(pc, probe_x, probe_y, traces)

    results: Dict[str, History] = {}
    for learner in (bp, pc):
        callback = None
        if learner is pc:
            record(0)                               # energy before training
            callback = record

        seed_everything(seed)                       # identical batch order
        trainer = Trainer(learner, train_loader, val_loader, seed=seed,
                          on_epoch_end=callback)
        history = trainer.fit(cfg.train.epochs)
        trainer.test(test_loader, history)          # test touched once
        history.save(out / f"{learner.name}_seed{seed}.json")
        results[learner.name] = history

    (out / f"pc_energy_traces_seed{seed}.json").write_text(json.dumps(traces))
    return results


def _mean_std(values: List[float]):
    arr = np.asarray(values, dtype=float)
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return float(arr.mean()), float(std)


def summarize(runs: List[Dict[str, History]]) -> dict:
    summary = {}
    for name in runs[0]:
        hs = [r[name] for r in runs]
        acc_m, acc_s = _mean_std([h.final["test_acc"] for h in hs])
        loss_m, loss_s = _mean_std([h.final["test_loss"] for h in hs])
        time_m, _ = _mean_std([np.mean(h.series("epoch_time")) for h in hs])
        summary[name] = {"test_acc_mean": acc_m, "test_acc_std": acc_s,
                         "test_loss_mean": loss_m, "test_loss_std": loss_s,
                         "epoch_time_mean": time_m, "n_seeds": len(hs)}
    return summary


def print_table(summary: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'learner':<20}{'test acc (%)':>20}{'test loss':>20}{'s / epoch':>14}")
    print("-" * 78)
    for name, s in summary.items():
        acc = f"{100 * s['test_acc_mean']:.2f} ± {100 * s['test_acc_std']:.2f}"
        loss = f"{s['test_loss_mean']:.4f} ± {s['test_loss_std']:.4f}"
        print(f"{name:<20}{acc:>20}{loss:>20}{s['epoch_time_mean']:>14.2f}")
    if {"backprop", "predictive_coding"} <= summary.keys():
        ratio = (summary["predictive_coding"]["epoch_time_mean"]
                 / summary["backprop"]["epoch_time_mean"])
        print(f"\nPC is {ratio:.1f}x slower per epoch than backprop.")
    print("=" * 78)


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    out = Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    runs = [run_seed(cfg, seed, out) for seed in cfg.train.seeds]

    summary = summarize(runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print_table(summary)


if __name__ == "__main__":
    main()