"""Gradient alignment: how close are PC's local updates to backprop's?

ANALYSIS ONLY. The backprop gradient below is computed with autograd purely
as a reference to compare against. It is never used to update any weights.

For each layer we report:
  cosine      cos(PC update, BP gradient)   -> 1.0 means same direction
  norm_ratio  ||PC update|| / ||BP gradient||  -> how the magnitudes compare

Three views are produced:
  1. alignment along training (fixed T)
  2. alignment vs number of inference steps T
  3. alignment vs target "nudge" beta (small output error => theory says
     PC ~ backprop). beta = 1 is the real one-hot target.

Run:  python -m analysis.gradient_alignment --epochs 3
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import torch

from config import ExperimentCfg
from data import get_loaders, to_one_hot
from models.base import squared_error_loss
from models.predictive import PCLearner
from training.trainer import Trainer, seed_everything


# ----------------------------------------------------------------------
# The two gradients being compared
# ----------------------------------------------------------------------
def backprop_weight_grads(learner: PCLearner, x: torch.Tensor,
                          target: torch.Tensor) -> List[torch.Tensor]:
    """Reference only: dL/dW from autograd. Uses torch.autograd.grad, so
    nothing is written to any parameter's .grad."""
    with torch.enable_grad():
        weights = [layer.weight for layer in learner.layers]
        loss = squared_error_loss(learner.forward(x), target)
        grads = torch.autograd.grad(loss, weights)
    return [g.detach() for g in grads]


def pc_weight_grads(learner: PCLearner, x: torch.Tensor,
                    target: torch.Tensor, steps: int) -> List[torch.Tensor]:
    """PC's local weight updates after `steps` inference iterations."""
    result = learner.infer(x, target, steps=steps)
    return [gW for gW, _ in learner.weight_gradients(result)]


def nudged_target(learner: PCLearner, x: torch.Tensor,
                  target: torch.Tensor, beta: float) -> torch.Tensor:
    """Move the target a fraction beta of the way from the network's own
    output to the true target. beta -> 0 means a vanishing output error."""
    with torch.no_grad():
        out = learner.forward(x)
    return out + beta * (target - out)


# ----------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------
def alignment(learner: PCLearner, x: torch.Tensor, y: torch.Tensor,
              steps: int, beta: float = 1.0) -> Dict[str, List[float]]:
    """Per-layer cosine similarity and norm ratio on one batch."""
    target = nudged_target(learner, x, to_one_hot(y).to(x.device), beta)
    pc = pc_weight_grads(learner, x, target, steps)
    bp = backprop_weight_grads(learner, x, target)

    cos, ratio = [], []
    for p, b in zip(pc, bp):
        cos.append(torch.nn.functional.cosine_similarity(
            p.flatten(), b.flatten(), dim=0).item())
        ratio.append((p.norm() / (b.norm() + 1e-12)).item())
    return {"cosine": cos, "norm_ratio": ratio}


def alignment_vs_steps(learner, x, y, steps_list, beta=1.0) -> dict:
    return {str(T): alignment(learner, x, y, steps=T, beta=beta)
            for T in steps_list}


def alignment_vs_nudge(learner, x, y, betas, steps) -> dict:
    return {str(b): alignment(learner, x, y, steps=steps, beta=b)
            for b in betas}


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def print_cosine_table(title: str, table: dict, key_name: str) -> None:
    n_layers = len(next(iter(table.values()))["cosine"])
    print(f"\n{title}")
    header = f"{key_name:>8} | " + " ".join(f"layer {i + 1:<3}" for i in range(n_layers))
    print(header)
    print("-" * len(header))
    for key, r in table.items():
        print(f"{key:>8} | " + " ".join(f"{c:>8.3f}" for c in r["cosine"]))


def run(cfg: ExperimentCfg, seed: int, epochs: int, steps_list: List[int],
        betas: List[float], out: Path) -> None:
    seed_everything(seed)
    train_loader, val_loader, _ = get_loaders(cfg.data, seed=seed)

    seed_everything(seed)
    pc = PCLearner(cfg.model, cfg.train, cfg.pc)

    px, py = next(iter(val_loader))              # fixed probe batch
    px, py = px.to(pc.device), py.to(pc.device)

    # 1) alignment along training, measured at the configured T
    along_training: Dict[str, dict] = {}

    def record(epoch: int) -> None:
        along_training[str(epoch)] = alignment(
            pc, px, py, steps=cfg.pc.inference_steps)

    record(0)                                    # untrained network
    trainer = Trainer(pc, train_loader, val_loader, seed=seed,
                      on_epoch_end=record)
    trainer.fit(epochs)

    # 2) and 3) sweeps on the trained network
    vs_steps = alignment_vs_steps(pc, px, py, steps_list)
    vs_nudge = alignment_vs_nudge(pc, px, py, betas,
                                  steps=max(steps_list))

    result = {"seed": seed, "epochs": epochs,
              "along_training": along_training,
              "vs_steps": vs_steps, "vs_nudge": vs_nudge}
    out.mkdir(parents=True, exist_ok=True)
    (out / "gradient_alignment.json").write_text(json.dumps(result, indent=2))

    print_cosine_table(f"Cosine vs epoch (T={cfg.pc.inference_steps})",
                       along_training, "epoch")
    print_cosine_table("Cosine vs inference steps T (beta=1)",
                       vs_steps, "T")
    print_cosine_table(f"Cosine vs target nudge beta (T={max(steps_list)})",
                       vs_nudge, "beta")


def main() -> None:
    p = argparse.ArgumentParser(description="PC vs BP gradient alignment")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, nargs="+",
                   default=[0, 1, 2, 5, 10, 20, 50, 100, 200])
    p.add_argument("--betas", type=float, nargs="+",
                   default=[1.0, 0.5, 0.1, 0.01, 0.001])
    p.add_argument("--state-lr", type=float, default=None)
    p.add_argument("--out-dir", default="./results")
    args = p.parse_args()

    cfg = ExperimentCfg()
    if args.state_lr:
        cfg = replace(cfg, pc=replace(cfg.pc, state_lr=args.state_lr))
    run(cfg, args.seed, args.epochs, args.steps, args.betas,
        Path(args.out_dir))


if __name__ == "__main__":
    main()