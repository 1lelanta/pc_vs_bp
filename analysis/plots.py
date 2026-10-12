"""Turn saved JSON results into report figures and a results table.

Reads only files written by the experiment scripts; never retrains.

    python -m analysis.plots
    python -m analysis.plots --default-dir ./results \
        --tuned-dir ./results_tuned --out-dir ./figures
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                      # headless: just write PNG files
import matplotlib.pyplot as plt
import numpy as np

from training.metrics import History

BP_COLOR, TUNED_COLOR, DEFAULT_COLOR = "#222222", "#1f77b4", "#ff7f0e"
LAYER_COLORS = ["#1b9e77", "#d95f02", "#7570b3"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False,
})


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------
def read_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.exists() else None


def load_histories(folder: Path, name: str) -> List[History]:
    return [History.load(p) for p in sorted(folder.glob(f"{name}_seed*.json"))]


def summary_entry(folder: Path, name: str) -> Optional[dict]:
    return (read_json(folder / "summary.json") or {}).get(name)


def build_runs(default_dir: Path, tuned_dir: Path) -> list:
    """[(label, color, histories, summary)] in plotting order."""
    runs = []
    bp_dir = tuned_dir if load_histories(tuned_dir, "backprop") else default_dir
    bp = load_histories(bp_dir, "backprop")
    if bp:
        runs.append(("Backprop", BP_COLOR, bp,
                     summary_entry(bp_dir, "backprop")))
    for folder, color in ((tuned_dir, TUNED_COLOR),
                          (default_dir, DEFAULT_COLOR)):
        pc = load_histories(folder, "predictive_coding")
        if not pc:
            continue
        pc_cfg = (read_json(folder / "config.json") or {}).get("pc", {})
        label = (f"PC (T={pc_cfg.get('inference_steps', '?')}, "
                 f"η={pc_cfg.get('state_lr', '?')})")
        runs.append((label, color, pc,
                     summary_entry(folder, "predictive_coding")))
    return runs


def layer_label(l: int, n: int) -> str:
    return f"layer {l + 1} (output)" if l == n - 1 else f"layer {l + 1} (hidden)"


def stack_cosines(table: dict) -> Tuple[List[float], np.ndarray]:
    keys = sorted(table, key=float)
    return [float(k) for k in keys], np.array([table[k]["cosine"] for k in keys])


def band(ax, histories, key, label, color, scale=1.0) -> None:
    arr = np.array([h.series(key) for h in histories]) * scale
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(mean)
    x = np.arange(1, arr.shape[1] + 1)
    ax.plot(x, mean, label=label, color=color, lw=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def fig_learning_curves(runs, out: Path) -> Optional[Path]:
    if not runs:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("val_acc", "Validation accuracy (%)", 100.0),
              ("val_loss", "Validation loss", 1.0),
              ("train_loss", "Training loss", 1.0)]
    for ax, (key, title, scale) in zip(axes, panels):
        for label, color, hist, _ in runs:
            band(ax, hist, key, label, color, scale)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
    axes[0].set_ylim(92, 98)
    axes[2].set_yscale("log")
    axes[0].legend(loc="lower right")
    fig.suptitle("Learning curves (mean ± std over seeds)")
    fig.tight_layout()
    path = out / "learning_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_energy(default_dir: Path, tuned_dir: Path, out: Path
               ) -> Optional[Path]:
    traces = None
    for folder in (default_dir, tuned_dir):       # prefer the longer T
        traces = read_json(folder / "pc_energy_traces_seed0.json")
        if traces:
            break
    if not traces:
        return None

    epochs = sorted(traces, key=int)
    first, last = epochs[0], epochs[-1]
    chosen = sorted({first, epochs[len(epochs) // 2], last}, key=int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ep in zip(axes[:2], (first, last)):
        arr = np.array(traces[ep])                # (T+1, L)
        for l in range(arr.shape[1]):
            ax.plot(np.arange(arr.shape[0]), arr[:, l], lw=2,
                    color=LAYER_COLORS[l % 3],
                    label=layer_label(l, arr.shape[1]))
        tag = " (untrained)" if ep == "0" else ""
        ax.set_title(f"Layer-wise energy, epoch {ep}{tag}")
        ax.set_xlabel("Inference step")
        ax.set_ylabel("Energy  E_l = ½‖ε_l‖²")
    axes[0].legend()

    for ep in chosen:
        total = np.array(traces[ep]).sum(axis=1)
        axes[2].plot(np.arange(len(total)), total, lw=2, label=f"epoch {ep}")
    axes[2].set_yscale("log")
    axes[2].set_title("Total energy F during inference")
    axes[2].set_xlabel("Inference step")
    axes[2].set_ylabel("F (log scale)")
    axes[2].legend()

    fig.suptitle("Energy dynamics (fixed probe batch, seed 0)")
    fig.tight_layout()
    path = out / "energy_dynamics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_alignment(default_dir: Path, out: Path) -> Optional[Path]:
    data = read_json(default_dir / "gradient_alignment.json")
    if data is None:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("vs_steps", "Inference steps T",
               "Cosine vs inference steps (β=1)", "symlog"),
              ("vs_nudge", "Target nudge β  (error shrinks →)",
               "Cosine vs target nudge", "log"),
              ("along_training", "Epoch",
               "Cosine along training", "linear")]
    for ax, (key, xlabel, title, scale) in zip(axes, panels):
        xs, arr = stack_cosines(data[key])
        for l in range(arr.shape[1]):
            ax.plot(xs, arr[:, l], marker="o", lw=2,
                    color=LAYER_COLORS[l % 3],
                    label=layer_label(l, arr.shape[1]))
        if scale == "symlog":
            ax.set_xscale("symlog", linthresh=1)
        elif scale == "log":
            ax.set_xscale("log")
            ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].set_ylabel("cos(PC weight update, backprop gradient)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Gradient alignment: PC vs backprop")
    fig.tight_layout()
    path = out / "gradient_alignment.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_ablations(default_dir: Path, out: Path) -> Optional[Path]:
    data = read_json(default_dir / "ablations.json")
    if data is None:
        return None
    ref = data["reference_backprop"]
    ref_acc = 100 * ref["val_acc"]
    sweep_T, sweep_lr = data["sweep_T"], data["sweep_lr"]
    low = 94.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) accuracy vs T
    ax = axes[0]
    ax.plot([r["T"] for r in sweep_T], [100 * r["val_acc"] for r in sweep_T],
            marker="o", lw=2, color=TUNED_COLOR, label="PC")
    ax.axhline(ref_acc, ls="--", color=BP_COLOR, label="Backprop")
    ax.set_xscale("log")
    ax.set_xlabel("Inference steps T")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"Accuracy vs T (state_lr={data['fixed_state_lr']})")
    ax.set_ylim(low, 97.2)
    ax.legend(loc="lower right")
    off = [r for r in sweep_T if 100 * r["val_acc"] < low]
    if off:
        ax.text(0.03, 0.05, "\n".join(
            f"T={r['T']}: {100 * r['val_acc']:.1f}% (off chart)" for r in off),
            transform=ax.transAxes, fontsize=8)

    # (b) accuracy vs state_lr
    ax = axes[1]
    ax.plot([r["state_lr"] for r in sweep_lr],
            [100 * r["val_acc"] for r in sweep_lr],
            marker="o", lw=2, color=DEFAULT_COLOR, label="PC")
    ax.axhline(ref_acc, ls="--", color=BP_COLOR, label="Backprop")
    ax.set_xscale("log")
    ax.set_xlabel("State learning rate η")
    ax.set_title(f"Accuracy vs η (T={data['fixed_T']})")
    ax.set_ylim(low, 97.2)
    off = [r for r in sweep_lr if 100 * r["val_acc"] < low]
    if off:
        ax.text(0.03, 0.05, "\n".join(
            f"η={r['state_lr']}: {100 * r['val_acc']:.1f}% (off chart)"
            for r in off), transform=ax.transAxes, fontsize=8)

    # (c) accuracy vs cost
    ax = axes[2]
    for rows, color, name in ((sweep_T, TUNED_COLOR, "T sweep"),
                              (sweep_lr, DEFAULT_COLOR, "η sweep")):
        ax.scatter([r["epoch_time"] for r in rows],
                   [100 * r["val_acc"] for r in rows],
                   color=color, label=name, s=40)
    ax.scatter([ref["epoch_time"]], [ref_acc], color=BP_COLOR, marker="*",
               s=160, label="Backprop")
    ax.set_xlabel("Seconds per epoch")
    ax.set_title("Accuracy vs compute cost")
    ax.set_ylim(low, 97.2)
    ax.legend(loc="lower right")

    fig.suptitle(f"PC hyperparameter ablations ({data['epochs']} epochs, "
                 f"seed {data['seed']}, validation set)")
    fig.tight_layout()
    path = out / "ablations.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_summary(runs, out: Path) -> Optional[Path]:
    rows = [(label, color, s) for label, color, _, s in runs if s]
    if not rows:
        return None
    labels = [r[0] for r in rows]
    colors = [r[1] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, [100 * r[2]["test_acc_mean"] for r in rows],
                yerr=[100 * r[2]["test_acc_std"] for r in rows],
                color=colors, capsize=5)
    axes[0].set_ylim(95, 97.5)
    axes[0].set_title("Test accuracy (%)")
    axes[1].bar(labels, [r[2]["epoch_time_mean"] for r in rows],
                color=colors)
    axes[1].set_title("Training time (seconds / epoch)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Final comparison (mean over seeds; error bars = std)")
    fig.tight_layout()
    path = out / "summary.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Results table
# ----------------------------------------------------------------------
def write_table(runs, out: Path) -> Optional[Path]:
    rows = [(label, s) for label, _, _, s in runs if s]
    if not rows:
        return None
    base = next((s["epoch_time_mean"] for l, s in rows if l == "Backprop"),
                rows[0][1]["epoch_time_mean"])
    lines = ["| Learner | Test accuracy (%) | Test loss | s / epoch | Cost vs BP |",
             "|---|---|---|---|---|"]
    for label, s in rows:
        lines.append(
            f"| {label} | {100 * s['test_acc_mean']:.2f} ± "
            f"{100 * s['test_acc_std']:.2f} | {s['test_loss_mean']:.4f} ± "
            f"{s['test_loss_std']:.4f} | {s['epoch_time_mean']:.2f} | "
            f"{s['epoch_time_mean'] / base:.1f}x |")
    text = "\n".join(lines)
    path = out / "results_table.md"
    path.write_text(text + "\n")
    print("\n" + text + "\n")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Make report figures")
    p.add_argument("--default-dir", default="./results")
    p.add_argument("--tuned-dir", default="./results_tuned")
    p.add_argument("--out-dir", default="./figures")
    args = p.parse_args()

    default_dir, tuned_dir = Path(args.default_dir), Path(args.tuned_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = build_runs(default_dir, tuned_dir)
    made = [
        fig_learning_curves(runs, out),
        fig_energy(default_dir, tuned_dir, out),
        fig_alignment(default_dir, out),
        fig_ablations(default_dir, out),
        fig_summary(runs, out),
        write_table(runs, out),
    ]
    for path in made:
        print(f"saved {path}" if path else "skipped (input files missing)")


if __name__ == "__main__":
    main()