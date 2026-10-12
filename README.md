# Backpropagation vs Predictive Coding

PyTorch implementation and comparison of two learning rules on MNIST: standard backpropagation (BP) and predictive coding (PC), where learning is driven by local prediction-error minimisation. Both learners share one architecture, one set of initial weights, one loss and one optimiser, so that any difference in results comes from the learning rule alone.

The project covers four requirements:

1. A neural network trained with standard backpropagation.
2. A predictive coding network that exposes layer-wise energy, layer-wise prediction errors, latent state updates and total system energy.
3. A comparison of both architectures (accuracy, convergence, cost, gradient alignment, hyperparameter sensitivity).
4. A discussion of the limitations of backpropagation that predictive coding addresses (see the written report).

## Results summary

Test-set results over seeds 0, 1 and 2, 10 epochs each (mean ± standard deviation):

| Learner | Test accuracy (%) | Test loss | s / epoch | Cost vs BP |
|---|---|---|---|---|
| Backpropagation | 96.98 ± 0.14 | 0.0313 ± 0.0018 | 0.89 | 1.0× |
| PC, default (T = 20, η = 0.1) | 95.84 ± 0.30 | 0.0515 ± 0.0022 | 5.54 | 6.2× |
| PC, tuned (T = 5, η = 0.1) | 96.51 ± 0.12 | 0.0420 ± 0.0029 | 2.24 | 2.3× |

- T is the number of inference steps per batch and η is the latent state step size. The tuned setting was selected on the validation set; the test set was evaluated once per run.
- PC weight updates align with the BP gradient (cosine similarity 0.96 to 0.97 per layer at equilibrium) but are not identical to it.
- Timings were measured on a laptop CPU with the dataset cached in memory. Epoch time covers training only.

Exact numbers depend on hardware, library versions and seeds.

## Project structure

```
pc_vs_bp/
├── config.py                  # frozen dataclasses: DataCfg, ModelCfg, PCCfg, TrainCfg, ExperimentCfg
├── data.py                    # get_loaders(), to_one_hot(); dataset cached as tensors
├── models/
│   ├── base.py                # LearnerBase, layer construction, activations, loss, optimiser factory
│   ├── backprop.py            # BackpropLearner (autograd)
│   └── predictive.py          # PCLearner (errors, energies, inference, local updates)
├── training/
│   ├── trainer.py             # model-agnostic training loop, seeding, evaluation
│   └── metrics.py             # WeightedMean, Timer, History (JSON-serialisable)
├── analysis/
│   ├── gradient_alignment.py  # cosine similarity between PC updates and BP gradients
│   ├── ablations.py           # sweeps over T and η on the validation set
│   └── plots.py               # figures and results table from saved JSON
├── experiments/
│   └── run_comparison.py      # trains both learners across seeds, saves all results
├── tests/
│   └── test_PC.py             # gradient, energy and separation tests
└── requirements.txt
```

## Setup

Requirements: Python 3.8 or newer. A GPU is not needed. The first run downloads MNIST (about 50 MB) into `./data`, which needs an internet connection.

```bash
cd pc_vs_bp
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pytest
```

The `--index-url` line installs the CPU-only PyTorch build. Activate the environment in every new terminal with `source .venv/bin/activate`.

## Usage

Run all commands from the project root with the environment active. The `-m` form keeps the imports working.

```bash
# 1. Unit tests
python -m pytest -v

# 2. Quick smoke test (one epoch, one seed)
python -m experiments.run_comparison --epochs 1 --seeds 0 --out-dir ./results_smoke

# 3. Full comparison with the default PC settings (T = 20, eta = 0.1) -> ./results
python -m experiments.run_comparison

# 4. Gradient alignment between PC updates and BP gradients -> ./results
python -m analysis.gradient_alignment --epochs 3

# 5. Hyperparameter ablations on the validation set -> ./results
python -m analysis.ablations --epochs 5

# 6. Full comparison with tuned PC settings -> ./results_tuned
python -m experiments.run_comparison --T 5 --state-lr 0.1 --out-dir ./results_tuned

# 7. Figures and results table -> ./figures
python -m analysis.plots
```

On a laptop CPU the full comparison takes a few minutes. The default PC settings are the slowest because every batch runs 20 inference steps.

### Command-line options

| Script | Option | Meaning |
|---|---|---|
| `experiments.run_comparison` | `--dataset`, `--epochs`, `--seeds`, `--T`, `--state-lr`, `--lr`, `--device`, `--out-dir` | Override the defaults in `config.py` |
| `analysis.gradient_alignment` | `--epochs`, `--seed`, `--steps`, `--betas`, `--state-lr`, `--out-dir` | Training length, inference-step list and target-nudge list |
| `analysis.ablations` | `--epochs`, `--seed`, `--train-size`, `--T-list`, `--lr-list`, `--fixed-T`, `--fixed-lr`, `--out-dir` | Sweep values; `--train-size` uses a random training subset for speed |
| `analysis.plots` | `--default-dir`, `--tuned-dir`, `--out-dir` | Result folders to read and figure output folder |

## Method

Notation: `x0` is the input (clamped), `xL` is the output state (clamped to the one-hot target during training), and `x1 … x(L-1)` are free latent states. Here L = 3.

| Quantity | Definition |
|---|---|
| Prediction | `mu_l = f(W_l x_(l-1) + b_l)`; linear at the output layer |
| Layer-wise prediction error | `eps_l = x_l − mu_l` |
| Layer-wise energy | `E_l = ½ ‖eps_l‖²` (batch-averaged) |
| Total system energy | `F = E_1 + E_2 + … + E_L` |
| Latent state update | `x_l ← x_l − η ∂F/∂x_l`, with `∂F/∂x_l = eps_l − W_(l+1)ᵀ (eps_(l+1) ⊙ f′(a_(l+1)))` |
| Weight update | `∂F/∂W_l = −(eps_l ⊙ f′(a_l)) x_(l-1)ᵀ` and `∂F/∂b_l = −(eps_l ⊙ f′(a_l))` |

One PC training step: initialise all states with a feedforward pass (hidden errors start at exactly zero), clamp the output to the target, run T inference steps with the weights frozen (all hidden states updated simultaneously, layer-wise energies recorded at every step), compute the local weight gradients from the settled states, and apply one optimiser step. The local gradients are written into the parameters' gradient fields, so BP and PC use the same optimiser. At test time PC predicts with a plain feedforward pass.

## Verification

`tests/test_PC.py` contains six tests:

1. Hand-derived latent state gradients equal autograd gradients of the total energy `F`.
2. Hand-derived weight and bias gradients equal autograd gradients of `F`.
3. During inference the hidden-layer energies are zero at step 0 and the total energy never increases.
4. PC training never calls `backward()` and leaves no autograd graph on the gradients.
5. Inference builds no autograd graph.
6. At equilibrium with a tiny target nudge, the state gradients are below 1e-9 and PC weight updates match a BP pass seeded with PC's own equilibrium output error (cosine similarity above 0.999).

## Outputs

Every script writes JSON next to the exact configuration used, so analysis can be repeated without retraining.

| Path | Content |
|---|---|
| `results*/config.json` | Full configuration of the run |
| `results*/backprop_seed*.json`, `predictive_coding_seed*.json` | Per-epoch training and validation metrics, final test metrics |
| `results*/pc_energy_traces_seed*.json` | Layer-wise energy per inference step, recorded after every epoch on a fixed probe batch |
| `results*/summary.json` | Mean and standard deviation over seeds |
| `results/gradient_alignment.json` | Cosine similarities by epoch, inference steps and target nudge |
| `results/ablations.json` | Validation results of the T and η sweeps |
| `figures/*.png`, `figures/results_table.md` | Learning curves, energy dynamics, gradient alignment, ablations, summary, results table |

## Configuration

All hyperparameters are defined in `config.py` as frozen dataclasses (`DataCfg`, `ModelCfg`, `PCCfg`, `TrainCfg`). Defaults: layer sizes 784-128-64-10, tanh activation, batch size 128, Adam with learning rate 0.001, 10 epochs, seeds 0 to 2, PC with T = 20 and η = 0.1. Command-line options override these values without editing the file. The activation can be changed to `relu` in `ModelCfg`; PC with ReLU has not been evaluated in this project.

## Design notes

- **Shared foundation.** `models/base.py` holds everything that must be identical between the learners: layer construction, initialisation, activation functions, loss and optimiser factory.
- **Separation of learning rules.** `backprop.py` and `predictive.py` do not import each other. `loss.backward()` appears only in the BP learner.
- **Model-agnostic trainer.** The trainer only calls `train_step` and `evaluate` and logs whatever metrics the learner returns, so PC's energies appear in the history without trainer changes.
- **Fair comparison.** Identical initial weights, identical batch order and an identical optimiser for both learners; tuning uses only the validation set, and the test set is evaluated once per run.

## Limitations

- One dataset (MNIST), one architecture and 10 epochs.
- Only PC hyperparameters were tuned; BP kept its default learning rate.
- Ablations use a single seed and five epochs.
- Both learners use Adam and squared error on one-hot targets; other losses and optimisers would give different numbers.
- The standard PC formulation implemented here still uses the transposed forward weights in the state update, so the weight-symmetry requirement of BP is not removed.