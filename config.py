"""Central configuration for the PC vs. Backprop study.

Every hyperparameter lives here. The rest of the code receives these
objects and never hard-codes a number, which keeps experiments
reproducible and easy to sweep.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass(frozen=True)
class DataCfg:
    dataset: str = "mnist"
    root: str = "./data"
    batch_size: int = 128
    val_fraction: float = 0.1 #  10% of data used for validation
    num_workers: int = 0


@dataclass(frozen=True)
class ModelCfg:
    # Same architecture is used by BOTH learners for a fair comparison.
    layer_sizes: Tuple[int, ...] = (784, 128, 64, 10)
    activation: str = "tanh"       # "tanh" or "relu"


@dataclass(frozen=True)
class PCCfg:
    inference_steps: int = 20      # T: latent-state updates per batch
    state_lr: float = 0.1          # eta_x: step size for latent states
    init_mode: str = "forward"     # initialise states with a forward pass


@dataclass(frozen=True)
class TrainCfg:
    epochs: int = 10
    lr: float = 1e-3               # weight learning rate (same for both)
    optimizer: str = "adam"        # "adam" or "sgd"
    seeds: Tuple[int, ...] = (0, 1, 2)
    device: str = "cpu"
    out_dir: str = "./results"


@dataclass(frozen=True)
class ExperimentCfg:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    pc: PCCfg = field(default_factory=PCCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def to_dict(self) -> dict:
        """Plain dict, so the exact config can be saved next to results."""
        return asdict(self)