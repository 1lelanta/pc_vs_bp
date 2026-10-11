"""Shared foundation for both learners.

Everything that must be IDENTICAL between backprop and predictive coding
lives here: layer construction, initialisation, activation functions,
the loss definition and the optimizer factory. What differs (the learning
rule) lives in each learner's own file.
"""
from abc import ABC, abstractmethod
from typing import Dict, Iterable

import torch
import torch.nn as nn

from config import ModelCfg, TrainCfg
from data import to_one_hot

# name -> (f, f'). PC needs the derivative f'(a) explicitly for its local
# rules; backprop only needs f. Defining both here keeps them in sync.
def _tanh_grad(a: torch.Tensor) -> torch.Tensor:
    t = torch.tanh(a)
    return 1.0 - t * t


def _relu_grad(a: torch.Tensor) -> torch.Tensor:
    return (a > 0).to(a.dtype)


ACTIVATIONS = {
    "tanh": (torch.tanh, _tanh_grad),
    "relu": (torch.relu, _relu_grad),
}


def make_layers(cfg: ModelCfg) -> nn.ModuleList:
    """Build the linear layers with Xavier init and zero biases.

    Called after seeding, so the same seed gives the same initial weights
    for both learners.
    """
    layers = nn.ModuleList()
    sizes = cfg.layer_sizes
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        lin = nn.Linear(n_in, n_out)
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
    return layers


def make_optimizer(params: Iterable[torch.nn.Parameter], cfg: TrainCfg
                   ) -> torch.optim.Optimizer:
    """Same optimizer factory for both learners."""
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr)
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


def squared_error_loss(out: torch.Tensor, target: torch.Tensor
                       ) -> torch.Tensor:
    """0.5 * ||out - target||^2, summed over outputs, averaged over batch.

    This is deliberately the same form as the output-layer energy in the
    PC network, which is what makes the two models directly comparable.
    """
    return 0.5 * ((out - target) ** 2).sum(dim=1).mean()


class LearnerBase(ABC):
    """Contract every learner must satisfy.

    The trainer only ever calls train_step / predict / evaluate, so it
    never needs to know which learning rule is running.
    """
    name: str = "base"

    def __init__(self, model_cfg: ModelCfg, train_cfg: TrainCfg):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.f, self.df = ACTIVATIONS[model_cfg.activation]
        self.layers = make_layers(model_cfg).to(self.device)

    # ----- must be implemented by each learner -----
    @abstractmethod
    def train_step(self, x: torch.Tensor, y: torch.Tensor
                   ) -> Dict[str, float]:
        """One update on a batch. x: (B, 784) floats, y: (B,) int labels.
        Returns a dict of scalar metrics, e.g. {"loss": ...}."""

    # ----- shared behaviour -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain feedforward pass. Last layer is linear (no activation)."""
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < last:
                x = self.f(x)
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Output scores (B, 10). Argmax gives the predicted class."""
        return self.forward(x.to(self.device))

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Loss and correct count on a batch (for the trainer to aggregate)."""
        x, y = x.to(self.device), y.to(self.device)
        out = self.predict(x)
        loss = squared_error_loss(out, to_one_hot(y))
        correct = (out.argmax(dim=1) == y).sum().item()
        return {"loss": loss.item(), "correct": correct, "n": y.numel()}

    def load_weights_from(self, other: "LearnerBase") -> None:
        """Copy another learner's weights (guarantees identical starts)."""
        self.layers.load_state_dict(other.layers.state_dict())