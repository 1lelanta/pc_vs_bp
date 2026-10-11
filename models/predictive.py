"""Predictive Coding learner.

Learning here uses NO autograd and NO backward(): everything runs under
torch.no_grad() and every update is built from local quantities
(a layer's own error and the error of the layer directly above it).

Notation: x_0 = input (clamped), x_L = target (clamped during training),
x_1 .. x_{L-1} = free latent states. layers[i] maps x_i -> x_{i+1}.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from config import ModelCfg, PCCfg, TrainCfg
from data import to_one_hot
from models.base import LearnerBase, make_optimizer


@dataclass
class InferenceResult:
    states: List[torch.Tensor]   # x_0 .. x_L
    errors: List[torch.Tensor]   # eps_1 .. eps_L
    derivs: List[torch.Tensor]   # f'(a_1) .. f'(a_L); ones at the output
    energy_trace: torch.Tensor   # (T+1, L): layer-wise energy at each step


class PCLearner(LearnerBase):
    name = "predictive_coding"

    def __init__(self, model_cfg: ModelCfg, train_cfg: TrainCfg,
                 pc_cfg: PCCfg):
        super().__init__(model_cfg, train_cfg)
        if pc_cfg.init_mode != "forward":
            raise NotImplementedError("Only init_mode='forward' is supported")
        self.pc_cfg = pc_cfg
        self.optimizer = make_optimizer(self.layers.parameters(), train_cfg)

    # ------------------------------------------------------------------
    # Predictions, errors, energies
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Initialise every state with a feedforward pass (so hidden
        prediction errors start at exactly zero)."""
        last = len(self.layers) - 1
        states = [x]
        for i, layer in enumerate(self.layers):
            a = layer(states[-1])
            states.append(self.f(a) if i < last else a)
        return states

    def _errors(self, states: List[torch.Tensor]
                ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Layer-wise prediction errors eps_l = x_l - mu_l and the
        activation derivatives f'(a_l) needed by the local rules."""
        last = len(self.layers) - 1
        errors, derivs = [], []
        for i, layer in enumerate(self.layers):
            a = layer(states[i])                    # pre-activation a_{i+1}
            if i < last:
                mu, d = self.f(a), self.df(a)
            else:                                   # linear output layer
                mu, d = a, torch.ones_like(a)
            errors.append(states[i + 1] - mu)
            derivs.append(d)
        return errors, derivs

    @staticmethod
    def layer_energies(errors: List[torch.Tensor]) -> torch.Tensor:
        """E_l = 0.5 * ||eps_l||^2, averaged over the batch. Shape (L,)."""
        return torch.stack([0.5 * (e ** 2).sum(dim=1).mean() for e in errors])

    def energy(self, states: List[torch.Tensor]) -> torch.Tensor:
        """Total system energy F = sum_l E_l. Deliberately NOT wrapped in
        no_grad, so tests can differentiate it with autograd and check our
        hand-derived gradients against it."""
        errors, _ = self._errors(states)
        return self.layer_energies(errors).sum()