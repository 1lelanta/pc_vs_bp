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

        # ------------------------------------------------------------------
    # Inference: relax the latent states by gradient descent on F
    # ------------------------------------------------------------------
    def state_gradients(self, errors: List[torch.Tensor],
                        derivs: List[torch.Tensor]) -> List[torch.Tensor]:
        """dF/dx_l for each HIDDEN state l = 1 .. L-1 (per sample):

            dF/dx_l = eps_l - W_{l+1}^T (eps_{l+1} * f'(a_{l+1}))

        Local: a state is pulled by its own error and by the error of the
        one layer it helps predict. Nothing else is involved."""
        deltas = [e * d for e, d in zip(errors, derivs)]
        grads = []
        for l in range(1, len(self.layers)):
            # deltas[l] is layer l+1's delta; layers[l].weight is W_{l+1}
            grads.append(errors[l - 1] - deltas[l] @ self.layers[l].weight)
        return grads

    @torch.no_grad()
    def infer(self, x: torch.Tensor, target: torch.Tensor,
              steps: int = None, state_lr: float = None) -> InferenceResult:
        """Clamp input and target, then run T state updates with the
        weights frozen. Records layer-wise energy at every step."""
        T = self.pc_cfg.inference_steps if steps is None else steps
        eta = self.pc_cfg.state_lr if state_lr is None else state_lr

        states = self._forward_states(x)
        states[-1] = target.clone()                 # clamp output to target

        trace = []
        for _ in range(T):
            errors, derivs = self._errors(states)
            trace.append(self.layer_energies(errors))
            grads = self.state_gradients(errors, derivs)   # all computed first
            for l, g in enumerate(grads, start=1):         # then applied together
                states[l] = states[l] - eta * g

        errors, derivs = self._errors(states)       # state of the system at t=T
        trace.append(self.layer_energies(errors))
        return InferenceResult(states, errors, derivs, torch.stack(trace))   

        # ------------------------------------------------------------------
    # Learning: one local weight update after inference has settled
    # ------------------------------------------------------------------
    def weight_gradients(self, result: InferenceResult
                         ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """dF/dW_l and dF/db_l, batch-averaged:

            dF/dW_l = -(eps_l * f'(a_l))^T x_{l-1}
            dF/db_l = -(eps_l * f'(a_l))

        Uses only layer l's own error and the activity just below it."""
        batch = result.states[0].size(0)
        grads = []
        for i in range(len(self.layers)):
            delta = result.errors[i] * result.derivs[i]
            gW = -(delta.t() @ result.states[i]) / batch
            gb = -delta.mean(dim=0)
            grads.append((gW, gb))
        return grads

    @torch.no_grad()
    def train_step(self, x: torch.Tensor, y: torch.Tensor
                   ) -> Dict[str, float]:
        x, y = x.to(self.device), y.to(self.device)
        target = to_one_hot(y)

        result = self.infer(x, target)              # 1) settle the states
        for layer, (gW, gb) in zip(self.layers, self.weight_gradients(result)):
            layer.weight.grad = gW                  # 2) hand local updates to
            layer.bias.grad = gb                    #    the same optimizer BP uses
        self.optimizer.step()

        trace = result.energy_trace
        return {
            # output-layer energy at t=0 == the plain squared-error loss,
            # so it is directly comparable to backprop's "loss"
            "loss": trace[0, -1].item(),
            # total energy F after inference has settled
            "energy": trace[-1].sum().item(),
        } 