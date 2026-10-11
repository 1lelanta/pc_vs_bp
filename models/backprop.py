"""Standard backpropagation learner (the baseline)."""
from typing import Dict

import torch

from config import ModelCfg, TrainCfg
from data import to_one_hot
from models.base import LearnerBase, make_optimizer, squared_error_loss


class BackpropLearner(LearnerBase):
    name = "backprop"

    def __init__(self, model_cfg: ModelCfg, train_cfg: TrainCfg):
        super().__init__(model_cfg, train_cfg)
        self.optimizer = make_optimizer(self.layers.parameters(), train_cfg)

    def train_step(self, x: torch.Tensor, y: torch.Tensor
                   ) -> Dict[str, float]:
        x, y = x.to(self.device), y.to(self.device)
        target = to_one_hot(y)

        self.optimizer.zero_grad()
        out = self.forward(x)                       # global forward pass
        loss = squared_error_loss(out, target)      # one global loss
        loss.backward()                             # global backward pass
        self.optimizer.step()

        return {"loss": loss.item()}