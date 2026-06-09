"""Exponential moving average of model parameters."""

import copy
from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    """Maintain an exponential moving average of model parameters.

    Usage::

        ema = EMA(model, decay=0.999)
        for batch in loader:
            loss.backward()
            optimizer.step()
            ema.update()

        with ema.shadow_context():
            evaluate(model)  # model uses EMA weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup = {}

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self) -> None:
        self._backup = {}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    @contextmanager
    def shadow_context(self):
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self):
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict) -> None:
        for name, tensor in state_dict.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)
