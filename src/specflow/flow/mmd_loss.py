"""Distribution-matching regularization for perturbation response samples."""

from typing import Iterable

import torch
from torch import nn


class MMDLoss(nn.Module):
    """Differentiable multi-bandwidth RBF maximum mean discrepancy."""

    def __init__(self, bandwidths: Iterable[float] = (0.1, 0.5, 1.0, 5.0)) -> None:
        super().__init__()
        values = tuple(float(value) for value in bandwidths)
        if not values or any(value <= 0 for value in values):
            raise ValueError("bandwidths must contain positive values")
        self.bandwidths = values

    def _kernel(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        distance_sq = torch.cdist(left, right) ** 2
        return sum(
            torch.exp(-distance_sq / (2.0 * bandwidth**2))
            for bandwidth in self.bandwidths
        )

    def forward(self, predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        predicted = predicted.reshape(-1, predicted.shape[-1]).float()
        observed = observed.reshape(-1, observed.shape[-1]).float()
        value = (
            self._kernel(predicted, predicted).mean()
            + self._kernel(observed, observed).mean()
            - 2.0 * self._kernel(predicted, observed).mean()
        )
        return torch.clamp(value, min=0.0)
