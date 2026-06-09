"""Continuous-time embeddings."""

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of scalar flow time followed by an MLP."""

    def __init__(self, d_model: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even for sinusoidal time embeddings")
        self.d_model = d_model
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        half = self.d_model // 2
        frequency = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / half
        )
        angles = time * frequency.unsqueeze(0)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return self.mlp(embedding)
