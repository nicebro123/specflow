"""Cell-level aggregation from gene tokens."""

import math

import torch
from torch import nn


class AttentivePooling(nn.Module):
    """Aggregate gene tokens and expose weights for interpretation."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))

    def forward(self, gene_tokens: torch.Tensor):
        scores = torch.einsum("bgd,d->bg", gene_tokens, self.query)
        weights = torch.softmax(scores, dim=-1)
        condition = torch.einsum("bg,bgd->bd", weights, gene_tokens)
        return condition, weights
