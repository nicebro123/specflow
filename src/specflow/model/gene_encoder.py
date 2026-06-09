"""Per-gene token encoding."""

from typing import Optional

import torch
from torch import nn


class GeneTokenEncoder(nn.Module):
    """Encode control expression, graph coordinates, targeting, and perturbation."""

    def __init__(self, spectral_dim: int, d_model: int, pert_dim: int = 0) -> None:
        super().__init__()
        self.pert_dim = pert_dim
        self.encoder = nn.Sequential(
            nn.Linear(spectral_dim + 2 + pert_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        ctrl_expr: torch.Tensor,
        spectral_embedding: torch.Tensor,
        pert_mask: torch.Tensor,
        pert_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [
            ctrl_expr.unsqueeze(-1),
            spectral_embedding,
            pert_mask.unsqueeze(-1),
        ]
        if self.pert_dim:
            if pert_embedding is None:
                raise ValueError("pert_embedding is required when pert_dim > 0")
            n_genes = spectral_embedding.shape[1]
            parts.append(pert_embedding.unsqueeze(1).expand(-1, n_genes, -1))
        features = torch.cat(parts, dim=-1)
        return self.encoder(features)
