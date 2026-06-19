"""Spectral-conditioned expression-space velocity field."""

from typing import Optional

import torch
from torch import nn

from specflow.model.time_embedding import TimeEmbedding


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class FiLM(nn.Module):
    """Feature-wise linear modulation from a conditioning vector.

    Produces a per-feature scale and shift broadcast across all genes. The
    projection is zero-initialized so modulation starts as identity
    (gamma = 0 -> scale 1, beta = 0), keeping early training stable.
    """

    def __init__(self, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1.0 + gamma).unsqueeze(1) * hidden + beta.unsqueeze(1)


class VelocityField(nn.Module):
    """Predict a flow velocity for each gene and each input cell."""

    def __init__(
        self,
        spectral_dim: int,
        d_model: int,
        hidden_dim: int,
        n_layers: int,
        pert_dim: int = 0,
        prop_dim: int = 0,
        contextual_prop_dim: int = 0,
    ) -> None:
        super().__init__()
        self.pert_dim = pert_dim
        self.prop_dim = prop_dim
        self.contextual_prop_dim = contextual_prop_dim
        self.time_embedding = TimeEmbedding(d_model)
        self.local_projection = nn.Sequential(
            nn.Linear(spectral_dim + 3 + prop_dim, hidden_dim),
            nn.SiLU(),
        )
        cond_dim = d_model * 2 + pert_dim
        self.global_projection = nn.Linear(cond_dim, hidden_dim)
        self.contextual_prop_projection = None
        if contextual_prop_dim:
            self.contextual_prop_projection = nn.Linear(
                contextual_prop_dim, hidden_dim, bias=False
            )
            nn.init.zeros_(self.contextual_prop_projection.weight)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(n_layers)]
        )
        self.films = nn.ModuleList(
            [FiLM(cond_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        cell_condition: torch.Tensor,
        spectral_embedding: torch.Tensor,
        ctrl_expr: torch.Tensor,
        pert_mask: torch.Tensor,
        pert_embedding: Optional[torch.Tensor] = None,
        propagation: Optional[torch.Tensor] = None,
        contextual_propagation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        time_feature = self.time_embedding(time)
        cond_parts = [cell_condition, time_feature]
        if self.pert_dim:
            if pert_embedding is None:
                raise ValueError("pert_embedding is required when pert_dim > 0")
            cond_parts.append(pert_embedding)
        cond = torch.cat(cond_parts, dim=-1)
        global_feature = self.global_projection(cond)
        local_parts = [
            x_t.unsqueeze(-1),
            ctrl_expr.unsqueeze(-1),
            spectral_embedding,
            pert_mask.unsqueeze(-1),
        ]
        if self.prop_dim:
            if propagation is None:
                raise ValueError("propagation is required when prop_dim > 0")
            local_parts.append(propagation)
        local_feature = torch.cat(local_parts, dim=-1)
        hidden = self.local_projection(local_feature) + global_feature.unsqueeze(1)
        if self.contextual_prop_dim:
            if contextual_propagation is None:
                raise ValueError(
                    "contextual_propagation is required when contextual_prop_dim > 0"
                )
            hidden = hidden + self.contextual_prop_projection(
                contextual_propagation
            )
        for block, film in zip(self.blocks, self.films):
            hidden = block(hidden)
            hidden = film(hidden, cond)
        return self.output(hidden).squeeze(-1)
