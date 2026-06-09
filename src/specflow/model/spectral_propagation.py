"""Learnable spectral filtering for perturbation-signal propagation.

This realizes the original "perturbation signal propagates along the regulatory
network" intuition correctly: instead of re-computing the eigendecomposition per
perturbation (negligible and direction-wrong for activation), the FIXED base
graph's eigenbasis is reused and a learnable spectral filter diffuses the
perturbation indicator to every gene.

Given base eigenvectors phi (G, k), eigenvalues lambda (k,), and a per-cell
perturbation indicator s (B, G):

    h = phi @ diag(g_theta(lambda)) @ phi^T @ s     -> (B, G, C)

for C learnable filters g_theta. h_i is gene i's "perturbation influence" (how
far downstream of the perturbed genes it lies on the graph), used as a per-gene
conditioning feature for the velocity field. Cost is O(G*k): eigenvectors are
precomputed once, only the small filter MLP on eigenvalues is learned.
"""

import torch
from torch import nn


class SpectralPropagation(nn.Module):
    def __init__(self, n_channels: int = 8, hidden_dim: int = 32) -> None:
        super().__init__()
        if n_channels < 1:
            raise ValueError("n_channels must be positive")
        self.n_channels = n_channels
        self.filter = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_channels),
        )
        self._has_basis = False

    @property
    def output_dim(self) -> int:
        return self.n_channels

    def set_basis(self, eigenvectors, eigenvalues) -> None:
        """Install the fixed base-graph spectrum (non-persistent buffers)."""
        eigvecs = torch.as_tensor(eigenvectors, dtype=torch.float32)
        eigvals = torch.as_tensor(eigenvalues, dtype=torch.float32).reshape(-1)
        if eigvecs.ndim != 2 or eigvecs.shape[1] != eigvals.shape[0]:
            raise ValueError(
                "eigenvectors must be (G, k) and eigenvalues (k,) with matching k"
            )
        device = self.filter[0].weight.device
        if self._has_basis:
            self.eigvecs = eigvecs.to(device)
            self.eigvals = eigvals.to(device)
        else:
            # persistent=False: derived from the graph, re-set at load time rather
            # than serialized into checkpoints.
            self.register_buffer("eigvecs", eigvecs.to(device), persistent=False)
            self.register_buffer("eigvals", eigvals.to(device), persistent=False)
            self._has_basis = True

    def forward(self, pert_mask: torch.Tensor) -> torch.Tensor:
        if not self._has_basis:
            raise RuntimeError(
                "SpectralPropagation requires set_basis(eigenvectors, eigenvalues)"
            )
        eigvecs = self.eigvecs.to(pert_mask.dtype)
        eigvals = self.eigvals.to(pert_mask.dtype)
        proj = pert_mask @ eigvecs                      # (B, k)
        gains = self.filter(eigvals.unsqueeze(-1))      # (k, C)
        scaled = proj.unsqueeze(-1) * gains.unsqueeze(0)  # (B, k, C)
        return torch.einsum("bkc,gk->bgc", scaled, eigvecs)  # (B, G, C)
