"""SpecFlow velocity model with static or dual-graph spectral conditioning."""

from typing import Mapping, Optional

import torch
from torch import nn

from specflow.model.cell_aggregator import AttentivePooling
from specflow.model.gene_encoder import GeneTokenEncoder
from specflow.model.spectral_fusion import DualGraphSpectralFusion
from specflow.model.spectral_propagation import SpectralPropagation
from specflow.model.velocity_field import VelocityField


class SpecFlow(nn.Module):
    """Predict velocities conditioned on control state and graph position.

    A tensor input retains the static coexpression MVP path. When
    ``dual_graph=True``, a dictionary with ``go`` and ``coexp`` eigenvectors is
    fused dynamically under the current gene-aligned perturbation mask.
    """

    def __init__(
        self,
        n_genes: int,
        spectral_dim: int,
        d_model: int = 128,
        hidden_dim: int = 256,
        n_velocity_layers: int = 3,
        dual_graph: bool = False,
        go_components: Optional[int] = None,
        coexp_components: Optional[int] = None,
        pert_dim: int = 32,
        graph_dim: int = 32,
        macro_ratio: float = 0.5,
        graph_mode: str = "dual",
        fusion_mode: str = "adaptive",
        scale_mode: str = "multi",
        use_spectral_embedding: bool = True,
        spectral_propagation: bool = False,
        propagation_channels: int = 8,
        propagation_scale: float = 1.0,
    ) -> None:
        super().__init__()
        propagation_scale = float(propagation_scale)
        if propagation_scale < 0:
            raise ValueError("propagation_scale must be non-negative")
        self.n_genes = n_genes
        self.spectral_dim = spectral_dim
        self.dual_graph = dual_graph
        self.use_spectral_embedding = use_spectral_embedding
        self.propagation_scale = propagation_scale
        self.spectral_fusion = None
        if dual_graph:
            if go_components is None or coexp_components is None:
                raise ValueError("dual_graph requires go_components and coexp_components")
            self.spectral_fusion = DualGraphSpectralFusion(
                n_genes=n_genes,
                go_components=go_components,
                coexp_components=coexp_components,
                spectral_dim=spectral_dim,
                pert_dim=pert_dim,
                graph_dim=graph_dim,
                macro_ratio=macro_ratio,
                graph_mode=graph_mode,
                fusion_mode=fusion_mode,
                scale_mode=scale_mode,
            )
        self.pert_dim = pert_dim
        self.pert_encoder = nn.Sequential(
            nn.Linear(n_genes, pert_dim),
            nn.SiLU(),
            nn.Linear(pert_dim, pert_dim),
        )
        self.propagation = (
            SpectralPropagation(n_channels=propagation_channels)
            if spectral_propagation
            else None
        )
        prop_dim = propagation_channels if spectral_propagation else 0
        self.gene_encoder = GeneTokenEncoder(spectral_dim, d_model, pert_dim=pert_dim)
        self.cell_aggregator = AttentivePooling(d_model)
        self.velocity_field = VelocityField(
            spectral_dim=spectral_dim,
            d_model=d_model,
            hidden_dim=hidden_dim,
            n_layers=n_velocity_layers,
            pert_dim=pert_dim,
            prop_dim=prop_dim,
        )

    def _expand_spectral(
        self, spectral_embedding: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if spectral_embedding.ndim == 2:
            spectral_embedding = spectral_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        if spectral_embedding.shape != (batch_size, self.n_genes, self.spectral_dim):
            raise ValueError(
                "spectral_embedding must have shape (G, K) or (B, G, K) "
                f"with G={self.n_genes} and K={self.spectral_dim}"
            )
        return spectral_embedding

    def _spectral_features(self, spectral_input, pert_mask: torch.Tensor):
        if torch.is_tensor(spectral_input):
            spectral = self._expand_spectral(spectral_input, pert_mask.shape[0])
            if not self.use_spectral_embedding:
                spectral = torch.zeros_like(spectral)
            return spectral, {}
        if not isinstance(spectral_input, Mapping) or set(spectral_input) != {"go", "coexp"}:
            raise ValueError("spectral input must be a tensor or a {'go', 'coexp'} mapping")
        if self.spectral_fusion is None:
            raise ValueError("dual graph inputs require SpecFlow(dual_graph=True)")
        fused, auxiliary = self.spectral_fusion(
            spectral_input["go"], spectral_input["coexp"], pert_mask
        )
        if not self.use_spectral_embedding:
            return torch.zeros_like(fused), auxiliary
        return fused, auxiliary

    def encode_condition(
        self,
        ctrl_expr: torch.Tensor,
        pert_mask: torch.Tensor,
        spectral_embedding,
    ):
        """Return cell-conditioning features and interpretation tensors."""
        spectral, fusion_auxiliary = self._spectral_features(
            spectral_embedding, pert_mask
        )
        pert_embedding = self.pert_encoder(pert_mask.float())
        tokens = self.gene_encoder(
            ctrl_expr, spectral, pert_mask.float(), pert_embedding
        )
        condition, attention = self.cell_aggregator(tokens)
        return {
            "spectral_embedding": spectral,
            "gene_tokens": tokens,
            "cell_condition": condition,
            "gene_attention": attention,
            "pert_embedding": pert_embedding,
            **fusion_auxiliary,
        }

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        ctrl_expr: torch.Tensor,
        pert_mask: torch.Tensor,
        spectral_embedding,
    ):
        if x_t.shape != ctrl_expr.shape or x_t.shape != pert_mask.shape:
            raise ValueError("x_t, ctrl_expr, and pert_mask must all have shape (B, G)")
        if x_t.shape[1] != self.n_genes:
            raise ValueError(f"expected {self.n_genes} genes, got {x_t.shape[1]}")

        features = self.encode_condition(ctrl_expr, pert_mask, spectral_embedding)
        spectral = features["spectral_embedding"]
        condition = features["cell_condition"]
        attention = features["gene_attention"]
        propagation = (
            self.propagation(pert_mask.float()) * self.propagation_scale
            if self.propagation is not None
            else None
        )
        velocity = self.velocity_field(
            x_t,
            time,
            condition,
            spectral,
            ctrl_expr,
            pert_mask.float(),
            features["pert_embedding"],
            propagation,
        )
        return velocity, attention
