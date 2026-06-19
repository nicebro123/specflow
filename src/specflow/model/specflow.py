"""SpecFlow velocity model with static or dual-graph spectral conditioning."""

import math
from typing import Mapping, Optional

import torch
from torch import nn

from specflow.model.cell_aggregator import AttentivePooling
from specflow.model.contextual_propagation import (
    ContextualLocalPropagation,
    GraphAwarePerturbationEncoder,
)
from specflow.model.gene_encoder import GeneTokenEncoder
from specflow.model.spectral_fusion import DualGraphSpectralFusion
from specflow.model.spectral_propagation import SpectralPropagation
from specflow.model.velocity_field import VelocityField


_PROPAGATION_GATE_MODES = {"none", "perturbation"}
_PERTURBATION_ENCODERS = {"legacy", "graph_pool"}
_PROPAGATION_VARIANTS = {"spectral", "contextual_local"}


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


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
        propagation_gate: str = "none",
        propagation_gate_init: float = 0.5,
        perturbation_encoder: str = "legacy",
        propagation_variant: str = "spectral",
        local_propagation_hops: int = 1,
        local_propagation_null_init: float = 0.9,
    ) -> None:
        super().__init__()
        propagation_scale = float(propagation_scale)
        if propagation_scale < 0:
            raise ValueError("propagation_scale must be non-negative")
        propagation_gate = str(propagation_gate).lower()
        if propagation_gate not in _PROPAGATION_GATE_MODES:
            allowed = ", ".join(sorted(_PROPAGATION_GATE_MODES))
            raise ValueError(f"propagation_gate must be one of: {allowed}")
        propagation_gate_init = float(propagation_gate_init)
        if not 0.0 < propagation_gate_init < 1.0:
            raise ValueError("propagation_gate_init must be between 0 and 1")
        if propagation_gate != "none" and not spectral_propagation:
            raise ValueError("propagation_gate requires spectral_propagation=True")
        perturbation_encoder = str(perturbation_encoder).lower()
        if perturbation_encoder not in _PERTURBATION_ENCODERS:
            allowed = ", ".join(sorted(_PERTURBATION_ENCODERS))
            raise ValueError(f"perturbation_encoder must be one of: {allowed}")
        propagation_variant = str(propagation_variant).lower()
        if propagation_variant not in _PROPAGATION_VARIANTS:
            allowed = ", ".join(sorted(_PROPAGATION_VARIANTS))
            raise ValueError(f"propagation_variant must be one of: {allowed}")
        if local_propagation_hops != 1:
            raise ValueError("local_propagation_hops currently supports only 1")
        if not 0.0 < local_propagation_null_init < 1.0:
            raise ValueError(
                "local_propagation_null_init must be between 0 and 1"
            )
        if perturbation_encoder == "graph_pool" and not dual_graph:
            raise ValueError("graph_pool perturbation encoding requires dual_graph=True")
        if propagation_variant == "contextual_local":
            if not spectral_propagation:
                raise ValueError(
                    "contextual_local propagation requires spectral_propagation=True"
                )
            if not dual_graph:
                raise ValueError(
                    "contextual_local propagation requires dual_graph=True"
                )
            if perturbation_encoder != "graph_pool":
                raise ValueError(
                    "contextual_local propagation requires perturbation_encoder='graph_pool'"
                )
            if propagation_gate != "none":
                raise ValueError(
                    "contextual_local propagation cannot use the legacy propagation_gate"
                )
            if propagation_channels != ContextualLocalPropagation.output_dim:
                raise ValueError(
                    "contextual_local propagation requires propagation_channels=2"
                )
        self.n_genes = n_genes
        self.spectral_dim = spectral_dim
        self.dual_graph = dual_graph
        self.use_spectral_embedding = use_spectral_embedding
        self.propagation_scale = propagation_scale
        self.propagation_gate_mode = propagation_gate
        self.propagation_gate_init = propagation_gate_init
        self.perturbation_encoder_mode = perturbation_encoder
        self.propagation_variant = propagation_variant
        self.local_propagation_hops = local_propagation_hops
        self.local_propagation_null_init = local_propagation_null_init
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
                perturbation_encoder=perturbation_encoder,
            )
        self.pert_dim = pert_dim
        self.pert_encoder = (
            nn.Sequential(
                nn.Linear(n_genes, pert_dim),
                nn.SiLU(),
                nn.Linear(pert_dim, pert_dim),
            )
            if perturbation_encoder == "legacy"
            else None
        )
        self.graph_pert_encoder = (
            GraphAwarePerturbationEncoder(
                n_genes=n_genes,
                go_components=go_components,
                coexp_components=coexp_components,
                graph_dim=graph_dim,
                pert_dim=pert_dim,
            )
            if perturbation_encoder == "graph_pool"
            else None
        )
        self.propagation = (
            SpectralPropagation(n_channels=propagation_channels)
            if spectral_propagation and propagation_variant == "spectral"
            else None
        )
        self.contextual_propagation = (
            ContextualLocalPropagation(
                n_genes=n_genes,
                d_model=d_model,
                pert_dim=pert_dim,
                null_init=local_propagation_null_init,
                scale=propagation_scale,
            )
            if spectral_propagation and propagation_variant == "contextual_local"
            else None
        )
        self.propagation_gate = None
        if propagation_gate == "perturbation" and self.propagation is not None:
            self.propagation_gate = nn.Linear(pert_dim, propagation_channels)
            nn.init.zeros_(self.propagation_gate.weight)
            nn.init.constant_(self.propagation_gate.bias, _logit(propagation_gate_init))
        prop_dim = (
            propagation_channels
            if spectral_propagation and propagation_variant == "spectral"
            else 0
        )
        contextual_prop_dim = (
            ContextualLocalPropagation.output_dim
            if self.contextual_propagation is not None
            else 0
        )
        self.gene_encoder = GeneTokenEncoder(spectral_dim, d_model, pert_dim=pert_dim)
        self.cell_aggregator = AttentivePooling(d_model)
        self.velocity_field = VelocityField(
            spectral_dim=spectral_dim,
            d_model=d_model,
            hidden_dim=hidden_dim,
            n_layers=n_velocity_layers,
            pert_dim=pert_dim,
            prop_dim=prop_dim,
            contextual_prop_dim=contextual_prop_dim,
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

    def _perturbation_features(
        self,
        spectral_input,
        pert_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.graph_pert_encoder is not None:
            return self.graph_pert_encoder(spectral_input, pert_mask)
        if self.pert_encoder is None:
            raise RuntimeError("perturbation encoder is not initialized")
        return self.pert_encoder(pert_mask.float())

    def _spectral_features(
        self,
        spectral_input,
        pert_mask: torch.Tensor,
        pert_embedding: torch.Tensor,
    ):
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
            spectral_input["go"],
            spectral_input["coexp"],
            pert_mask,
            perturbation_embedding=(
                pert_embedding
                if self.perturbation_encoder_mode == "graph_pool"
                else None
            ),
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
        pert_embedding = self._perturbation_features(
            spectral_embedding, pert_mask
        )
        spectral, fusion_auxiliary = self._spectral_features(
            spectral_embedding, pert_mask, pert_embedding
        )
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

    def _propagation_features(
        self, pert_mask: torch.Tensor, pert_embedding: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.propagation is None:
            return None
        propagation = self.propagation(pert_mask.float()) * self.propagation_scale
        if self.propagation_gate is not None:
            gate = torch.sigmoid(self.propagation_gate(pert_embedding))
            propagation = propagation * gate.to(propagation.dtype).unsqueeze(1)
        return propagation

    def reset_routing_stats(self) -> None:
        if self.contextual_propagation is not None:
            self.contextual_propagation.reset_routing_stats()

    def routing_summary(self) -> dict:
        if self.contextual_propagation is None:
            return {}
        return self.contextual_propagation.routing_summary()

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
        propagation = self._propagation_features(
            pert_mask, features["pert_embedding"]
        )
        contextual_propagation = None
        if self.contextual_propagation is not None:
            contextual_propagation, _ = self.contextual_propagation(
                pert_mask.float(),
                features["gene_tokens"],
                features["pert_embedding"],
            )
        velocity_args = (
            x_t,
            time,
            condition,
            spectral,
            ctrl_expr,
            pert_mask.float(),
            features["pert_embedding"],
            propagation,
        )
        if contextual_propagation is None:
            velocity = self.velocity_field(*velocity_args)
        else:
            velocity = self.velocity_field(
                *velocity_args,
                contextual_propagation=contextual_propagation,
            )
        return velocity, attention
