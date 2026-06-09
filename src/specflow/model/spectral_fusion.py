"""Perturbation-conditioned multi-scale dual-graph spectral fusion."""

from typing import Tuple

import torch
from torch import nn

from specflow.model.sign_net import SignNet


class ScaleAttention(nn.Module):
    """Fuse low-frequency and high-frequency coordinates within one graph."""

    def __init__(self, input_dim: int, pert_dim: int, output_dim: int) -> None:
        super().__init__()
        self.macro_projection = nn.Linear(input_dim, output_dim)
        self.micro_projection = nn.Linear(input_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2 + pert_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        macro: torch.Tensor,
        micro: torch.Tensor,
        perturbation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        macro_feature = self.macro_projection(macro)
        micro_feature = self.micro_projection(micro)
        condition = perturbation.unsqueeze(1).expand(-1, macro.shape[1], -1)
        weight = self.gate(
            torch.cat((macro_feature, micro_feature, condition), dim=-1)
        )
        return weight * macro_feature + (1.0 - weight) * micro_feature, weight


class GraphScaleEncoder(nn.Module):
    """Run SignNet before fusing genuine low/high-frequency partitions."""

    def __init__(
        self,
        n_components: int,
        macro_ratio: float,
        graph_dim: int,
        pert_dim: int,
        sign_hidden_dim: int,
        sign_component_dim: int,
        scale_mode: str = "multi",
    ) -> None:
        super().__init__()
        if n_components < 2:
            raise ValueError("multi-scale fusion requires at least two components")
        if not 0.0 < macro_ratio < 1.0:
            raise ValueError("macro_ratio must lie strictly between 0 and 1")
        if scale_mode not in {"multi", "macro", "micro"}:
            raise ValueError("scale_mode must be 'multi', 'macro', or 'micro'")
        self.scale_mode = scale_mode
        self.macro_components = max(1, min(n_components - 1, round(n_components * macro_ratio)))
        self.micro_components = n_components - self.macro_components
        self.macro_encoder = SignNet(
            self.macro_components, sign_hidden_dim, sign_component_dim
        )
        self.micro_encoder = SignNet(
            self.micro_components, sign_hidden_dim, sign_component_dim
        )
        self.macro_projection = nn.Linear(self.macro_encoder.output_dim, graph_dim)
        self.micro_projection = nn.Linear(self.micro_encoder.output_dim, graph_dim)
        self.scale_attention = ScaleAttention(graph_dim, pert_dim, graph_dim)

    def forward(
        self, eigenvectors: torch.Tensor, perturbation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        macro = eigenvectors[..., : self.macro_components]
        micro = eigenvectors[..., self.macro_components :]
        macro = self.macro_projection(self.macro_encoder(macro))
        micro = self.micro_projection(self.micro_encoder(micro))
        if self.scale_mode == "macro":
            return macro, torch.ones(
                (*macro.shape[:-1], 1), dtype=macro.dtype, device=macro.device
            )
        if self.scale_mode == "micro":
            return micro, torch.zeros(
                (*micro.shape[:-1], 1), dtype=micro.dtype, device=micro.device
            )
        return self.scale_attention(macro, micro, perturbation)


class CrossGraphFusion(nn.Module):
    """Choose between functional and expression-derived topology per gene."""

    def __init__(
        self,
        graph_dim: int,
        pert_dim: int,
        output_dim: int,
        graph_mode: str = "dual",
        fusion_mode: str = "adaptive",
    ) -> None:
        super().__init__()
        if graph_mode not in {"dual", "go", "coexp", "none"}:
            raise ValueError("graph_mode must be 'dual', 'go', 'coexp', or 'none'")
        if fusion_mode not in {"adaptive", "mean", "concat"}:
            raise ValueError("fusion_mode must be 'adaptive', 'mean', or 'concat'")
        self.graph_mode = graph_mode
        self.fusion_mode = fusion_mode
        self.go_projection = nn.Linear(graph_dim, output_dim)
        self.coexp_projection = nn.Linear(graph_dim, output_dim)
        self.go_score = nn.Linear(output_dim + pert_dim, 1)
        self.coexp_score = nn.Linear(output_dim + pert_dim, 1)
        self.concat_projection = nn.Linear(output_dim * 2, output_dim)

    def forward(
        self,
        go_embedding: torch.Tensor,
        coexp_embedding: torch.Tensor,
        perturbation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        go_feature = self.go_projection(go_embedding)
        coexp_feature = self.coexp_projection(coexp_embedding)
        if self.graph_mode == "go":
            weights = torch.zeros(
                (*go_feature.shape[:-1], 2),
                device=go_feature.device,
                dtype=go_feature.dtype,
            )
            weights[..., 0] = 1.0
            return go_feature, weights
        if self.graph_mode == "coexp":
            weights = torch.zeros(
                (*coexp_feature.shape[:-1], 2),
                device=coexp_feature.device,
                dtype=coexp_feature.dtype,
            )
            weights[..., 1] = 1.0
            return coexp_feature, weights
        if self.graph_mode == "none":
            weights = torch.full(
                (*go_feature.shape[:-1], 2),
                0.5,
                device=go_feature.device,
                dtype=go_feature.dtype,
            )
            return torch.zeros_like(go_feature), weights
        if self.fusion_mode == "mean":
            weights = torch.full(
                (*go_feature.shape[:-1], 2),
                0.5,
                device=go_feature.device,
                dtype=go_feature.dtype,
            )
            return 0.5 * (go_feature + coexp_feature), weights
        if self.fusion_mode == "concat":
            weights = torch.full(
                (*go_feature.shape[:-1], 2),
                0.5,
                device=go_feature.device,
                dtype=go_feature.dtype,
            )
            return self.concat_projection(
                torch.cat((go_feature, coexp_feature), dim=-1)
            ), weights
        condition = perturbation.unsqueeze(1).expand(-1, go_feature.shape[1], -1)
        scores = torch.cat(
            (
                self.go_score(torch.cat((go_feature, condition), dim=-1)),
                self.coexp_score(torch.cat((coexp_feature, condition), dim=-1)),
            ),
            dim=-1,
        )
        weights = torch.softmax(scores, dim=-1)
        fused = weights[..., 0:1] * go_feature + weights[..., 1:2] * coexp_feature
        return fused, weights


class DualGraphSpectralFusion(nn.Module):
    """Apply multi-scale SignNet encoding and adaptive GO/coexpression fusion."""

    def __init__(
        self,
        n_genes: int,
        go_components: int,
        coexp_components: int,
        spectral_dim: int,
        pert_dim: int = 32,
        graph_dim: int = 32,
        macro_ratio: float = 0.5,
        sign_hidden_dim: int = 32,
        sign_component_dim: int = 4,
        graph_mode: str = "dual",
        fusion_mode: str = "adaptive",
        scale_mode: str = "multi",
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.go_components = go_components
        self.coexp_components = coexp_components
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(n_genes, pert_dim),
            nn.SiLU(),
            nn.Linear(pert_dim, pert_dim),
        )
        self.go_encoder = GraphScaleEncoder(
            go_components,
            macro_ratio,
            graph_dim,
            pert_dim,
            sign_hidden_dim,
            sign_component_dim,
            scale_mode,
        )
        self.coexp_encoder = GraphScaleEncoder(
            coexp_components,
            macro_ratio,
            graph_dim,
            pert_dim,
            sign_hidden_dim,
            sign_component_dim,
            scale_mode,
        )
        self.cross_graph = CrossGraphFusion(
            graph_dim,
            pert_dim,
            spectral_dim,
            graph_mode=graph_mode,
            fusion_mode=fusion_mode,
        )

    def _batch_spectral(
        self, values: torch.Tensor, components: int, batch_size: int
    ) -> torch.Tensor:
        if values.ndim == 2:
            values = values.unsqueeze(0).expand(batch_size, -1, -1)
        if values.shape != (batch_size, self.n_genes, components):
            raise ValueError(
                "dual-graph eigenvectors must have shape (G, K) or (B, G, K)"
            )
        return values

    def forward(
        self,
        go_eigenvectors: torch.Tensor,
        coexp_eigenvectors: torch.Tensor,
        pert_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size = pert_mask.shape[0]
        go_eigenvectors = self._batch_spectral(
            go_eigenvectors, self.go_components, batch_size
        )
        coexp_eigenvectors = self._batch_spectral(
            coexp_eigenvectors, self.coexp_components, batch_size
        )
        perturbation = self.perturbation_encoder(pert_mask.float())
        go_embedding, go_scale = self.go_encoder(go_eigenvectors, perturbation)
        coexp_embedding, coexp_scale = self.coexp_encoder(
            coexp_eigenvectors, perturbation
        )
        fused, cross_graph = self.cross_graph(
            go_embedding, coexp_embedding, perturbation
        )
        return fused, {
            "go_scale": go_scale,
            "coexp_scale": coexp_scale,
            "cross_graph": cross_graph,
        }
