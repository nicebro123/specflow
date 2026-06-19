"""Graph-aware perturbation encoding and contextual local propagation."""

import math
from typing import Mapping, Tuple

import numpy as np
from scipy import sparse
import torch
from torch import nn

from specflow.model.sign_net import SignNet


class GraphAwarePerturbationEncoder(nn.Module):
    """Encode targeted genes from sign-invariant dual-graph coordinates."""

    def __init__(
        self,
        n_genes: int,
        go_components: int,
        coexp_components: int,
        graph_dim: int,
        pert_dim: int,
        sign_hidden_dim: int = 32,
        sign_component_dim: int = 4,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.go_components = go_components
        self.coexp_components = coexp_components
        self.go_encoder = SignNet(
            go_components, sign_hidden_dim, sign_component_dim
        )
        self.coexp_encoder = SignNet(
            coexp_components, sign_hidden_dim, sign_component_dim
        )
        self.go_projection = nn.Linear(self.go_encoder.output_dim, graph_dim)
        self.coexp_projection = nn.Linear(
            self.coexp_encoder.output_dim, graph_dim
        )
        pooled_dim = graph_dim * 4 + 1
        self.output = nn.Sequential(
            nn.Linear(pooled_dim, pert_dim),
            nn.SiLU(),
            nn.Linear(pert_dim, pert_dim),
        )

    def _batch_spectral(
        self,
        values: torch.Tensor,
        components: int,
        batch_size: int,
    ) -> torch.Tensor:
        if values.ndim == 2:
            values = values.unsqueeze(0).expand(batch_size, -1, -1)
        if values.shape != (batch_size, self.n_genes, components):
            raise ValueError(
                "dual-graph eigenvectors must have shape (G, K) or (B, G, K)"
            )
        return values

    @staticmethod
    def _encode_structure(
        values: torch.Tensor,
        encoder: SignNet,
        projection: nn.Linear,
    ) -> torch.Tensor:
        # Static spectra arrive as an expanded view. Encode the shared graph once
        # instead of repeating SignNet over every cell in the batch.
        if values.ndim == 3 and values.stride(0) == 0:
            encoded = projection(encoder(values[:1]))
            return encoded.expand(values.shape[0], -1, -1)
        return projection(encoder(values))

    @staticmethod
    def _pool(
        structure: torch.Tensor,
        mask: torch.Tensor,
        count: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.einsum("bg,bgd->bd", mask, structure) / count
        second_moment = (
            torch.einsum("bg,bgd->bd", mask, structure.square()) / count
        )
        return mean, second_moment

    def forward(
        self,
        spectral_input: Mapping[str, torch.Tensor],
        pert_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(spectral_input, Mapping) or set(spectral_input) != {
            "go",
            "coexp",
        }:
            raise ValueError(
                "graph-aware perturbation encoding requires {'go', 'coexp'} spectra"
            )
        if pert_mask.ndim != 2 or pert_mask.shape[1] != self.n_genes:
            raise ValueError(f"pert_mask must have shape (B, {self.n_genes})")
        batch_size = pert_mask.shape[0]
        go = self._batch_spectral(
            spectral_input["go"], self.go_components, batch_size
        )
        coexp = self._batch_spectral(
            spectral_input["coexp"], self.coexp_components, batch_size
        )
        go_structure = self._encode_structure(
            go, self.go_encoder, self.go_projection
        )
        coexp_structure = self._encode_structure(
            coexp, self.coexp_encoder, self.coexp_projection
        )
        mask = pert_mask.to(go_structure.dtype)
        raw_count = mask.sum(dim=1, keepdim=True)
        count = raw_count.clamp_min(1.0)
        go_mean, go_second = self._pool(go_structure, mask, count)
        coexp_mean, coexp_second = self._pool(coexp_structure, mask, count)
        count_feature = torch.log1p(raw_count)
        pooled = torch.cat(
            (
                go_mean,
                go_second,
                coexp_mean,
                coexp_second,
                count_feature,
            ),
            dim=-1,
        )
        encoded = self.output(pooled)
        return encoded * (raw_count > 0).to(encoded.dtype)


class ContextualLocalPropagation(nn.Module):
    """Route one-hop GO/coexpression influence or choose no propagation."""

    output_dim = 2
    route_names = ("null", "go", "coexp")

    def __init__(
        self,
        n_genes: int,
        d_model: int,
        pert_dim: int,
        hidden_dim: int = 64,
        null_init: float = 0.9,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < null_init < 1.0:
            raise ValueError("null_init must be between 0 and 1")
        if scale < 0:
            raise ValueError("scale must be non-negative")
        self.n_genes = n_genes
        self.null_init = float(null_init)
        self.scale = float(scale)
        self.router = nn.Sequential(
            nn.LayerNorm(d_model + pert_dim),
            nn.Linear(d_model + pert_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        final = self.router[-1]
        nn.init.zeros_(final.weight)
        side_probability = (1.0 - null_init) / 2.0
        with torch.no_grad():
            final.bias.copy_(
                torch.tensor(
                    [
                        math.log(null_init),
                        math.log(side_probability),
                        math.log(side_probability),
                    ],
                    dtype=final.bias.dtype,
                )
            )
        self._has_graphs = False
        self.register_buffer(
            "_routing_mass", torch.zeros(3), persistent=False
        )
        self.register_buffer(
            "_routing_count", torch.zeros(()), persistent=False
        )

    @staticmethod
    def _normalized_adjacency(adjacency) -> torch.Tensor:
        graph = sparse.csr_matrix(adjacency, dtype=np.float32)
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("adjacency must be square")
        if (graph - graph.T).nnz:
            raise ValueError("adjacency must be symmetric")
        graph = graph.copy()
        graph.setdiag(0.0)
        graph.eliminate_zeros()
        degree = np.asarray(graph.sum(axis=1)).reshape(-1)
        inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        positive = degree > 0
        inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
        normalized = sparse.diags(inv_sqrt) @ graph @ sparse.diags(inv_sqrt)
        coo = normalized.tocoo()
        indices = torch.from_numpy(
            np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
        )
        values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
        return torch.sparse_coo_tensor(
            indices, values, size=coo.shape
        ).coalesce()

    def set_graphs(self, go_adjacency, coexp_adjacency) -> None:
        go = self._normalized_adjacency(go_adjacency)
        coexp = self._normalized_adjacency(coexp_adjacency)
        expected = (self.n_genes, self.n_genes)
        if tuple(go.shape) != expected or tuple(coexp.shape) != expected:
            raise ValueError(
                f"graph shapes must both equal {expected}"
            )
        device = self.router[-1].weight.device
        go = go.to(device)
        coexp = coexp.to(device)
        if self._has_graphs:
            self.go_adjacency = go
            self.coexp_adjacency = coexp
        else:
            self.register_buffer(
                "go_adjacency", go, persistent=False
            )
            self.register_buffer(
                "coexp_adjacency", coexp, persistent=False
            )
            self._has_graphs = True

    @staticmethod
    def _propagate(
        adjacency: torch.Tensor,
        pert_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=pert_mask.device.type, enabled=False):
            propagated = torch.sparse.mm(
                adjacency.float(),
                pert_mask.float().transpose(0, 1),
            ).transpose(0, 1)
        propagated = propagated.to(pert_mask.dtype)
        return propagated * (1.0 - pert_mask)

    def reset_routing_stats(self) -> None:
        self._routing_mass.zero_()
        self._routing_count.zero_()

    def routing_summary(self) -> dict:
        count = float(self._routing_count.item())
        if count <= 0:
            return {name: 0.0 for name in self.route_names}
        values = (self._routing_mass / count).detach().cpu().tolist()
        return dict(zip(self.route_names, values))

    def forward(
        self,
        pert_mask: torch.Tensor,
        gene_tokens: torch.Tensor,
        pert_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._has_graphs:
            raise RuntimeError(
                "ContextualLocalPropagation requires set_graphs(go, coexp)"
            )
        if pert_mask.shape != gene_tokens.shape[:2]:
            raise ValueError("pert_mask and gene_tokens must align on (B, G)")
        condition = pert_embedding.unsqueeze(1).expand(
            -1, self.n_genes, -1
        )
        probabilities = torch.softmax(
            self.router(torch.cat((gene_tokens, condition), dim=-1)),
            dim=-1,
        )
        go_candidate = self._propagate(self.go_adjacency, pert_mask)
        coexp_candidate = self._propagate(
            self.coexp_adjacency, pert_mask
        )
        influence = torch.stack(
            (
                go_candidate * probabilities[..., 1],
                coexp_candidate * probabilities[..., 2],
            ),
            dim=-1,
        )
        influence = influence * self.scale
        if not self.training:
            self._routing_mass.add_(
                probabilities.detach().sum(dim=(0, 1))
            )
            self._routing_count.add_(
                probabilities.new_tensor(
                    probabilities.shape[0] * probabilities.shape[1]
                )
            )
        return influence, probabilities
