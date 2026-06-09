"""Perturbation-dependent edge attenuation for gene graphs."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class PerturbationAwareGraphModifier:
    """Attenuate all edges incident to directly perturbed genes once."""

    alpha_go: float = 0.1
    alpha_coexp: float = 0.05

    def __post_init__(self) -> None:
        for graph_type, alpha in self.alpha.items():
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"{graph_type} attenuation must lie in [0, 1]")

    @property
    def alpha(self):
        return {"go": self.alpha_go, "coexp": self.alpha_coexp}

    def modify(
        self,
        adjacency: sparse.spmatrix,
        pert_mask: np.ndarray,
        graph_type: str,
    ) -> sparse.csr_matrix:
        if graph_type not in self.alpha:
            raise ValueError(f"unknown graph_type: {graph_type!r}")
        graph = sparse.coo_matrix(adjacency, dtype=np.float64)
        mask = np.asarray(pert_mask).reshape(-1) > 0
        if graph.shape[0] != graph.shape[1] or graph.shape[0] != mask.size:
            raise ValueError("pert_mask must align with a square adjacency matrix")
        if not mask.any():
            return graph.tocsr()

        affected = mask[graph.row] | mask[graph.col]
        data = graph.data.copy()
        data[affected] *= self.alpha[graph_type]
        modified = sparse.csr_matrix((data, (graph.row, graph.col)), shape=graph.shape)
        modified.eliminate_zeros()
        return modified

    def modify_pair(self, graphs, pert_mask: np.ndarray):
        return {
            graph_type: self.modify(adjacency, pert_mask, graph_type)
            for graph_type, adjacency in graphs.items()
        }
