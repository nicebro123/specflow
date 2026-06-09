"""Control-derived coexpression graph construction."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class CoexpressionGraphBuilder:
    """Build an absolute-Pearson sparse graph from control expression."""

    k_neighbors: int = 20
    threshold: float = 0.3

    def build(self, control_expression: np.ndarray) -> sparse.csr_matrix:
        expression = np.asarray(control_expression, dtype=np.float64)
        if expression.ndim != 2:
            raise ValueError("control_expression must have shape (N_cells, G_genes)")
        if expression.shape[0] < 2 or expression.shape[1] < 2:
            raise ValueError("at least two cells and two genes are required")
        if self.k_neighbors < 0:
            raise ValueError("k_neighbors must be non-negative")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")

        correlation = np.corrcoef(expression, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.abs(correlation)
        np.fill_diagonal(weights, 0.0)

        keep = weights >= self.threshold
        n_genes = weights.shape[0]
        k = min(self.k_neighbors, n_genes - 1)
        if k:
            topk_idx = np.argpartition(weights, -k, axis=1)[:, -k:]
            rows = np.arange(n_genes)[:, None]
            keep[rows, topk_idx] = weights[rows, topk_idx] > 0

        directed = np.where(keep, weights, 0.0)
        symmetric = 0.5 * (directed + directed.T)
        np.fill_diagonal(symmetric, 0.0)
        graph = sparse.csr_matrix(symmetric)
        graph.eliminate_zeros()
        return graph
