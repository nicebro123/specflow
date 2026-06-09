"""Random-walk Laplacian spectral embedding.

Internally solves the symmetric normalized Laplacian (numerically stable
symmetric eigensolver) then converts to random-walk eigenvectors via the
identity  phi_rw = D^{-1/2} phi_sym.  Eigenvalues are identical for both
formulations.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


@dataclass(frozen=True)
class SpectralResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass
class SpectralEmbedding:
    """Compute non-trivial eigenvectors of the random-walk Laplacian.

    The random-walk Laplacian L_rw = I - D^{-1}W has the same spectrum as the
    symmetric normalized Laplacian L_sym = I - D^{-1/2}WD^{-1/2}, but its
    eigenvectors encode transition-probability geometry rather than
    degree-weighted positions.  We exploit the spectral equivalence by solving
    L_sym (stable, symmetric) and recovering L_rw eigenvectors via
    phi_rw = D^{-1/2} phi_sym.
    """

    n_components: int = 32

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _symmetric_normalized_laplacian(
        self, adjacency: sparse.spmatrix
    ) -> sparse.csr_matrix:
        """L_sym = I - D^{-1/2} W D^{-1/2} (used internally for the eigensolver)."""
        graph = sparse.csr_matrix(adjacency, dtype=np.float64)
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("adjacency must be square")
        n_nodes = graph.shape[0]
        if n_nodes < 2:
            raise ValueError("adjacency must contain at least two nodes")
        if self.n_components < 1 or self.n_components >= n_nodes:
            raise ValueError("n_components must be in [1, n_nodes - 1]")
        difference = graph - graph.T
        if difference.nnz and not np.allclose(difference.data, 0.0, atol=1e-7):
            raise ValueError("adjacency must be symmetric")

        degrees = np.asarray(graph.sum(axis=1)).ravel()
        inv_sqrt_degree = np.zeros_like(degrees)
        positive = degrees > 0
        inv_sqrt_degree[positive] = 1.0 / np.sqrt(degrees[positive])
        normalizer = sparse.diags(inv_sqrt_degree)
        return sparse.eye(n_nodes, format="csr") - normalizer @ graph @ normalizer

    @staticmethod
    def _inv_sqrt_degree(adjacency: sparse.spmatrix) -> np.ndarray:
        degrees = np.asarray(
            sparse.csr_matrix(adjacency, dtype=np.float64).sum(axis=1)
        ).ravel()
        out = np.zeros_like(degrees)
        pos = degrees > 0
        out[pos] = 1.0 / np.sqrt(degrees[pos])
        return out

    @staticmethod
    def _sqrt_degree(adjacency: sparse.spmatrix) -> np.ndarray:
        degrees = np.asarray(
            sparse.csr_matrix(adjacency, dtype=np.float64).sum(axis=1)
        ).ravel()
        out = np.zeros_like(degrees)
        pos = degrees > 0
        out[pos] = np.sqrt(degrees[pos])
        return out

    def _to_rw_basis(
        self, sym_vectors: np.ndarray, adjacency: sparse.spmatrix
    ) -> np.ndarray:
        """phi_rw = D^{-1/2} phi_sym."""
        return self._inv_sqrt_degree(adjacency)[:, None] * sym_vectors

    def _to_sym_basis(
        self, rw_vectors: np.ndarray, adjacency: sparse.spmatrix
    ) -> np.ndarray:
        """phi_sym = D^{1/2} phi_rw."""
        return self._sqrt_degree(adjacency)[:, None] * rw_vectors

    # ------------------------------------------------------------------
    # Public API  (kept backward-compatible)
    # ------------------------------------------------------------------

    def normalized_laplacian(self, adjacency: sparse.spmatrix) -> sparse.csr_matrix:
        """Return L_sym.  Retained for external callers; internally prefer
        ``_symmetric_normalized_laplacian``."""
        return self._symmetric_normalized_laplacian(adjacency)

    def fit(self, adjacency: sparse.spmatrix) -> SpectralResult:
        graph = sparse.csr_matrix(adjacency, dtype=np.float64)
        n_nodes = graph.shape[0]
        laplacian = self._symmetric_normalized_laplacian(graph)

        requested = self.n_components + 1
        if requested >= n_nodes:
            values, vectors = np.linalg.eigh(laplacian.toarray())
        else:
            values, vectors = sparse_linalg.eigsh(laplacian, k=requested, which="SM")

        order = np.argsort(values)
        values = np.maximum(values[order], 0.0)
        vectors = vectors[:, order]

        # Skip the constant connected-component mode.
        values = values[1 : self.n_components + 1]
        sym_vectors = vectors[:, 1 : self.n_components + 1]

        # L_sym eigenvectors → L_rw eigenvectors
        rw_vectors = self._to_rw_basis(sym_vectors, graph)

        return SpectralResult(
            eigenvalues=values.astype(np.float32),
            eigenvectors=rw_vectors.astype(np.float32),
        )

    def fit_transform(self, adjacency: sparse.spmatrix) -> np.ndarray:
        return self.fit(adjacency).eigenvectors

    def fit_with_perturbation_update(
        self,
        base_adjacency: sparse.spmatrix,
        modified_adjacency: sparse.spmatrix,
        base_result: SpectralResult = None,
    ) -> SpectralResult:
        """Approximate modified eigenvectors with first-order perturbation.

        The correction is projected onto the retained non-trivial base
        eigenspace.  Perturbation theory is applied in the L_sym basis
        (symmetric, numerically stable) and the result is converted back
        to L_rw eigenvectors.  Exact ``fit`` remains preferable when graph
        modifications are large or eigenvalues are nearly repeated.
        """
        base_result = base_result or self.fit(base_adjacency)
        values = base_result.eigenvalues.astype(np.float64)
        rw_vectors = base_result.eigenvectors.astype(np.float64)

        # Convert stored L_rw eigenvectors back to L_sym basis for
        # perturbation theory (requires symmetry of the operator).
        sym_vectors = self._to_sym_basis(rw_vectors, base_adjacency)

        delta = self._symmetric_normalized_laplacian(
            modified_adjacency
        ) - self._symmetric_normalized_laplacian(base_adjacency)

        projected = sym_vectors.T @ (delta @ sym_vectors)
        corrected = sym_vectors.copy()
        for column in range(values.size):
            for other in range(values.size):
                if column == other:
                    continue
                denominator = values[column] - values[other]
                if abs(denominator) > 1e-8:
                    corrected[:, column] += (
                        projected[other, column] / denominator
                    ) * sym_vectors[:, other]
        corrected, _ = np.linalg.qr(corrected)

        modified_laplacian = self._symmetric_normalized_laplacian(modified_adjacency)
        updated_values = np.asarray(
            [
                corrected[:, index].T @ (modified_laplacian @ corrected[:, index])
                for index in range(corrected.shape[1])
            ]
        )
        order = np.argsort(updated_values)

        # Convert corrected L_sym eigenvectors to L_rw basis using the
        # *modified* graph's degree matrix.
        rw_corrected = self._to_rw_basis(corrected[:, order], modified_adjacency)

        return SpectralResult(
            eigenvalues=np.maximum(updated_values[order], 0.0).astype(np.float32),
            eigenvectors=rw_corrected.astype(np.float32),
        )
