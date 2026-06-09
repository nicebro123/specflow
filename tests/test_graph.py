import numpy as np

from specflow.graph.coexp_graph import CoexpressionGraphBuilder
from specflow.graph.spectral_embedding import SpectralEmbedding


def test_coexpression_graph_is_sparse_symmetric_and_has_no_self_edges():
    rng = np.random.default_rng(3)
    expression = rng.normal(size=(40, 10))
    graph = CoexpressionGraphBuilder(k_neighbors=3, threshold=0.6).build(expression)

    assert graph.shape == (10, 10)
    np.testing.assert_allclose(graph.toarray(), graph.toarray().T)
    np.testing.assert_array_equal(graph.diagonal(), np.zeros(10))
    assert graph.nnz > 0


def test_spectral_embedding_returns_sorted_finite_components():
    rng = np.random.default_rng(4)
    expression = rng.normal(size=(50, 12))
    graph = CoexpressionGraphBuilder(k_neighbors=4, threshold=0.2).build(expression)
    result = SpectralEmbedding(n_components=4).fit(graph)

    assert result.eigenvectors.shape == (12, 4)
    assert result.eigenvalues.shape == (4,)
    assert np.isfinite(result.eigenvectors).all()
    assert np.all(np.diff(result.eigenvalues) >= -1e-6)
    assert np.all(result.eigenvalues >= 0)
