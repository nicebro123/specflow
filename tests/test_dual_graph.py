import numpy as np
from scipy import sparse
import torch

from specflow.flow.ode_solver import EulerSampler
from specflow.graph.go_graph import GOGraphBuilder
from specflow.graph.perturbation_aware import PerturbationAwareGraphModifier
from specflow.graph.spectral_cache import SpectralCache
from specflow.model.sign_net import SignNet
from specflow.model.specflow import SpecFlow
from specflow.training.trainer import SpecFlowTrainer


def test_go_graph_parses_namespace_filtered_gaf(tmp_path):
    gaf = tmp_path / "annotations.gaf"
    gaf.write_text(
        "!gaf-version: 2.2\n"
        "DB\t1\tA\t\tGO:0001\tREF\tEXP\t\tP\tA\t\tprotein\ttaxon:9606\t20200101\tDB\n"
        "DB\t2\tB\t\tGO:0001\tREF\tEXP\t\tP\tB\t\tprotein\ttaxon:9606\t20200101\tDB\n"
        "DB\t3\tC\t\tGO:0001\tREF\tEXP\t\tF\tC\t\tprotein\ttaxon:9606\t20200101\tDB\n",
        encoding="utf-8",
    )
    graph = GOGraphBuilder(
        ["A", "B", "C"],
        annotation_file=str(gaf),
        k_neighbors=2,
        namespace="biological_process",
    ).build()

    assert graph[0, 1] > 0
    assert graph[0, 2] == 0
    np.testing.assert_allclose(graph.toarray(), graph.toarray().T)


def test_perturbation_edges_are_attenuated_once_for_combination_targets():
    graph = sparse.csr_matrix(
        np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    )
    modifier = PerturbationAwareGraphModifier(alpha_go=0.2, alpha_coexp=0.1)
    modified = modifier.modify(graph, np.array([1, 1, 0]), "go").toarray()

    np.testing.assert_allclose(
        [modified[0, 1], modified[0, 2], modified[1, 2]],
        [0.2, 0.4, 0.6],
    )


def _small_cache(tmp_path=None, use_approximation=False, n_components=2):
    go = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0.5, 0, 0],
                [1, 0, 1, 0.2, 0],
                [0.5, 1, 0, 1, 0.4],
                [0, 0.2, 1, 0, 1],
                [0, 0, 0.4, 1, 0],
            ],
            dtype=float,
        )
    )
    coexp = sparse.csr_matrix(
        np.array(
            [
                [0, 0.8, 0.2, 0, 0.1],
                [0.8, 0, 0.7, 0.1, 0],
                [0.2, 0.7, 0, 0.6, 0.3],
                [0, 0.1, 0.6, 0, 0.9],
                [0.1, 0, 0.3, 0.9, 0],
            ]
        )
    )
    return SpectralCache(
        {"go": go, "coexp": coexp},
        {"go": n_components, "coexp": n_components},
        cache_dir=str(tmp_path) if tmp_path else None,
        use_approximation=use_approximation,
    )


def test_spectral_cache_batches_condition_specific_dual_graph_embeddings(tmp_path):
    cache = _small_cache(tmp_path)
    masks = torch.tensor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float32)
    spectral = cache.batch_embeddings(masks)

    assert spectral["go"].shape == (2, 5, 2)
    assert spectral["coexp"].shape == (2, 5, 2)
    assert np.isfinite(spectral["go"].numpy()).all()
    assert cache.get(masks[1], "go") is cache.get(masks[1], "go")
    assert len(list(tmp_path.glob("*.npz"))) >= 2


def test_spectral_cache_optional_first_order_update_is_finite():
    cache = _small_cache(use_approximation=True)
    spectral = cache.batch_embeddings(torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32))

    assert spectral["go"].shape == (1, 5, 2)
    assert torch.isfinite(spectral["go"]).all()
    assert torch.isfinite(spectral["coexp"]).all()


def test_disk_cache_separates_different_spectral_dimensions(tmp_path):
    mask = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32)
    two_component = _small_cache(tmp_path, n_components=2).batch_embeddings(mask)
    three_component = _small_cache(tmp_path, n_components=3).batch_embeddings(mask)

    assert two_component["go"].shape[-1] == 2
    assert three_component["go"].shape[-1] == 3


def test_sign_net_is_invariant_to_eigenvector_sign_flips():
    torch.manual_seed(11)
    sign_net = SignNet(n_components=3, hidden_dim=8, component_dim=2)
    eigenvectors = torch.randn(2, 6, 3)

    torch.testing.assert_close(sign_net(eigenvectors), sign_net(-eigenvectors))


def test_dual_graph_provider_trains_and_samples_with_mmd():
    torch.manual_seed(12)
    cache = _small_cache()
    model = SpecFlow(
        n_genes=5,
        spectral_dim=4,
        d_model=12,
        hidden_dim=16,
        n_velocity_layers=1,
        dual_graph=True,
        go_components=2,
        coexp_components=2,
        graph_dim=8,
        pert_dim=8,
    )
    trainer = SpecFlowTrainer(
        model,
        cache.batch_embeddings,
        sigma=0.1,
        learning_rate=1e-3,
        mmd_weight=0.02,
        mmd_interval=1,
        mmd_steps=2,
        device=torch.device("cpu"),
    )
    batch = {
        "ctrl_expr": torch.randn(4, 5),
        "pert_expr": torch.randn(4, 5),
        "pert_mask": torch.tensor(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            dtype=torch.float32,
        ),
    }
    history = trainer.fit([batch], n_epochs=1)
    samples = EulerSampler(model, sigma=0.1, n_steps=2).sample(
        batch["ctrl_expr"], batch["pert_mask"], cache.batch_embeddings, n_samples=2
    )

    assert np.isfinite(history[0]["train_loss"])
    assert samples.shape == (2, 4, 5)


def test_delta_correlation_loss_trains_with_condition_groups():
    torch.manual_seed(14)
    cache = _small_cache()
    model = SpecFlow(
        n_genes=5,
        spectral_dim=4,
        d_model=12,
        hidden_dim=16,
        n_velocity_layers=1,
        dual_graph=True,
        go_components=2,
        coexp_components=2,
        graph_dim=8,
        pert_dim=8,
    )
    trainer = SpecFlowTrainer(
        model,
        cache.batch_embeddings,
        sigma=0.1,
        learning_rate=1e-3,
        mmd_weight=0.0,
        delta_corr_weight=0.03,
        device=torch.device("cpu"),
    )
    batch = {
        "ctrl_expr": torch.randn(4, 5),
        "pert_expr": torch.randn(4, 5),
        "pert_mask": torch.tensor(
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]],
            dtype=torch.float32,
        ),
        "condition": ["A", "A", "B", "B"],
    }

    loss = trainer.train_batch(batch)

    assert np.isfinite(loss)
