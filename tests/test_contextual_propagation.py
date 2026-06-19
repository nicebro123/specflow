import numpy as np
import pytest
from scipy import sparse
import torch

from specflow.config import SpecFlowConfig
from specflow.model.contextual_propagation import (
    ContextualLocalPropagation,
    GraphAwarePerturbationEncoder,
)
from specflow.model.specflow import SpecFlow
from specflow.model.velocity_field import VelocityField


def _spectra(batch_size=3, n_genes=5, components=3):
    torch.manual_seed(41)
    return {
        "go": torch.randn(batch_size, n_genes, components),
        "coexp": torch.randn(batch_size, n_genes, components),
    }


def test_graph_aware_perturbation_encoder_is_sign_invariant():
    encoder = GraphAwarePerturbationEncoder(
        n_genes=5,
        go_components=3,
        coexp_components=3,
        graph_dim=4,
        pert_dim=6,
    )
    encoder.eval()
    spectra = _spectra()
    mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    encoded = encoder(spectra, mask)
    flipped = encoder(
        {"go": -spectra["go"], "coexp": -spectra["coexp"]},
        mask,
    )

    torch.testing.assert_close(encoded, flipped)


def test_graph_aware_encoder_handles_single_double_and_empty_masks():
    encoder = GraphAwarePerturbationEncoder(
        n_genes=5,
        go_components=3,
        coexp_components=3,
        graph_dim=4,
        pert_dim=6,
    )
    mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    encoded = encoder(_spectra(), mask)

    assert encoded.shape == (3, 6)
    assert torch.isfinite(encoded).all()
    torch.testing.assert_close(encoded[2], torch.zeros_like(encoded[2]))


def test_contextual_local_propagation_routes_one_hop_and_excludes_targets():
    module = ContextualLocalPropagation(
        n_genes=3,
        d_model=4,
        pert_dim=2,
        hidden_dim=5,
        null_init=0.9,
    )
    go = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    coexp = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    module.set_graphs(go, coexp)
    mask = torch.tensor([[1.0, 0.0, 0.0]])

    influence, probabilities = module(
        mask,
        torch.randn(1, 3, 4),
        torch.randn(1, 2),
    )

    assert influence.shape == (1, 3, 2)
    torch.testing.assert_close(
        probabilities[0, 0],
        torch.tensor([0.9, 0.05, 0.05]),
    )
    torch.testing.assert_close(influence[:, 0], torch.zeros(1, 2))
    assert influence[0, 1, 0] > 0
    assert influence[0, 2, 1] > 0
    assert influence[0, 2, 0] == 0
    assert influence[0, 1, 1] == 0


def test_contextual_local_scale_zero_strictly_disables_influence():
    module = ContextualLocalPropagation(
        n_genes=3,
        d_model=4,
        pert_dim=2,
        null_init=0.9,
        scale=0.0,
    )
    graph = sparse.csr_matrix(
        np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    )
    module.set_graphs(graph, graph)

    influence, _ = module(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.randn(1, 3, 4),
        torch.randn(1, 2),
    )

    torch.testing.assert_close(influence, torch.zeros_like(influence))


def test_zero_initialized_contextual_adapter_preserves_velocity_output():
    torch.manual_seed(43)
    field = VelocityField(
        spectral_dim=3,
        d_model=4,
        hidden_dim=7,
        n_layers=2,
        pert_dim=2,
        contextual_prop_dim=2,
    )
    batch_size, n_genes = 2, 5
    args = (
        torch.randn(batch_size, n_genes),
        torch.rand(batch_size, 1),
        torch.randn(batch_size, 4),
        torch.randn(batch_size, n_genes, 3),
        torch.randn(batch_size, n_genes),
        torch.zeros(batch_size, n_genes),
        torch.randn(batch_size, 2),
    )

    without_influence = field(
        *args,
        contextual_propagation=torch.zeros(batch_size, n_genes, 2),
    )
    with_influence = field(
        *args,
        contextual_propagation=torch.randn(batch_size, n_genes, 2),
    )

    torch.testing.assert_close(without_influence, with_influence)


def test_legacy_model_state_dict_loads_strictly_without_contextual_keys():
    torch.manual_seed(47)
    original = SpecFlow(
        n_genes=8,
        spectral_dim=3,
        d_model=8,
        hidden_dim=12,
        dual_graph=False,
        spectral_propagation=True,
        propagation_channels=2,
    )
    state = original.state_dict()
    restored = SpecFlow(
        n_genes=8,
        spectral_dim=3,
        d_model=8,
        hidden_dim=12,
        dual_graph=False,
        spectral_propagation=True,
        propagation_channels=2,
    )

    restored.load_state_dict(state, strict=True)

    assert not any("contextual" in key for key in state)
    assert not any("graph_pert_encoder" in key for key in state)


def test_contextual_configuration_defaults_and_invalid_combinations():
    default = SpecFlowConfig.from_dict({})
    assert default.model.perturbation_encoder == "legacy"
    assert default.model.propagation_variant == "spectral"
    assert default.model.local_propagation_hops == 1
    assert default.model.local_propagation_null_init == 0.9

    valid = SpecFlowConfig.from_dict(
        {
            "model": {
                "dual_graph": True,
                "perturbation_encoder": "graph_pool",
                "spectral_propagation": True,
                "propagation_variant": "contextual_local",
                "propagation_channels": 2,
            }
        }
    )
    assert valid.model.propagation_variant == "contextual_local"

    with pytest.raises(ValueError, match="graph_pool"):
        SpecFlowConfig.from_dict(
            {"model": {"dual_graph": False, "perturbation_encoder": "graph_pool"}}
        )
    with pytest.raises(ValueError, match="spectral_propagation"):
        SpecFlowConfig.from_dict(
            {
                "model": {
                    "perturbation_encoder": "graph_pool",
                    "propagation_variant": "contextual_local",
                    "propagation_channels": 2,
                }
            }
        )
    with pytest.raises(ValueError, match="propagation_gate"):
        SpecFlowConfig.from_dict(
            {
                "model": {
                    "perturbation_encoder": "graph_pool",
                    "spectral_propagation": True,
                    "propagation_variant": "contextual_local",
                    "propagation_channels": 2,
                    "propagation_gate": "perturbation",
                }
            }
        )
