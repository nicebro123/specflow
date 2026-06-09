import torch

from specflow.flow.flow_matching import ControlAnchoredFlowMatching
from specflow.flow.ode_solver import EulerSampler
from specflow.model.specflow import SpecFlow


def _build_batch(batch_size=5, n_genes=8):
    return {
        "ctrl_expr": torch.randn(batch_size, n_genes),
        "pert_expr": torch.randn(batch_size, n_genes),
        "pert_mask": torch.randint(0, 2, (batch_size, n_genes)).float(),
    }


def test_model_and_flow_loss_have_expected_shapes_and_gradients():
    torch.manual_seed(5)
    batch = _build_batch()
    spectral = torch.randn(8, 3)
    model = SpecFlow(8, 3, d_model=16, hidden_dim=24, n_velocity_layers=2)

    output = ControlAnchoredFlowMatching(sigma=0.2).compute_loss(
        model, batch, spectral
    )
    assert output.prediction.shape == (5, 8)
    assert output.target.shape == (5, 8)
    assert output.target_delta.shape == (5, 8)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_euler_sampler_returns_requested_population_shape():
    batch = _build_batch(batch_size=3, n_genes=8)
    model = SpecFlow(8, 3, d_model=16, hidden_dim=24, n_velocity_layers=1)
    samples = EulerSampler(model, sigma=0.1, n_steps=3).sample(
        batch["ctrl_expr"], batch["pert_mask"], torch.randn(8, 3), n_samples=4
    )
    assert samples.shape == (4, 3, 8)
