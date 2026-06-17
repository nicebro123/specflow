import torch
import pytest

from specflow.config import SpecFlowConfig
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


def test_no_spectral_embedding_zeroes_spectral_features():
    torch.manual_seed(7)
    model = SpecFlow(
        8,
        3,
        d_model=16,
        hidden_dim=24,
        n_velocity_layers=1,
        use_spectral_embedding=False,
    )
    features = model.encode_condition(
        torch.randn(2, 8),
        torch.randint(0, 2, (2, 8)).float(),
        torch.randn(8, 3),
    )

    assert torch.count_nonzero(features["spectral_embedding"]) == 0


class _RecordingVelocityField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_propagation = None

    def forward(
        self,
        x_t,
        time,
        cell_condition,
        spectral_embedding,
        ctrl_expr,
        pert_mask,
        pert_embedding=None,
        propagation=None,
    ):
        self.seen_propagation = propagation.detach().clone()
        return torch.zeros_like(x_t)


def test_spectral_propagation_scale_scales_forward_features():
    torch.manual_seed(13)
    model = SpecFlow(
        8,
        3,
        d_model=16,
        hidden_dim=24,
        n_velocity_layers=1,
        spectral_propagation=True,
        propagation_channels=2,
        propagation_scale=0.25,
    )
    eigenvectors = torch.randn(8, 3)
    eigenvalues = torch.tensor([0.1, 0.5, 1.0])
    model.propagation.set_basis(eigenvectors, eigenvalues)
    recorder = _RecordingVelocityField()
    model.velocity_field = recorder
    ctrl = torch.randn(2, 8)
    pert = torch.randint(0, 2, (2, 8)).float()
    spectral = torch.randn(8, 3)
    raw_propagation = model.propagation(pert)

    output, _ = model(ctrl, torch.tensor([0.2, 0.7]), ctrl, pert, spectral)

    torch.testing.assert_close(output, torch.zeros_like(ctrl))
    torch.testing.assert_close(
        recorder.seen_propagation, raw_propagation * 0.25
    )


def test_perturbation_gate_scales_propagation_channels_from_initial_gate():
    torch.manual_seed(19)
    model = SpecFlow(
        8,
        3,
        d_model=16,
        hidden_dim=24,
        n_velocity_layers=1,
        spectral_propagation=True,
        propagation_channels=2,
        propagation_scale=0.5,
        propagation_gate="perturbation",
        propagation_gate_init=0.25,
    )
    model.propagation.set_basis(torch.randn(8, 3), torch.tensor([0.1, 0.5, 1.0]))
    recorder = _RecordingVelocityField()
    model.velocity_field = recorder
    ctrl = torch.randn(2, 8)
    pert = torch.randint(0, 2, (2, 8)).float()
    raw_propagation = model.propagation(pert)

    model(ctrl, torch.tensor([0.2, 0.7]), ctrl, pert, torch.randn(8, 3))

    assert model.propagation_gate_mode == "perturbation"
    torch.testing.assert_close(
        recorder.seen_propagation, raw_propagation * 0.5 * 0.25
    )


def test_zero_spectral_propagation_scale_keeps_shape_and_zeroes_features():
    torch.manual_seed(17)
    model = SpecFlow(
        8,
        3,
        d_model=16,
        hidden_dim=24,
        n_velocity_layers=1,
        spectral_propagation=True,
        propagation_channels=2,
        propagation_scale=0.0,
    )
    model.propagation.set_basis(torch.randn(8, 3), torch.tensor([0.1, 0.5, 1.0]))
    recorder = _RecordingVelocityField()
    model.velocity_field = recorder
    ctrl = torch.randn(2, 8)
    pert = torch.randint(0, 2, (2, 8)).float()

    model(ctrl, torch.tensor([0.2, 0.7]), ctrl, pert, torch.randn(8, 3))

    assert recorder.seen_propagation.shape == (2, 8, 2)
    torch.testing.assert_close(
        recorder.seen_propagation, torch.zeros_like(recorder.seen_propagation)
    )


def test_spectral_propagation_scale_config_defaults_and_validation():
    config = SpecFlowConfig.from_dict({"model": {"spectral_propagation": True}})
    assert config.model.propagation_scale == 1.0
    assert config.model.propagation_gate == "none"
    assert config.model.propagation_gate_init == 0.5

    with pytest.raises(ValueError, match="propagation_scale"):
        SpecFlowConfig.from_dict({"model": {"propagation_scale": -0.1}})
    with pytest.raises(ValueError, match="propagation_gate"):
        SpecFlowConfig.from_dict({"model": {"propagation_gate": "bad"}})
    with pytest.raises(ValueError, match="propagation_gate_init"):
        SpecFlowConfig.from_dict({"model": {"propagation_gate_init": 1.0}})
    with pytest.raises(ValueError, match="spectral_propagation"):
        SpecFlowConfig.from_dict({"model": {"propagation_gate": "perturbation"}})


def test_negative_spectral_propagation_scale_is_rejected():
    with pytest.raises(ValueError, match="propagation_scale"):
        SpecFlow(8, 3, propagation_scale=-1.0)


def test_perturbation_gate_requires_spectral_propagation():
    with pytest.raises(ValueError, match="spectral_propagation"):
        SpecFlow(8, 3, propagation_gate="perturbation")


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_controls = None

    def forward(self, x_t, time, ctrl_expr, pert_mask, spectral_embedding):
        self.seen_controls = ctrl_expr.detach().clone()
        return torch.zeros_like(x_t), None


def test_flow_can_disable_control_anchor():
    batch = {
        "ctrl_expr": torch.ones(2, 3),
        "pert_expr": torch.full((2, 3), 4.0),
        "pert_mask": torch.zeros(2, 3),
    }
    model = _RecordingModel()

    output = ControlAnchoredFlowMatching(
        sigma=0.0,
        control_anchor=False,
    ).compute_loss(model, batch, torch.randn(3, 2))

    torch.testing.assert_close(model.seen_controls, torch.zeros(2, 3))
    torch.testing.assert_close(output.target_delta, batch["pert_expr"])


def test_euler_sampler_can_disable_control_anchor():
    model = _RecordingModel()
    samples = EulerSampler(
        model,
        sigma=0.0,
        n_steps=2,
        control_anchor=False,
    ).sample(torch.ones(2, 3), torch.zeros(2, 3), torch.randn(3, 2))

    torch.testing.assert_close(model.seen_controls, torch.zeros(2, 3))
    torch.testing.assert_close(samples, torch.zeros(1, 2, 3))


def test_euler_sampler_returns_requested_population_shape():
    batch = _build_batch(batch_size=3, n_genes=8)
    model = SpecFlow(8, 3, d_model=16, hidden_dim=24, n_velocity_layers=1)
    samples = EulerSampler(model, sigma=0.1, n_steps=3).sample(
        batch["ctrl_expr"], batch["pert_mask"], torch.randn(8, 3), n_samples=4
    )
    assert samples.shape == (4, 3, 8)
