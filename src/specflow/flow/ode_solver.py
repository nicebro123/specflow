"""Lightweight ODE inference for static and dynamic SpecFlow conditions."""

from typing import Mapping

import torch


class EulerSampler:
    """Integrate the learned velocity field using explicit Euler steps."""

    def __init__(
        self,
        model,
        sigma: float = 0.5,
        n_steps: int = 50,
        control_anchor: bool = True,
    ) -> None:
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        self.model = model
        self.sigma = sigma
        self.n_steps = n_steps
        self.control_anchor = control_anchor

    @torch.no_grad()
    def sample(
        self,
        ctrl_expr: torch.Tensor,
        pert_mask: torch.Tensor,
        spectral_embedding,
        n_samples: int = 1,
    ) -> torch.Tensor:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        original_batch = ctrl_expr.shape[0]
        controls = ctrl_expr.repeat(n_samples, 1)
        anchor = controls if self.control_anchor else torch.zeros_like(controls)
        masks = pert_mask.repeat(n_samples, 1)
        state = anchor + self.sigma * torch.randn_like(controls)
        spectral = self._spectral_for_samples(
            spectral_embedding, masks, ctrl_expr.shape[0], n_samples
        )

        training = self.model.training
        self.model.eval()
        try:
            state = self._integrate(state, anchor, masks, spectral)
        finally:
            self.model.train(training)
        # Predictions live in log1p space (>= 0); clamp the unconstrained Euler
        # output so sampled expression is valid (consistent across the internal
        # evaluator, pearson_delta monitoring, and the cell_eval protocol).
        state = state.clamp_min(0.0)
        return state.reshape(n_samples, original_batch, -1)

    def sample_with_grad(
        self,
        ctrl_expr: torch.Tensor,
        pert_mask: torch.Tensor,
        spectral_embedding,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Generate samples while retaining gradients for MMD regularization."""
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        controls = ctrl_expr.repeat(n_samples, 1)
        anchor = controls if self.control_anchor else torch.zeros_like(controls)
        masks = pert_mask.repeat(n_samples, 1)
        state = anchor + self.sigma * torch.randn_like(controls)
        spectral = self._spectral_for_samples(
            spectral_embedding, masks, ctrl_expr.shape[0], n_samples
        )
        result = self._integrate(state, anchor, masks, spectral)
        return result.reshape(n_samples, ctrl_expr.shape[0], -1)

    def _spectral_for_samples(
        self, spectral_source, masks: torch.Tensor, original_batch: int, n_samples: int
    ):
        if callable(spectral_source):
            return spectral_source(masks)

        def repeat_batch(values: torch.Tensor) -> torch.Tensor:
            if values.ndim == 3 and values.shape[0] == original_batch and n_samples > 1:
                return values.repeat(n_samples, 1, 1)
            return values

        if torch.is_tensor(spectral_source):
            return repeat_batch(spectral_source)
        if isinstance(spectral_source, Mapping):
            return {key: repeat_batch(value) for key, value in spectral_source.items()}
        raise ValueError("spectral source must be a tensor, mapping, or callable")

    def _integrate(
        self,
        state: torch.Tensor,
        controls: torch.Tensor,
        masks: torch.Tensor,
        spectral,
    ) -> torch.Tensor:
        step_size = 1.0 / self.n_steps
        for step in range(self.n_steps):
            time = torch.full(
                (state.shape[0], 1),
                step * step_size,
                device=state.device,
                dtype=state.dtype,
            )
            velocity, _ = self.model(state, time, controls, masks, spectral)
            state = state + step_size * velocity
        return state
