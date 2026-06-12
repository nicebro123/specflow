"""Control-state anchored conditional flow matching."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class FlowLossOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    target_delta: torch.Tensor
    interpolated_state: torch.Tensor


def _ot_align_targets(ctrl_expr, target_expr, conditions):
    """Reorder perturbed cells to their optimal-transport control partners.

    Within each condition group, solve the balanced assignment (Hungarian) that
    minimizes total squared distance between control and perturbed cells, and
    permute the targets accordingly. Replaces random control->perturbed pairing
    with a coupling that respects expression similarity.
    """
    from scipy.optimize import linear_sum_assignment

    batch_size = ctrl_expr.shape[0]
    if conditions is None:
        groups = {None: list(range(batch_size))}
    else:
        groups = {}
        for idx, condition in enumerate(conditions):
            groups.setdefault(condition, []).append(idx)

    order = list(range(batch_size))
    for indices in groups.values():
        if len(indices) < 2:
            continue
        sub_ctrl = ctrl_expr[indices]
        sub_target = target_expr[indices]
        cost = torch.cdist(sub_ctrl.float(), sub_target.float()).pow(2)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        for row, col in zip(rows, cols):
            order[indices[row]] = indices[col]
    return target_expr[order]


class ControlAnchoredFlowMatching:
    """Train a velocity model on paths starting near measured controls."""

    def __init__(
        self,
        sigma: float = 0.5,
        ot_coupling: bool = False,
        control_anchor: bool = True,
    ) -> None:
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.sigma = sigma
        self.ot_coupling = ot_coupling
        self.control_anchor = control_anchor

    def compute_loss(self, model, batch, spectral_embedding: torch.Tensor) -> FlowLossOutput:
        ctrl_expr = batch["ctrl_expr"]
        target_expr = batch["pert_expr"]
        pert_mask = batch["pert_mask"]
        if ctrl_expr.shape != target_expr.shape:
            raise ValueError("control and perturbed expression must have identical shape")

        if self.ot_coupling:
            target_expr = _ot_align_targets(
                ctrl_expr, target_expr, batch.get("condition")
            )

        anchor_expr = ctrl_expr if self.control_anchor else torch.zeros_like(ctrl_expr)
        time = torch.rand(
            ctrl_expr.shape[0], 1, device=ctrl_expr.device, dtype=ctrl_expr.dtype
        )
        target_delta = target_expr - anchor_expr
        x_0 = anchor_expr + self.sigma * torch.randn_like(ctrl_expr)
        x_t = (1.0 - time) * x_0 + time * target_expr
        target_velocity = target_expr - x_0
        predicted_velocity, _ = model(
            x_t, time, anchor_expr, pert_mask, spectral_embedding
        )
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return FlowLossOutput(loss, predicted_velocity, target_velocity, target_delta, x_t)
