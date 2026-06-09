"""Simple non-parametric baselines for perturbation prediction."""

from typing import Iterable, Mapping, Sequence

import numpy as np


def condition_targets(
    condition: str,
    separator: str = "+",
    control_labels: Iterable[str] = ("ctrl", "control", "non-targeting"),
) -> tuple[str, ...]:
    """Return non-control perturbation tokens from a condition label."""

    controls = {str(label).strip().lower() for label in control_labels}
    targets = []
    for token in str(condition).split(separator):
        token = token.strip()
        if token and token.lower() not in controls:
            targets.append(token)
    return tuple(targets)


def build_single_delta_lookup(
    perturbed_by_condition: Mapping[str, np.ndarray],
    control_expression: np.ndarray,
    train_conditions: Sequence[str],
    separator: str = "+",
    control_labels: Iterable[str] = ("ctrl", "control", "non-targeting"),
) -> dict[str, np.ndarray]:
    """Estimate one single-perturbation mean delta per target from train data."""

    control_mean = np.asarray(control_expression, dtype=np.float32).mean(axis=0)
    deltas: dict[str, np.ndarray] = {}
    for condition in train_conditions:
        targets = condition_targets(
            condition,
            separator=separator,
            control_labels=control_labels,
        )
        if len(targets) != 1 or condition not in perturbed_by_condition:
            continue
        expression = np.asarray(perturbed_by_condition[condition], dtype=np.float32)
        if expression.size == 0:
            continue
        deltas[targets[0]] = expression.mean(axis=0) - control_mean
    return deltas


def predict_control_baseline(control_samples: np.ndarray) -> np.ndarray:
    """Predict each perturbation as sampled control cells."""

    return np.asarray(control_samples, dtype=np.float32).copy()


def predict_additive_baseline(
    condition: str,
    control_samples: np.ndarray,
    single_deltas: Mapping[str, np.ndarray],
    separator: str = "+",
    control_labels: Iterable[str] = ("ctrl", "control", "non-targeting"),
    missing_single: str = "error",
    clamp_min: float | None = 0.0,
) -> np.ndarray:
    """Predict a condition as control plus the sum of train single deltas."""

    if missing_single not in {"error", "zero"}:
        raise ValueError("missing_single must be 'error' or 'zero'")

    controls = np.asarray(control_samples, dtype=np.float32)
    delta = np.zeros(controls.shape[1], dtype=np.float32)
    missing = []
    for target in condition_targets(
        condition,
        separator=separator,
        control_labels=control_labels,
    ):
        single_delta = single_deltas.get(target)
        if single_delta is None:
            missing.append(target)
            continue
        delta = delta + np.asarray(single_delta, dtype=np.float32)

    if missing and missing_single == "error":
        joined = ", ".join(missing)
        raise KeyError(
            f"no train single-perturbation delta available for {condition!r}: {joined}"
        )

    prediction = controls + delta[None, :]
    if clamp_min is not None:
        prediction = np.maximum(prediction, float(clamp_min))
    return prediction.astype(np.float32, copy=False)
