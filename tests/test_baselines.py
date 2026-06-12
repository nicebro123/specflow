import importlib.util
from pathlib import Path

import numpy as np
import pytest

BASELINES_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "specflow"
    / "evaluation"
    / "baselines.py"
)
spec = importlib.util.spec_from_file_location("baselines_under_test", BASELINES_PATH)
baselines = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baselines)

build_single_delta_lookup = baselines.build_single_delta_lookup
condition_targets = baselines.condition_targets
predict_additive_baseline = baselines.predict_additive_baseline
predict_control_baseline = baselines.predict_control_baseline


def test_condition_targets_removes_control_tokens():
    assert condition_targets("A+ctrl", control_labels=["ctrl", "control"]) == ("A",)
    assert condition_targets("control+B", control_labels=["ctrl", "control"]) == ("B",)
    assert condition_targets("A+B", control_labels=["ctrl", "control"]) == ("A", "B")


def test_additive_baseline_uses_train_single_deltas_without_leakage():
    controls = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    perturbed = {
        "A+ctrl": np.array([[2.0, 4.0], [4.0, 6.0]], dtype=np.float32),
        "B+ctrl": np.array([[3.0, 1.0], [5.0, 3.0]], dtype=np.float32),
        "A+B": np.array([[9.0, 9.0]], dtype=np.float32),
    }

    deltas = build_single_delta_lookup(
        perturbed,
        controls,
        train_conditions=["A+ctrl", "B+ctrl"],
        control_labels=["ctrl"],
    )
    pred = predict_additive_baseline(
        "A+B",
        controls,
        deltas,
        control_labels=["ctrl"],
        clamp_min=None,
    )

    np.testing.assert_allclose(deltas["A"], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(deltas["B"], np.array([2.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(pred, controls + np.array([3.0, 1.0], dtype=np.float32))


def test_additive_baseline_missing_single_errors_by_default():
    with pytest.raises(KeyError):
        predict_additive_baseline(
            "A+B",
            np.ones((2, 3), dtype=np.float32),
            {"A": np.ones(3, dtype=np.float32)},
        )


def test_additive_baseline_missing_single_zero_uses_available_deltas():
    controls = np.ones((2, 3), dtype=np.float32)
    pred = predict_additive_baseline(
        "A+B",
        controls,
        {"A": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        missing_single="zero",
        clamp_min=None,
    )

    np.testing.assert_allclose(
        pred,
        np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], dtype=np.float32),
    )


def test_control_baseline_returns_copy():
    controls = np.ones((2, 3), dtype=np.float32)
    pred = predict_control_baseline(controls)
    pred[0, 0] = 9.0
    assert controls[0, 0] == 1.0
