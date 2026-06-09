import numpy as np
import pytest

from specflow.data.preprocessing import build_perturbation_map, create_splits


def test_build_perturbation_map_is_gene_aligned_for_combinations():
    mapping = build_perturbation_map(["A", "A+C", "ctrl"], ["A", "B", "C"])

    np.testing.assert_array_equal(mapping["A"], np.array([1, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(
        mapping["A+C"], np.array([1, 0, 1], dtype=np.float32)
    )
    np.testing.assert_array_equal(mapping["ctrl"], np.zeros(3, dtype=np.float32))


def test_build_perturbation_map_rejects_unmodeled_targets():
    with pytest.raises(ValueError, match="absent"):
        build_perturbation_map(["UNKNOWN"], ["A", "B"])


def test_additive_split_keeps_all_single_perturbations_in_train():
    conditions = ["A", "B", "A+B", "A+C", "B+C"]
    split = create_splits(conditions, setting="additive", seed=1, test_fraction=0.34)

    assert "A" in split["train"]
    assert "B" in split["train"]
    assert set(split["test"]).issubset({"A+B", "A+C", "B+C"})


def test_control_tokens_do_not_turn_single_perturbations_into_combinations():
    conditions = ["A+ctrl", "B+ctrl", "A+B"]
    masks = build_perturbation_map(conditions, ["A", "B"])
    split = create_splits(conditions, setting="additive", seed=1, test_fraction=1.0)

    np.testing.assert_array_equal(masks["A+ctrl"], np.array([1, 0], dtype=np.float32))
    assert "A+ctrl" in split["train"]
    assert "B+ctrl" in split["train"]
    assert split["test"] == ["A+B"]


def test_drug_target_mapping_resolves_to_gene_aligned_mask():
    mapping = build_perturbation_map(
        ["drug_a+drug_b"],
        ["G1", "G2", "G3"],
        target_map={"drug_a": ["G1"], "drug_b": ["G3"]},
    )

    np.testing.assert_array_equal(
        mapping["drug_a+drug_b"], np.array([1, 0, 1], dtype=np.float32)
    )
