import csv
import json
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import torch

from specflow.config import SpecFlowConfig
from specflow.data.benchmark import load_benchmark_h5ad
from specflow.data.dataset import make_dataloader
from specflow.experiment import ExperimentRunner


def _write_gaf(path, genes):
    lines = ["!gaf-version: 2.2\n"]
    for index, gene in enumerate(genes):
        lines.append(
            f"DB\t{index}\t{gene}\t\tGO:MODULE\tREF\tEXP\t\tP\t{gene}\t\tprotein\ttaxon:9606\t20200101\tDB\n"
        )
        if index < 3:
            lines.append(
                f"DB\t{index}\t{gene}\t\tGO:FIRST\tREF\tEXP\t\tP\t{gene}\t\tprotein\ttaxon:9606\t20200101\tDB\n"
            )
    path.write_text("".join(lines), encoding="utf-8")


def _write_fixture(tmp_path):
    rng = np.random.default_rng(30)
    genes = [f"G{i}" for i in range(6)]
    conditions = ["ctrl", "G0+ctrl", "G1+ctrl", "G2+ctrl", "G0+G1"]
    rows = []
    labels = []
    base = rng.normal(size=(8, len(genes))).astype(np.float32)
    for condition in conditions:
        if condition == "ctrl":
            values = base
        else:
            targets = [token for token in condition.split("+") if token != "ctrl"]
            shift = np.array([1.0 if gene in targets else 0.0 for gene in genes])
            values = base + shift + rng.normal(scale=0.05, size=base.shape)
        rows.append(values.astype(np.float32))
        labels.extend([condition] * values.shape[0])
    h5ad = tmp_path / "tiny.h5ad"
    ad.AnnData(
        X=np.vstack(rows),
        obs=pd.DataFrame(
            {"condition": labels},
            index=[f"cell_{index}" for index in range(len(labels))],
        ),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(h5ad)
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["G0+ctrl", "G1+ctrl"],
                "val": ["G2+ctrl"],
                "test": ["G0+G1"],
            }
        ),
        encoding="utf-8",
    )
    gaf = tmp_path / "go.gaf"
    _write_gaf(gaf, genes)
    return h5ad, split_path, gaf


def test_h5ad_adapter_and_grouped_batches_use_external_splits(tmp_path):
    h5ad, split_path, _ = _write_fixture(tmp_path)
    prepared = load_benchmark_h5ad(
        str(h5ad),
        condition_key="condition",
        control_labels=["ctrl"],
        split_path=str(split_path),
    )
    loader = make_dataloader(
        prepared.dataset("train", samples_per_condition=4),
        batch_size=4,
        group_by_condition=True,
        shuffle=False,
    )
    batches = list(loader)

    assert prepared.splits["test"] == ["G0+G1"]
    np.testing.assert_array_equal(
        prepared.perturbation_map["G0+ctrl"],
        np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    assert all(len(set(batch["condition"])) == 1 for batch in batches)


def test_experiment_runner_passes_propagation_scale_to_model(tmp_path):
    h5ad, split_path, _ = _write_fixture(tmp_path)
    config = SpecFlowConfig.from_dict(
        {
            "data": {
                "h5ad_path": str(h5ad),
                "condition_key": "condition",
                "control_labels": ["ctrl"],
                "split_path": str(split_path),
            },
            "model": {
                "dual_graph": False,
                "spectral_dim": 2,
                "spectral_propagation": True,
                "propagation_channels": 3,
                "propagation_scale": 0.25,
                "propagation_gate": "perturbation",
                "propagation_gate_init": 0.75,
            },
        }
    )
    runner = ExperimentRunner.from_config(config)

    model = runner.build_model()

    assert model.propagation_scale == 0.25
    assert model.propagation_gate_mode == "perturbation"
    assert model.propagation_gate_init == 0.75
    assert model.propagation_gate is not None


def test_experiment_runner_trains_checkpoints_and_evaluates_tiny_h5ad(tmp_path):
    torch.manual_seed(31)
    h5ad, split_path, gaf = _write_fixture(tmp_path)
    output = tmp_path / "outputs"
    config = SpecFlowConfig.from_dict(
        {
            "data": {
                "h5ad_path": str(h5ad),
                "condition_key": "condition",
                "control_labels": ["ctrl"],
                "split_path": str(split_path),
                "samples_per_condition": 4,
                "seed": 31,
            },
            "graph": {
                "go": {
                    "annotation_file": str(gaf),
                    "namespace": "biological_process",
                    "k_neighbors": 3,
                },
                "coexp": {"k_neighbors": 2, "threshold": 0.1},
                "perturbation": {"alpha_go": 0.3, "alpha_coexp": 0.1},
            },
            "spectral": {
                "go_components": 2,
                "coexp_components": 2,
                "cache_dir": "cache",
            },
            "model": {
                "dual_graph": True,
                "spectral_dim": 4,
                "d_model": 12,
                "hidden_dim": 16,
                "n_velocity_layers": 1,
                "graph_dim": 8,
                "pert_dim": 8,
            },
            "flow": {"sigma": 0.1, "mmd_weight": 0.0},
            "training": {
                "batch_size": 4,
                "max_epochs": 1,
                "learning_rate": 0.001,
                "patience": 1,
            },
            "inference": {
                "n_samples": 1,
                "n_control_cells": 4,
                "ode_steps": 2,
                "de_top_k": 3,
            },
            "output": {
                "output_dir": str(output),
                "checkpoint_name": "tiny.pt",
            },
        }
    )
    runner = ExperimentRunner.from_config(config)
    training = runner.train(n_epochs=1)
    evaluation = runner.evaluate()

    assert (output / "tiny.pt").exists()
    assert (output / "graphs" / "go.npz").exists()
    assert training["epochs_completed"] == 1
    assert "mse" in evaluation["aggregate"]
    assert np.isfinite(evaluation["aggregate"]["de_spearman"])
    assert evaluation["results_csv"] == str(output / "results.csv")
    with (output / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["perturbation"] == "G0+G1"
    assert "pearson_delta" in rows[0]
    assert "discrimination_score_l2" in rows[0]


def test_experiment_runner_scdfm_evaluation_uses_cell_eval(tmp_path, monkeypatch):
    torch.manual_seed(33)
    h5ad, split_path, gaf = _write_fixture(tmp_path)
    output = tmp_path / "scdfm_outputs"
    config = SpecFlowConfig.from_dict(
        {
            "data": {
                "h5ad_path": str(h5ad),
                "condition_key": "condition",
                "control_labels": ["ctrl"],
                "split_path": str(split_path),
                "samples_per_condition": 4,
                "seed": 33,
            },
            "graph": {
                "go": {
                    "annotation_file": str(gaf),
                    "namespace": "biological_process",
                    "k_neighbors": 3,
                },
                "coexp": {"k_neighbors": 2, "threshold": 0.1},
            },
            "spectral": {
                "go_components": 2,
                "coexp_components": 2,
                "cache_dir": "cache",
            },
            "model": {
                "dual_graph": True,
                "spectral_dim": 4,
                "d_model": 12,
                "hidden_dim": 16,
                "n_velocity_layers": 1,
                "graph_dim": 8,
                "pert_dim": 8,
            },
            "flow": {"sigma": 0.1, "mmd_weight": 0.0},
            "training": {
                "batch_size": 4,
                "max_epochs": 1,
                "learning_rate": 0.001,
                "show_progress": False,
            },
            "inference": {
                "n_control_cells": 4,
                "ode_steps": 2,
            },
            "output": {
                "output_dir": str(output),
                "checkpoint_name": "tiny.pt",
            },
        }
    )
    runner = ExperimentRunner.from_config(config)
    training = runner.train(n_epochs=1)
    calls = {}

    class FakeFrame:
        def __init__(self, value):
            self.value = value

        def write_csv(self, path):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.value)

    class FakeMetricsEvaluator:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def compute(self):
            return FakeFrame("metric,value\nmse,0.2\n"), FakeFrame(
                "metric,value\nmse,0.2\n"
            )

    fake_module = types.ModuleType("cell_eval")
    fake_module.MetricsEvaluator = FakeMetricsEvaluator
    monkeypatch.setitem(sys.modules, "cell_eval", fake_module)

    evaluation = runner.evaluate_scdfm(
        checkpoint=training["checkpoint"],
        split_path=str(split_path),
        infer_top_gene=3,
        n_control_cells=4,
        num_threads=2,
    )

    assert evaluation["protocol"] == "scdfm"
    assert evaluation["test_conditions"] == ["G0+G1"]
    assert (output / "pred.h5ad").exists()
    assert (output / "real.h5ad").exists()
    assert evaluation["results_csv"] == str(output / "results.csv")
    assert evaluation["agg_results_csv"] == str(output / "agg_results.csv")
    assert (output / "scdfm_evaluation_summary.json").exists()
    assert calls["control_pert"] == "control"
    assert calls["num_threads"] == 2
    assert calls["adata_pred"].obs["perturbation"].isin(["control", "G0+G1"]).all()


def test_experiment_runner_supports_step_based_training(tmp_path):
    torch.manual_seed(32)
    h5ad, split_path, gaf = _write_fixture(tmp_path)
    output = tmp_path / "step_outputs"
    config = SpecFlowConfig.from_dict(
        {
            "data": {
                "h5ad_path": str(h5ad),
                "condition_key": "condition",
                "control_labels": ["ctrl"],
                "split_path": str(split_path),
                "samples_per_condition": 4,
                "seed": 32,
            },
            "graph": {
                "go": {
                    "annotation_file": str(gaf),
                    "namespace": "biological_process",
                    "k_neighbors": 3,
                },
                "coexp": {"k_neighbors": 2, "threshold": 0.1},
            },
            "spectral": {
                "go_components": 2,
                "coexp_components": 2,
                "cache_dir": "cache",
            },
            "model": {
                "dual_graph": True,
                "spectral_dim": 4,
                "d_model": 12,
                "hidden_dim": 16,
                "n_velocity_layers": 1,
                "graph_dim": 8,
                "pert_dim": 8,
            },
            "flow": {"sigma": 0.1, "mmd_weight": 0.0},
            "training": {
                "batch_size": 4,
                "max_steps": 3,
                "eval_every_steps": 2,
                "learning_rate": 0.001,
                "warmup_steps": 0,
                "show_progress": False,
            },
            "output": {
                "output_dir": str(output),
                "checkpoint_name": "tiny.pt",
            },
        }
    )
    runner = ExperimentRunner.from_config(config)

    training = runner.train()

    assert training["mode"] == "steps"
    assert training["steps_completed"] == 3
    assert [row["step"] for row in training["history"]] == [2, 3]
    assert (output / "tiny.pt").exists()
    assert (output / "checkpoints" / "step_2.pt").exists()
    assert (output / "checkpoints" / "step_3.pt").exists()
