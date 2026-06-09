import pickle
import sys
import types

import anndata as ad
import numpy as np

from specflow.evaluation.scdfm_protocol import (
    load_scdfm_split,
    run_cell_eval,
    scdfm_norman_split,
    select_scdfm_eval_genes,
    write_scdfm_anndata,
)


def test_scdfm_norman_split_is_fold_deterministic_and_holds_out_combos():
    conditions = [
        "ctrl",
        "A+ctrl",
        "B+ctrl",
        "A+B",
        "C+D",
        "E+F",
        "G+H",
    ]

    first = scdfm_norman_split(conditions, fold=1)
    second = scdfm_norman_split(conditions, fold=1)

    assert first == second
    assert len(first["test"]) == 1
    assert "ctrl" not in first["test"][0]
    assert set(first["train"]).isdisjoint(first["test"])


def test_load_scdfm_split_reads_folded_pickle(tmp_path):
    split_path = tmp_path / "split_results.pkl"
    with split_path.open("wb") as handle:
        pickle.dump(
            [
                {"train": ["A+ctrl"], "test": ["A+B"]},
                {"train": ["B+ctrl"], "test": ["C+D"]},
            ],
            handle,
        )

    split = load_scdfm_split(str(split_path), fold=1)

    assert split["train"] == ["B+ctrl"]
    assert split["test"] == ["C+D"]
    assert split["val"] == []


def test_select_scdfm_eval_genes_uses_test_cells_after_model_gene_alignment(tmp_path):
    rng = np.random.default_rng(3)
    genes = [f"G{i}" for i in range(6)]
    adata = ad.AnnData(
        X=rng.gamma(shape=2.0, scale=1.0, size=(12, 6)).astype(np.float32),
        obs={
            "condition": [
                "ctrl",
                "ctrl",
                "G0+G1",
                "G0+G1",
                "G2+G3",
                "G2+G3",
                "G4+ctrl",
                "G4+ctrl",
                "G5+ctrl",
                "G5+ctrl",
                "G1+G2",
                "G1+G2",
            ]
        },
        var={"symbol": genes},
    )
    adata.var_names = [f"ENSG{i}" for i in range(6)]
    h5ad_path = tmp_path / "tiny.h5ad"
    adata.write_h5ad(h5ad_path)

    eval_genes = select_scdfm_eval_genes(
        str(h5ad_path),
        model_gene_names=genes,
        test_conditions=["G0+G1", "G2+G3"],
        gene_key="symbol",
        infer_top_gene=3,
    )

    assert 1 <= len(eval_genes) <= 3
    assert set(eval_genes).issubset(set(genes))


def test_write_scdfm_anndata_outputs_cell_eval_inputs(tmp_path):
    eval_genes = ["G0", "G1", "G2"]
    paths = write_scdfm_anndata(
        str(tmp_path),
        predicted_by_condition={"A+B": np.ones((2, 3), dtype=np.float32)},
        observed_by_condition={"A+B": np.zeros((4, 3), dtype=np.float32)},
        control_expression=np.full((3, 3), 0.5, dtype=np.float32),
        eval_gene_names=eval_genes,
    )

    pred = ad.read_h5ad(paths["pred_h5ad"])
    real = ad.read_h5ad(paths["real_h5ad"])

    assert pred.shape == (5, 3)
    assert real.shape == (7, 3)
    assert list(pred.var_names) == eval_genes
    assert pred.obs["perturbation"].tolist() == ["control"] * 3 + ["A+B"] * 2


def test_run_cell_eval_invokes_metrics_evaluator_and_writes_outputs(
    tmp_path, monkeypatch
):
    paths = write_scdfm_anndata(
        str(tmp_path),
        predicted_by_condition={"A+B": np.ones((2, 3), dtype=np.float32)},
        observed_by_condition={"A+B": np.zeros((2, 3), dtype=np.float32)},
        control_expression=np.full((2, 3), 0.5, dtype=np.float32),
        eval_gene_names=["G0", "G1", "G2"],
    )
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
            return FakeFrame("metric,value\nmse,0.1\n"), FakeFrame(
                "metric,value\nmse,0.1\n"
            )

    fake_module = types.ModuleType("cell_eval")
    fake_module.MetricsEvaluator = FakeMetricsEvaluator
    monkeypatch.setitem(sys.modules, "cell_eval", fake_module)

    result = run_cell_eval(
        paths["pred_h5ad"],
        paths["real_h5ad"],
        str(tmp_path),
        control_pert="control",
        num_threads=2,
    )

    assert calls["control_pert"] == "control"
    assert calls["pert_col"] == "perturbation"
    assert calls["num_threads"] == 2
    assert calls["adata_pred"].shape == (4, 3)
    assert calls["adata_real"].shape == (4, 3)
    assert result["results_csv"] == str(tmp_path / "results.csv")
    assert result["agg_results_csv"] == str(tmp_path / "agg_results.csv")
    assert (tmp_path / "results.csv").read_text(encoding="utf-8").startswith("metric")
