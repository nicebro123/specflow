import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_holdout_subsets import (  # noqa: E402
    condition_subset,
    condition_targets,
    summarize_root,
    write_summary,
)


def _write_results(run_dir: Path, fold: int, rows):
    run_dir.mkdir(parents=True)
    (run_dir / "scdfm_evaluation_summary.json").write_text(
        json.dumps({"fold": fold}),
        encoding="utf-8",
    )
    fieldnames = [
        "perturbation",
        "de_spearman_lfc_sig",
        "pearson_delta",
        "mse",
        "mae",
        "discrimination_score_cosine",
    ]
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_condition_targets_remove_control_tokens():
    assert condition_targets("A+ctrl", ["ctrl", "control"]) == ("A",)
    assert condition_targets("control+B", ["ctrl", "control"]) == ("B",)
    assert condition_targets("A+B", ["ctrl", "control"]) == ("A", "B")
    assert condition_subset("ctrl", ["ctrl", "control"]) is None


def test_summarize_root_splits_single_and_double(tmp_path):
    _write_results(
        tmp_path / "00_holdout_full_f0_s42_g0",
        0,
        [
            {
                "perturbation": "A+ctrl",
                "de_spearman_lfc_sig": "0.2",
                "pearson_delta": "0.4",
                "mse": "0.10",
                "mae": "0.20",
                "discrimination_score_cosine": "0.7",
            },
            {
                "perturbation": "B+ctrl",
                "de_spearman_lfc_sig": "0.4",
                "pearson_delta": "0.6",
                "mse": "0.20",
                "mae": "0.30",
                "discrimination_score_cosine": "0.8",
            },
            {
                "perturbation": "A+B",
                "de_spearman_lfc_sig": "0.8",
                "pearson_delta": "0.9",
                "mse": "0.30",
                "mae": "0.40",
                "discrimination_score_cosine": "0.9",
            },
        ],
    )
    _write_results(
        tmp_path / "01_holdout_full_f1_s42_g1",
        1,
        [
            {
                "perturbation": "C+ctrl",
                "de_spearman_lfc_sig": "0.6",
                "pearson_delta": "0.8",
                "mse": "0.40",
                "mae": "0.50",
                "discrimination_score_cosine": "0.6",
            },
            {
                "perturbation": "C+D",
                "de_spearman_lfc_sig": "1.0",
                "pearson_delta": "0.7",
                "mse": "0.50",
                "mae": "0.60",
                "discrimination_score_cosine": "0.5",
            },
        ],
    )

    rows = summarize_root(tmp_path)
    by_key = {(row["subset"], row["statistic"], row["fold"]): row for row in rows}

    assert by_key[("Single", "fold_mean", "0")]["n_conditions"] == "2"
    assert by_key[("Single", "fold_mean", "0")]["de_spearman"] == "0.300000"
    assert by_key[("Double", "fold_mean", "1")]["pearson_delta"] == "0.700000"

    aggregate = {(row["subset"], row["statistic"]): row for row in rows if not row["fold"]}
    assert aggregate[("Single", "mean")]["de_spearman"] == "0.450000"
    assert aggregate[("Double", "mean")]["mse"] == "0.400000"
    assert aggregate[("Single", "std")]["pearson_delta"] == "0.212132"

    output = tmp_path / "summary.csv"
    write_summary(output, rows)
    written = list(csv.DictReader(output.open()))
    assert len(written) == len(rows)
