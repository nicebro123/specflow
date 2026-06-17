"""Summarize SpecFlow experiment outputs into one CSV.

Each run directory is expected to contain some or all of:

  agg_results.csv
  results.csv
  training_history.json
  training_summary.json
  data_summary.json
  scdfm_evaluation_summary.json
  baseline_summary.json
  extra_metrics_summary.json

Example:

  python scripts/summarize_experiments.py \
    --root outputs \
    --output outputs/experiment_summary.csv
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


MEAN_METRICS = [
    "pearson_delta",
    "mse",
    "mae",
    "de_spearman_sig",
    "de_spearman_lfc_sig",
    "de_direction_match",
    "pr_auc",
    "roc_auc",
    "overlap_at_N",
    "precision_at_N",
    "de_sig_genes_recall",
    "de_nsig_counts_real",
    "de_nsig_counts_pred",
    "discrimination_score_l1",
    "discrimination_score_l2",
    "discrimination_score_cosine",
    "pearson_edistance",
    "clustering_agreement",
]

STD_METRICS = [
    "pearson_delta",
    "mse",
    "mae",
    "de_spearman_lfc_sig",
    "de_direction_match",
]


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_agg_results(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row.get("statistic", "")): row
        for row in rows
        if row.get("statistic")
    }


def _to_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_counts(data_summary: Dict[str, object]) -> Dict[str, object]:
    splits = data_summary.get("splits")
    if not isinstance(splits, dict):
        return {}
    return {
        f"{name}_conditions": len(values)
        for name, values in splits.items()
        if isinstance(values, list)
    }


def _best_history_row(history: List[Dict[str, object]]) -> Dict[str, object]:
    candidates = [
        row
        for row in history
        if _to_float(row.get("val_pearson_delta")) is not None
    ]
    if candidates:
        return max(candidates, key=lambda row: float(row["val_pearson_delta"]))
    candidates = [
        row
        for row in history
        if _to_float(row.get("validation_loss")) is not None
    ]
    if candidates:
        return min(candidates, key=lambda row: float(row["validation_loss"]))
    return history[-1] if history else {}


def _history_fields(run_dir: Path) -> Dict[str, object]:
    history = _read_json(run_dir / "training_history.json")
    if not isinstance(history, list):
        history = []
    best = _best_history_row(history)
    last = history[-1] if history else {}
    return {
        "history_rows": len(history),
        "best_step": best.get("step"),
        "best_epoch": best.get("epoch"),
        "best_val_pearson_delta": best.get("val_pearson_delta"),
        "best_validation_loss": best.get("validation_loss"),
        "best_train_loss": best.get("train_loss"),
        "final_step": last.get("step"),
        "final_epoch": last.get("epoch"),
        "final_val_pearson_delta": last.get("val_pearson_delta"),
        "final_validation_loss": last.get("validation_loss"),
        "final_train_loss": last.get("train_loss"),
        "final_lr": last.get("lr"),
    }


def _run_status(run_dir: Path) -> str:
    if (run_dir / "agg_results.csv").exists():
        return "complete"
    if (run_dir / "training_summary.json").exists():
        return "trained_no_cell_eval"
    if (run_dir / "train.log").exists():
        return "log_only"
    if (run_dir / "run_config.yaml").exists():
        return "planned"
    return "incomplete"


def _config_fields(run_dir: Path) -> Dict[str, object]:
    config = _read_json(run_dir / "run_config.json")
    if not config and (run_dir / "run_config.yaml").exists():
        try:
            import yaml

            with (run_dir / "run_config.yaml").open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        except ImportError:
            config = {}
    if not isinstance(config, dict):
        return {}
    data = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    flow = config.get("flow", {}) if isinstance(config.get("flow"), dict) else {}
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    training = (
        config.get("training", {})
        if isinstance(config.get("training"), dict)
        else {}
    )
    inference = (
        config.get("inference", {})
        if isinstance(config.get("inference"), dict)
        else {}
    )
    return {
        "cfg_fold": data.get("split_fold"),
        "cfg_seed": data.get("seed"),
        "cfg_samples_per_condition": data.get("samples_per_condition"),
        "cfg_sigma": flow.get("sigma"),
        "cfg_ot_coupling": flow.get("ot_coupling"),
        "cfg_mmd_weight": flow.get("mmd_weight"),
        "cfg_delta_corr_weight": flow.get("delta_corr_weight"),
        "cfg_spectral_propagation": model.get("spectral_propagation"),
        "cfg_propagation_channels": model.get("propagation_channels"),
        "cfg_propagation_scale": model.get("propagation_scale"),
        "cfg_propagation_gate": model.get("propagation_gate"),
        "cfg_propagation_gate_init": model.get("propagation_gate_init"),
        "cfg_graph_mode": model.get("graph_mode"),
        "cfg_batch_size": training.get("batch_size"),
        "cfg_max_steps": training.get("max_steps"),
        "cfg_learning_rate": training.get("learning_rate"),
        "cfg_n_control_cells": inference.get("n_control_cells"),
        "cfg_ode_steps": inference.get("ode_steps"),
    }


def _summarize_run(run_dir: Path, root: Path) -> Dict[str, object]:
    data_summary = _read_json(run_dir / "data_summary.json")
    training_summary = _read_json(run_dir / "training_summary.json")
    eval_summary = _read_json(run_dir / "scdfm_evaluation_summary.json")
    baseline_summary = _read_json(run_dir / "baseline_summary.json")
    extra_metrics = _read_json(run_dir / "extra_metrics_summary.json")
    agg = _read_agg_results(run_dir / "agg_results.csv")
    mean = agg.get("mean", {})
    std = agg.get("std", {})
    stat = run_dir.stat()

    row: Dict[str, object] = {
        "run_name": run_dir.name,
        "method": baseline_summary.get("baseline") or "SpecFlow",
        "source": baseline_summary.get("source") or "local_cell_eval",
        "run_dir": str(run_dir),
        "relative_run_dir": str(run_dir.relative_to(root)),
        "status": _run_status(run_dir),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "checkpoint": training_summary.get("checkpoint") or eval_summary.get("checkpoint"),
        "mode": training_summary.get("mode"),
        "steps_completed": training_summary.get("steps_completed"),
        "eval_every_steps": training_summary.get("eval_every_steps"),
        "selection_score": training_summary.get("best_loss"),
        "fold": eval_summary.get("fold"),
        "split_method": eval_summary.get("split_method"),
        "n_test_conditions": eval_summary.get("n_test_conditions"),
        "n_eval_genes": eval_summary.get("n_eval_genes"),
        "l2_mean": extra_metrics.get("l2_mean"),
        "l2_mean_std": extra_metrics.get("l2_mean_std"),
        "pearson_delta_hat": extra_metrics.get("pearson_delta_hat"),
        "pearson_delta_hat_std": extra_metrics.get("pearson_delta_hat_std"),
        "pearson_delta_hat20": extra_metrics.get("pearson_delta_hat20"),
        "pearson_delta_hat20_std": extra_metrics.get("pearson_delta_hat20_std"),
        "extra_metric_reference": extra_metrics.get("reference"),
        "n_genes": data_summary.get("n_genes"),
        "n_controls": data_summary.get("n_controls"),
        "n_conditions": data_summary.get("n_conditions"),
    }
    graph_edges = data_summary.get("graph_edges")
    if isinstance(graph_edges, dict):
        for name, value in graph_edges.items():
            row[f"{name}_edges"] = value
    row.update(_config_fields(run_dir))
    row.update(_split_counts(data_summary))
    row.update(_history_fields(run_dir))
    for metric in MEAN_METRICS:
        row[metric] = mean.get(metric)
    for metric in STD_METRICS:
        row[f"{metric}_std"] = std.get(metric)
    return row


def _candidate_run_dirs(root: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/agg_results.csv" if recursive else "*/agg_results.csv"
    seen = set()
    for path in root.glob(pattern):
        run_dir = path.parent
        if run_dir not in seen:
            seen.add(run_dir)
            yield run_dir

    pattern = "**/training_summary.json" if recursive else "*/training_summary.json"
    for path in root.glob(pattern):
        run_dir = path.parent
        if run_dir not in seen:
            seen.add(run_dir)
            yield run_dir

    pattern = "**/run_config.yaml" if recursive else "*/run_config.yaml"
    for path in root.glob(pattern):
        run_dir = path.parent
        if run_dir not in seen:
            seen.add(run_dir)
            yield run_dir


def _fieldnames(rows: List[Dict[str, object]]) -> List[str]:
    preferred = [
        "run_name",
        "method",
        "source",
        "status",
        "relative_run_dir",
        "modified_time",
        "l2_mean",
        "pearson_delta",
        "mse",
        "mae",
        "de_spearman_lfc_sig",
        "de_direction_match",
        "pr_auc",
        "roc_auc",
        "discrimination_score_l2",
        "pearson_delta_hat",
        "pearson_delta_hat20",
        "extra_metric_reference",
        "best_val_pearson_delta",
        "final_val_pearson_delta",
        "steps_completed",
        "fold",
        "cfg_fold",
        "cfg_seed",
        "cfg_sigma",
        "cfg_delta_corr_weight",
        "cfg_mmd_weight",
        "cfg_ot_coupling",
        "cfg_spectral_propagation",
        "cfg_propagation_channels",
        "cfg_propagation_scale",
        "cfg_propagation_gate",
        "cfg_propagation_gate_init",
        "cfg_graph_mode",
        "n_test_conditions",
        "n_eval_genes",
    ]
    all_fields = []
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    return preferred + [key for key in all_fields if key not in preferred]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="outputs",
        help="Directory containing run output folders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Summary CSV path. Defaults to <root>/experiment_summary.csv.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan one directory level under --root.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))
    output = Path(args.output).resolve() if args.output else root / "experiment_summary.csv"

    rows = [
        _summarize_run(run_dir, root)
        for run_dir in sorted(_candidate_run_dirs(root, recursive=not args.no_recursive))
    ]
    rows.sort(
        key=lambda row: (
            row.get("status") != "complete",
            -(_to_float(row.get("pearson_delta")) or float("-inf")),
            str(row.get("run_name")),
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows) if rows else ["run_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
