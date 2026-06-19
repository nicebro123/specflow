#!/usr/bin/env python3
"""Collect every code_0602_opo experiment into run-level and condition-level CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


ARTIFACT_NAMES = {
    "run_config.yaml",
    "run_config.json",
    "train.log",
    "training_summary.json",
    "agg_results.csv",
    "results.csv",
    "baseline_summary.json",
}

ERROR_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|RuntimeError|FileNotFoundError|"
    r"ValueError|Killed|out of memory",
    re.IGNORECASE,
)
STEP_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_config(run_dir: Path) -> dict[str, Any]:
    json_path = run_dir / "run_config.json"
    if json_path.exists():
        value = read_json(json_path)
        return value if isinstance(value, dict) else {}
    yaml_path = run_dir / "run_config.yaml"
    if yaml_path.exists():
        try:
            value = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            return value if isinstance(value, dict) else {}
        except (OSError, yaml.YAMLError):
            return {}
    return {}


def flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            output.update(flatten_scalars(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        output[prefix] = json.dumps(value, ensure_ascii=False)
    elif value is None or isinstance(value, (str, int, float, bool)):
        output[prefix] = value
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def candidate_run_dirs(root: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for name in ARTIFACT_NAMES:
        for path in root.glob(f"**/{name}"):
            run_dirs.add(path.parent)
    return sorted(run_dirs)


def active_commands() -> str:
    try:
        return subprocess.check_output(
            ["ps", "-eo", "args"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""


def log_state(log_path: Path) -> tuple[int, int | None, int | None, float | None]:
    if not log_path.exists():
        return 0, None, None, None
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 2 * 1024 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return 0, None, None, None
    error_count = len(ERROR_PATTERN.findall(text))
    lines = text.replace("\r", "\n").splitlines()
    step_lines = [line for line in lines if "steps:" in line]
    matches = STEP_PATTERN.findall(step_lines[-1]) if step_lines else []
    if not matches:
        return error_count, None, None, None
    current, total = map(int, matches[-1])
    percent = 100.0 * current / total if total else None
    return error_count, current, total, percent


def run_status(
    run_dir: Path,
    commands: str,
    error_count: int,
) -> str:
    if (run_dir / "agg_results.csv").exists():
        return "complete"
    if str(run_dir) in commands:
        return "running"
    if error_count:
        return "failed"
    if (run_dir / "training_summary.json").exists():
        return "trained_no_cell_eval"
    if (run_dir / "train.log").exists():
        return "stopped_or_log_only"
    if (run_dir / "run_config.yaml").exists() or (run_dir / "run_config.json").exists():
        return "planned"
    return "incomplete"


def study_and_category(run_dir: Path, root: Path) -> tuple[str, str]:
    relative = run_dir.relative_to(root)
    study = relative.parts[0] if len(relative.parts) > 1 else ""
    lower = study.lower()
    if "baseline" in lower:
        category = "baseline"
    elif "combosciplex" in lower:
        category = "combosciplex"
    elif "holdout" in lower:
        category = "holdout"
    elif "hparam" in lower:
        category = "hyperparameter"
    elif (
        "prop_scale" in lower
        or "propagation" in lower
        or "adaptive_prop" in lower
        or "adaptive_gate" in lower
    ):
        category = "propagation"
    elif "core_component" in lower or "ablation" in lower:
        category = "ablation"
    elif "additive" in lower:
        category = "additive"
    elif "continuous_queue" in lower:
        category = "queue"
    else:
        category = "other"
    return study, category


def training_fields(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "training_summary.json")
    if not isinstance(summary, dict):
        summary = {}
    history = read_json(run_dir / "training_history.json")
    if not isinstance(history, list):
        history = summary.get("history", [])
    if not isinstance(history, list):
        history = []
    valid_history = [row for row in history if isinstance(row, dict)]
    best_rows = [
        row
        for row in valid_history
        if isinstance(row.get("val_pearson_delta"), (int, float))
    ]
    best = max(best_rows, key=lambda row: row["val_pearson_delta"]) if best_rows else {}
    last = valid_history[-1] if valid_history else {}
    return {
        "checkpoint": summary.get("checkpoint"),
        "training_mode": summary.get("mode"),
        "steps_completed": summary.get("steps_completed"),
        "selection_score": summary.get("best_loss"),
        "history_rows": len(valid_history),
        "best_step": best.get("step"),
        "best_val_pearson_delta": best.get("val_pearson_delta"),
        "best_validation_loss": best.get("validation_loss"),
        "final_step": last.get("step"),
        "final_train_loss": last.get("train_loss"),
        "final_validation_loss": last.get("validation_loss"),
        "final_val_pearson_delta": last.get("val_pearson_delta"),
        "final_lr": last.get("lr"),
    }


def aggregate_fields(run_dir: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for row in read_csv(run_dir / "agg_results.csv"):
        statistic = row.get("statistic")
        if not statistic:
            continue
        for key, value in row.items():
            if key == "statistic":
                continue
            if statistic == "mean":
                output[key] = value
            elif statistic in {"std", "count", "min", "max", "50%"}:
                suffix = "median" if statistic == "50%" else statistic
                output[f"{key}_{suffix}"] = value
    return output


def evaluation_fields(run_dir: Path) -> dict[str, Any]:
    evaluation = read_json(run_dir / "scdfm_evaluation_summary.json")
    baseline = read_json(run_dir / "baseline_summary.json")
    extra = read_json(run_dir / "extra_metrics_summary.json")
    data = read_json(run_dir / "data_summary.json")
    if not isinstance(evaluation, dict):
        evaluation = {}
    if not isinstance(baseline, dict):
        baseline = {}
    if not isinstance(extra, dict):
        extra = {}
    if not isinstance(data, dict):
        data = {}
    output = {
        "method": baseline.get("baseline") or "SpecFlow",
        "result_source": baseline.get("source") or "local_cell_eval",
        "eval_protocol": evaluation.get("protocol"),
        "eval_fold": evaluation.get("fold"),
        "eval_split_method": evaluation.get("split_method"),
        "n_test_conditions": evaluation.get("n_test_conditions"),
        "n_eval_genes": evaluation.get("n_eval_genes"),
        "n_genes": data.get("n_genes"),
        "n_controls": data.get("n_controls"),
        "n_conditions": data.get("n_conditions"),
    }
    for key, value in extra.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[f"extra_{key}"] = value
    splits = data.get("splits")
    if isinstance(splits, dict):
        for key, value in splits.items():
            if isinstance(value, list):
                output[f"{key}_conditions"] = len(value)
    graph_edges = data.get("graph_edges")
    if isinstance(graph_edges, dict):
        for key, value in graph_edges.items():
            output[f"{key}_edges"] = value
    return output


def ordered_fields(rows: list[dict[str, Any]], preferred: Iterable[str]) -> list[str]:
    all_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    return list(preferred) + [key for key in all_fields if key not in preferred]


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: Iterable[str]) -> None:
    fields = ordered_fields(rows, preferred)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--runs-output", required=True)
    parser.add_argument("--conditions-output", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    commands = active_commands()
    run_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    for run_dir in candidate_run_dirs(root):
        study, category = study_and_category(run_dir, root)
        error_count, progress_step, progress_total, progress_pct = log_state(
            run_dir / "train.log"
        )
        stat = run_dir.stat()
        row: dict[str, Any] = {
            "study": study,
            "category": category,
            "run_name": run_dir.name,
            "relative_run_dir": str(run_dir.relative_to(root)),
            "absolute_run_dir": str(run_dir),
            "status": run_status(run_dir, commands, error_count),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(
                timespec="seconds"
            ),
            "log_error_matches": error_count,
            "progress_step": progress_step,
            "progress_total": progress_total,
            "progress_pct": progress_pct,
            "has_best_checkpoint": (run_dir / "best.pt").exists(),
            "has_results": (run_dir / "results.csv").exists(),
            "has_agg_results": (run_dir / "agg_results.csv").exists(),
        }
        row.update(
            {
                f"cfg_{key}": value
                for key, value in flatten_scalars(read_config(run_dir)).items()
            }
        )
        row.update(training_fields(run_dir))
        row.update(evaluation_fields(run_dir))
        row.update(aggregate_fields(run_dir))
        run_rows.append(row)

        for result in read_csv(run_dir / "results.csv"):
            condition_rows.append(
                {
                    "study": study,
                    "category": category,
                    "run_name": run_dir.name,
                    "relative_run_dir": str(run_dir.relative_to(root)),
                    "method": row.get("method"),
                    "status": row.get("status"),
                    "cfg_data_dataset": row.get("cfg_data_dataset"),
                    "cfg_data_setting": row.get("cfg_data_setting"),
                    "cfg_data_split_fold": row.get("cfg_data_split_fold"),
                    **result,
                }
            )

    status_order = {
        "complete": 0,
        "running": 1,
        "trained_no_cell_eval": 2,
        "failed": 3,
        "stopped_or_log_only": 4,
        "planned": 5,
        "incomplete": 6,
    }
    run_rows.sort(
        key=lambda row: (
            row["study"],
            status_order.get(str(row["status"]), 99),
            row["run_name"],
        )
    )
    condition_rows.sort(
        key=lambda row: (
            str(row.get("study", "")),
            str(row.get("run_name", "")),
            str(row.get("condition", row.get("perturbation", ""))),
        )
    )

    run_preferred = [
        "study",
        "category",
        "run_name",
        "status",
        "method",
        "relative_run_dir",
        "modified_time",
        "cfg_data_dataset",
        "cfg_data_setting",
        "cfg_data_split_path",
        "cfg_data_split_fold",
        "cfg_data_seed",
        "eval_split_method",
        "eval_fold",
        "progress_step",
        "progress_total",
        "progress_pct",
        "steps_completed",
        "pearson_delta",
        "mse_delta",
        "mae_delta",
        "mse",
        "mae",
        "de_spearman_sig",
        "de_spearman_lfc_sig",
        "de_direction_match",
        "de_sig_genes_recall",
        "overlap_at_N",
        "precision_at_N",
        "pr_auc",
        "roc_auc",
        "discrimination_score_l1",
        "discrimination_score_l2",
        "discrimination_score_cosine",
        "pearson_edistance",
        "clustering_agreement",
        "cfg_model_spectral_propagation",
        "cfg_model_propagation_channels",
        "cfg_model_propagation_scale",
        "cfg_model_propagation_gate",
        "cfg_model_propagation_gate_init",
        "cfg_model_graph_mode",
        "cfg_flow_sigma",
        "cfg_flow_ot_coupling",
        "cfg_flow_mmd_weight",
        "cfg_flow_delta_corr_weight",
        "cfg_training_batch_size",
        "cfg_training_max_steps",
        "cfg_training_learning_rate",
        "cfg_inference_n_control_cells",
        "n_test_conditions",
        "n_eval_genes",
        "has_best_checkpoint",
        "has_results",
        "has_agg_results",
        "log_error_matches",
    ]
    condition_preferred = [
        "study",
        "category",
        "run_name",
        "method",
        "status",
        "cfg_data_dataset",
        "cfg_data_setting",
        "cfg_data_split_fold",
        "condition",
        "perturbation",
    ]
    write_csv(Path(args.runs_output), run_rows, run_preferred)
    write_csv(Path(args.conditions_output), condition_rows, condition_preferred)
    print(
        json.dumps(
            {
                "runs": len(run_rows),
                "conditions": len(condition_rows),
                "runs_output": str(Path(args.runs_output).resolve()),
                "conditions_output": str(Path(args.conditions_output).resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
