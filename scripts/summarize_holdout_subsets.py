"""Summarize Norman holdout results separately for single and double tests.

The scDFM Norman holdout table reports unseen single-perturbation and
double-perturbation results separately. SpecFlow's cell_eval output is
per-condition in each run's ``results.csv``; this script post-processes those
files without retraining.

Example:

  python scripts/summarize_holdout_subsets.py \
    --root outputs/20260603_holdout_full_0602
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_CONTROL_LABELS = ("ctrl", "control", "non-targeting")

METRIC_COLUMNS = [
    "de_spearman",
    "pearson_delta",
    "mse",
    "mae",
    "ds",
    "discrimination_score_l1",
    "discrimination_score_l2",
    "discrimination_score_cosine",
    "pr_auc",
    "roc_auc",
    "de_spearman_sig",
    "de_direction_match",
    "de_sig_genes_recall",
    "de_nsig_counts_real",
    "de_nsig_counts_pred",
]

RESULT_METRIC_MAP = {
    "de_spearman": "de_spearman_lfc_sig",
    "pearson_delta": "pearson_delta",
    "mse": "mse",
    "mae": "mae",
    # Keep the paper-facing DS column aligned with the DS value used in our
    # running summaries: cell_eval's cosine discrimination score.
    "ds": "discrimination_score_cosine",
    "discrimination_score_l1": "discrimination_score_l1",
    "discrimination_score_l2": "discrimination_score_l2",
    "discrimination_score_cosine": "discrimination_score_cosine",
    "pr_auc": "pr_auc",
    "roc_auc": "roc_auc",
    "de_spearman_sig": "de_spearman_sig",
    "de_direction_match": "de_direction_match",
    "de_sig_genes_recall": "de_sig_genes_recall",
    "de_nsig_counts_real": "de_nsig_counts_real",
    "de_nsig_counts_pred": "de_nsig_counts_pred",
}

OUTPUT_COLUMNS = [
    "setting",
    "subset",
    "statistic",
    "method",
    "source",
    "run_name",
    "fold",
    "n_conditions",
    *METRIC_COLUMNS,
    "notes",
]


def condition_targets(
    condition: str,
    control_labels: Iterable[str] = DEFAULT_CONTROL_LABELS,
    separator: str = "+",
) -> Tuple[str, ...]:
    """Return non-control target tokens from a condition label."""
    controls = {str(label).strip().lower() for label in control_labels}
    targets = []
    for token in str(condition).split(separator):
        token = token.strip()
        if token and token.lower() not in controls:
            targets.append(token)
    return tuple(targets)


def condition_subset(
    condition: str,
    control_labels: Iterable[str] = DEFAULT_CONTROL_LABELS,
    separator: str = "+",
) -> Optional[str]:
    """Classify a perturbation condition as Single, Double, or Other."""
    n_targets = len(condition_targets(condition, control_labels, separator))
    if n_targets == 1:
        return "Single"
    if n_targets == 2:
        return "Double"
    if n_targets > 2:
        return "Other"
    return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _read_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _to_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _format(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def _mean_metric(rows: Sequence[Mapping[str, str]], metric: str) -> Optional[float]:
    source_column = RESULT_METRIC_MAP[metric]
    values = [
        parsed
        for row in rows
        if (parsed := _to_float(row.get(source_column))) is not None
    ]
    return mean(values) if values else None


def _fold_from_run(run_dir: Path) -> str:
    summary = _read_json(run_dir / "scdfm_evaluation_summary.json")
    if summary.get("fold") is not None:
        return str(summary["fold"])

    config = _read_yaml(run_dir / "run_config.yaml")
    data = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    if data.get("split_fold") is not None:
        return str(data["split_fold"])

    match = re.search(r"_f(\d+)(?:_|$)", run_dir.name)
    return match.group(1) if match else ""


def _row_from_conditions(
    *,
    setting: str,
    subset: str,
    statistic: str,
    method: str,
    source: str,
    run_name: str,
    fold: str,
    conditions: Sequence[Mapping[str, str]],
    notes: str = "",
) -> Dict[str, str]:
    row = {
        "setting": setting,
        "subset": subset,
        "statistic": statistic,
        "method": method,
        "source": source,
        "run_name": run_name,
        "fold": fold,
        "n_conditions": str(len(conditions)),
        "notes": notes,
    }
    for metric in METRIC_COLUMNS:
        row[metric] = _format(_mean_metric(conditions, metric))
    return row


def summarize_run(
    run_dir: Path,
    *,
    setting: str,
    method: str,
    source: str,
    control_labels: Iterable[str] = DEFAULT_CONTROL_LABELS,
    separator: str = "+",
    include_other: bool = False,
) -> List[Dict[str, str]]:
    """Return Single/Double fold-level rows for one run directory."""
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return []

    grouped: Dict[str, List[Mapping[str, str]]] = {
        "Single": [],
        "Double": [],
    }
    if include_other:
        grouped["Other"] = []

    for row in _read_csv(results_path):
        subset = condition_subset(
            row.get("perturbation", ""),
            control_labels=control_labels,
            separator=separator,
        )
        if subset in grouped:
            grouped[subset].append(row)

    output = []
    fold = _fold_from_run(run_dir)
    for subset in ["Single", "Double", "Other"]:
        rows = grouped.get(subset, [])
        if not rows:
            continue
        output.append(
            _row_from_conditions(
                setting=setting,
                subset=subset,
                statistic="fold_mean",
                method=method,
                source=source,
                run_name=run_dir.name,
                fold=fold,
                conditions=rows,
            )
        )
    return output


def _aggregate_rows(
    fold_rows: Sequence[Mapping[str, str]],
    *,
    setting: str,
    method: str,
    source: str,
) -> List[Dict[str, str]]:
    output = []
    subsets = sorted({row["subset"] for row in fold_rows})
    for subset in subsets:
        rows = [row for row in fold_rows if row["subset"] == subset]
        for statistic in ["mean", "std"]:
            aggregate = {
                "setting": setting,
                "subset": subset,
                "statistic": statistic,
                "method": method,
                "source": source,
                "run_name": "",
                "fold": "",
                "n_conditions": (
                    str(sum(int(row["n_conditions"]) for row in rows))
                    if statistic == "mean"
                    else ""
                ),
                "notes": (
                    "mean_across_fold_means"
                    if statistic == "mean"
                    else "std_across_fold_means"
                ),
            }
            for metric in METRIC_COLUMNS:
                values = [
                    parsed
                    for row in rows
                    if (parsed := _to_float(row.get(metric))) is not None
                ]
                if statistic == "mean":
                    aggregate[metric] = _format(mean(values) if values else None)
                else:
                    aggregate[metric] = _format(stdev(values) if len(values) > 1 else None)
            output.append(aggregate)
    return output


def summarize_root(
    root: Path,
    *,
    setting: str = "norman_holdout",
    method: str = "SpecFlow",
    source: str = "local_cell_eval",
    control_labels: Iterable[str] = DEFAULT_CONTROL_LABELS,
    separator: str = "+",
    include_other: bool = False,
) -> List[Dict[str, str]]:
    fold_rows: List[Dict[str, str]] = []
    for run_dir in sorted(root.glob("[0-9][0-9]_*")):
        if run_dir.is_dir():
            fold_rows.extend(
                summarize_run(
                    run_dir,
                    setting=setting,
                    method=method,
                    source=source,
                    control_labels=control_labels,
                    separator=separator,
                    include_other=include_other,
                )
            )
    return [
        *_aggregate_rows(
            fold_rows,
            setting=setting,
            method=method,
            source=source,
        ),
        *fold_rows,
    ]


def write_summary(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/20260603_holdout_full_0602")
    parser.add_argument("--output", default=None)
    parser.add_argument("--setting", default="norman_holdout")
    parser.add_argument("--method", default="SpecFlow")
    parser.add_argument("--source", default="local_cell_eval")
    parser.add_argument(
        "--control-labels",
        nargs="+",
        default=list(DEFAULT_CONTROL_LABELS),
        help="Condition tokens treated as controls when classifying perturbations.",
    )
    parser.add_argument("--separator", default="+")
    parser.add_argument("--include-other", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output = (
        Path(args.output)
        if args.output is not None
        else root / "holdout_single_double_summary.csv"
    )
    rows = summarize_root(
        root,
        setting=args.setting,
        method=args.method,
        source=args.source,
        control_labels=args.control_labels,
        separator=args.separator,
        include_other=args.include_other,
    )
    write_summary(output, rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
