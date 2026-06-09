"""Merge local SpecFlow summaries with paper-reported baseline metrics."""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


OUTPUT_COLUMNS = [
    "setting",
    "subset",
    "method",
    "source",
    "l2",
    "mse",
    "mae",
    "de_spearman",
    "pearson_delta",
    "ds",
    "pearson_delta_hat",
    "pearson_delta_hat20",
    "notes",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _local_rows(
    rows: Iterable[Dict[str, str]],
    setting: str,
    subset: str,
    method: str,
    include_incomplete: bool,
) -> List[Dict[str, str]]:
    output = []
    for row in rows:
        if not include_incomplete and row.get("status") != "complete":
            continue
        local_method = row.get("method") or method
        source = row.get("source") or "local_cell_eval"
        output.append(
            {
                "setting": setting,
                "subset": subset,
                "method": local_method,
                "source": source,
                "l2": row.get("l2_mean", ""),
                "mse": row.get("mse", ""),
                "mae": row.get("mae", ""),
                "de_spearman": row.get("de_spearman_lfc_sig", ""),
                "pearson_delta": row.get("pearson_delta", ""),
                "ds": row.get("discrimination_score_l2", ""),
                "pearson_delta_hat": row.get("pearson_delta_hat", ""),
                "pearson_delta_hat20": row.get("pearson_delta_hat20", ""),
                "notes": row.get("run_name", ""),
            }
        )
    return output


def _filter_paper_rows(
    rows: Iterable[Dict[str, str]],
    setting: str,
    subset: str,
) -> List[Dict[str, str]]:
    output = []
    for row in rows:
        if row.get("setting") != setting:
            continue
        if subset and row.get("subset", "") not in {"", subset}:
            continue
        output.append({column: row.get(column, "") for column in OUTPUT_COLUMNS})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-summary", required=True)
    parser.add_argument("--paper-baselines", default="paper_baselines/scdfm_reported_metrics.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--setting", default="norman_additive")
    parser.add_argument("--subset", default="")
    parser.add_argument("--method", default="SpecFlow")
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()

    local = _read_csv(Path(args.local_summary))
    paper = _read_csv(Path(args.paper_baselines))
    rows = _filter_paper_rows(paper, args.setting, args.subset)
    rows.extend(
        _local_rows(
            local,
            setting=args.setting,
            subset=args.subset,
            method=args.method,
            include_incomplete=args.include_incomplete,
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
