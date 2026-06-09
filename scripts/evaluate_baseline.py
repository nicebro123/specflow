"""Evaluate control or additive baselines with the scDFM cell_eval protocol."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from specflow.config import SpecFlowConfig
from specflow.evaluation.baselines import (
    build_single_delta_lookup,
    predict_additive_baseline,
    predict_control_baseline,
)
from specflow.evaluation.scdfm_protocol import (
    resolve_scdfm_split,
    run_cell_eval,
    select_scdfm_eval_genes,
    write_scdfm_anndata,
)
from specflow.experiment import ExperimentRunner


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _sample_control_indices(
    control_expression: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sample_count = min(int(n_cells), control_expression.shape[0])
    return rng.permutation(control_expression.shape[0])[:sample_count]


def _apply_split_overrides(
    config: SpecFlowConfig,
    split_path: Optional[str],
    fold: Optional[int],
) -> None:
    if split_path is not None:
        config.data.split_path = split_path
    if fold is not None:
        config.data.split_fold = fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/norman.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline", choices=("control", "additive"), required=True)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--split-method",
        choices=("additive", "combinations"),
        default=None,
        help="scDFM split method. Defaults to data.setting when applicable.",
    )
    parser.add_argument("--infer-top-gene", type=int, default=1000)
    parser.add_argument("--n-control-cells", type=int, default=None)
    parser.add_argument("--control-pert", default="control")
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--missing-single",
        choices=("error", "zero"),
        default="error",
        help=(
            "Additive baseline behavior when a test target has no train single "
            "delta. Use 'error' for strict no-leakage evaluation."
        ),
    )
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument(
        "--write-anndata-only",
        action="store_true",
        help="Write pred.h5ad and real.h5ad without running cell_eval.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config = SpecFlowConfig.from_yaml(args.config)
    _apply_split_overrides(config, args.split_path, args.fold)
    runner = ExperimentRunner.from_config(config, output_dir=str(output_dir))

    resolved_fold = config.data.split_fold
    resolved_split_path = config.data.split_path
    setting = config.data.setting
    resolved_split_method = args.split_method or (
        setting if setting in {"additive", "combinations"} else "additive"
    )
    split = resolve_scdfm_split(
        resolved_split_path,
        runner.data.perturbed_by_condition.keys(),
        dataset=config.data.dataset,
        fold=resolved_fold,
        split_method=resolved_split_method,
    )
    test_conditions = [
        condition
        for condition in split["test"]
        if condition in runner.data.perturbed_by_condition
    ]
    missing_conditions = sorted(set(split["test"]) - set(test_conditions))
    if missing_conditions:
        preview = ", ".join(missing_conditions[:10])
        raise ValueError(f"split contains conditions absent from data: {preview}")
    if not test_conditions:
        raise ValueError("split has no test conditions")

    eval_gene_names = select_scdfm_eval_genes(
        config.data.h5ad_path,
        runner.data.gene_names,
        test_conditions,
        condition_key=config.data.condition_key,
        gene_key=config.data.gene_key,
        control_labels=config.data.control_labels,
        infer_top_gene=args.infer_top_gene,
    )
    gene_to_idx = {gene: idx for idx, gene in enumerate(runner.data.gene_names)}
    eval_indices = np.array([gene_to_idx[gene] for gene in eval_gene_names])

    sample_cells = (
        config.inference.n_control_cells
        if args.n_control_cells is None
        else args.n_control_cells
    )
    resolved_seed = config.data.seed if args.seed is None else args.seed
    rng = np.random.default_rng(resolved_seed)
    train_conditions = list(runner.data.splits.get("train", []))
    single_deltas = build_single_delta_lookup(
        runner.data.perturbed_by_condition,
        runner.data.control_expression,
        train_conditions,
        separator=config.data.separator,
        control_labels=config.data.control_labels,
    )

    predicted: Dict[str, np.ndarray] = {}
    observed: Dict[str, np.ndarray] = {}
    for condition in test_conditions:
        control_indices = _sample_control_indices(
            runner.data.control_expression,
            sample_cells,
            rng,
        )
        controls = runner.data.control_expression[control_indices]
        if args.baseline == "control":
            samples = predict_control_baseline(controls)
        else:
            samples = predict_additive_baseline(
                condition,
                controls,
                single_deltas,
                separator=config.data.separator,
                control_labels=config.data.control_labels,
                missing_single=args.missing_single,
                clamp_min=args.clamp_min,
            )
        predicted[condition] = samples[:, eval_indices]
        observed[condition] = runner.data.perturbed_by_condition[condition][
            :, eval_indices
        ]

    paths = write_scdfm_anndata(
        str(output_dir),
        predicted_by_condition=predicted,
        observed_by_condition=observed,
        control_expression=runner.data.control_expression[:, eval_indices],
        eval_gene_names=eval_gene_names,
        control_pert=args.control_pert,
    )
    result = {
        "protocol": "scdfm",
        "baseline": args.baseline,
        "source": "local_baseline",
        "fold": resolved_fold,
        "split_method": resolved_split_method,
        "n_test_conditions": len(test_conditions),
        "n_eval_genes": len(eval_gene_names),
        "n_train_conditions": len(train_conditions),
        "n_single_deltas": len(single_deltas),
        "test_conditions": test_conditions,
        **paths,
    }
    if not args.write_anndata_only:
        result.update(
            run_cell_eval(
                paths["pred_h5ad"],
                paths["real_h5ad"],
                str(output_dir),
                control_pert=args.control_pert,
                num_threads=args.num_threads,
            )
        )

    _write_json(output_dir / "baseline_summary.json", result)
    _write_json(output_dir / "scdfm_evaluation_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
