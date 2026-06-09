"""Evaluate a SpecFlow checkpoint with the scDFM cell_eval protocol."""

import argparse
import json

from specflow.config import SpecFlowConfig
from specflow.experiment import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/norman.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="scDFM split fold. Defaults to data.split_fold from the config.",
    )
    parser.add_argument(
        "--split-method",
        choices=("additive", "combinations"),
        default=None,
        help="scDFM split method. Defaults to data.setting when applicable.",
    )
    parser.add_argument("--infer-top-gene", type=int, default=1000)
    parser.add_argument(
        "--n-control-cells",
        type=int,
        default=None,
        help="Number of control cells to sample per perturbation. Defaults to config.",
    )
    parser.add_argument("--control-pert", default="control")
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--write-anndata-only",
        action="store_true",
        help="Write pred.h5ad and real.h5ad without running cell_eval.",
    )
    args = parser.parse_args()

    config = SpecFlowConfig.from_yaml(args.config)
    runner = ExperimentRunner.from_config(config, output_dir=args.output_dir)
    result = runner.evaluate_scdfm(
        checkpoint=args.checkpoint,
        split_path=args.split_path,
        fold=args.fold,
        split_method=args.split_method,
        infer_top_gene=args.infer_top_gene,
        n_control_cells=args.n_control_cells,
        control_pert=args.control_pert,
        num_threads=args.num_threads,
        seed=args.seed,
        write_anndata_only=args.write_anndata_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
