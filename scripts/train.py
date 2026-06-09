"""Train SpecFlow from a benchmark configuration file."""

import argparse
import json

from specflow.config import SpecFlowConfig, parse_override_value
from specflow.experiment import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override any config field with a dotted key, e.g. "
            "--set flow.sigma=0.3 --set model.graph_mode=go. Repeatable. "
            "Values are parsed with YAML rules (numbers/booleans/null/strings)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument(
        "--partition",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument(
        "--evaluation-protocol",
        choices=("cell_eval", "internal"),
        default="cell_eval",
        help=(
            "Post-training evaluation protocol. cell_eval is the "
            "scDFM-compatible default."
        ),
    )
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--split-method",
        choices=("additive", "combinations"),
        default=None,
    )
    parser.add_argument("--infer-top-gene", type=int, default=1000)
    parser.add_argument("--n-control-cells", type=int, default=None)
    parser.add_argument("--control-pert", default="control")
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument(
        "--write-anndata-only",
        action="store_true",
        help="For cell_eval protocol, only write pred.h5ad and real.h5ad.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only train; do not run post-training evaluation or write results.csv.",
    )
    args = parser.parse_args()

    overrides = {}
    for item in args.overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = parse_override_value(raw)
    # Route split selection through the config so training and the scDFM
    # evaluation always use the *same* fold/split. Otherwise --fold would only
    # change evaluation while training stayed on the config's split_fold.
    if args.fold is not None:
        overrides["data.split_fold"] = args.fold
    if args.split_path is not None:
        overrides["data.split_path"] = args.split_path

    runner = ExperimentRunner.from_config(
        SpecFlowConfig.from_yaml(args.config, overrides=overrides),
        output_dir=args.output_dir,
    )
    result = {
        "training": runner.train(
            n_epochs=args.epochs,
            max_steps=args.max_steps,
            eval_every_steps=args.eval_every_steps,
        )
    }
    if not args.skip_evaluation:
        if args.evaluation_protocol == "internal":
            result["evaluation"] = runner.evaluate(partition=args.partition)
        else:
            result["evaluation"] = runner.evaluate_scdfm(
                checkpoint=result["training"]["checkpoint"],
                split_path=args.split_path,
                fold=args.fold,
                split_method=args.split_method,
                infer_top_gene=args.infer_top_gene,
                n_control_cells=args.n_control_cells,
                control_pert=args.control_pert,
                num_threads=args.num_threads,
                write_anndata_only=args.write_anndata_only,
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
