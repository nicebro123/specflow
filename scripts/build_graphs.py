"""Build GO/coexpression graphs and precompute condition spectra."""

import argparse
import json

from specflow.config import SpecFlowConfig
from specflow.experiment import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    runner = ExperimentRunner.from_config(
        SpecFlowConfig.from_yaml(args.config), output_dir=args.output_dir
    )
    print(json.dumps(runner.precompute_spectra(), indent=2))


if __name__ == "__main__":
    main()
