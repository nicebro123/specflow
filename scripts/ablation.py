"""Materialize and optionally run supported SpecFlow ablation configurations."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from specflow.config import SpecFlowConfig
from specflow.experiment import ExperimentRunner


def variants(base):
    definitions = {
        "no_mmd": {"flow": {"mmd_weight": 0.0}},
        "static_graph": {
            "graph": {"perturbation": {"alpha_go": 1.0, "alpha_coexp": 1.0}}
        },
        "sigma_0_1": {"flow": {"sigma": 0.1}},
        "sigma_1_0": {"flow": {"sigma": 1.0}},
        "exact_spectrum": {"spectral": {"use_perturbation_approx": False}},
        "approx_spectrum": {"spectral": {"use_perturbation_approx": True}},
    }

    def merge(target, override):
        for key, value in override.items():
            if isinstance(value, dict):
                target[key] = merge(target.get(key, {}), value)
            else:
                target[key] = value
        return target

    return {
        name: merge(copy.deepcopy(base), override)
        for name, override in definitions.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-root", default="outputs/ablations")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle) or {}
    output_root = Path(args.output_root)
    results = {}
    for name, raw_config in variants(base).items():
        raw_config.setdefault("output", {})["output_dir"] = str(output_root / name)
        config_path = output_root / name / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw_config, handle, sort_keys=False)
        results[name] = {"config": str(config_path)}
        if args.run:
            runner = ExperimentRunner.from_config(SpecFlowConfig.from_dict(raw_config))
            results[name]["training"] = runner.train(n_epochs=args.epochs)
            results[name]["evaluation"] = runner.evaluate()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
