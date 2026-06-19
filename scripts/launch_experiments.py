"""Launch or dry-run SpecFlow experiment batches.

The launcher is intentionally conservative:

* each run gets an immutable generated config saved under the run directory;
* CUDA_VISIBLE_DEVICES is set per run from the experiment spec;
* existing run directories are skipped only when they contain training artifacts;
* --dry-run is the default so command generation can be audited before launch.

Example:

  python scripts/launch_experiments.py \
    --spec configs/experiments/ablation_4gpu.yaml \
    --dry-run

  python scripts/launch_experiments.py \
    --spec configs/experiments/ablation_4gpu.yaml \
    --launch
"""

import argparse
import os
import shlex
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import yaml


DEFAULT_TRAIN_SCRIPT = "scripts/train.py"
RUN_ARTIFACTS = {
    "agg_results.csv",
    "results.csv",
    "training_summary.json",
    "scdfm_evaluation_summary.json",
    "train.log",
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _set_by_path(payload: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    if not dotted_key or dotted_key.startswith(".") or dotted_key.endswith("."):
        raise ValueError(f"invalid override key: {dotted_key!r}")
    current: MutableMapping[str, Any] = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        child = current[part]
        if not isinstance(child, MutableMapping):
            raise ValueError(
                f"cannot set {dotted_key!r}: {part!r} is not a mapping"
            )
        current = child
    current[parts[-1]] = value


def _apply_overrides(
    base_config: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    config = deepcopy(dict(base_config))
    for key, value in (overrides or {}).items():
        _set_by_path(config, str(key), value)
    return config


def _slug(value: str) -> str:
    allowed = []
    for char in value.strip().lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_", ".", "+"}:
            allowed.append(char)
        elif char.isspace():
            allowed.append("_")
    result = "".join(allowed).strip("._-")
    return result or "run"


def _format_run_name(
    index: int,
    experiment: Mapping[str, Any],
    include_gpu: bool = True,
) -> str:
    name = _slug(str(experiment.get("name") or f"run_{index:02d}"))
    fold = experiment.get("fold")
    seed = experiment.get("seed")
    gpu = experiment.get("gpu")
    parts = [f"{index:02d}_{name}"]
    if fold is not None:
        parts.append(f"f{fold}")
    if seed is not None:
        parts.append(f"s{seed}")
    if include_gpu and gpu is not None:
        parts.append(f"g{gpu}")
    return "_".join(parts)


def _quote_command(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _has_run_artifacts(run_dir: Path) -> bool:
    return any((run_dir / name).exists() for name in RUN_ARTIFACTS)


def _is_existing_status(status: str) -> bool:
    return status.startswith("exists")


def _experiment_list(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    experiments = spec.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("spec must contain a non-empty 'experiments' list")
    defaults = spec.get("defaults", {}) or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("spec 'defaults' must be a mapping")
    output = []
    for idx, item in enumerate(experiments):
        if not isinstance(item, dict):
            raise ValueError(f"experiments[{idx}] must be a mapping")
        # Spec-level defaults fill in fields the experiment omits, so common
        # values (seed, gpu, fold, ...) need not be repeated per experiment.
        merged = dict(defaults)
        merged.update(item)
        if "name" not in merged:
            raise ValueError(f"experiments[{idx}] is missing required field 'name'")
        output.append(merged)
    return output


def _resolve_study_dir(spec: Mapping[str, Any], spec_path: Path) -> Path:
    root = Path(str(spec.get("output_root", "outputs"))).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    study_name = spec.get("study_name")
    if not study_name:
        stamp = datetime.now().strftime("%Y%m%d")
        study_name = f"{stamp}_{spec_path.stem}"
    return root / _slug(str(study_name))


def _build_run(
    spec: Mapping[str, Any],
    spec_path: Path,
    experiment: Mapping[str, Any],
    index: int,
    overwrite: bool,
) -> Dict[str, Any]:
    base_config_path = Path(str(experiment.get("config", spec.get("base_config", "configs/norman.yaml"))))
    if not base_config_path.is_absolute():
        base_config_path = (Path.cwd() / base_config_path).resolve()
    base_config = _read_yaml(base_config_path)

    overrides = dict(spec.get("overrides", {}) or {})
    overrides.update(experiment.get("overrides", {}) or {})
    if experiment.get("fold") is not None:
        overrides["data.split_fold"] = int(experiment["fold"])
    if experiment.get("seed") is not None:
        overrides["data.seed"] = int(experiment["seed"])
    if experiment.get("max_steps") is not None:
        overrides["training.max_steps"] = int(experiment["max_steps"])
    if experiment.get("batch_size") is not None:
        overrides["training.batch_size"] = int(experiment["batch_size"])
    if experiment.get("learning_rate") is not None:
        overrides["training.learning_rate"] = float(experiment["learning_rate"])

    config = _apply_overrides(base_config, overrides)
    study_dir = _resolve_study_dir(spec, spec_path)
    run_name = str(experiment.get("run_name") or _format_run_name(index, experiment))
    run_dir = study_dir / run_name
    config_path = run_dir / "run_config.yaml"
    log_path = run_dir / "train.log"

    if run_dir.exists() and _has_run_artifacts(run_dir) and not overwrite:
        status = "exists"
        if config_path.exists():
            try:
                if _read_yaml(config_path) != config:
                    status = "exists_stale_config"
            except Exception:
                status = "exists_unreadable_config"
    else:
        status = "ready"

    train_script = Path(str(spec.get("train_script", DEFAULT_TRAIN_SCRIPT)))
    python_executable = str(spec.get("python", "python"))
    command = [
        python_executable,
        str(train_script),
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
    ]
    for extra in experiment.get("extra_args", spec.get("extra_args", [])) or []:
        command.append(str(extra))
    if experiment.get("skip_evaluation", spec.get("skip_evaluation", False)):
        command.append("--skip-evaluation")

    return {
        "index": index,
        "name": experiment["name"],
        "gpu": experiment.get("gpu"),
        "status": status,
        "base_config": str(base_config_path),
        "run_dir": run_dir,
        "config_path": config_path,
        "log_path": log_path,
        "config": config,
        "command": command,
        "overrides": overrides,
    }


def _write_manifest(study_dir: Path, runs: List[Mapping[str, Any]], spec_path: Path) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "spec": str(spec_path.resolve()),
        "runs": [
            {
                "index": run["index"],
                "name": run["name"],
                "gpu": run["gpu"],
                "status": run["status"],
                "run_dir": str(run["run_dir"]),
                "config_path": str(run["config_path"]),
                "log_path": str(run["log_path"]),
                "overrides": run["overrides"],
            }
            for run in runs
        ],
    }
    _write_yaml(study_dir / "launch_manifest.yaml", manifest)


def _shell_command(run: Mapping[str, Any]) -> str:
    command = _quote_command(run["command"])
    log_path = shlex.quote(str(run["log_path"]))
    run_dir = shlex.quote(str(run["run_dir"]))
    source_path = shlex.quote(str((Path.cwd() / "src").resolve()))
    env_prefix = f"PYTHONPATH={source_path}${{PYTHONPATH:+:$PYTHONPATH}} "
    env_prefix += (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(str(run['gpu']))} "
        if run.get("gpu") is not None
        else ""
    )
    return f"mkdir -p {run_dir} && {env_prefix}{command} 2>&1 | tee {log_path}"


def _write_gpu_scripts(
    study_dir: Path,
    runs: List[Mapping[str, Any]],
    overwrite: bool = False,
) -> None:
    by_gpu: Dict[str, List[Mapping[str, Any]]] = {}
    for run in runs:
        gpu = "cpu" if run.get("gpu") is None else str(run["gpu"])
        by_gpu.setdefault(gpu, []).append(run)

    tmux_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for gpu, gpu_runs in sorted(by_gpu.items(), key=lambda item: item[0]):
        script_path = study_dir / f"run_gpu{gpu}.sh"
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(Path.cwd()))}",
            "",
        ]
        for run in gpu_runs:
            # A failed run must not abort the whole GPU queue (set -e), so each
            # run is guarded with `|| echo FAILED`. The card keeps working
            # through the remaining experiments without manual intervention.
            failure_note = (
                f"echo '[{run['index']:02d}] FAILED: {run['run_dir']}'"
            )
            if _is_existing_status(str(run["status"])) and not overwrite:
                lines.extend(
                    [
                        f"echo '[{run['index']:02d}] {run['name']} -> {run['run_dir']}'",
                        (
                            f"echo 'skip existing ({run['status']}): "
                            f"{run['run_dir']}'; "
                            "echo 'remove the run directory or pass --overwrite "
                            "to regenerate and rerun it'"
                        ),
                        "",
                    ]
                )
                continue
            skip_complete = (
                f"if [ -f {shlex.quote(str(run['run_dir'] / 'agg_results.csv'))} ]; "
                f"then echo 'skip complete: {run['run_dir']}'; else"
            )
            if overwrite:
                skip_complete = "if false; then :; else"
            lines.extend(
                [
                    f"echo '[{run['index']:02d}] {run['name']} -> {run['run_dir']}'",
                    skip_complete,
                    f"  {_shell_command(run)} || {failure_note}",
                    "fi",
                    "",
                ]
            )
        script_path.write_text("\n".join(lines), encoding="utf-8")
        script_path.chmod(0o755)

        session = f"sf_{study_dir.name}_g{gpu}"[:80]
        tmux_lines.append(
            f"tmux new -d -s {shlex.quote(session)} {shlex.quote('bash ' + str(script_path))}"
        )
    tmux_path = study_dir / "launch_tmux.sh"
    tmux_path.write_text("\n".join(tmux_lines) + "\n", encoding="utf-8")
    tmux_path.chmod(0o755)


def _print_run(run: Mapping[str, Any]) -> None:
    env_prefix = (
        f"CUDA_VISIBLE_DEVICES={run['gpu']} "
        if run.get("gpu") is not None
        else ""
    )
    print(f"[{run['index']:02d}] {run['name']} -> {run['run_dir']}")
    print(f"     status: {run['status']}, gpu: {run.get('gpu')}")
    print(f"     config: {run['config_path']}")
    if run["status"] == "exists_stale_config":
        print(
            "     warning: existing run artifacts use a different run_config.yaml; "
            "remove the run directory or pass --overwrite before rerunning"
        )
    elif run["status"] == "exists_unreadable_config":
        print(
            "     warning: existing run_config.yaml could not be read; "
            "remove the run directory or pass --overwrite before rerunning"
        )
    print(f"     command: {env_prefix}{_quote_command(run['command'])}")
    print(f"     shell: {_shell_command(run)}")


def _launch_run(run: Mapping[str, Any]) -> int:
    if _is_existing_status(str(run["status"])):
        print(f"SKIP existing run: {run['run_dir']}")
        return 0

    run_dir: Path = run["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(run["config_path"], run["config"])

    env = os.environ.copy()
    source_path = str((Path.cwd() / "src").resolve())
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_path + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else source_path
    )
    if run.get("gpu") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(run["gpu"])
    with run["log_path"].open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            run["command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="YAML experiment spec.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print commands and write generated configs without launching. Default.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Actually run experiments sequentially in this process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow launching into an existing non-empty run directory.",
    )
    parser.add_argument(
        "--no-gpu-scripts",
        action="store_true",
        help="Do not write per-GPU shell scripts and launch_tmux.sh.",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    spec = _read_yaml(spec_path)
    experiments = _experiment_list(spec)
    runs = [
        _build_run(spec, spec_path, experiment, index, overwrite=args.overwrite)
        for index, experiment in enumerate(experiments)
    ]
    study_dir = _resolve_study_dir(spec, spec_path)
    study_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run["run_dir"].mkdir(parents=True, exist_ok=True)
        if not _is_existing_status(str(run["status"])) or args.overwrite:
            _write_yaml(run["config_path"], run["config"])
        _print_run(run)
    _write_manifest(study_dir, runs, spec_path)
    if not args.no_gpu_scripts:
        _write_gpu_scripts(study_dir, runs, overwrite=args.overwrite)

    if not args.launch:
        print(f"Dry run complete. Study directory: {study_dir}")
        print("Use --launch to execute these runs sequentially.")
        if not args.no_gpu_scripts:
            print(f"Per-GPU scripts: {study_dir}/run_gpu*.sh")
            print(f"Tmux launcher: {study_dir}/launch_tmux.sh")
        return

    failures = []
    for run in runs:
        code = _launch_run(run)
        if code != 0:
            failures.append((run["name"], code))
            print(f"FAILED {run['name']} with exit code {code}")
    succeeded = len(runs) - len(failures)
    print(f"\nLaunch complete: {succeeded}/{len(runs)} runs succeeded.")
    if failures:
        for name, code in failures:
            print(f"  FAILED {name} (exit {code})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
