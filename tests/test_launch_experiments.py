import importlib.util
from pathlib import Path

import yaml


def _load_launcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "launch_experiments.py"
    spec = importlib.util.spec_from_file_location("launch_experiments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_dotted_overrides_preserves_unrelated_values():
    launcher = _load_launcher()
    base = {
        "flow": {"sigma": 0.2, "delta_corr_weight": 0.03},
        "model": {"spectral_propagation": True},
    }

    result = launcher._apply_overrides(
        base,
        {
            "flow.delta_corr_weight": 0.0,
            "model.spectral_propagation": False,
        },
    )

    assert result["flow"]["sigma"] == 0.2
    assert result["flow"]["delta_corr_weight"] == 0.0
    assert result["model"]["spectral_propagation"] is False
    assert base["flow"]["delta_corr_weight"] == 0.03


def test_defaults_fill_omitted_experiment_fields():
    launcher = _load_launcher()
    spec = {
        "defaults": {"seed": 42, "gpu": 0, "fold": 1},
        "experiments": [
            {"name": "full"},
            {"name": "go_only", "gpu": 1, "overrides": {"model.graph_mode": "go"}},
        ],
    }

    experiments = launcher._experiment_list(spec)

    # Omitted fields come from defaults.
    assert experiments[0] == {"name": "full", "seed": 42, "gpu": 0, "fold": 1}
    # Explicit per-experiment fields win over defaults.
    assert experiments[1]["gpu"] == 1
    assert experiments[1]["seed"] == 42 and experiments[1]["fold"] == 1
    assert experiments[1]["overrides"] == {"model.graph_mode": "go"}


def test_dry_run_config_only_directory_is_still_ready(tmp_path, monkeypatch):
    launcher = _load_launcher()
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("outputs/study/00_full_f1_s42_g0").mkdir(parents=True)
    Path("outputs/study/00_full_f1_s42_g0/run_config.yaml").write_text(
        "data: {}\n", encoding="utf-8"
    )
    base_config = {
        "data": {"split_fold": 1, "seed": 42},
        "flow": {"delta_corr_weight": 0.03},
        "model": {"spectral_propagation": True},
    }
    Path("configs/norman.yaml").write_text(
        yaml.safe_dump(base_config), encoding="utf-8"
    )
    spec = {
        "study_name": "study",
        "output_root": "outputs",
        "base_config": "configs/norman.yaml",
        "experiments": [{"name": "full", "gpu": 0, "fold": 1, "seed": 42}],
    }

    run = launcher._build_run(
        spec,
        tmp_path / "batch.yaml",
        spec["experiments"][0],
        index=0,
        overwrite=False,
    )
    assert run["status"] == "ready"
    launcher._write_yaml(run["config_path"], run["config"])

    Path("outputs/study/00_full_f1_s42_g0/train.log").write_text(
        "started\n", encoding="utf-8"
    )
    run = launcher._build_run(
        spec,
        tmp_path / "batch.yaml",
        spec["experiments"][0],
        index=0,
        overwrite=False,
    )
    assert run["status"] == "exists"


def test_existing_artifact_with_changed_config_is_marked_stale(tmp_path, monkeypatch):
    launcher = _load_launcher()
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    run_dir = Path("outputs/study/00_full_f1_s42_g0")
    run_dir.mkdir(parents=True)
    run_dir.joinpath("run_config.yaml").write_text(
        yaml.safe_dump({"data": {"split_path": "old.json"}}),
        encoding="utf-8",
    )
    run_dir.joinpath("train.log").write_text("started\n", encoding="utf-8")
    Path("configs/norman.yaml").write_text(
        yaml.safe_dump({"data": {"split_path": "new.pkl"}}),
        encoding="utf-8",
    )
    spec = {
        "study_name": "study",
        "output_root": "outputs",
        "base_config": "configs/norman.yaml",
        "experiments": [{"name": "full", "gpu": 0, "fold": 1, "seed": 42}],
    }

    run = launcher._build_run(
        spec,
        tmp_path / "batch.yaml",
        spec["experiments"][0],
        index=0,
        overwrite=False,
    )

    assert run["status"] == "exists_stale_config"
    assert launcher._is_existing_status(run["status"])
