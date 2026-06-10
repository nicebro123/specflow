import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _tokens(condition, controls):
    return [
        token
        for token in condition.split("+")
        if token and token.lower() not in controls
    ]


def test_combosciplex_protocol_files_exist_and_cover_split_tokens():
    config = yaml.safe_load((ROOT / "configs/combosciplex.yaml").read_text())
    data = config["data"]
    target_map_path = ROOT / data["target_map_path"]
    split_path = ROOT / data["split_path"]

    assert target_map_path.exists()
    assert split_path.exists()
    assert data["preprocess"] is True
    assert data["n_top_genes"] == 5000
    assert "control+control" in data["control_labels"]

    target_map = yaml.safe_load(target_map_path.read_text())
    split = json.loads(split_path.read_text())
    train = split["train"]
    val = split.get("val", [])
    test = split["test"]
    controls = {label.lower() for label in data["control_labels"]}

    assert train
    assert test
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))

    drugs = {
        token
        for condition in train + val + test
        for token in _tokens(condition, controls)
    }
    assert drugs <= set(target_map)
    assert all(target_map[drug] for drug in drugs)


def test_holdout_configs_use_generated_folded_pickle():
    assert (ROOT / "scripts/build_holdout_split.py").exists()
    for config_path in [
        ROOT / "configs/norman_holdout.yaml",
        ROOT / "configs/experiments/holdout_full_4gpu.yaml",
        ROOT / "configs/experiments/holdout_ablation_4gpu.yaml",
    ]:
        text = config_path.read_text()
        assert "data/splits/norman_holdout.pkl" in text
        assert "data/splits/norman_holdout.json" not in text
