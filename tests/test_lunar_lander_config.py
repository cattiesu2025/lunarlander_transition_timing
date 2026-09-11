import json
from pathlib import Path

import pytest
import yaml

from experiments.lunar_lander_braking.config import (
    file_hash,
    load_config,
    load_manifest,
    write_freeze_bundle,
)
from experiments.lunar_lander_braking.run import verify_frozen_inputs


CONFIG = Path("experiments/lunar_lander_braking/configs/pilot.yaml")
ECON_1M_CONFIG = Path("experiments/lunar_lander_braking/configs/pilot_1m.yaml")
V5_CURRENT_CONFIG = Path("experiments/lunar_lander_braking/configs/pilot_v5_current.yaml")
V5_HIGH_CONFIG = Path("experiments/lunar_lander_braking/configs/pilot_v5_high.yaml")
V6_HIGH_CONFIG = Path("experiments/lunar_lander_braking/configs/pilot_v6_high_latest.yaml")
DEVELOPMENT = Path("experiments/lunar_lander_braking/grids/development.json")
DEVELOPMENT_HIGH = Path("experiments/lunar_lander_braking/grids/development_high.json")
HELD_OUT = Path("experiments/lunar_lander_braking/grids/held_out.json")
HELD_OUT_HIGH = Path("experiments/lunar_lander_braking/grids/held_out_high.json")


def test_manifests_are_complete_and_axes_do_not_overlap():
    load_config(CONFIG)
    development = load_manifest(DEVELOPMENT)
    held_out = load_manifest(HELD_OUT)
    assert len(development) == len(held_out) == 18
    for field in ("height_above_pad", "vy", "x_offset"):
        assert set(row[field] for row in development).isdisjoint(
            row[field] for row in held_out
        )


def test_freeze_bundle_detects_post_freeze_drift(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_text("value: 1\n", encoding="utf-8")
    bundle_dir = tmp_path / "frozen"
    bundle = write_freeze_bundle(source, [], bundle_dir)
    manifest = bundle_dir / "manifest.sha256.json"
    assert bundle["files"]["config.yaml"] == file_hash(source)
    verify_frozen_inputs(manifest, [source])
    source.write_text("value: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_inputs(manifest, [source])
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema_version"] == 1


def test_freeze_bundle_refuses_overwrite(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError):
        write_freeze_bundle(CONFIG, [], output)


@pytest.mark.parametrize("time_scale", [None, 0, -1])
def test_config_requires_positive_time_scale(tmp_path, time_scale):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if time_scale is None:
        config["reward"].pop("time_scale")
    else:
        config["reward"]["time_scale"] = time_scale
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=r"reward\.time_scale must be a positive number"):
        load_config(path)


@pytest.mark.parametrize("settling_scale", [None, 0, -1])
def test_config_requires_positive_settling_actuation_scale(tmp_path, settling_scale):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if settling_scale is None:
        config["reward"].pop("settling_actuation_scale")
    else:
        config["reward"]["settling_actuation_scale"] = settling_scale
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=r"reward\.settling_actuation_scale must be a positive number"):
        load_config(path)


def test_config_declares_sb3_double_dqn_override():
    config = load_config(CONFIG)
    assert config["agent"]["framework"] == "stable_baselines3"
    assert config["agent"]["algorithm"] == "double_dqn_train_override"
    assert config["agent"]["gradient_steps"] == 1


def test_econ_1m_pilot_changes_only_name_and_training_budget():
    base = load_config(CONFIG)
    extended = load_config(ECON_1M_CONFIG)
    assert extended["experiment"]["name"] == "lunar_lander_braking_pilot_v4_econ_1m"
    assert extended["experiment"]["train_steps"] == 1_000_000
    base_experiment = {**base["experiment"], "name": extended["experiment"]["name"], "train_steps": 1_000_000}
    assert extended["experiment"] == base_experiment
    assert {key: value for key, value in extended.items() if key != "experiment"} == {
        key: value for key, value in base.items() if key != "experiment"
    }


def test_v5_arms_change_only_declared_checkpoint_and_height_fields():
    v4 = load_config(ECON_1M_CONFIG)
    current = load_config(V5_CURRENT_CONFIG)
    high = load_config(V5_HIGH_CONFIG)
    expected_steps = [600_000, 700_000, 800_000, 900_000, 1_000_000]
    assert current["experiment"]["checkpoint_steps"] == expected_steps
    assert high["experiment"]["checkpoint_steps"] == expected_steps

    v4_experiment = {
        **v4["experiment"],
        "name": current["experiment"]["name"],
        "checkpoint_steps": expected_steps,
    }
    assert current["experiment"] == v4_experiment
    assert {key: value for key, value in current.items() if key != "experiment"} == {
        key: value for key, value in v4.items() if key != "experiment"
    }

    expected_high = {**current, "experiment": {**current["experiment"], "name": high["experiment"]["name"]}}
    expected_high["environment"] = {
        **current["environment"],
        "training_distribution": {
            **current["environment"]["training_distribution"],
            "height_above_pad": [7.8, 12.2],
        },
    }
    assert high == expected_high


def test_high_development_grid_only_changes_ids_and_heights():
    current = load_manifest(DEVELOPMENT)
    high = load_manifest(DEVELOPMENT_HIGH)
    assert len(high) == 18
    height_map = {5.0: 8.0, 6.0: 10.0, 7.0: 12.0}
    for low_row, high_row in zip(current, high):
        assert high_row["height_above_pad"] == height_map[low_row["height_above_pad"]]
        assert high_row["scenario_id"].startswith("devhigh_")
        assert {
            key: value for key, value in high_row.items()
            if key not in {"scenario_id", "height_above_pad"}
        } == {
            key: value for key, value in low_row.items()
            if key not in {"scenario_id", "height_above_pad"}
        }


def test_v6_uses_v5_high_training_contract_with_new_experiment_name():
    v5 = load_config(V5_HIGH_CONFIG)
    v6 = load_config(V6_HIGH_CONFIG)
    expected = {
        **v5,
        "experiment": {**v5["experiment"], "name": v6["experiment"]["name"]},
        "selection": v6["selection"],
    }
    assert v6 == expected
    assert v6["selection"] == {
        "rule": "latest_eligible_checkpoint",
        "interventions": ["original", "low_descent_speed"],
        "minimum_landed_per_intervention": 15,
        "minimum_primary_events_per_intervention": 15,
        "uses_onset_time": False,
    }


def test_high_development_and_held_out_axes_do_not_overlap():
    development = load_manifest(DEVELOPMENT_HIGH)
    held_out = load_manifest(HELD_OUT_HIGH)
    assert len(development) == len(held_out) == 18
    for field in ("height_above_pad", "vy", "x_offset", "terrain_seed", "noise_seed"):
        assert set(row[field] for row in development).isdisjoint(
            row[field] for row in held_out
        )
    training_height = load_config(V6_HIGH_CONFIG)["environment"][
        "training_distribution"
    ]["height_above_pad"]
    low, high = training_height
    assert all(low <= row["height_above_pad"] <= high for row in held_out)


@pytest.mark.parametrize(
    "checkpoint_steps,match",
    [
        ([600_000, 600_000], "sorted and unique"),
        ([700_000, 600_000], "sorted and unique"),
        ([600_002], "align"),
        ([1_100_000], "cannot exceed"),
    ],
)
def test_checkpoint_schedule_validation(tmp_path, checkpoint_steps, match):
    config = yaml.safe_load(V5_CURRENT_CONFIG.read_text(encoding="utf-8"))
    config["experiment"]["checkpoint_steps"] = checkpoint_steps
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(path)
