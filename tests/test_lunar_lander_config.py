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
DEVELOPMENT = Path("experiments/lunar_lander_braking/grids/development.json")
HELD_OUT = Path("experiments/lunar_lander_braking/grids/held_out.json")


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
