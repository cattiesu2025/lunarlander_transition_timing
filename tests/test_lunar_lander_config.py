import json
from pathlib import Path

import pytest

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
