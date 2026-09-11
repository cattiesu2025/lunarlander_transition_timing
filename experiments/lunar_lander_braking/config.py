"""Configuration, manifest validation, and immutable freeze bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


REQUIRED_CONFIG_SECTIONS = {
    "experiment",
    "environment",
    "reward",
    "detector",
    "agent",
    "evaluation",
}
REQUIRED_SCENARIO_FIELDS = {
    "scenario_id",
    "height_above_pad",
    "vx",
    "vy",
    "x_offset",
    "angle",
    "angular_velocity",
    "terrain_seed",
    "noise_seed",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    missing = REQUIRED_CONFIG_SECTIONS - config.keys()
    if missing:
        raise ValueError(f"Configuration missing sections: {sorted(missing)}")
    conditions = config["reward"].get("conditions", {})
    if set(conditions) != {"DESC", "BAL", "ECON"}:
        raise ValueError("Reward conditions must be exactly DESC, BAL, and ECON")
    for field in (
        "downward_reference_speed",
        "downward_scale",
        "main_engine_scale",
        "time_scale",
        "settling_actuation_scale",
    ):
        try:
            value = float(config["reward"][field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"reward.{field} must be a positive number") from error
        if value <= 0:
            raise ValueError(f"reward.{field} must be a positive number")
    if int(config["experiment"].get("action_repeat", 1)) != 1:
        raise ValueError("Protocol v1 requires action_repeat=1")
    agent = config["agent"]
    if agent.get("framework") != "stable_baselines3":
        raise ValueError("agent.framework must be stable_baselines3")
    if agent.get("algorithm") != "double_dqn_train_override":
        raise ValueError("agent.algorithm must be double_dqn_train_override")
    if int(agent.get("gradient_steps", 0)) <= 0:
        raise ValueError("agent.gradient_steps must be a positive integer")
    checkpoint_steps = config["experiment"].get("checkpoint_steps")
    if checkpoint_steps is not None:
        if (
            not isinstance(checkpoint_steps, list)
            or not checkpoint_steps
            or any(
                not isinstance(step, int) or isinstance(step, bool) or step <= 0
                for step in checkpoint_steps
            )
        ):
            raise ValueError("experiment.checkpoint_steps must be positive integers")
        if checkpoint_steps != sorted(set(checkpoint_steps)):
            raise ValueError("experiment.checkpoint_steps must be sorted and unique")
        train_steps = int(config["experiment"]["train_steps"])
        if checkpoint_steps[-1] > train_steps:
            raise ValueError("experiment.checkpoint_steps cannot exceed train_steps")
        train_frequency = int(agent["train_frequency"])
        if any(step % train_frequency for step in checkpoint_steps):
            raise ValueError(
                "experiment.checkpoint_steps must align with agent.train_frequency"
            )
    selection = config.get("selection")
    if selection is not None:
        if selection.get("rule") != "latest_eligible_checkpoint":
            raise ValueError("selection.rule must be latest_eligible_checkpoint")
        if selection.get("interventions") != ["original", "low_descent_speed"]:
            raise ValueError(
                "selection.interventions must be original and low_descent_speed"
            )
        if selection.get("uses_onset_time") is not False:
            raise ValueError("selection.uses_onset_time must be false")
        for field in (
            "minimum_landed_per_intervention",
            "minimum_primary_events_per_intervention",
        ):
            value = selection.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= 18
            ):
                raise ValueError(f"selection.{field} must be an integer in [1, 18]")
    return config


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a non-empty JSON list")
    ids: set[str] = set()
    for index, row in enumerate(rows):
        missing = REQUIRED_SCENARIO_FIELDS - set(row)
        if missing:
            raise ValueError(f"Manifest row {index} missing: {sorted(missing)}")
        scenario_id = str(row["scenario_id"])
        if scenario_id in ids:
            raise ValueError(f"Duplicate scenario_id: {scenario_id}")
        ids.add(scenario_id)
        if float(row["height_above_pad"]) <= 0 or float(row["vy"]) >= 0:
            raise ValueError(f"Scenario {scenario_id} must begin above the pad and descending")
    return rows

def write_freeze_bundle(
    config_path: str | Path,
    manifest_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    inputs = [Path(config_path), *(Path(path) for path in manifest_paths)]
    checksums: dict[str, str] = {}
    for source in inputs:
        target = output / source.name
        shutil.copyfile(source, target)
        checksums[target.name] = file_hash(target)
    bundle = {"schema_version": 1, "files": checksums}
    (output / "manifest.sha256.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle
