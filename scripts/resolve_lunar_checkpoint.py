#!/usr/bin/env python3
"""Resolve one hash-locked selected checkpoint for sealed evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.lunar_lander_braking.config import (  # noqa: E402
    canonical_hash,
    file_hash,
    load_config,
)
from experiments.lunar_lander_braking.run import verify_frozen_inputs  # noqa: E402


CONDITIONS = ("DESC", "BAL", "ECON")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def resolve_checkpoint(
    config_path: str | Path,
    selection_path: str | Path,
    selection_lock_path: str | Path,
    freeze_manifest_path: str | Path,
    condition: str,
    seed: int,
) -> Path:
    config = load_config(config_path)
    selection_path = Path(selection_path)
    lock = _load_json(selection_lock_path)
    if lock.get("file") != selection_path.name:
        raise ValueError("Selection lock names a different file")
    if lock.get("sha256") != file_hash(selection_path):
        raise ValueError("Selected-checkpoint manifest hash mismatch")

    verify_frozen_inputs(freeze_manifest_path, [config_path])
    payload = _load_json(selection_path)
    if payload.get("schema_version") != 2:
        raise ValueError("Unsupported selected-checkpoint manifest schema")
    if payload.get("selection_rule") != "latest_eligible_checkpoint":
        raise ValueError("Unexpected selection rule")
    if payload.get("config_hash") != canonical_hash(config):
        raise ValueError("Selection/config hash mismatch")
    if payload.get("freeze_manifest_sha256") != file_hash(freeze_manifest_path):
        raise ValueError("Selection/freeze-manifest hash mismatch")
    frozen = _load_json(freeze_manifest_path)
    frozen_development_hash = frozen.get("files", {}).get("development_high.json")
    if payload.get("development_manifest_sha256") != frozen_development_hash:
        raise ValueError("Selection/development-manifest hash mismatch")
    if payload.get("gate") != {
        "interventions": config["selection"]["interventions"],
        "minimum_landed_per_intervention": config["selection"][
            "minimum_landed_per_intervention"
        ],
        "minimum_primary_events_per_intervention": config["selection"][
            "minimum_primary_events_per_intervention"
        ],
        "uses_onset_time": False,
    }:
        raise ValueError("Selection gate does not match the frozen config")

    selected = payload.get("selected_models")
    if not isinstance(selected, list):
        raise ValueError("selected_models must be a list")
    expected_keys = {
        (candidate_condition, int(candidate_seed))
        for candidate_condition in CONDITIONS
        for candidate_seed in config["experiment"]["seeds"]
    }
    keys = [(row.get("condition"), int(row.get("seed", -1))) for row in selected]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("Selected-model keys are missing, duplicated, or unexpected")

    matches = [
        row
        for row in selected
        if row["condition"] == condition and int(row["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one selected checkpoint for {condition} seed {seed}")
    row = matches[0]
    checkpoint = Path(row["checkpoint_path"])
    if not checkpoint.is_file():
        raise ValueError(f"Selected checkpoint is missing: {checkpoint}")
    if file_hash(checkpoint) != row.get("checkpoint_sha256"):
        raise ValueError(f"Selected checkpoint hash mismatch: {checkpoint}")
    metadata_path = checkpoint.with_name("metadata.json")
    metadata = _load_json(metadata_path)
    expected_metadata = {
        "checkpoint_sha256": row["checkpoint_sha256"],
        "config_hash": canonical_hash(config),
        "condition": condition,
        "seed": int(seed),
        "training_steps": int(row["checkpoint_step"]),
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"Selected checkpoint metadata mismatch: {field}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    checkpoint = resolve_checkpoint(
        args.config,
        args.selection,
        args.selection_lock,
        args.freeze_manifest,
        args.condition,
        args.seed,
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
