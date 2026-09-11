#!/usr/bin/env python3
"""Select each model's latest task-valid checkpoint on the development grid."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.lunar_lander_braking.config import (  # noqa: E402
    canonical_hash,
    file_hash,
    load_config,
    load_manifest,
)
from experiments.lunar_lander_braking.selection import (  # noqa: E402
    INTERVENTIONS,
    MIN_LANDED,
    MIN_PRIMARY_EVENTS,
    checkpoint_gate_summary,
    latest_eligible_checkpoint,
)
from experiments.lunar_lander_braking.run import verify_frozen_inputs  # noqa: E402


CONDITIONS = ("DESC", "BAL", "ECON")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def verify_artifacts(
    checkpoint: Path,
    evaluation_dir: Path,
    rows: list[dict[str, Any]],
    config_hash: str,
    condition: str,
    seed: int,
    step: int,
    expected_trajectories: int,
) -> str:
    metadata_path = checkpoint.with_name("metadata.json")
    candidates_path = evaluation_dir / "candidates.jsonl"
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise ValueError(f"Missing checkpoint or metadata: {checkpoint}")
    if not candidates_path.is_file():
        raise ValueError(f"Missing candidates file: {candidates_path}")
    trajectories = list((evaluation_dir / "trajectories").glob("*.jsonl.gz"))
    if len(trajectories) != expected_trajectories:
        raise ValueError(
            f"Expected {expected_trajectories} trajectories at {evaluation_dir}, "
            f"found {len(trajectories)}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint_hash = file_hash(checkpoint)
    expected_metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "config_hash": config_hash,
        "condition": condition,
        "seed": int(seed),
        "training_steps": int(step),
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Checkpoint metadata mismatch for {field} at {checkpoint}"
            )
    if {row.get("checkpoint_sha256") for row in rows} != {checkpoint_hash}:
        raise ValueError(f"Evaluation/checkpoint hash mismatch at {evaluation_dir}")
    if {row.get("config_hash") for row in rows} != {config_hash}:
        raise ValueError(f"Evaluation/config hash mismatch at {evaluation_dir}")
    return checkpoint_hash


def select_models(
    config: dict[str, Any],
    manifest: list[dict[str, Any]],
    input_dir: Path,
    train_root: Path,
    conditions: tuple[str, ...] = CONDITIONS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_ids = [str(row["scenario_id"]) for row in manifest]
    steps = [int(step) for step in config["experiment"]["checkpoint_steps"]]
    seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    config_hash = canonical_hash(config)
    selection = config.get("selection", {})
    minimum_landed = int(
        selection.get("minimum_landed_per_intervention", MIN_LANDED)
    )
    minimum_primary_events = int(
        selection.get(
            "minimum_primary_events_per_intervention", MIN_PRIMARY_EVENTS
        )
    )
    audit: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    for condition in conditions:
        for seed in seeds:
            model_summaries: list[dict[str, Any]] = []
            for step in steps:
                evaluation_dir = input_dir / f"step_{step}" / condition / f"seed_{seed}"
                episodes_path = evaluation_dir / "episodes.jsonl"
                if not episodes_path.is_file():
                    raise ValueError(f"Missing evaluation episodes: {episodes_path}")
                rows = read_jsonl(episodes_path)
                checkpoint = (
                    train_root
                    / condition
                    / f"seed_{seed}"
                    / "checkpoints"
                    / f"step_{step}"
                    / "model.zip"
                )
                checkpoint_hash = verify_artifacts(
                    checkpoint,
                    evaluation_dir,
                    rows,
                    config_hash,
                    condition,
                    seed,
                    step,
                    len(scenario_ids) * len(INTERVENTIONS),
                )
                summary = checkpoint_gate_summary(
                    rows,
                    scenario_ids,
                    condition,
                    seed,
                    minimum_landed,
                    minimum_primary_events,
                )
                if not summary["integrity_complete"]:
                    raise ValueError(
                        f"Incomplete/technical evaluation at {evaluation_dir}: {summary}"
                    )
                summary.update(
                    {
                        "checkpoint_step": step,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                        "selected": False,
                    }
                )
                model_summaries.append(summary)

            chosen = latest_eligible_checkpoint(model_summaries)
            if chosen is not None:
                chosen["selected"] = True
                selected.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "checkpoint_step": int(chosen["checkpoint_step"]),
                        "checkpoint_path": chosen["checkpoint_path"],
                        "checkpoint_sha256": chosen["checkpoint_sha256"],
                    }
                )
            audit.extend(sorted(model_summaries, key=lambda row: int(row["checkpoint_step"]), reverse=True))
    return audit, selected


def write_outputs(
    output_dir: Path,
    audit: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    minimum_landed: int,
    minimum_primary_events: int,
    config_hash: str,
    development_manifest_sha256: str,
    freeze_manifest_sha256: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    payload = {
        "schema_version": 2,
        "selection_rule": "latest_eligible_checkpoint",
        "config_hash": config_hash,
        "development_manifest_sha256": development_manifest_sha256,
        "freeze_manifest_sha256": freeze_manifest_sha256,
        "gate": {
            "interventions": list(INTERVENTIONS),
            "minimum_landed_per_intervention": minimum_landed,
            "minimum_primary_events_per_intervention": minimum_primary_events,
            "uses_onset_time": False,
        },
        "selected_models": selected,
    }
    (output_dir / "selected_checkpoints.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected_path = output_dir / "selected_checkpoints.json"
    lock = {
        "schema_version": 1,
        "file": selected_path.name,
        "sha256": file_hash(selected_path),
    }
    (output_dir / "selected_checkpoints.sha256.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--freeze-manifest",
        help="Required for formal selection; verifies the config and development grid.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="Subset used only for pilot/smoke audits; maintained v6 uses all three.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_manifest(args.manifest)
    formal = config["experiment"]["phase"] == "formal"
    if formal and not args.freeze_manifest:
        raise SystemExit("Formal selection requires --freeze-manifest")
    if args.freeze_manifest:
        verify_frozen_inputs(args.freeze_manifest, [args.config, args.manifest])
    output_dir = Path(args.output)
    if formal and output_dir.exists():
        raise SystemExit(f"Formal selection output already exists: {output_dir}")
    audit, selected = select_models(
        config,
        manifest,
        Path(args.input),
        Path(args.train_root),
        tuple(args.conditions),
    )
    selection = config.get("selection", {})
    write_outputs(
        output_dir,
        audit,
        selected,
        int(selection.get("minimum_landed_per_intervention", MIN_LANDED)),
        int(
            selection.get(
                "minimum_primary_events_per_intervention", MIN_PRIMARY_EVENTS
            )
        ),
        canonical_hash(config),
        file_hash(args.manifest),
        file_hash(args.freeze_manifest) if args.freeze_manifest else None,
    )
    expected_models = len(args.conditions) * len(config["experiment"]["seeds"])
    if len(selected) != expected_models:
        missing = sorted(
            set(
                (condition, int(seed))
                for condition in args.conditions
                for seed in config["experiment"]["seeds"]
            )
            - {(row["condition"], int(row["seed"])) for row in selected}
        )
        raise SystemExit(f"No eligible checkpoint for: {missing}")
    print(output_dir / "selected_checkpoints.json")


if __name__ == "__main__":
    main()
