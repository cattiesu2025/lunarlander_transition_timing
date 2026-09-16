#!/usr/bin/env python3
"""Reproduce descriptive checks for the amended v7 held-out evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


CONDITIONS = ("DESC", "BAL", "ECON")
INTERVENTIONS = ("original", "low_descent_speed")
DETECTORS = (
    "first_fire",
    "window_0.20",
    "sustained_fire",
    "window_0.40",
    "effective_braking",
    "effective_braking_window_0.20",
    "effective_braking_window_0.40",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_seed_values(values: list[float]) -> dict:
    """Use the frozen seed bootstrap settings as descriptive uncertainty."""
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(260916)
    draws = np.median(array[rng.integers(0, len(array), (10_000, len(array)))], axis=1)
    return {
        "median_s": float(np.median(array)),
        "descriptive_bootstrap_95_ci_s": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "positive_seeds": int(np.sum(array > 1e-12)),
        "negative_seeds": int(np.sum(array < -1e-12)),
        "tied_seeds": int(np.sum(np.abs(array) <= 1e-12)),
        "seed_medians_s": [float(value) for value in array],
    }


def detector_result(row: dict, detector: str) -> dict:
    result = row["detectors"][detector]
    if not result["observed"]:
        raise ValueError(f"Missing {detector} event in a held-out episode")
    return result


def build_report(root: Path, amendment: Path) -> dict:
    selection_path = root / "selection_amended_original_only" / "selected_checkpoints.json"
    selection = json.loads(selection_path.read_text())
    lock = json.loads(selection_path.with_name("selected_checkpoints.sha256.json").read_text())
    if lock["sha256"] != sha256(selection_path) or selection["amendment_sha256"] != sha256(amendment):
        raise ValueError("Selection lock or protocol amendment hash mismatch")
    if selection.get("schema_version") != 3 or selection.get("gate", {}).get("interventions") != ["original"]:
        raise ValueError("Expected the amended original-only selection")
    selected = {
        (row["condition"], int(row["seed"])): row["checkpoint_sha256"]
        for row in selection["selected_models"]
    }
    seeds = sorted({seed for _, seed in selected})
    expected_models = {(condition, seed) for condition in CONDITIONS for seed in seeds}
    if len(selection["selected_models"]) != len(expected_models) or set(selected) != expected_models:
        raise ValueError("Amended selection is not complete")

    held_out = root / "held_out_eval_amended_original_only"
    episode_paths = sorted(held_out.rglob("episodes.jsonl"))
    rows = [json.loads(line) for path in episode_paths for line in path.open(encoding="utf-8")]
    manifest = json.loads((root / "frozen" / "held_out_persistent_v7.json").read_text())
    scenes = sorted(str(row["scenario_id"]) for row in manifest)
    expected = {
        (condition, seed, scene, intervention)
        for condition, seed in expected_models
        for scene in scenes
        for intervention in INTERVENTIONS
    }
    keys = [
        (row["condition"], int(row["seed"]), str(row["scenario_id"]), row["intervention"])
        for row in rows
    ]
    if len(episode_paths) != len(expected_models) or len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Held-out episodes are missing, duplicated, or unexpected")
    if any(row["outcome"] == "technical_error" for row in rows):
        raise ValueError("Held-out episodes contain a technical error")
    if any(row.get("checkpoint_sha256") != selected[(row["condition"], int(row["seed"]))] for row in rows):
        raise ValueError("Held-out checkpoint hash does not match the amended selection")
    if any(row.get("config_hash") != selection["config_hash"] for row in rows):
        raise ValueError("Held-out config hash does not match the amended selection")
    by_key = dict(zip(keys, rows))

    contrasts = {}
    for intervention in INTERVENTIONS:
        contrasts[intervention] = {}
        for detector in DETECTORS:
            seed_values = []
            joint_events = []
            for seed in seeds:
                differences = []
                for scene in scenes:
                    desc = detector_result(by_key["DESC", seed, scene, intervention], detector)
                    econ = detector_result(by_key["ECON", seed, scene, intervention], detector)
                    differences.append(econ["onset_time_s"] - desc["onset_time_s"])
                joint_events.append(len(differences))
                seed_values.append(float(np.median(differences)))
            contrasts[intervention][detector] = {
                **summarize_seed_values(seed_values),
                "joint_events_per_seed": joint_events,
                "events_per_condition": {
                    condition: sum(
                        detector_result(row, detector)["observed"]
                        for row in rows
                        if row["condition"] == condition and row["intervention"] == intervention
                    )
                    for condition in CONDITIONS
                },
            }

    intervention_delays = {}
    endpoint_diagnostics = {}
    outcomes = {}
    for condition in CONDITIONS:
        seed_values = []
        for seed in seeds:
            differences = [
                detector_result(by_key[condition, seed, scene, "low_descent_speed"], "sustained_fire")["onset_time_s"]
                - detector_result(by_key[condition, seed, scene, "original"], "sustained_fire")["onset_time_s"]
                for scene in scenes
            ]
            seed_values.append(float(np.median(differences)))
        intervention_delays[condition] = {
            **summarize_seed_values(seed_values),
            "joint_events": len(seeds) * len(scenes),
        }

        original = [row for row in rows if row["condition"] == condition and row["intervention"] == "original"]
        speed_reductions = []
        effective_minus_sustained = []
        same_onset = 0
        for row in original:
            sustained = detector_result(row, "sustained_fire")
            effective = detector_result(row, "effective_braking")
            candidate = next(item for item in sustained["candidates"] if item["step"] == sustained["onset_step"])
            speed_reductions.append(float(candidate["speed_reduction"]))
            effective_minus_sustained.append(effective["onset_time_s"] - sustained["onset_time_s"])
            same_onset += effective["onset_step"] == sustained["onset_step"]
        endpoint_diagnostics[condition] = {
            "same_sustained_and_effective_onset": same_onset,
            "speed_reduction_at_sustained_onset_ge_0_40": sum(value >= 0.40 for value in speed_reductions),
            "median_speed_reduction_at_sustained_onset": float(np.median(speed_reductions)),
            "median_effective_minus_sustained_s": float(np.median(effective_minus_sustained)),
        }
        outcomes[condition] = {
            intervention: dict(Counter(
                row["outcome"] for row in rows
                if row["condition"] == condition and row["intervention"] == intervention
            ))
            for intervention in INTERVENTIONS
        }

    aggregate_path = root / "aggregate_amended_original_only" / "summary.json"
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("protocol_amendment", {}).get("selection_manifest_sha256") != sha256(selection_path):
        raise ValueError("Aggregate does not reference the amended selection")
    official = aggregate["confirmatory_econ_minus_desc_s"]["estimate"]
    reproduced = contrasts["original"]["sustained_fire"]["median_s"]
    if not np.isclose(official, reproduced, atol=1e-12):
        raise ValueError("Recomputed primary contrast differs from the frozen aggregate")
    episode_digest = hashlib.sha256()
    for path in episode_paths:
        episode_digest.update(str(path.relative_to(held_out)).encode("utf-8"))
        episode_digest.update(b"\0")
        episode_digest.update(sha256(path).encode("ascii"))
        episode_digest.update(b"\n")
    return {
        "analysis_status": "Supplementary descriptive checks; only the original-scenario sustained-fire contrast is the amended v7 primary result",
        "source_sha256": {
            "aggregate_summary": sha256(aggregate_path),
            "selection_manifest": sha256(selection_path),
            "protocol_amendment": sha256(amendment),
            "episode_files_combined": episode_digest.hexdigest(),
        },
        "integrity": {"models": len(expected_models), "episode_files": len(episode_paths), "episodes": len(rows), "scenes_per_model_intervention": len(scenes)},
        "econ_minus_desc_contrasts": contrasts,
        "low_speed_minus_original_sustained_fire_delays": intervention_delays,
        "original_endpoint_diagnostics": endpoint_diagnostics,
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/lunar_lander_braking_formal_v3_persistent_onset"))
    parser.add_argument("--amendment", type=Path, default=Path("experiments/lunar_lander_braking/amendment_v7_original_only.md"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root, args.amendment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
