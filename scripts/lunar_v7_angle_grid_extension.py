#!/usr/bin/env python3
"""Run six additional v7 angles and aggregate the complete nine-angle grid."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STUDY = Path("outputs/lunar_lander_braking_formal_v3_persistent_onset")
PRIOR_EXTENSION = STUDY / "v7_tilt_extension"
EXTENSION = STUDY / "v7_angle_grid_extension"
CONFIG = Path("experiments/lunar_lander_braking/configs/formal_v7_persistent_onset.yaml")
BASE_GRID = Path("experiments/lunar_lander_braking/grids/held_out_persistent_v7.json")
PROTOCOL = Path("experiments/lunar_lander_braking/protocol_v7_angle_grid_extension.md")
SELECTION = STUDY / "selection_amended_original_only/selected_checkpoints.json"
SELECTION_LOCK = SELECTION.with_name("selected_checkpoints.sha256.json")
FREEZE = STUDY / "frozen/manifest.sha256.json"
AMENDMENT = Path("experiments/lunar_lander_braking/amendment_v7_original_only.md")

NEW_ANGLES = (-3, -2, -1, 1, 2, 3)
ALL_ANGLES = tuple(range(-4, 5))
CONDITIONS = ("DESC", "BAL", "ECON")
INTERVENTIONS = ("original", "low_descent_speed")
SEEDS = tuple(range(2001, 2021))
BOOTSTRAP_SEED = 260916
BOOTSTRAP_SAMPLES = 10_000


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def suffix(degrees: int) -> str:
    return ("m" if degrees < 0 else "p" if degrees > 0 else "z") + f"{abs(degrees):02d}"


def split_scenario_id(scenario_id: str) -> tuple[str, int]:
    base, encoded = scenario_id.rsplit("_tilt_", 1)
    sign = -1 if encoded[0] == "m" else 1
    return base, sign * int(encoded[1:])


def build_grid(base: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(base) != 18 or len({row["scenario_id"] for row in base}) != 18:
        raise ValueError("Expected 18 unique base scenarios")
    result = []
    for row in base:
        if row["angle"] != 0 or row["angular_velocity"] != 0:
            raise ValueError("Expected upright base scenarios with zero angular velocity")
        for degrees in NEW_ANGLES:
            result.append(
                {
                    **row,
                    "scenario_id": f"{row['scenario_id']}_tilt_{suffix(degrees)}",
                    "angle": math.radians(degrees),
                }
            )
    return result


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare() -> None:
    prior_lock = PRIOR_EXTENSION / "manifest.sha256.json"
    prior_diagnostics = list(PRIOR_EXTENSION.glob("eval/*/seed_*/posture_diagnostics.json"))
    if not prior_lock.is_file() or len(prior_diagnostics) != 60:
        raise ValueError("Complete three-angle v7_tilt_extension results are required")
    inputs = [
        CONFIG, BASE_GRID, PROTOCOL, SELECTION, SELECTION_LOCK, FREEZE, AMENDMENT,
        Path(__file__).relative_to(ROOT), prior_lock,
    ]
    hashes = {str(path): sha(path) for path in inputs}
    grid = build_grid(json.loads(BASE_GRID.read_text(encoding="utf-8")))
    EXTENSION.mkdir(parents=True, exist_ok=False)
    manifest = EXTENSION / "angle_grid_scenarios.json"
    write_json(manifest, grid)
    hashes[str(manifest)] = sha(manifest)
    write_json(
        EXTENSION / "manifest.sha256.json",
        {
            "study_version": "v7",
            "status": "post-result exploratory nine-angle extension",
            "new_angles_degrees": list(NEW_ANGLES),
            "combined_angles_degrees": list(ALL_ANGLES),
            "files": {CONFIG.name: sha(CONFIG), manifest.name: sha(manifest)},
            "source_files": hashes,
        },
    )
    print(f"Prepared {len(grid)} new matched scenarios at {EXTENSION}")


def posture_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    airborne = []
    for row in records:
        if row.get("contact_before") or row.get("contact"):
            break
        airborne.append(row)
    if not airborne:
        return {"precontact_steps": 0}
    positions = [airborne[0]["x_before"], *(row["x"] for row in airborne)]
    return {
        "precontact_steps": len(airborne),
        "precontact_side_actions": sum(row["action"] in (1, 3) for row in airborne),
        "precontact_peak_abs_angle_deg": max(abs(math.degrees(row["angle"])) for row in airborne),
        "precontact_final_angle_deg": math.degrees(airborne[-1]["angle"]),
        "precontact_x_range": max(positions) - min(positions),
    }


def evaluate(condition: str, seed: int) -> None:
    lock_path = EXTENSION / "manifest.sha256.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for path, expected in lock["source_files"].items():
        if sha(path) != expected:
            raise ValueError(f"Extension input changed: {path}")
    from scripts.resolve_lunar_checkpoint import resolve_checkpoint
    from experiments.lunar_lander_braking.run import command_evaluate

    checkpoint = resolve_checkpoint(
        CONFIG, SELECTION, SELECTION_LOCK, FREEZE, condition, seed, AMENDMENT
    )
    output = EXTENSION / "eval" / condition / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "provenance.json",
        {"extension_lock_sha256": sha(lock_path), "checkpoint_sha256": sha(checkpoint)},
    )
    command_evaluate(
        argparse.Namespace(
            config=CONFIG,
            manifest=EXTENSION / "angle_grid_scenarios.json",
            freeze_manifest=lock_path,
            checkpoint=checkpoint,
            condition=condition,
            seed=seed,
            output=output,
        )
    )
    compact = []
    for line in (output / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        episode = json.loads(line)
        with gzip.open(episode["trajectory"], "rt", encoding="utf-8") as handle:
            records = [json.loads(item) for item in handle]
        compact.append(
            {
                key: episode[key]
                for key in (
                    "condition", "seed", "scenario_id", "intervention", "outcome",
                    "primary_observed", "primary_onset_time_s", "checkpoint_sha256",
                )
            }
            | posture_summary(records)
        )
    write_json(output / "posture_diagnostics.json", compact)


def projected_rows(path: Path) -> list[dict[str, Any]]:
    keep = {
        "condition", "seed", "scenario_id", "intervention", "outcome",
        "primary_observed", "primary_onset_time_s", "precontact_steps",
        "precontact_side_actions", "precontact_peak_abs_angle_deg",
        "precontact_final_angle_deg", "precontact_x_range",
    }
    return [{key: row[key] for key in keep if key in row} for row in json.loads(path.read_text())]


def load_combined_rows() -> list[dict[str, Any]]:
    rows = []
    for root in (PRIOR_EXTENSION, EXTENSION):
        paths = sorted(root.glob("eval/*/seed_*/posture_diagnostics.json"))
        if len(paths) != 60:
            raise ValueError(f"Expected 60 completed model evaluations under {root}; found {len(paths)}")
        lock_hash = sha(root / "manifest.sha256.json")
        provenance = sorted(root.glob("eval/*/seed_*/provenance.json"))
        if len(provenance) != 60 or any(
            json.loads(path.read_text())["extension_lock_sha256"] != lock_hash
            for path in provenance
        ):
            raise ValueError(f"Provenance mismatch under {root}")
        for path in paths:
            rows.extend(projected_rows(path))
    for row in rows:
        row["base_scene"], row["angle_degrees"] = split_scenario_id(row["scenario_id"])
    keys = [
        (row["condition"], row["seed"], row["scenario_id"], row["intervention"])
        for row in rows
    ]
    expected = len(CONDITIONS) * len(SEEDS) * 18 * len(ALL_ANGLES) * len(INTERVENTIONS)
    if len(rows) != expected or len(set(keys)) != expected:
        raise ValueError(f"Expected {expected} unique combined episodes; found {len(rows)}")
    if {row["angle_degrees"] for row in rows} != set(ALL_ANGLES):
        raise ValueError("Combined results do not contain the complete nine-angle grid")
    if any(row["outcome"] == "technical_error" for row in rows):
        raise ValueError("Combined results contain technical errors")
    return rows


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def contrast(rows: list[dict[str, Any]], intervention: str, angles: tuple[int, ...]) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["intervention"] == intervention and row["angle_degrees"] in angles
    ]
    lookup = {
        (row["condition"], row["seed"], row["base_scene"], row["angle_degrees"]): row
        for row in selected
    }
    seed_medians = []
    joint_counts = []
    scenes = sorted({row["base_scene"] for row in selected})
    for seed in SEEDS:
        differences = []
        for scene in scenes:
            for angle in angles:
                desc = lookup[("DESC", seed, scene, angle)]
                econ = lookup[("ECON", seed, scene, angle)]
                if desc["primary_observed"] and econ["primary_observed"]:
                    differences.append(econ["primary_onset_time_s"] - desc["primary_onset_time_s"])
        joint_counts.append(len(differences))
        if not differences:
            raise ValueError(f"No joint events for seed {seed}")
        seed_medians.append(statistics.median(differences))
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap = [
        statistics.median(rng.choices(seed_medians, k=len(seed_medians)))
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return {
        "angles_degrees": list(angles),
        "estimate_s": statistics.median(seed_medians),
        "ci95_s": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "positive_negative_zero_seeds": [
            sum(value > 0 for value in seed_medians),
            sum(value < 0 for value in seed_medians),
            sum(value == 0 for value in seed_medians),
        ],
        "joint_events_per_seed": joint_counts,
        "seed_medians_s": seed_medians,
    }


def aggregate() -> None:
    rows = load_combined_rows()
    output = EXTENSION / "aggregate"
    output.mkdir(parents=True, exist_ok=False)
    contrasts = {}
    outcomes = {}
    for intervention in INTERVENTIONS:
        contrasts[intervention] = {
            "combined": contrast(rows, intervention, ALL_ANGLES),
            "by_angle": {
                str(angle): contrast(rows, intervention, (angle,)) for angle in ALL_ANGLES
            },
        }
        outcomes[intervention] = {}
        for angle in ALL_ANGLES:
            outcomes[intervention][str(angle)] = {}
            for condition in CONDITIONS:
                group = [
                    row for row in rows
                    if row["intervention"] == intervention
                    and row["angle_degrees"] == angle
                    and row["condition"] == condition
                ]
                outcomes[intervention][str(angle)][condition] = {
                    "episodes": len(group),
                    "primary_observed": sum(row["primary_observed"] for row in group),
                    "outcomes": dict(Counter(row["outcome"] for row in group)),
                }
    summary = {
        "schema_version": 1,
        "status": "post-result exploratory nine-angle extension",
        "episodes": len(rows),
        "technical_errors": 0,
        "angles_degrees": list(ALL_ANGLES),
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
        "contrasts": contrasts,
        "outcomes": outcomes,
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# v7 nine-angle extension results",
        "",
        "Status: post-result exploratory extension. The three-angle and six-angle runs are combined",
        "hierarchically: matched scene-angle differences are summarized within each training seed,",
        "then the 20 seed medians are summarized across seeds.",
        "",
        f"Integrity: {len(rows):,} unique episodes; 0 technical errors.",
        "",
        "| Initial angle | Original ECON-DESC (s) | Half-speed ECON-DESC (s) |",
        "| ---: | ---: | ---: |",
    ]
    for angle in ALL_ANGLES:
        original = contrasts["original"]["by_angle"][str(angle)]
        low = contrasts["low_descent_speed"]["by_angle"][str(angle)]
        lines.append(
            f"| {angle:+d}° | {original['estimate_s']:+.3f} "
            f"[{original['ci95_s'][0]:+.3f}, {original['ci95_s'][1]:+.3f}] | "
            f"{low['estimate_s']:+.3f} [{low['ci95_s'][0]:+.3f}, {low['ci95_s'][1]:+.3f}] |"
        )
    lines.extend(["", "## Combined nine-angle estimate", ""])
    for intervention, label in (
        ("original", "Original descent speed"),
        ("low_descent_speed", "Half initial descent speed"),
    ):
        value = contrasts[intervention]["combined"]
        signs = value["positive_negative_zero_seeds"]
        lines.append(
            f"- {label}: ECON-DESC {value['estimate_s']:+.3f} s "
            f"(descriptive 95% seed-bootstrap interval "
            f"[{value['ci95_s'][0]:+.3f}, {value['ci95_s'][1]:+.3f}]); "
            f"seed directions {signs[0]}/{signs[1]}/{signs[2]} positive/negative/zero."
        )
    lines.extend(
        [
            "",
            "## Outcome and event denominators",
            "",
            "| Intervention | Angle | Condition | Primary | Landed | Crash | Timeout | Out of bounds |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for intervention in INTERVENTIONS:
        for angle in ALL_ANGLES:
            for condition in CONDITIONS:
                value = outcomes[intervention][str(angle)][condition]
                counts = value["outcomes"]
                lines.append(
                    f"| {intervention} | {angle:+d}° | {condition} | "
                    f"{value['primary_observed']}/{value['episodes']} | "
                    f"{counts.get('landed', 0)} | {counts.get('crash', 0)} | "
                    f"{counts.get('timeout', 0)} | {counts.get('out_of_bounds', 0)} |"
                )
    lines.extend(
        [
            "",
            "The extension does not create a new independent confirmatory test. Outcome and event",
            "denominators in `summary.json` must be reported alongside timing estimates.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--condition", choices=CONDITIONS, required=True)
    evaluation.add_argument("--seed", type=int, choices=SEEDS, required=True)
    commands.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "evaluate":
        evaluate(args.condition, args.seed)
    else:
        aggregate()


if __name__ == "__main__":
    main()
