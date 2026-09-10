"""CLI for calibration probes, freezing, training, and fixed-model evaluation."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Callable

from .agent import DoubleDQNAgent, train_condition
from .config import canonical_hash, file_hash, load_config, load_manifest, write_freeze_bundle
from .env import Scenario, make_env, rollout
from .events import DetectorConfig, detect_first_fire, detect_onset, detect_sustained_fire


PACKAGE = Path(__file__).resolve().parent


def verify_frozen_inputs(bundle_path: str | Path, paths: list[str | Path]) -> None:
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    expected = bundle.get("files", {})
    for value in paths:
        path = Path(value)
        if path.name not in expected:
            raise ValueError(f"Frozen bundle does not declare {path.name}")
        if file_hash(path) != expected[path.name]:
            raise ValueError(f"Frozen input hash mismatch: {path}")


def detector_configs(config: dict[str, Any]) -> dict[str, DetectorConfig]:
    primary = config["detector"]["primary"]
    values = {"primary": DetectorConfig(**primary)}
    for window in config["detector"]["sensitivity_windows_s"]:
        values[f"window_{float(window):.2f}"] = DetectorConfig(
            window_s=float(window),
            firing_fraction=float(primary["firing_fraction"]),
            epsilon_v=float(primary["epsilon_v"]),
        )
    return values


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def command_freeze(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    formal = config["experiment"]["phase"] == "formal"
    if formal and not args.confirm_calibrated:
        raise SystemExit("Formal freeze requires --confirm-calibrated after documented probe/pilot review")
    if formal and (not args.pip_lock or not (args.conda_lock or args.python_lock)):
        raise SystemExit(
            "Formal freeze requires --pip-lock and either --conda-lock or --python-lock"
        )
    inputs = [args.development_manifest, args.held_out_manifest, PACKAGE / "protocol.md"]
    project_root = PACKAGE.parents[1]
    inputs.extend((project_root / "environment.yml", project_root / "requirements.txt"))
    if args.conda_lock:
        inputs.append(args.conda_lock)
    if args.python_lock:
        inputs.append(args.python_lock)
    if args.pip_lock:
        inputs.append(args.pip_lock)
    bundle = write_freeze_bundle(
        args.config,
        inputs,
        args.output,
    )
    print(json.dumps(bundle, indent=2, sort_keys=True))


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    formal = config["experiment"]["phase"] == "formal"
    if formal and not args.freeze_manifest:
        raise SystemExit("Formal training requires --freeze-manifest")
    if args.freeze_manifest:
        verify_frozen_inputs(args.freeze_manifest, [args.config])
    if args.seed not in config["experiment"]["seeds"] and not args.allow_smoke_seed:
        raise SystemExit(f"Seed {args.seed} is not declared; use --allow-smoke-seed only for smoke tests")
    checkpoint = train_condition(config, args.condition, args.seed, args.output, args.device, args.steps)
    print(checkpoint)


def _evaluate_rollout(
    config: dict[str, Any], checkpoint: Path, condition: str, seed: int,
    scenario: Scenario, intervention: str, output: Path,
) -> dict[str, Any]:
    external_metadata_path = checkpoint.with_name("metadata.json")
    if not external_metadata_path.exists():
        raise ValueError(f"Missing checkpoint metadata: {external_metadata_path}")
    external_metadata = json.loads(external_metadata_path.read_text(encoding="utf-8"))
    checkpoint_hash = file_hash(checkpoint)
    if external_metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Checkpoint SHA-256 mismatch")
    agent, metadata = DoubleDQNAgent.load(checkpoint)
    if metadata["condition"] != condition or int(metadata["seed"]) != seed:
        raise ValueError("Checkpoint metadata does not match requested condition/seed")
    if metadata["config_hash"] != canonical_hash(config):
        raise ValueError("Checkpoint/config hash mismatch")
    env = make_env(config, condition)
    result = rollout(env, scenario, lambda obs: agent.act(obs, greedy=True))
    env.close()
    configs = detector_configs(config)
    primary = detect_onset(result["records"], configs["primary"])
    detectors: dict[str, Any] = {
        "primary": primary.to_dict(),
        "first_fire": detect_first_fire(result["records"]).to_dict(),
        "sustained_fire": detect_sustained_fire(result["records"], configs["primary"]).to_dict(),
    }
    for label, detector_config in configs.items():
        if label != "primary":
            detectors[label] = detect_onset(result["records"], detector_config).to_dict()
    trajectory = output / "trajectories" / f"{scenario.scenario_id}_{intervention}.jsonl.gz"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(trajectory, "wt", encoding="utf-8") as handle:
        for row in result["records"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "schema_version": 1,
        "condition": condition,
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "intervention": intervention,
        "outcome": result["outcome"],
        "custom_return": result["custom_return"],
        "native_return_diagnostic": result["native_return_diagnostic"],
        "primary_observed": primary.observed,
        "primary_onset_time_s": primary.onset_time_s,
        "detectors": detectors,
        "checkpoint_sha256": checkpoint_hash,
        "config_hash": canonical_hash(config),
        "trajectory": str(trajectory),
    }


def command_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    manifest = load_manifest(args.manifest)
    formal = config["experiment"]["phase"] == "formal"
    if formal and not args.freeze_manifest:
        raise SystemExit("Formal evaluation requires --freeze-manifest")
    if args.freeze_manifest:
        verify_frozen_inputs(args.freeze_manifest, [args.config, args.manifest])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    technical_errors = 0
    factor = float(config["evaluation"]["intervention_speed_factor"])
    for manifest_row in manifest:
        original = Scenario.from_dict(manifest_row)
        for intervention, scenario in (
            ("original", original),
            ("low_descent_speed", Scenario(**{**original.__dict__, "vy": original.vy * factor})),
        ):
            try:
                row = _evaluate_rollout(
                    config, Path(args.checkpoint), args.condition, args.seed,
                    scenario, intervention, output,
                )
            except Exception as error:
                technical_errors += 1
                row = {
                    "schema_version": 1,
                    "condition": args.condition,
                    "seed": args.seed,
                    "scenario_id": scenario.scenario_id,
                    "intervention": intervention,
                    "outcome": "technical_error",
                    "primary_observed": False,
                    "primary_onset_time_s": None,
                    "detectors": {},
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "config_hash": canonical_hash(config),
                }
            rows.append(row)
            for label, detection in row["detectors"].items():
                for candidate in detection["candidates"]:
                    candidate_rows.append({
                        "condition": args.condition, "seed": args.seed,
                        "scenario_id": scenario.scenario_id, "intervention": intervention,
                        "detector": label, **candidate,
                    })
    _write_jsonl(output / "episodes.jsonl", rows)
    _write_jsonl(output / "candidates.jsonl", candidate_rows)
    print(output / "episodes.jsonl")
    if technical_errors:
        raise SystemExit(f"Evaluation recorded {technical_errors} technical errors")


def command_probe(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    manifest = load_manifest(args.manifest)
    selected = manifest if args.scenario_index is None else [manifest[args.scenario_index]]
    output = Path(args.output)
    rows = []
    primary_config = detector_configs(config)["primary"]
    for manifest_row in selected:
        scenario = Scenario.from_dict(manifest_row)
        fired = {"value": False}
        def single_fire(_):
            action = 0 if fired["value"] else 2
            fired["value"] = True
            return action
        policies: dict[str, Callable[[Any], int]] = {
            "no_fire": lambda _: 0,
            "sustained_main": lambda _: 2,
            "single_fire": single_fire,
        }
        for name, policy in policies.items():
            env = make_env(config, "BAL")
            result = rollout(env, scenario, policy)
            env.close()
            detection = detect_onset(result["records"], primary_config)
            window = result["records"][: primary_config.window_steps]
            window_reduction = None
            if len(window) == primary_config.window_steps:
                window_reduction = max(0.0, -window[0]["vy_before"]) - max(0.0, -window[-1]["vy"])
            rows.append({
                "probe": name, "scenario": scenario.__dict__, "outcome": result["outcome"],
                "detection": detection.to_dict(),
                "initial_vy": result["records"][0]["vy_before"],
                "final_vy": result["records"][-1]["vy"],
                "first_window_speed_reduction": window_reduction,
            })
    _write_jsonl(output, rows)
    reductions = {
        name: [row["first_window_speed_reduction"] for row in rows if row["probe"] == name and row["first_window_speed_reduction"] is not None]
        for name in ("no_fire", "sustained_main", "single_fire")
    }
    separated = bool(reductions["no_fire"] and reductions["sustained_main"] and max(reductions["no_fire"]) < min(reductions["sustained_main"]))
    summary = {
        "scenarios": len(selected),
        "configured_epsilon_v": primary_config.epsilon_v,
        "speed_reduction_ranges": {
            name: {"minimum": min(values), "maximum": max(values)} if values else None
            for name, values in reductions.items()
        },
        "no_fire_vs_sustained_separated": separated,
        "nonnegative_separation_midpoint": (
            (max(0.0, max(reductions["no_fire"])) + min(reductions["sustained_main"])) / 2
            if separated else None
        ),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


def reference_policy(observation) -> int:
    """Gymnasium's documented discrete heuristic for recoverability screening."""
    angle_target = min(
        0.4,
        max(-0.4, float(observation[0]) * 0.5 + float(observation[2])),
    )
    hover_target = 0.55 * abs(float(observation[0]))
    angle_todo = (angle_target - float(observation[4])) * 0.5 - float(observation[5])
    hover_todo = (
        (hover_target - float(observation[1])) * 0.5
        - float(observation[3]) * 0.5
    )
    if observation[6] or observation[7]:
        angle_todo = 0.0
        hover_todo = -float(observation[3]) * 0.5
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2
    if angle_todo < -0.05:
        return 3
    if angle_todo > 0.05:
        return 1
    return 0


def command_recoverability(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    rows: list[dict[str, Any]] = []
    primary_config = detector_configs(config)["primary"]
    for manifest_row in load_manifest(args.manifest):
        scenario = Scenario.from_dict(manifest_row)
        env = make_env(config, "BAL")
        result = rollout(env, scenario, reference_policy)
        env.close()
        detection = detect_onset(result["records"], primary_config)
        rows.append({
            "scenario_id": scenario.scenario_id,
            "outcome": result["outcome"],
            "primary_observed": detection.observed,
            "primary_onset_time_s": detection.onset_time_s,
        })
    output = Path(args.output)
    _write_jsonl(output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    summary = {
        "scenarios": len(rows),
        "outcome_counts": counts,
        "primary_events": sum(row["primary_observed"] for row in rows),
        "immediate_events": sum(row["primary_onset_time_s"] == 0.0 for row in rows),
        "interpretation": "The reference controller is a recoverability screen, not an optimality proof.",
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common_config = str(PACKAGE / "configs" / "pilot.yaml")
    common_manifest = str(PACKAGE / "grids" / "development.json")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--config", default=common_config)
    freeze.add_argument("--development-manifest", default=common_manifest)
    freeze.add_argument("--held-out-manifest", default=str(PACKAGE / "grids" / "held_out.json"))
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--confirm-calibrated", action="store_true")
    freeze.add_argument("--conda-lock", help="Output of `conda list --explicit`")
    freeze.add_argument("--python-lock", help="Python module/version record for a venv")
    freeze.add_argument("--pip-lock", help="Output of `python -m pip freeze`")
    freeze.set_defaults(func=command_freeze)
    train = sub.add_parser("train")
    train.add_argument("--config", default=common_config)
    train.add_argument("--condition", choices=["DESC", "BAL", "ECON"], required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--steps", type=int)
    train.add_argument("--allow-smoke-seed", action="store_true")
    train.add_argument("--freeze-manifest", help="Path to frozen manifest.sha256.json")
    train.set_defaults(func=command_train)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", default=common_config)
    evaluate.add_argument("--manifest", default=str(PACKAGE / "grids" / "held_out.json"))
    evaluate.add_argument("--condition", choices=["DESC", "BAL", "ECON"], required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--freeze-manifest", help="Path to frozen manifest.sha256.json")
    evaluate.set_defaults(func=command_evaluate)
    probe = sub.add_parser("probe")
    probe.add_argument("--config", default=common_config)
    probe.add_argument("--manifest", default=common_manifest)
    probe.add_argument("--scenario-index", type=int, help="Omit to calibrate over the full manifest")
    probe.add_argument("--output", required=True)
    probe.set_defaults(func=command_probe)
    recoverability = sub.add_parser("recoverability")
    recoverability.add_argument("--config", default=common_config)
    recoverability.add_argument("--manifest", default=common_manifest)
    recoverability.add_argument("--output", required=True)
    recoverability.set_defaults(func=command_recoverability)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
