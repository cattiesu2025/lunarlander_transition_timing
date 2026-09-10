#!/usr/bin/env python3
"""Aggregate the frozen LunarLander study without imputing absent events."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.lunar_lander_braking.config import load_config  # noqa: E402


KEY = ["condition", "seed", "scenario_id", "intervention"]


def read_jsonl(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    if not rows:
        raise ValueError("No episode rows found")
    return pd.DataFrame(rows)


def validate_completeness(
    frame: pd.DataFrame, seeds: list[int], scenario_ids: list[str]
) -> dict[str, Any]:
    required = set(KEY + ["outcome", "primary_observed", "primary_onset_time_s"])
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing episode columns: {sorted(missing_columns)}")
    duplicates = frame.duplicated(KEY, keep=False)
    duplicate_rows = frame.loc[duplicates, KEY].to_dict("records")
    expected = {
        (condition, seed, scenario, intervention)
        for condition in ("DESC", "BAL", "ECON")
        for seed in seeds
        for scenario in scenario_ids
        for intervention in ("original", "low_descent_speed")
    }
    actual = {tuple(row) for row in frame[KEY].itertuples(index=False, name=None)}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    technical = int((frame["outcome"] == "technical_error").sum())
    return {
        "complete": not duplicate_rows and not missing and not unexpected and technical == 0,
        "expected_rows": len(expected),
        "actual_rows": len(frame),
        "missing": [dict(zip(KEY, row)) for row in missing],
        "unexpected": [dict(zip(KEY, row)) for row in unexpected],
        "duplicates": duplicate_rows,
        "technical_errors": technical,
    }


def seed_contrasts(frame: pd.DataFrame, minimum_joint_events: int) -> pd.DataFrame:
    original = frame[frame["intervention"] == "original"].drop_duplicates(KEY, keep=False)
    rows: list[dict[str, Any]] = []
    for seed, seed_frame in original.groupby("seed"):
        desc = seed_frame[seed_frame["condition"] == "DESC"].set_index("scenario_id")
        econ = seed_frame[seed_frame["condition"] == "ECON"].set_index("scenario_id")
        common_ids = sorted(set(desc.index) & set(econ.index))
        observed_ids = [
            scenario for scenario in common_ids
            if bool(desc.at[scenario, "primary_observed"])
            and bool(econ.at[scenario, "primary_observed"])
        ]
        differences = [
            float(econ.at[scenario, "primary_onset_time_s"])
            - float(desc.at[scenario, "primary_onset_time_s"])
            for scenario in observed_ids
        ]
        rows.append({
            "seed": int(seed),
            "joint_events": len(differences),
            "estimable": len(differences) >= minimum_joint_events,
            "median_econ_minus_desc_s": float(np.median(differences)) if differences else np.nan,
        })
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def bootstrap_seed_median(values: np.ndarray, samples: int, seed: int) -> dict[str, float]:
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite seed estimates")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    draws = np.median(values[indices], axis=1)
    return {
        "estimate": float(np.median(values)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
    }


def exact_sign_test(values: np.ndarray) -> dict[str, Any]:
    nonzero = values[values != 0]
    positives = int((nonzero > 0).sum())
    return {
        "positive": positives,
        "negative": int((nonzero < 0).sum()),
        "ties": int((values == 0).sum()),
        "two_sided_p": float(binomtest(positives, len(nonzero), 0.5).pvalue) if len(nonzero) else 1.0,
    }


def intervention_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (condition, seed), group in frame.groupby(["condition", "seed"]):
        original = group[group["intervention"] == "original"].set_index("scenario_id")
        low = group[group["intervention"] == "low_descent_speed"].set_index("scenario_id")
        ids = sorted(set(original.index) & set(low.index))
        transitions = Counter(
            f"{bool(original.at[i, 'primary_observed'])}->{bool(low.at[i, 'primary_observed'])}"
            for i in ids
        )
        common = [i for i in ids if bool(original.at[i, "primary_observed"]) and bool(low.at[i, "primary_observed"])]
        delays = [
            float(low.at[i, "primary_onset_time_s"]) - float(original.at[i, "primary_onset_time_s"])
            for i in common
        ]
        rows.append({
            "condition": condition,
            "seed": int(seed),
            "original_event_rate": float(original.loc[ids, "primary_observed"].mean()),
            "low_speed_event_rate": float(low.loc[ids, "primary_observed"].mean()),
            "event_rate_difference": float(low.loc[ids, "primary_observed"].mean() - original.loc[ids, "primary_observed"].mean()),
            "transitions": dict(sorted(transitions.items())),
            "joint_events": len(common),
            "median_delay_joint_events_s": float(np.median(delays)) if delays else None,
        })
    return rows


def detector_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Report event denominators and onset medians for every stored detector."""
    expanded: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        for detector, result in row.get("detectors", {}).items():
            expanded.append({
                "condition": row["condition"],
                "intervention": row["intervention"],
                "detector": detector,
                "observed": bool(result["observed"]),
                "onset_time_s": result["onset_time_s"],
            })
    if not expanded:
        return []
    detector_frame = pd.DataFrame(expanded)
    rows: list[dict[str, Any]] = []
    for keys, group in detector_frame.groupby(["condition", "intervention", "detector"]):
        observed = group[group["observed"]]
        rows.append({
            "condition": keys[0],
            "intervention": keys[1],
            "detector": keys[2],
            "events": int(group["observed"].sum()),
            "episodes": int(len(group)),
            "median_onset_time_s": float(observed["onset_time_s"].median()) if len(observed) else None,
        })
    return rows


def aggregate(config: dict[str, Any], manifest: list[dict[str, Any]], frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    seeds = [int(value) for value in config["experiment"]["seeds"]]
    scenario_ids = [str(row["scenario_id"]) for row in manifest]
    integrity = validate_completeness(frame, seeds, scenario_ids)
    minimum = int(config["evaluation"]["minimum_joint_events"])
    contrasts = seed_contrasts(frame, minimum)
    all_seeds_estimable = (
        set(contrasts["seed"]) == set(seeds)
        and len(contrasts) == len(seeds)
        and bool(contrasts["estimable"].all())
    )
    confirmatory = None
    sign_test = None
    formal_phase = config["experiment"].get("phase", "formal") == "formal"
    if formal_phase and integrity["complete"] and all_seeds_estimable:
        values = contrasts["median_econ_minus_desc_s"].to_numpy(float)
        confirmatory = bootstrap_seed_median(
            values,
            int(config["evaluation"]["bootstrap_samples"]),
            int(config["evaluation"]["bootstrap_seed"]),
        )
        confirmatory["supports_L1"] = bool(confirmatory["ci_lower"] > 0)
        sign_test = exact_sign_test(values)
    outcome_counts = (
        frame.groupby(["condition", "intervention", "outcome"]).size().rename("count").reset_index().to_dict("records")
    )
    event_counts = (
        frame.groupby(["condition", "intervention"])["primary_observed"]
        .agg(["sum", "count"]).reset_index().to_dict("records")
    )
    summary = {
        "schema_version": 1,
        "analysis_phase": config["experiment"].get("phase", "unspecified"),
        "confirmatory_eligible_phase": formal_phase,
        "integrity": integrity,
        "all_seeds_estimable": all_seeds_estimable,
        "minimum_joint_events": minimum,
        "confirmatory_econ_minus_desc_s": confirmatory,
        "exact_sign_test": sign_test,
        "event_counts": event_counts,
        "detector_comparison": detector_summary(frame),
        "outcome_counts": outcome_counts,
        "intervention": intervention_summary(frame),
        "interpretation": "Confirmatory result is null for pilot data or unless integrity is complete and every declared seed is estimable.",
    }
    return summary, contrasts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True, help="Directory recursively containing episodes.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    with Path(args.manifest).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    paths = sorted(Path(args.input).rglob("episodes.jsonl"))
    frame = read_jsonl(paths)
    summary, contrasts = aggregate(config, manifest, frame)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contrasts.to_csv(output / "seed_contrasts.csv", index=False)
    print(output / "summary.json")


if __name__ == "__main__":
    main()
