"""Task-validity-only checkpoint gate for LunarLander models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


INTERVENTIONS = ("original", "low_descent_speed")
MIN_LANDED = 15
MIN_PRIMARY_EVENTS = 15


def checkpoint_gate_summary(
    rows: Sequence[Mapping[str, Any]],
    scenario_ids: Sequence[str],
    condition: str,
    seed: int,
    minimum_landed: int = MIN_LANDED,
    minimum_primary_events: int = MIN_PRIMARY_EVENTS,
) -> dict[str, Any]:
    """Summarize task validity without reading any ONSET time value."""
    required = {
        "condition",
        "seed",
        "scenario_id",
        "intervention",
        "outcome",
        "primary_observed",
    }
    expected = {
        (scenario_id, intervention)
        for scenario_id in scenario_ids
        for intervention in INTERVENTIONS
    }
    keys: list[tuple[str, str]] = []
    malformed = 0
    wrong_model = 0
    technical_errors = 0
    for row in rows:
        if not required.issubset(row):
            malformed += 1
            continue
        if row["condition"] != condition or int(row["seed"]) != int(seed):
            wrong_model += 1
        keys.append((str(row["scenario_id"]), str(row["intervention"])))
        technical_errors += int(row["outcome"] == "technical_error")

    duplicates = sum(count - 1 for count in Counter(keys).values() if count > 1)
    actual = set(keys)
    missing = len(expected - actual)
    unexpected = len(actual - expected)
    integrity_complete = not any(
        (malformed, wrong_model, technical_errors, duplicates, missing, unexpected)
    ) and len(rows) == len(expected)

    summary: dict[str, Any] = {
        "condition": condition,
        "seed": int(seed),
        "integrity_complete": integrity_complete,
        "technical_errors": technical_errors,
        "malformed_rows": malformed,
        "wrong_model_rows": wrong_model,
        "duplicate_rows": duplicates,
        "missing_rows": missing,
        "unexpected_rows": unexpected,
    }
    for intervention in INTERVENTIONS:
        selected = [
            row
            for row in rows
            if required.issubset(row)
            and row["condition"] == condition
            and int(row["seed"]) == int(seed)
            and row["intervention"] == intervention
            and (str(row["scenario_id"]), intervention) in expected
        ]
        prefix = intervention
        summary[f"{prefix}_landed"] = sum(
            row["outcome"] == "landed" for row in selected
        )
        summary[f"{prefix}_primary_events"] = sum(
            bool(row["primary_observed"]) for row in selected
        )

    summary["eligible"] = bool(
        integrity_complete
        and all(
            summary[f"{intervention}_landed"] >= minimum_landed
            and summary[f"{intervention}_primary_events"] >= minimum_primary_events
            for intervention in INTERVENTIONS
        )
    )
    return summary


def latest_eligible_checkpoint(
    summaries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Return the newest eligible summary, matching the highway walk-back rule."""
    eligible = [summary for summary in summaries if bool(summary["eligible"])]
    if not eligible:
        return None
    return max(eligible, key=lambda summary: int(summary["checkpoint_step"]))
