import importlib.util
from pathlib import Path

import pandas as pd


SPEC = importlib.util.spec_from_file_location("aggregate_lunar", Path("scripts/aggregate_lunar_lander.py"))
aggregate_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate_module)


def synthetic_frame():
    rows = []
    for condition in ("DESC", "BAL", "ECON"):
        for seed in (1, 2):
            for scenario in ("a", "b"):
                for intervention in ("original", "low_descent_speed"):
                    onset = 1.0
                    if condition == "ECON":
                        onset += 0.4 + 0.1 * seed
                    if intervention == "low_descent_speed":
                        onset += 0.2
                    rows.append({
                        "condition": condition, "seed": seed, "scenario_id": scenario,
                        "intervention": intervention, "outcome": "landed",
                        "primary_observed": True, "primary_onset_time_s": onset,
                    })
    return pd.DataFrame(rows)


def test_complete_seed_level_estimand_and_bootstrap():
    frame = synthetic_frame()
    config = {
        "experiment": {"seeds": [1, 2]},
        "evaluation": {"minimum_joint_events": 2, "bootstrap_samples": 200, "bootstrap_seed": 9},
    }
    summary, contrasts = aggregate_module.aggregate(config, [{"scenario_id": "a"}, {"scenario_id": "b"}], frame)
    assert summary["integrity"]["complete"]
    assert summary["all_seeds_estimable"]
    assert summary["confirmatory_econ_minus_desc_s"]["estimate"] == 0.55
    assert contrasts["joint_events"].tolist() == [2, 2]


def test_missing_episode_blocks_confirmatory_result():
    frame = synthetic_frame().iloc[:-1]
    config = {
        "experiment": {"seeds": [1, 2]},
        "evaluation": {"minimum_joint_events": 2, "bootstrap_samples": 20, "bootstrap_seed": 9},
    }
    summary, _ = aggregate_module.aggregate(config, [{"scenario_id": "a"}, {"scenario_id": "b"}], frame)
    assert not summary["integrity"]["complete"]
    assert summary["confirmatory_econ_minus_desc_s"] is None


def test_complete_pilot_is_never_confirmatory():
    frame = synthetic_frame()
    config = {
        "experiment": {"seeds": [1, 2], "phase": "pilot"},
        "evaluation": {"minimum_joint_events": 2, "bootstrap_samples": 20, "bootstrap_seed": 9},
    }
    summary, _ = aggregate_module.aggregate(
        config, [{"scenario_id": "a"}, {"scenario_id": "b"}], frame
    )
    assert summary["integrity"]["complete"]
    assert not summary["confirmatory_eligible_phase"]
    assert summary["confirmatory_econ_minus_desc_s"] is None
