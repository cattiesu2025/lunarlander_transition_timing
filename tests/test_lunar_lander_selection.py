from experiments.lunar_lander_braking.selection import (
    checkpoint_gate_summary,
    latest_eligible_checkpoint,
)


def rows(landed=18, events=18, onset_time=0.5):
    values = []
    for intervention in ("original", "low_descent_speed"):
        for index in range(18):
            values.append({
                "condition": "ECON",
                "seed": 101,
                "scenario_id": f"s{index}",
                "intervention": intervention,
                "outcome": "landed" if index < landed else "timeout",
                "primary_observed": index < events,
                "primary_onset_time_s": onset_time,
            })
    return values


def test_gate_requires_landing_and_event_floors_in_both_interventions():
    scenario_ids = [f"s{index}" for index in range(18)]
    assert checkpoint_gate_summary(rows(), scenario_ids, "ECON", 101)["eligible"]
    assert not checkpoint_gate_summary(
        rows(landed=14), scenario_ids, "ECON", 101
    )["eligible"]
    assert not checkpoint_gate_summary(
        rows(events=14), scenario_ids, "ECON", 101
    )["eligible"]


def test_gate_does_not_read_onset_time_values():
    scenario_ids = [f"s{index}" for index in range(18)]
    early = checkpoint_gate_summary(rows(onset_time=-999.0), scenario_ids, "ECON", 101)
    late = checkpoint_gate_summary(rows(onset_time=999.0), scenario_ids, "ECON", 101)
    assert early == late


def test_gate_rejects_incomplete_or_technical_evaluation():
    scenario_ids = [f"s{index}" for index in range(18)]
    incomplete = rows()[:-1]
    assert not checkpoint_gate_summary(
        incomplete, scenario_ids, "ECON", 101
    )["eligible"]
    technical = rows()
    technical[0]["outcome"] = "technical_error"
    assert not checkpoint_gate_summary(
        technical, scenario_ids, "ECON", 101
    )["eligible"]


def test_selection_returns_latest_eligible_checkpoint():
    summaries = [
        {"checkpoint_step": 600_000, "eligible": True},
        {"checkpoint_step": 700_000, "eligible": False},
        {"checkpoint_step": 800_000, "eligible": True},
        {"checkpoint_step": 900_000, "eligible": False},
    ]
    assert latest_eligible_checkpoint(summaries)["checkpoint_step"] == 800_000
    assert latest_eligible_checkpoint(
        [{"checkpoint_step": 1_000_000, "eligible": False}]
    ) is None
