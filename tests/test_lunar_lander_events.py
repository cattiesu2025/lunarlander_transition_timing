import pytest

from experiments.lunar_lander_braking.events import (
    DetectorConfig,
    detect_first_fire,
    detect_onset,
    detect_sustained_fire,
)


def records(actions, start_vy=-2.0, gain_per_step=0.01, contact_step=None):
    rows = []
    vy = start_vy
    for step, action in enumerate(actions):
        before = vy
        vy += gain_per_step if action == 2 else -0.002
        contact = step == contact_step
        rows.append({
            "step": step, "time_s": step * 0.02, "action": action,
            "vy_before": before, "vy": vy, "contact_before": False,
            "contact": contact, "crash": False, "out_of_bounds": False,
            "technical_error": False, "terminated": contact, "truncated": False,
        })
    return rows


def test_primary_onset_uses_candidate_time_and_motion_confirmation():
    config = DetectorConfig(window_s=0.30, firing_fraction=0.60, epsilon_v=0.08)
    result = detect_onset(records([2] * 15), config)
    assert result.observed
    assert result.onset_time_s == 0.0
    assert result.confirmation_time_s == pytest.approx(0.30)


def test_single_ignition_is_not_confirmed_but_auxiliary_detects_it():
    rows = records([2] + [0] * 14)
    config = DetectorConfig()
    assert not detect_onset(rows, config).observed
    assert detect_first_fire(rows).observed
    assert not detect_sustained_fire(rows, config).observed


def test_contact_and_incomplete_windows_are_rejected():
    config = DetectorConfig()
    contact = detect_onset(records([2] * 15, contact_step=10), config)
    assert not contact.observed
    assert "invalid_window_state" in contact.candidates[0].rejection_reasons
    incomplete = detect_onset(records([0] * 10 + [2]), config)
    assert "incomplete_window" in incomplete.candidates[0].rejection_reasons


def test_window_must_be_integral_number_of_steps():
    with pytest.raises(ValueError):
        DetectorConfig(window_s=0.31)
