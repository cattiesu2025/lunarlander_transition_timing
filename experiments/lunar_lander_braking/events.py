"""Powered-braking ONSET and auxiliary event detectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


MAIN_ENGINE_ACTION = 2


@dataclass(frozen=True)
class DetectorConfig:
    window_s: float = 0.30
    firing_fraction: float = 0.60
    epsilon_v: float = 0.08
    dt: float = 0.02

    def __post_init__(self) -> None:
        if self.window_s <= 0 or self.dt <= 0:
            raise ValueError("window_s and dt must be positive")
        if not 0 < self.firing_fraction <= 1:
            raise ValueError("firing_fraction must lie in (0, 1]")
        exact = self.window_s / self.dt
        if abs(exact - round(exact)) > 1e-9:
            raise ValueError("window_s must be an integer multiple of dt")
        if self.epsilon_v < 0:
            raise ValueError("epsilon_v must be non-negative")

    @property
    def window_steps(self) -> int:
        return round(self.window_s / self.dt)


@dataclass
class Candidate:
    step: int
    onset_time_s: float
    confirmation_time_s: float | None
    firing_fraction: float | None
    speed_reduction: float | None
    confirmed: bool
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    detector: str
    observed: bool
    onset_step: int | None
    onset_time_s: float | None
    confirmation_time_s: float | None
    candidates: list[Candidate]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["candidates"] = [asdict(candidate) for candidate in self.candidates]
        return value


def _is_invalid(record: dict[str, Any]) -> bool:
    return bool(
        record.get("contact_before")
        or record.get("contact")
        or record.get("crash")
        or record.get("out_of_bounds")
        or record.get("technical_error")
    )


def detect_onset(
    records: Iterable[dict[str, Any]], config: DetectorConfig
) -> DetectionResult:
    steps = list(records)
    candidates: list[Candidate] = []
    first_confirmed: Candidate | None = None
    n = config.window_steps
    for index, record in enumerate(steps):
        if int(record.get("action", -1)) != MAIN_ENGINE_ACTION:
            continue
        reasons: list[str] = []
        if float(record.get("vy_before", 0.0)) >= 0:
            reasons.append("not_descending")
        if record.get("contact_before"):
            reasons.append("contact_at_candidate")
        window = steps[index : index + n]
        if len(window) < n:
            reasons.append("incomplete_window")
            firing = None
            reduction = None
            confirmation = None
        else:
            firing = sum(int(item.get("action", -1)) == MAIN_ENGINE_ACTION for item in window) / n
            start_down = max(0.0, -float(record["vy_before"]))
            end_down = max(0.0, -float(window[-1]["vy"]))
            reduction = start_down - end_down
            confirmation = float(record["time_s"]) + config.window_s
            if firing < config.firing_fraction:
                reasons.append("insufficient_firing_fraction")
            if reduction < config.epsilon_v:
                reasons.append("insufficient_speed_reduction")
            if any(_is_invalid(item) for item in window):
                reasons.append("invalid_window_state")
            if any(item.get("terminated") or item.get("truncated") for item in window[:-1]):
                reasons.append("window_ends_after_episode")
        candidate = Candidate(
            step=int(record["step"]),
            onset_time_s=float(record["time_s"]),
            confirmation_time_s=confirmation,
            firing_fraction=firing,
            speed_reduction=reduction,
            confirmed=not reasons,
            rejection_reasons=reasons,
        )
        candidates.append(candidate)
        if candidate.confirmed and first_confirmed is None:
            first_confirmed = candidate
    return DetectionResult(
        detector="onset",
        observed=first_confirmed is not None,
        onset_step=first_confirmed.step if first_confirmed else None,
        onset_time_s=first_confirmed.onset_time_s if first_confirmed else None,
        confirmation_time_s=first_confirmed.confirmation_time_s if first_confirmed else None,
        candidates=candidates,
    )


def detect_first_fire(records: Iterable[dict[str, Any]]) -> DetectionResult:
    for record in records:
        if int(record.get("action", -1)) == MAIN_ENGINE_ACTION:
            candidate = Candidate(
                step=int(record["step"]),
                onset_time_s=float(record["time_s"]),
                confirmation_time_s=float(record["time_s"]),
                firing_fraction=1.0,
                speed_reduction=None,
                confirmed=True,
            )
            return DetectionResult(
                "first_fire", True, candidate.step, candidate.onset_time_s,
                candidate.confirmation_time_s, [candidate]
            )
    return DetectionResult("first_fire", False, None, None, None, [])


def detect_sustained_fire(
    records: Iterable[dict[str, Any]], config: DetectorConfig
) -> DetectionResult:
    relaxed = DetectorConfig(
        window_s=config.window_s,
        firing_fraction=config.firing_fraction,
        epsilon_v=0.0,
        dt=config.dt,
    )
    result = detect_onset(records, relaxed)
    for candidate in result.candidates:
        candidate.rejection_reasons = [
            reason for reason in candidate.rejection_reasons
            if reason != "insufficient_speed_reduction"
        ]
        candidate.confirmed = not candidate.rejection_reasons
    first = next((item for item in result.candidates if item.confirmed), None)
    return DetectionResult(
        "sustained_fire",
        first is not None,
        first.step if first else None,
        first.onset_time_s if first else None,
        first.confirmation_time_s if first else None,
        result.candidates,
    )
