"""Controlled-reset, reward-modified Gymnasium LunarLander."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np


FPS = 50
DT = 1.0 / FPS
VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE = 30.0
LEG_DOWN = 18


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    height_above_pad: float
    vx: float
    vy: float
    x_offset: float
    angle: float
    angular_velocity: float
    terrain_seed: int
    noise_seed: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Scenario":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RewardWeights:
    downward_speed: float
    main_engine: float


def _rotate(vector: tuple[float, float], angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]


class ControlledLunarLander(gym.Wrapper):
    """A wrapper that resets the actual Box2D assembly to a recorded scenario."""

    def __init__(
        self,
        env: gym.Env,
        reward_weights: RewardWeights,
        downward_reference_speed: float = 2.0,
        downward_scale: float = 10.0,
        main_engine_scale: float = 10.0,
        time_scale: float = 5.0,
        settling_actuation_scale: float = 5.0,
    ) -> None:
        super().__init__(env)
        if downward_reference_speed <= 0:
            raise ValueError("downward_reference_speed must be positive")
        self.reward_weights = reward_weights
        self.downward_reference_speed = float(downward_reference_speed)
        self.downward_scale = float(downward_scale)
        self.main_engine_scale = float(main_engine_scale)
        self.time_scale = float(time_scale)
        self.settling_actuation_scale = float(settling_actuation_scale)
        self.scenario: Scenario | None = None
        self._previous_common_shaping = 0.0
        self._step_index = 0

    @property
    def base(self):
        return self.env.unwrapped

    def _state(self) -> np.ndarray:
        base = self.base
        pos, vel = base.lander.position, base.lander.linearVelocity
        return np.asarray(
            [
                (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
                (pos.y - (base.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
                vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
                vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
                base.lander.angle,
                20.0 * base.lander.angularVelocity / FPS,
                float(base.legs[0].ground_contact),
                float(base.legs[1].ground_contact),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _common_shaping(state: np.ndarray) -> float:
        return float(
            -100.0 * np.hypot(state[0], state[1])
            - 100.0 * abs(state[4])
            + 10.0 * state[6]
            + 10.0 * state[7]
        )

    def _physical_info(self, action: int | None = None) -> dict[str, Any]:
        base = self.base
        state = self._state()
        return {
            "scenario_id": self.scenario.scenario_id if self.scenario else None,
            "dt": DT,
            "step": self._step_index,
            "time_s": self._step_index * DT,
            "action": action,
            "x": float(base.lander.position.x),
            "height_above_pad": float(base.lander.position.y - base.helipad_y),
            "vx": float(base.lander.linearVelocity.x),
            "vy": float(base.lander.linearVelocity.y),
            "angle": float(base.lander.angle),
            "angular_velocity": float(base.lander.angularVelocity),
            "left_contact": bool(base.legs[0].ground_contact),
            "right_contact": bool(base.legs[1].ground_contact),
            "contact": bool(base.legs[0].ground_contact or base.legs[1].ground_contact),
            "game_over": bool(base.game_over),
            "out_of_bounds": bool(abs(float(state[0])) >= 1.0),
            "lander_awake": bool(base.lander.awake),
            "units": {"position": "box2d_world_unit", "velocity": "box2d_world_unit_per_s"},
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = options or {}
        scenario_value = options.get("scenario")
        if scenario_value is None:
            raise ValueError("Controlled reset requires options={'scenario': ...}")
        self.scenario = scenario_value if isinstance(scenario_value, Scenario) else Scenario.from_dict(scenario_value)
        self.env.reset(seed=self.scenario.terrain_seed)
        base = self.base
        lander = base.lander
        old_angle = float(lander.angle)
        relative = [
            (float(leg.position.x - lander.position.x), float(leg.position.y - lander.position.y), float(leg.angle - old_angle))
            for leg in base.legs
        ]
        target_x = VIEWPORT_W / SCALE / 2 + self.scenario.x_offset
        target_y = base.helipad_y + self.scenario.height_above_pad
        lander.position = (target_x, target_y)
        lander.angle = self.scenario.angle
        lander.linearVelocity = (self.scenario.vx, self.scenario.vy)
        lander.angularVelocity = self.scenario.angular_velocity
        lander.awake = True
        delta_angle = self.scenario.angle - old_angle
        for leg, (rx, ry, relative_angle) in zip(base.legs, relative):
            rotated_x, rotated_y = _rotate((rx, ry), delta_angle)
            leg.position = (target_x + rotated_x, target_y + rotated_y)
            leg.angle = self.scenario.angle + relative_angle
            leg.linearVelocity = (
                self.scenario.vx - self.scenario.angular_velocity * rotated_y,
                self.scenario.vy + self.scenario.angular_velocity * rotated_x,
            )
            leg.angularVelocity = self.scenario.angular_velocity
            leg.ground_contact = False
            leg.awake = True
        base.game_over = False
        base.prev_shaping = None
        base.np_random = np.random.default_rng(self.scenario.noise_seed)
        self._step_index = 0
        state = self._state()
        self._previous_common_shaping = self._common_shaping(state)
        info = self._physical_info()
        info["scenario"] = asdict(self.scenario)
        return state, info

    def step(self, action: int):
        before = self._physical_info(action)
        state, native_reward, terminated, truncated, native_info = self.env.step(action)
        self._step_index += 1
        state = self._state()
        current_shaping = self._common_shaping(state)
        common = current_shaping - self._previous_common_shaping
        self._previous_common_shaping = current_shaping
        base = self.base
        landed = bool(terminated and not base.game_over and abs(float(state[0])) < 1.0 and not base.lander.awake)
        crash = bool(terminated and not landed and not abs(float(state[0])) >= 1.0)
        if terminated:
            common = 100.0 if landed else -100.0
        side_cost = -0.03 if action in (1, 3) else 0.0
        downward = max(0.0, -float(base.lander.linearVelocity.y))
        downward_cost = -self.reward_weights.downward_speed * self.downward_scale * (downward / self.downward_reference_speed) ** 2 * DT
        main_cost = -self.reward_weights.main_engine * self.main_engine_scale * float(action == 2) * DT
        time_cost = -self.time_scale * DT
        settling_actuation_cost = -self.settling_actuation_scale * DT * float(
            action != 0 and before["left_contact"] and before["right_contact"]
        )
        reward = common + side_cost + downward_cost + main_cost + time_cost + settling_actuation_cost
        info = dict(native_info)
        info.update(self._physical_info(action))
        info["step"] = self._step_index - 1
        info["time_s"] = (self._step_index - 1) * DT
        info.update(
            {
                "vy_before": before["vy"],
                "vx_before": before["vx"],
                "x_before": before["x"],
                "height_above_pad_before": before["height_above_pad"],
                "contact_before": before["contact"],
                "landed": landed,
                "crash": crash,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "native_reward_diagnostic": float(native_reward),
                "reward_components": {
                    "common": common,
                    "side_engine": side_cost,
                    "downward_speed": downward_cost,
                    "main_engine": main_cost,
                    "time": time_cost,
                    "settling_actuation": settling_actuation_cost,
                    "total": reward,
                },
            }
        )
        return state, float(reward), terminated, truncated, info


def make_env(config: dict[str, Any], condition: str) -> ControlledLunarLander:
    reward = config["reward"]
    weights = RewardWeights(**reward["conditions"][condition])
    base = gym.make(
        "LunarLander-v3",
        gravity=float(config["environment"]["gravity"]),
        enable_wind=bool(config["environment"]["enable_wind"]),
        continuous=False,
        max_episode_steps=int(config["experiment"]["max_episode_steps"]),
    )
    return ControlledLunarLander(
        base,
        weights,
        downward_reference_speed=float(reward["downward_reference_speed"]),
        downward_scale=float(reward["downward_scale"]),
        main_engine_scale=float(reward["main_engine_scale"]),
        time_scale=float(reward["time_scale"]),
        settling_actuation_scale=float(reward["settling_actuation_scale"]),
    )


def classify_outcome(terminated: bool, truncated: bool, info: dict[str, Any]) -> str:
    if info.get("technical_error"):
        return "technical_error"
    if info.get("landed"):
        return "landed"
    if info.get("out_of_bounds"):
        return "out_of_bounds"
    if info.get("crash") or (terminated and not info.get("landed")):
        return "crash"
    if truncated:
        return "timeout"
    return "in_progress"


def rollout(
    env: ControlledLunarLander,
    scenario: Scenario,
    policy: Callable[[np.ndarray], int],
) -> dict[str, Any]:
    observation, reset_info = env.reset(options={"scenario": scenario})
    records: list[dict[str, Any]] = []
    total_reward = 0.0
    terminated = truncated = False
    last_info = reset_info
    while not (terminated or truncated):
        action = int(policy(observation))
        observation, reward, terminated, truncated, last_info = env.step(action)
        total_reward += reward
        records.append(dict(last_info))
    return {
        "scenario": asdict(scenario),
        "records": records,
        "outcome": classify_outcome(terminated, truncated, last_info),
        "custom_return": total_reward,
        "native_return_diagnostic": sum(item["native_reward_diagnostic"] for item in records),
    }
