import numpy as np
import pytest

pytest.importorskip("Box2D")

from experiments.lunar_lander_braking.config import load_config, load_manifest
from experiments.lunar_lander_braking.env import DT, Scenario, make_env


CONFIG = "experiments/lunar_lander_braking/configs/pilot.yaml"
MANIFEST = "experiments/lunar_lander_braking/grids/development.json"


def test_controlled_reset_changes_physics_and_preserves_leg_assembly():
    config = load_config(CONFIG)
    scenario = Scenario.from_dict(load_manifest(MANIFEST)[8])
    env = make_env(config, "BAL")
    observation, info = env.reset(options={"scenario": scenario})
    assert observation.shape == (8,)
    assert info["height_above_pad"] == pytest.approx(scenario.height_above_pad)
    assert info["vy"] == pytest.approx(scenario.vy)
    assert not info["contact"]
    base = env.base
    distances = [np.hypot(leg.position.x - base.lander.position.x, leg.position.y - base.lander.position.y) for leg in base.legs]
    assert all(0.1 < value < 2.0 for value in distances)
    env.close()


def test_reward_difference_is_only_declared_replacement_weights():
    config = load_config(CONFIG)
    scenario = Scenario.from_dict(load_manifest(MANIFEST)[8])
    env_desc = make_env(config, "DESC")
    env_econ = make_env(config, "ECON")
    env_desc.reset(options={"scenario": scenario})
    env_econ.reset(options={"scenario": scenario})
    _, _, _, _, desc = env_desc.step(2)
    _, _, _, _, econ = env_econ.step(2)
    assert desc["reward_components"]["common"] == pytest.approx(econ["reward_components"]["common"])
    assert desc["reward_components"]["side_engine"] == econ["reward_components"]["side_engine"]
    assert desc["reward_components"]["time"] == pytest.approx(-config["reward"]["time_scale"] * DT)
    assert desc["reward_components"]["time"] == econ["reward_components"]["time"]
    assert desc["reward_components"]["downward_speed"] != econ["reward_components"]["downward_speed"]
    assert desc["reward_components"]["main_engine"] != econ["reward_components"]["main_engine"]
    for result in (desc, econ):
        components = result["reward_components"]
        assert components["total"] == pytest.approx(
            components["common"]
            + components["side_engine"]
            + components["downward_speed"]
            + components["main_engine"]
            + components["time"]
        )
    env_desc.close()
    env_econ.close()


def test_time_penalty_matches_failure_scale_at_timeout_horizon():
    config = load_config(CONFIG)
    reward = config["reward"]
    horizon = config["experiment"]["max_episode_steps"]
    assert reward["time_scale"] * DT == pytest.approx(0.1)
    assert reward["time_scale"] * DT * horizon == pytest.approx(100.0)
