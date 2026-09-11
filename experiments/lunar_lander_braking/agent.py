"""Stable-Baselines3 DQN with a Double-DQN training update."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from torch.nn import functional as F

from .config import canonical_hash, file_hash
from .env import Scenario, make_env


def _double_dqn_next_values(
    online_q_values: th.Tensor, target_q_values: th.Tensor
) -> th.Tensor:
    """Select next actions online and evaluate those actions with the target net."""
    next_actions = online_q_values.argmax(dim=1, keepdim=True)
    return target_q_values.gather(dim=1, index=next_actions)


class DoubleDQN(DQN):
    """SB3 DQN whose train method uses the Double-DQN target."""

    _last_train_loss: float | None = None

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Run SB3's DQN update with decoupled next-action selection/evaluation."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if getattr(replay_data, "discounts", None) is not None
                else self.gamma
            )

            with th.no_grad():
                online_next_q = self.q_net(replay_data.next_observations)
                target_next_q = self.q_net_target(replay_data.next_observations)
                next_q_values = _double_dqn_next_values(
                    online_next_q, target_next_q
                )
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(float(loss.item()))

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self._last_train_loss = float(np.mean(losses)) if losses else None
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", self._last_train_loss)


def sample_training_scenario(
    config: dict[str, Any], seed: int, episode: int
) -> Scenario:
    bounds = config["environment"]["training_distribution"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 0x4C4C]))
    uniform = lambda key: float(rng.uniform(*map(float, bounds[key])))
    return Scenario(
        scenario_id=f"train_s{seed}_e{episode:06d}",
        height_above_pad=uniform("height_above_pad"),
        vx=uniform("horizontal_speed"),
        vy=-uniform("downward_speed"),
        x_offset=uniform("x_offset"),
        angle=uniform("angle"),
        angular_velocity=uniform("angular_velocity"),
        terrain_seed=int(rng.integers(0, 2**31 - 1)),
        noise_seed=int(rng.integers(0, 2**31 - 1)),
    )


class TrainingScenarioEnv(gym.Wrapper):
    """Inject deterministic episode-indexed scenarios into SB3 resets."""

    def __init__(self, env: gym.Env, config: dict[str, Any], seed: int) -> None:
        super().__init__(env)
        self.config = config
        self.training_seed = int(seed)
        self.next_episode = 0
        self.current_episode = -1
        self.current_scenario: Scenario | None = None
        self.episode_return = 0.0
        self.episode_length = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        supplied = dict(options or {})
        if "scenario" not in supplied:
            self.current_episode = self.next_episode
            self.next_episode += 1
            self.current_scenario = sample_training_scenario(
                self.config, self.training_seed, self.current_episode
            )
            supplied["scenario"] = self.current_scenario
        self.episode_return = 0.0
        self.episode_length = 0
        return self.env.reset(seed=seed, options=supplied)

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += float(reward)
        self.episode_length += 1
        if terminated or truncated:
            info = dict(info)
            assert self.current_scenario is not None
            info["training_episode"] = {
                "episode": self.current_episode,
                "return": self.episode_return,
                "length": self.episode_length,
                "scenario": asdict(self.current_scenario),
            }
        return observation, reward, terminated, truncated, info


class TrainingMetricsCallback(BaseCallback):
    """Persist episode-level metrics in the existing JSONL schema."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(verbose=0)
        self.path = Path(path)
        self.handle = None
        self.completed_episodes = 0

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("training_episode")
            if episode is None:
                continue
            row = {
                "episode": int(episode["episode"]),
                "steps": int(self.num_timesteps),
                "return": float(episode["return"]),
                "episode_length": int(episode["length"]),
                "mean_loss": getattr(self.model, "_last_train_loss", None),
                "epsilon": float(getattr(self.model, "exploration_rate", 0.0)),
                "scenario": episode["scenario"],
            }
            assert self.handle is not None
            self.handle.write(json.dumps(row, sort_keys=True) + "\n")
            self.handle.flush()
            self.completed_episodes += 1
        return True

    def _on_training_end(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _checkpoint_metadata(
    model: DoubleDQN,
    config: dict[str, Any],
    condition: str,
    seed: int,
    episodes: int,
    role: str,
) -> dict[str, Any]:
    return {
        "algorithm": "DoubleDQN_overridden_SB3_DQN_train",
        "stable_baselines3_version": stable_baselines3.__version__,
        "condition": condition,
        "seed": int(seed),
        "training_steps": int(model.num_timesteps),
        "episodes": int(episodes),
        "config_hash": canonical_hash(config),
        "checkpoint_role": role,
        "checkpoint_format": "stable_baselines3_zip",
    }


def _save_checkpoint(
    model: DoubleDQN,
    checkpoint: Path,
    config: dict[str, Any],
    condition: str,
    seed: int,
    episodes: int,
    role: str,
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    metadata = _checkpoint_metadata(
        model, config, condition, seed, episodes, role
    )
    metadata["checkpoint_sha256"] = file_hash(checkpoint)
    checkpoint.with_name("metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class PeriodicCheckpointCallback(BaseCallback):
    """Save predeclared fixed-step snapshots with independently hashed metadata."""

    def __init__(
        self,
        output: Path,
        steps: list[int],
        config: dict[str, Any],
        condition: str,
        seed: int,
        metrics: TrainingMetricsCallback,
    ) -> None:
        super().__init__(verbose=0)
        self.output = output
        self.pending_steps = list(steps)
        self.config = config
        self.condition = condition
        self.seed = int(seed)
        self.metrics = metrics

    def _on_step(self) -> bool:
        while self.pending_steps and self.num_timesteps >= self.pending_steps[0]:
            planned_step = self.pending_steps.pop(0)
            checkpoint = (
                self.output
                / "checkpoints"
                / f"step_{planned_step}"
                / "model.zip"
            )
            _save_checkpoint(
                self.model,
                checkpoint,
                self.config,
                self.condition,
                self.seed,
                self.metrics.completed_episodes,
                "scheduled_fixed_budget",
            )
        return True


def train_condition(
    config: dict[str, Any],
    condition: str,
    seed: int,
    output_dir: str | Path,
    device: str = "cpu",
    train_steps: int | None = None,
) -> Path:
    if condition not in config["reward"]["conditions"]:
        raise ValueError(f"Unknown condition: {condition}")

    total_steps = int(
        train_steps if train_steps is not None else config["experiment"]["train_steps"]
    )
    agent_config = config["agent"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training_env = TrainingScenarioEnv(make_env(config, condition), config, seed)
    metrics_callback = TrainingMetricsCallback(output / "training_metrics.jsonl")
    requested_checkpoints = [
        int(step)
        for step in config["experiment"].get("checkpoint_steps", [])
        if int(step) <= total_steps
    ]
    periodic_callback = PeriodicCheckpointCallback(
        output,
        [step for step in requested_checkpoints if step < total_steps],
        config,
        condition,
        seed,
        metrics_callback,
    )

    model = DoubleDQN(
        "MlpPolicy",
        training_env,
        learning_rate=float(agent_config["learning_rate"]),
        buffer_size=int(agent_config["replay_capacity"]),
        learning_starts=int(agent_config["learning_starts"]),
        batch_size=int(agent_config["batch_size"]),
        gamma=float(agent_config["gamma"]),
        train_freq=int(agent_config["train_frequency"]),
        gradient_steps=int(agent_config["gradient_steps"]),
        replay_buffer_kwargs={"handle_timeout_termination": False},
        target_update_interval=int(agent_config["target_update_interval"]),
        exploration_fraction=(
            float(agent_config["epsilon_decay_steps"]) / total_steps
        ),
        exploration_initial_eps=float(agent_config["epsilon_start"]),
        exploration_final_eps=float(agent_config["epsilon_end"]),
        max_grad_norm=float(agent_config["gradient_clip"]),
        policy_kwargs={"net_arch": [int(x) for x in agent_config["hidden_sizes"]]},
        seed=int(seed),
        device=device,
        verbose=0,
    )
    callbacks = CallbackList([metrics_callback, periodic_callback])
    model.learn(total_timesteps=total_steps, callback=callbacks, log_interval=None)

    checkpoint = output / "final.zip"
    _save_checkpoint(
        model,
        checkpoint,
        config,
        condition,
        seed,
        metrics_callback.completed_episodes,
        "final_fixed_budget",
    )
    if total_steps in requested_checkpoints:
        _save_checkpoint(
            model,
            output / "checkpoints" / f"step_{total_steps}" / "model.zip",
            config,
            condition,
            seed,
            metrics_callback.completed_episodes,
            "scheduled_fixed_budget",
        )
    model.get_env().close()
    return checkpoint
