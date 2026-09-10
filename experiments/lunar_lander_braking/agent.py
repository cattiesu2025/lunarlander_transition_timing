"""Minimal, reproducible MLP Double DQN implementation."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import canonical_hash, file_hash
from .env import Scenario, make_env


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, actions: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = obs_dim
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, actions))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.states = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, obs_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )


class DoubleDQNAgent:
    def __init__(self, obs_dim: int, actions: int, config: dict[str, Any], seed: int, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.actions = actions
        self.config = config
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        hidden = [int(value) for value in config["hidden_sizes"]]
        self.online = QNetwork(obs_dim, actions, hidden).to(self.device)
        self.target = QNetwork(obs_dim, actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(config["learning_rate"]))
        self.updates = 0

    def epsilon(self, step: int) -> float:
        fraction = min(1.0, step / float(self.config["epsilon_decay_steps"]))
        return float(self.config["epsilon_start"] + fraction * (self.config["epsilon_end"] - self.config["epsilon_start"]))

    def act(self, observation: np.ndarray, step: int = 0, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon(step):
            return int(self.rng.integers(self.actions))
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def update(self, replay: ReplayBuffer) -> float:
        batch = int(self.config["batch_size"])
        states, actions, rewards, next_states, dones = replay.sample(batch, self.device)
        predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + float(self.config["gamma"]) * (1.0 - dones) * next_values
        loss = nn.functional.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), float(self.config["gradient_clip"]))
        self.optimizer.step()
        self.updates += 1
        if self.updates % int(self.config["target_update_interval"]) == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "updates": self.updates,
                "agent_config": self.config,
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, obs_dim: int = 8, actions: int = 4, device: str = "cpu") -> tuple["DoubleDQNAgent", dict[str, Any]]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        metadata = checkpoint["metadata"]
        agent = cls(obs_dim, actions, checkpoint["agent_config"], int(metadata["seed"]), device)
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.updates = int(checkpoint["updates"])
        return agent, metadata


def sample_training_scenario(config: dict[str, Any], seed: int, episode: int) -> Scenario:
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


def train_condition(
    config: dict[str, Any], condition: str, seed: int, output_dir: str | Path,
    device: str = "cpu", train_steps: int | None = None,
) -> Path:
    if condition not in config["reward"]["conditions"]:
        raise ValueError(f"Unknown condition: {condition}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(config, condition)
    agent = DoubleDQNAgent(8, 4, config["agent"], seed, device)
    replay = ReplayBuffer(int(config["agent"]["replay_capacity"]), 8, seed + 17)
    total_steps = int(train_steps if train_steps is not None else config["experiment"]["train_steps"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training_metrics.jsonl"
    step = 0
    episode = 0
    with metrics_path.open("w", encoding="utf-8") as metrics:
        while step < total_steps:
            scenario = sample_training_scenario(config, seed, episode)
            obs, _ = env.reset(options={"scenario": scenario})
            terminated = truncated = False
            episode_return = 0.0
            losses: list[float] = []
            while not (terminated or truncated) and step < total_steps:
                action = agent.act(obs, step)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                replay.add(obs, action, reward, next_obs, terminated or truncated)
                obs = next_obs
                episode_return += reward
                step += 1
                if (
                    step >= int(config["agent"]["learning_starts"])
                    and step % int(config["agent"]["train_frequency"]) == 0
                    and replay.size >= int(config["agent"]["batch_size"])
                ):
                    losses.append(agent.update(replay))
            metrics.write(json.dumps({
                "episode": episode,
                "steps": step,
                "return": episode_return,
                "mean_loss": float(np.mean(losses)) if losses else None,
                "epsilon": agent.epsilon(step),
                "scenario": asdict(scenario),
            }, sort_keys=True) + "\n")
            episode += 1
    metadata = {
        "condition": condition,
        "seed": seed,
        "training_steps": total_steps,
        "episodes": episode,
        "config_hash": canonical_hash(config),
        "checkpoint_role": "final_fixed_budget",
    }
    checkpoint = output / "final.pt"
    agent.save(checkpoint, metadata)
    metadata["checkpoint_sha256"] = file_hash(checkpoint)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env.close()
    return checkpoint
