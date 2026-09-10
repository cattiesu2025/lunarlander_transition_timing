import torch

from stable_baselines3 import DQN

from experiments.lunar_lander_braking.agent import (
    DoubleDQN,
    _double_dqn_next_values,
)


def test_double_dqn_is_sb3_dqn_with_overridden_train():
    assert issubclass(DoubleDQN, DQN)
    assert DoubleDQN.train is not DQN.train


def test_double_dqn_selects_online_action_and_evaluates_with_target():
    online = torch.tensor([[1.0, 3.0], [5.0, 2.0]])
    target = torch.tensor([[10.0, 7.0], [4.0, 9.0]])
    values = _double_dqn_next_values(online, target)
    assert values.tolist() == [[7.0], [4.0]]
    assert values.tolist() != target.max(dim=1, keepdim=True).values.tolist()
