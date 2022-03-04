from typing import List
import abc

import torch
import torch.nn as nn
import numpy as np
from .torch_utils import maybe_numpy
from .distributions import TanhGaussian


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def get_actions(self, states):
        pass

    def reset(self, indices=None):
        pass


class BaseStatefulPolicy(BasePolicy):
    state_dtype: List

    @abc.abstractmethod
    def get_state(self):
        pass

    def step(self):
        pass


class NetPolicy(nn.Module, BasePolicy):
    @maybe_numpy
    def get_actions(self, states):
        return self(states).sample()


class DetNetPolicy(NetPolicy):
    @maybe_numpy
    def get_actions(self, states):
        return self(states)


class UniformPolicy(NetPolicy):
    def __init__(self, dim_action):
        super().__init__()
        self.dim_action = dim_action

    def forward(self, states):
        return torch.rand(states.shape[:-1] + (self.dim_action,), device=states.device) * 2 - 1


class EpsGreedy(BasePolicy):
    def __init__(self, policy: BasePolicy, action_space, eps):
        self.policy = policy
        self.action_space = action_space
        self.eps = eps

    def get_actions(self, states):
        n = len(states)
        actions = np.empty(n, dtype=np.int32)

        noises = np.random.rand(n)
        for i in range(n):
            if noises[i] < self.eps:
                actions[i] = self.action_space.sample()

        indices = np.where(noises >= self.eps)[0]
        if len(indices):
            actions[indices] = self.policy.get_actions(states[indices])
        return actions


class MaxQPolicy(DetNetPolicy):
    def forward(self, observations):
        return super().forward(observations).argmax(dim=-1)


class TanhGaussianPolicy(NetPolicy):
    STD_BOUNDS = (-20, 2)

    def forward(self, states):
        outputs = super().forward(states)
        dim_action = outputs.shape[-1] // 2
        mean, log_std = outputs[..., :dim_action], outputs[..., dim_action:]
        log_std = log_std.clamp(*self.STD_BOUNDS)
        return TanhGaussian(mean, log_std.exp())


class MeanPolicy(DetNetPolicy):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, states):
        return self.policy(states).mean


class AddGaussianNoise(NetPolicy):
    def __init__(self, policy: NetPolicy, mean, std):
        super().__init__()
        self.policy = policy
        self.mean = mean
        self.std = std

    def forward(self, states):
        actions = self.policy(states)
        if isinstance(actions, TanhGaussian):
            return TanhGaussian(actions.mean + self.mean, actions.stddev * self.std)
        noises = torch.randn(*actions.shape, device=states.device) * self.std + self.mean
        return actions + noises


__all__ = [
    'BasePolicy', 'NetPolicy', 'UniformPolicy', 'DetNetPolicy', 'AddGaussianNoise',
    'BaseStatefulPolicy', 'MaxQPolicy', 'EpsGreedy', 'TanhGaussianPolicy', 'MeanPolicy',
]
