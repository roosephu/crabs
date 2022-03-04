import gym
import numpy as np


class QuantizeAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env, n_bins):
        super().__init__(env)
        if isinstance(n_bins, int):
            n_bins = [n_bins] * env.action_space.shape[0]
        self.action_space = gym.spaces.MultiDiscrete(n_bins)
        self.n_bins = np.array(n_bins)
        self.lo = self.env.action_space.low
        self.hi = self.env.action_space.high

    def action(self, action):
        action = (self.hi - self.lo) / (self.n_bins - 1) * action + self.lo
        return action

    def reverse_action(self, action):
        return np.floor((action - self.lo) / ((self.hi - self.lo) / (self.n_bins - 1)))
