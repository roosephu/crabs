import gym
import numpy as np


class MultiDiscreteToDiscrete(gym.ActionWrapper):
    def __init__(self, env, n_bins=2):
        super().__init__(env)
        self.shape = np.full([env.action_space.shape[0]], n_bins)
        self.action_space = gym.spaces.Discrete(int(np.product(self.shape)))

    def action(self, action):
        return np.array(np.unravel_index(action, self.shape)) / (self.shape - 1.) * 2. - 1.

    def reverse_action(self, action):
        action = np.floor((action + 1.) / 2. * (self.shape - 1.))
        action = np.ravel_multi_index(action.T, self.shape)
        return action
