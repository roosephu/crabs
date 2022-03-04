import gym
import numpy as np


class NormalizeAndPermute(gym.ObservationWrapper):
    def __init__(self, env):
        lo = env.observation_space.low
        hi = env.observation_space.high
        shape = env.observation_space.shape
        assert len(shape) == 3 and np.all(lo == 0) and np.all(hi == 255)  # image
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0., 1., shape=(shape[2], shape[0], shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.transpose(observation, [2, 0, 1]).astype(np.float32) / 255.
