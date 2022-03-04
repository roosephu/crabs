import gym
import numpy as np


class CastDtype(gym.Wrapper):
    def __init__(self, env, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        self.observation_space.dtype = np.dtype(self.dtype)
        if isinstance(self.action_space, gym.spaces.Box):
            self.action_space.dtype = np.dtype(self.dtype)
    
    def reset(self):
        return self.observation(super().reset())
        
    def step(self, action):
        next_observation, reward, done, info = super().step(action)
        return self.observation(next_observation), self.dtype(reward), done, info

    def observation(self, observation):
        return np.asarray(observation).astype(self.dtype)
