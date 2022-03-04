import gym
import numpy as np
import torch.nn as nn
from ..multi_layer_perception import MultiLayerPerceptron


class ProbabilisticModel(gym.vector.VectorEnv):
    def __init__(self, n_envs, state_space, action_space):
        super().__init__(n_envs, state_space, action_space)
        dim_state = state_space.shape[0]
        dim_action = state_space.shape[0]
        self.mlp = MultiLayerPerceptron([dim_state + dim_action, 256, 256, dim_state], nn.ReLU)
        self._states = np.zeros(n_envs, dim_state)

    def reset_wait(self):
        pass

    def step_wait(self, actions):
        pass
