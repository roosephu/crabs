import gym.wrappers
import numpy as np


class RescaleAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env, lo, hi):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        old_lo = env.action_space.low
        old_hi = env.action_space.high
        new_lo = np.full_like(old_lo, lo)
        new_hi = np.full_like(old_hi, hi)
        self.action_space = gym.spaces.Box(low=new_lo, high=new_hi)

        self.w = (old_hi - old_lo) / (new_hi - new_lo)
        self.b = old_lo - new_lo * self.w

    def action(self, action):
        return self.b + action * self.w

    def reverse_action(self, action):
        return (action - self.b) / self.w

    def mb_step(self, observations, actions, next_observations):
        return self.env.mb_step(observations, self.action(actions), next_observations)


# class RescaleAction(gym.wrappers.RescaleAction):
#     def mb_step(self, states, actions, next_states):
#         return self.env.mb_step(states, self.action(actions), next_states)
