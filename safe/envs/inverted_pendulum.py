from gym import register
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from .safe_env_spec import SafeEnv, interval_barrier
from gym.utils.ezpickle import EzPickle
import numpy as np


class SafeInvertedPendulumEnv(InvertedPendulumEnv, SafeEnv):
    episode_unsafe = False

    def __init__(self, threshold=0.2, task='upright', random_reset=False, violation_penalty=10):
        self.threshold = threshold
        self.task = task
        self.random_reset = random_reset
        self.violation_penalty = violation_penalty
        super().__init__()
        EzPickle.__init__(self, threshold=threshold, task=task, random_reset=random_reset)  # deepcopy calls `get_state`

    def reset_model(self):
        if self.random_reset:
            qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
            qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
            self.set_state(qpos, qvel)
        else:
            self.set_state(self.init_qpos, self.init_qvel)
        self.episode_unsafe = False
        return self._get_obs()

    def _get_obs(self):
        return super()._get_obs().astype(np.float32)

    def step(self, a):
        a = np.clip(a, -1, 1)
        next_state, _, done, info = super().step(a)
        # reward = (next_state[0]**2 + next_state[1]**2)  # + a[0]**2 * 0.01
        # reward = next_state[1]**2  # + a[0]**2 * 0.01

        if self.task == 'upright':
            reward = -next_state[1]**2
        elif self.task == 'swing':
            reward = next_state[1]**2
        elif self.task == 'move':
            reward = next_state[0]**2
        else:
            assert 0
        
        if abs(next_state[..., 1]) > self.threshold or abs(next_state[..., 0]) > 0.9:
            # breakpoint()
            self.episode_unsafe = True
            reward -= self.violation_penalty
        info['episode.unsafe'] = self.episode_unsafe
        return next_state, reward, self.episode_unsafe, info

    def is_state_safe(self, states):
        # return states[..., 1].abs() <= self.threshold
        return self.barrier_fn(states) <= 1.0

    def barrier_fn(self, states):
        return interval_barrier(states[..., 1], -self.threshold, self.threshold).maximum(interval_barrier(states[..., 0], -0.9, 0.9))

    def reward_fn(self, states, actions, next_states):
        return -(next_states[..., 0]**2 + next_states[..., 1]**2) - actions[..., 0]**2 * 0.01


class SafeInvertedPendulumSwingEnv(SafeInvertedPendulumEnv):
    def __init__(self):
        super().__init__(threshold=1.5, task='swing')


class SafeInvertedPendulumMoveEnv(SafeInvertedPendulumEnv):
    def __init__(self):
        super().__init__(threshold=0.2, task='move')


register('SafeInvertedPendulum-v2', entry_point=SafeInvertedPendulumEnv, max_episode_steps=1000)
register('SafeInvertedPendulumSwing-v2', entry_point=SafeInvertedPendulumSwingEnv, max_episode_steps=1000)
register('SafeInvertedPendulumMove-v2', entry_point=SafeInvertedPendulumMoveEnv, max_episode_steps=1000)
