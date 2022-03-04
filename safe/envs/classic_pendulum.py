from gym import register
import gym.spaces as spaces
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
import numpy as np
from safe.envs.safe_env_spec import SafeEnv, interval_barrier
import torch


class SafeClassicPendulum(PendulumEnv, SafeEnv):
    def __init__(self, init_state, threshold, goal_state=(0, 0), max_torque=2.0, obs_type='state', task='upright', **kwargs):
        self.init_state = np.array(init_state, dtype=np.float32)
        self.goal_state = goal_state
        self.threshold = threshold
        self.obs_type = obs_type
        self.task = task
        super().__init__(**kwargs)

        if obs_type == 'state':
            high = np.array([np.pi / 2, self.max_speed])
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        elif obs_type == 'observation':
            high = np.array([1, 1, self.max_speed])
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        else:
            assert 0

        self.max_torque = max_torque
        self.action_space = spaces.Box(low=-max_torque, high=max_torque, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        th, thdot = self.state
        if self.obs_type == 'state':
            return np.array([angle_normalize(th), thdot], dtype=np.float32)
        else:
            return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)

    def reset(self):
        self.state = self.init_state
        self.last_u = None
        self.episode_unsafe = False
        return self._get_obs()

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        # costs = (angle_normalize(th) - self.goal_state[0]) ** 2 + \
        #     0.1 * (thdot - self.goal_state[1]) ** 2  # + 0.001 * (u ** 2)
        costs = (angle_normalize(th) - self.goal_state[0]) ** 2

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot], np.float32)
        if abs(newth) > self.threshold:
            # costs = 1000
            self.episode_unsafe = True
            done = True
        else:
            done = False
        return self._get_obs(), -costs, False, {'episode.unsafe': self.episode_unsafe}

    def reward_fn(self, states, actions, next_states):
        th, thdot = self.parse_state(states)
        max_torque = self.max_torque

        actions = actions.clamp(-1, 1)[..., 0] * max_torque
        goal_th, goal_thdot = self.goal_state
        costs = (th - goal_th) ** 2 + .1 * (thdot - goal_thdot) ** 2 + .001 * actions ** 2
        costs = torch.where(self.is_state_safe(next_states), costs, torch.tensor(1000., device=costs.device))

        return -costs

    def trans_fn(self, states: torch.Tensor, u: torch.Tensor):
        th, thdot = self.parse_state(states)

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = u.clamp(-1, 1)[..., 0] * self.max_torque

        newthdot = thdot + (-3 * self.g / (2 * l) * (th + np.pi).sin() + 3. / (m * l ** 2) * u) * dt
        newth = angle_normalize(th + newthdot * dt)
        newthdot = newthdot.clamp(-self.max_speed, self.max_speed)

        dims = list(range(1, states.ndim)) + [0]
        if self.obs_type == 'state':
            return torch.stack([newth, newthdot]).permute(dims)
        return torch.stack([newth.cos(), newth.sin(), newthdot]).permute(dims)

    def parse_state(self, states):
        if self.obs_type == 'state':
            thdot = states[..., 1]
            th = states[..., 0]
        else:
            thdot = states[..., 2]
            th = torch.atan2(states[..., 1], states[..., 0])
        return th, thdot

    def is_state_safe(self, states: torch.Tensor):
        th, thdot = self.parse_state(states)
        return th.abs() <= self.threshold

    def barrier_fn(self, states: torch.Tensor):
        th, thdot = self.parse_state(states)
        b1 = interval_barrier(th, -self.threshold, self.threshold)
        return b1


register('MyPendulum-v0', entry_point=SafeClassicPendulum, max_episode_steps=200)
