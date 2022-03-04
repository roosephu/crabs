from typing import List, Dict
import torch
import numpy as np


class EpisodeReturn:
    return_: float
    coef: float

    def __init__(self, discount=1.0):
        self.discount = discount

    def reset(self, _):
        self.return_ = 0.
        self.coef = 1.

    def step(self, transition):
        self.return_ += transition['reward'] * self.coef
        self.coef *= self.discount

    def emit(self) -> Dict[str, float]:
        key = 'return' if self.discount == 1.0 else f'return_{self.discount}'
        return {key: self.return_}


class EpisodeLength:
    length: int

    def reset(self, _):
        self.length = 0

    def step(self, _):
        self.length += 1

    def emit(self) -> Dict[str, float]:
        return {'length': self.length}


class ExtractLastInfo:
    info: dict

    def __init__(self, *keys):
        self.keys = keys

    def reset(self, _):
        self.info = {}

    def step(self, transition):
        self.info = transition['info']

    def emit(self) -> Dict[str, float]:
        return {key: self.info[key] for key in self.keys if key in self.info}


class RunnerWithModel:
    def __init__(self, fns, horizon, dim_state, stats=(), *, n=1, device='cpu'):
        self.fns = fns
        self.horizon = horizon
        self.n = n

        self.device = device
        self.states = torch.zeros(n, dim_state, dtype=torch.float32, device=device)
        self.n_steps = torch.zeros(n, dtype=torch.int64, device=device)
        self.should_reset = torch.ones(n, dtype=torch.bool, device=device)

        self.ep_stats = [[stat() for stat in stats] for _ in range(self.n)]

    def _check_reset(self):
        if not self.should_reset.any().item():
            return
        self.states = self.states.detach().clone()
        for i in range(self.n):
            if self.should_reset[i]:
                self.states[i] = state = self.fns['reset']()
                self.n_steps[i] = 0
                self.should_reset[i] = False
                for stat in self.ep_stats[i]:
                    stat.reset(state)

    def reset(self):
        self.should_reset[:] = True

    @torch.no_grad()
    def run(self, policy, n_samples, buffer=None):
        n_steps = n_samples // self.n
        ep_infos = []

        for _ in range(n_steps):
            self._check_reset()

            self.n_steps += 1
            states = self.states
            actions = policy(states)

            next_states = self.fns['transition'](self.states, actions)
            dones = self.fns['done'](states, actions, next_states)
            rewards = self.fns['reward'](states, actions, next_states)
            self.states = next_states

            timeouts = self.n_steps == self.horizon
            self.should_reset = timeouts | dones

            for i in range(self.n):
                transition = {'state': states[i], 'action': actions[i], 'reward': rewards[i],
                              'next_state': next_states[i], 'done': dones[i], 'timeout': timeouts[i], 'info': {}}
                if buffer is not None:
                    buffer.add_transition(transition)
                for stat in self.ep_stats[i]:
                    stat.step(transition)
                if self.should_reset[i]:
                    ep_info = {}
                    for stat in self.ep_stats[i]:
                        ep_info.update(stat.emit())
                    ep_infos.append(ep_info)
        return ep_infos


def merge_episode_stats(dicts: List[Dict]) -> Dict[str, np.ndarray]:
    ret = {}
    for info in dicts:
        for key, value in info.items():
            if key not in ret:
                ret[key] = []
            ret[key].append(value)
    return {key: np.array(value) for key, value in ret.items()}


class SimpleRunner:
    def __init__(self, make_env, n, make_stats=lambda: (), *, device=torch.device('cpu')):
        self.n = n
        self.envs = [make_env() for _ in range(n)]
        self.device = device
        self.states = [None] * n

        self.ep_stats = [make_stats() for _ in range(n)]
        assert 'TimeLimit' in repr(self.envs[0])

    def reset(self):
        self.states = [None] * self.n

    def _check_reset(self):
        for i in range(self.n):
            if self.states[i] is None:
                self.states[i] = state = self.envs[i].reset()
                for stat in self.ep_stats[i]:
                    stat.reset(state)

    @torch.no_grad()
    def run(self, policy, n_samples, buffer=None) -> List[Dict[str, float]]:
        n_steps = n_samples // self.n
        ep_stats = []

        for _ in range(n_steps):
            self._check_reset()

            actions = policy(torch.as_tensor(np.array(self.states), dtype=torch.float32, device=self.device))
            if isinstance(actions, torch.distributions.Distribution):
                actions = actions.sample()

            for i in range(self.n):
                next_state, reward, done, info = self.envs[i].step(actions[i].detach().cpu().numpy())
                timeout = bool(info.get('TimeLimit.truncated', False))
                done = bool(done) & ~timeout

                transition = {'state': self.states[i], 'action': actions[i], 'reward': reward,
                              'next_state': next_state, 'done': done, 'timeout': timeout, 'info': info}
                if buffer is not None:
                    buffer.add_transition(transition)
                for stat in self.ep_stats[i]:
                    stat.step(transition)

                if timeout or done:
                    self.states[i] = None
                    ep_stat = {}
                    for stat in self.ep_stats[i]:
                        ep_stat.update(stat.emit())
                    ep_stats.append(ep_stat)
                else:
                    self.states[i] = next_state

        return ep_stats
