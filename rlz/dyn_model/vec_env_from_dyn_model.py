import gym
import numpy as np

from .base_dyn_model import BaseDynModel


class VecEnvFromDynModel(gym.vector.VectorEnv):
    observations: np.ndarray

    def __init__(self, model: BaseDynModel, observation_space, action_space,
                 *, num_envs=1, extra_fn=None, init_fn=None):
        super().__init__(num_envs, observation_space, action_space)
        self.model = model
        self.extra_fn = extra_fn
        self.init_fn = init_fn if init_fn else model.get_init_states

        self.observations = np.zeros((num_envs, *observation_space.shape), dtype=np.float32)

    def reset(self, indices=None):
        # self.observations = self.model.get_init_observations(n_envs=self.num_envs)
        if indices is None:
            indices = range(self.num_envs)
        if len(indices) > 0:
            self.observations = self.observations.copy()
            self.observations[indices] = self.init_fn(len(indices))
        return self.observations

    def step(self, actions):
        next_observations, rewards, dones, infos = self.model.get_steps(self.observations, actions)
        if self.extra_fn is not None:
            _rewards, _dones, _infos = self.extra_fn(self.observations, actions, next_observations)
            rewards = rewards or _rewards
            dones = dones or _dones
            if _infos is not None:
                for info, _info in zip(infos, _infos):
                    info.update(_info)
        self.observations = next_observations

        return self.observations, rewards, dones, infos

    def __repr__(self):
        return f'VecEnv({self.model.__class__.__name__}, {self.num_envs})'

    def close(self, **kwargs):
        pass


class VectorWrapper(gym.vector.VectorEnv):
    def __init__(self, vec_env: VecEnvFromDynModel, *, max_episode_steps=0, extract_keys=()):
        super().__init__(vec_env.num_envs, vec_env.single_observation_space, vec_env.single_action_space)
        self.vec_env = vec_env
        self.max_episode_steps = max_episode_steps
        self.returns = np.zeros(self.num_envs)
        self.n_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.extract_keys = extract_keys

    def reset(self, indices=None):
        # import inspect
        # has_arg = inspect.signature(self.vec_env.reset)
        if indices is None:
            self.n_steps[:] = 0
            self.returns[:] = 0
        else:
            self.n_steps[indices] = 0
            self.returns[indices] = 0
        return self.vec_env.reset(indices=indices)

    def step(self, actions):
        next_states, rewards, dones, infos = self.vec_env.step(actions)
        self.n_steps += 1
        self.returns += rewards

        indices = []
        for i in range(self.num_envs):
            if not dones[i] and self.n_steps[i] != self.max_episode_steps:
                continue

            indices.append(i)
            if 'episode' not in infos[i]:
                infos[i]['episode'] = {}
            infos[i]['episode'].update({
                'return': self.returns[i],
                '_terminal_state': next_states[i],
                **{key: infos[i][key] for key in self.extract_keys if key in infos[i]},
            })
            infos[i].update({
                'TimeLimit.truncated': not dones[i] and self.n_steps[i] == self.max_episode_steps,
            })
            self.n_steps[i] = 0
            self.returns[i] = 0.

        if len(indices) > 0:
            next_states = next_states.copy()
            next_states[indices] = self.vec_env.reset(indices=indices)
        return next_states, rewards, dones, infos

    def __repr__(self):
        return f'Wrapped({self.vec_env})'

    def __getattr__(self, item):
        return getattr(self.vec_env, item)

    def close(self, **kwargs):
        pass
