from typing import Dict, List, NamedTuple, DefaultDict
from collections import deque

import torch
import numpy as np
import gym

from . import BaseVFn
from rlz.dataset import Dataset


def compute_advantage(samples: Dataset, gamma: float = 1., lambda_: float = 1., vfn: BaseVFn = None, n_envs=1):
    assert lambda_ == 1. or vfn is not None, "vfn shouldn't be None if lambda != 1."

    n_steps = len(samples) // n_envs
    samples = samples.reshape((n_steps, n_envs))
    use_next_vf = ~samples['done']
    if 'timeout' in samples.dtype.names:
        use_next_adv = ~(samples['done'] | samples['timeout'])
    else:
        use_next_adv = ~samples['done']

    if lambda_ != 1.:
        next_values = vfn.get_values(samples[-1]['next_state'])
        values = vfn.get_values(samples.reshape(-1)['state']).reshape(n_steps, n_envs)
    else:
        next_values = np.zeros(n_envs)
        values = np.zeros((n_steps, n_envs))
    advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
    last_gae_lambda = 0.

    for t in reversed(range(n_steps)):
        delta = samples[t]['reward'] + gamma * next_values * use_next_vf[t] - values[t]
        advantages[t] = last_gae_lambda = delta + gamma * lambda_ * last_gae_lambda * use_next_adv[t]
        next_values = values[t]
    return advantages.reshape(-1), values.reshape(-1)


def gen_dtype(env: gym.Env, fields: str, dtype='f4'):
    if isinstance(env, gym.vector.VectorEnv):
        action_space = env.single_action_space
        observation_space = env.single_observation_space
    else:
        action_space = env.action_space
        observation_space = env.observation_space

    dtypes = {
        'state': ('state', observation_space.dtype, observation_space.shape),
        'action': ('action', action_space.dtype, action_space.shape),
        'next_state': ('next_state', observation_space.dtype, observation_space.shape),
        'reward': ('reward', dtype),
        'done': ('done', 'bool'),
        'timeout': ('timeout', 'bool'),
        'return_': ('return_', dtype),
        'advantage': ('advantage', dtype),
    }
    return np.dtype([dtypes[field] for field in fields.split(' ')], align=True)


def verify_reward_fn(env, n_samples, eps=1e-5):
    import lunzi as lz

    dataset = Dataset(gen_dtype(env, 'state action next_state reward done'), n_samples)
    state = env.reset()
    for i in range(n_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset[i] = (state, action, next_state, reward, done)

        state = next_state
        if done:
            state = env.reset()

    rewards_, dones_ = env.mb_step(dataset.state, dataset.action, dataset.next_state)
    diff = dataset.reward - rewards_
    l_inf = np.abs(diff).max()
    lz.log.warning(f'reward function difference: {l_inf:.6f}')

    assert np.allclose(dones_, dataset.done)
    assert l_inf < eps


to_torch_dtype_dict = {
    np.bool       : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128,
    bool: torch.bool,
    int: torch.int64,
    float: torch.float32,
}
