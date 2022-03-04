import abc
import torch
import torch.nn.functional as F
import numpy as np


class SafeEnv(abc.ABC):
    @abc.abstractmethod
    def is_state_safe(self, states: torch.Tensor):
        pass

    @abc.abstractmethod
    def barrier_fn(self, states: torch.Tensor):
        pass

    def reward_fn(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor):
        pass

    def trans_fn(self, states: torch.Tensor, actions: torch.Tensor):
        pass

    def done_fn(self, states, actions, next_states):
        pass


def nonneg_barrier(x):
    return F.softplus(-3 * x)


# def interval_barrier(x, lb, rb, eps=1e-6):
#     x = (x - lb) / (rb - lb)
#     b = -((x + eps) * (1 - x + eps) / (0.5 + eps)**2).log()
#     b_min, b_max = 0, -np.log(4 * eps)
#     grad = 1. / eps - 1
#     out = grad * torch.max(-x, x - 1)
#     return torch.where(torch.as_tensor((0 < x) & (x < 1)), b, b_max + out)


def interval_barrier(x, lb, rb, eps=1e-2, grad=None):
    x = (x - lb) / (rb - lb) * 2 - 1
    b = -((1 + x + eps) * (1 - x + eps) / (1 + eps)**2).log()
    b_min, b_max = 0, -np.log(eps * (2 + eps) / (1 + eps)**2)
    if grad is None:
        grad = 2. / eps / (2 + eps)
    out = grad * (abs(x) - 1)
    return torch.where(torch.as_tensor((-1 < x) & (x < 1)), b / b_max, 1 + out)
