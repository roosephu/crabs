import abc

import torch.nn as nn
from .torch_utils import maybe_numpy


class BaseQFn(abc.ABC):
    @abc.abstractmethod
    def get_q(self, states, actions):
        pass


class NetQFn(nn.Module, BaseQFn):
    @maybe_numpy
    def get_q(self, states, actions):
        return self(states, actions)


__all__ = ['BaseQFn', 'NetQFn']
