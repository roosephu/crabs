import abc
import torch.nn as nn
from rlz.torch_utils import maybe_numpy


class BaseVFn(abc.ABC):
    @abc.abstractmethod
    def get_values(self, states):
        pass


class NetVFn(nn.Module, BaseVFn):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, states):
        return self.net(states)

    @maybe_numpy
    def get_values(self, states):
        return self(states)


__all__ = ['BaseVFn', 'NetVFn']
