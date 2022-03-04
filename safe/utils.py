import numpy as np
import torch
import torch.nn.functional as F
import lunzi as lz
import matplotlib.pyplot as plt
import pickle


def softminus(x: torch.Tensor):
    return -F.softplus(-x)


def swish(x):
    return x * x.sigmoid()


def barrier(states: torch.Tensor):
    max_angle = +np.pi / 2
    min_angle = -np.pi / 2

    # def interval_barrier(x, lb, rb):
    #     eps = 1e-30
    #     b = -((x - lb + eps) * (rb - x + eps)).log() + 2 * np.log((rb - lb) / 2)
    #     return torch.where(torch.as_tensor((lb < x) & (x < rb)), b, torch.tensor(100., device=x.device))

    def interval_barrier(x, lb, rb):
        x = (x - lb) / (rb - lb)
        eps = 1e-6
        b = -((x + eps) * (1 - x + eps) / (0.5 + eps)**2).log()
        b_min, b_max = 0, -np.log(4 * eps)
        grad = 1. / eps - 1
        out = grad * torch.max(-x, x - 1)
        return torch.where(torch.as_tensor((0 < x) & (x < 1)), b, b_max + out)

    b1 = interval_barrier(states[..., 0], min_angle, max_angle)
    return b1
    # b2 = interval_barrier(states[..., 1], -1, 1)
    # return (b1 + b2) / 2
