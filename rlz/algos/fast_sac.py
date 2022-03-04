from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Distribution
import einops

import lunzi as lz
from .layers import convert_to_batch_module

from ..trainer import BaseTrainer
from .utils import make_target_network, polyak_average


C1 = np.log(2 * np.pi) / 2
C2 = 2 * np.log(2.)


def sample_tanh_gaussian_fast(mean, std):
    x = torch.randn_like(mean)
    y = mean + x * std
    log_py = -x * x / 2 - std.log() - C1
    z = y.tanh()
    log_jacobian = C2 - 2 * (y + F.softplus(-2. * y))
    log_pz = log_py - log_jacobian
    log_pz = log_pz.reshape(len(z), -1).sum(-1)
    return z, log_pz


class FLAGS(lz.BaseFLAGS):
    lr = 3e-4
    alpha = 0.0  # set to 0.0 to enable auto alpha tuning
    gamma = 0.99
    batch_size = 256

    tau = 5e-3
    target_update = 1

    n_grad_iters = 1


class BatchQFn(nn.Module):
    def __init__(self, qfns):
        super().__init__()
        self.batch_qfns = convert_to_batch_module(qfns)

    def forward(self, states, actions):
        inputs = torch.cat([states, actions], dim=-1)
        inputs = einops.repeat(inputs, 'b i -> 2 b i')
        qv = self.batch_qfns(inputs)
        return qv


class FastSoftActorCritic(nn.Module, BaseTrainer):
    FLAGS = FLAGS

    def __init__(self, policy: nn.Module, qfns: List[nn.Module], target_entropy=None,
                 *, device='cpu', sampler=None, name='SAC', **kwargs):
        super().__init__()
        self.sampler = sampler
        self.FLAGS = FLAGS.copy().merge(kwargs)
        self.name = name

        self.qfns = BatchQFn(qfns)
        self.detached_qfns = qfns
        self.qfns_target = make_target_network(self.qfns)

        self.policy = policy
        self.qfn_opt = Adam(self.qfns.parameters(), self.FLAGS.lr)
        self.policy_opt = Adam(self.policy.parameters(), self.FLAGS.lr)

        self.n_updates = 0
        if self.FLAGS.alpha == 0.0:
            assert target_entropy is not None, '`target_entropy` must be specified when learning alpha'
            self.log_alpha = nn.Parameter(torch.tensor(0.0), True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], self.FLAGS.lr)
            self.auto_entropy = True
            self.target_entropy = target_entropy
        else:
            self.auto_entropy = False
            self.alpha = torch.tensor(self.FLAGS.alpha)

        self.init_trainer(device=device)

    def get_alpha(self):
        if self.auto_entropy:
            return self.log_alpha.exp()
        return self.alpha

    def configure_train_dataloader(self):
        while True:
            yield self.sampler(self.FLAGS.batch_size)

    @lz.timer
    def training_step(self, batch, batch_idx):
        distribution = self.policy(torch.concat([batch['state'], batch['next_state']], dim=0))
        _actions, _log_prob_actions = sample_tanh_gaussian_fast(
            distribution.base_dist.loc, distribution.base_dist.scale)
        a_curr, a_next = einops.rearrange(_actions, '(E b) ... -> E b ...', E=2)
        log_prob_a_curr, log_prob_a_next = einops.rearrange(_log_prob_actions, '(E b) ... -> E b ...', E=2)

        with torch.no_grad():
            alpha = self.get_alpha()
            min_next_qf, _ = self.qfns_target(batch['next_state'], a_next).min(dim=0)
            qf_ = (batch['reward'] + (1 - batch['done'].float()) * self.FLAGS.gamma *
                   (min_next_qf - alpha * log_prob_a_next))

        min_qf = self.qfns(batch['state'], a_curr).min(dim=0)[0]
        policy_loss = (alpha * log_prob_a_curr - min_qf).mean()
        self.minimize(self.policy_opt, policy_loss)

        qfn_losses = F.mse_loss(self.qfns(batch['state'], batch['action']), einops.repeat(qf_, 'i -> 2 i'))
        self.minimize(self.qfn_opt, qfn_losses.sum())

        if self.auto_entropy:
            alpha_loss = -self.log_alpha * (log_prob_a_curr.mean() + self.target_entropy).detach()
            self.minimize(self.alpha_opt, alpha_loss)

        if self.n_batches % self.FLAGS.target_update == 0:
            polyak_average(self.qfns, self.qfns_target, self.FLAGS.tau)

        return {'loss': qfn_losses.detach().cpu().numpy().round(6), 'alpha': self.get_alpha().item()}

    def post_step(self, output):
        if self.n_batches % 1000 == 0:
            loss = output['loss']
            alpha = output['alpha']
            lz.log.info(f'[{self.name}] # {self.n_batches}: qfn loss = {loss:.6f}, alpha = {alpha:.6f}')

    def sync(self):  # sync self.qfns to self.detached_qfns
        raise NotImplementedError
