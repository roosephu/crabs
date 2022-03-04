from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Distribution

import lunzi as lz

from ..qfn import NetQFn
from ..policy import NetPolicy
from ..trainer import BaseTrainer
from .utils import make_target_network, polyak_average


def sample_with_log_prob(distribution):
    sampling = distribution.base_dist.rsample()
    ret = sampling.tanh()
    log_jacobian = 2. * (np.log(2.) - sampling - F.softplus(-2. * sampling))
    log_probs = distribution.base_dist.log_prob(sampling) - log_jacobian
    log_probs = log_probs.reshape(len(ret), -1).sum(-1)
    return ret, log_probs


class FLAGS(lz.BaseFLAGS):
    lr = 3e-4
    alpha = 0.0  # set to 0.0 to enable auto alpha tuning
    gamma = 0.99
    batch_size = 256

    tau = 5e-3
    target_update = 1

    n_grad_iters = 1


class SACTrainer(nn.Module, BaseTrainer):
    FLAGS = FLAGS

    def __init__(self, policy: NetPolicy, qfns: List[NetQFn], target_entropy=None,
                 *, device='cpu', sampler=None, name='SAC', **kwargs):
        super().__init__()
        self.sampler = sampler
        self.FLAGS = FLAGS.copy().merge(kwargs)
        self.name = name

        self.qfns = nn.ModuleList(qfns)
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

        with torch.no_grad():
            alpha = self.get_alpha()
            next_actions, log_prob_actions = sample_with_log_prob(self.policy(batch['next_state']))
            next_qfs = [qfn_target(batch['next_state'], next_actions) for qfn_target in self.qfns_target]
            min_next_qf = torch.min(torch.stack(next_qfs), dim=0)[0]
            qf_ = (batch['reward'] + (1 - batch['done'].float()) * self.FLAGS.gamma *
                   (min_next_qf - alpha * log_prob_actions))

        qfn_losses = torch.stack([F.mse_loss(qfn(batch['state'], batch['action']), qf_) for qfn in self.qfns])
        self.minimize(self.qfn_opt, qfn_losses.sum())

        actions, log_prob_actions = sample_with_log_prob(self.policy(batch['state']))
        min_qf, _ = torch.min(torch.stack([qfn(batch['state'], actions) for qfn in self.qfns]), dim=0)

        policy_loss = (alpha.clamp(min=1e-3) * log_prob_actions - min_qf).mean()
        self.minimize(self.policy_opt, policy_loss)

        if self.auto_entropy:
            alpha_loss = -self.log_alpha * (log_prob_actions.mean() + self.target_entropy).detach()
            self.minimize(self.alpha_opt, alpha_loss)

        if self.n_batches % self.FLAGS.target_update == 0:
            polyak_average(self.qfns, self.qfns_target, self.FLAGS.tau)

        return {'loss': qfn_losses.detach().cpu().numpy().round(6), 'alpha': self.get_alpha().item()}

    def post_step(self, output):
        if self.n_batches % 1000 == 0:
            loss = output['loss']
            alpha = output['alpha']
            lz.log.info(f'[{self.name}] # {self.n_batches}: qfn loss = {loss}, alpha = {alpha:.6f}')
