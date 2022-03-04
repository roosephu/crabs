from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Distribution

import lunzi as lz

from rlz.qfn import NetQFn
from rlz.policy import NetPolicy
from rlz.trainer import BaseTrainer
from rlz.algos.utils import make_target_network, polyak_average


def sample_with_log_prob(distribution):
    sampling = distribution.base_dist.rsample()
    ret = sampling.tanh()
    log_jacobian = 2. * (np.log(2.) - sampling - F.softplus(-2. * sampling))
    log_probs = distribution.base_dist.log_prob(sampling) - log_jacobian
    log_probs = log_probs.reshape(len(ret), -1).sum(-1)
    return ret, log_probs


class FLAGS(lz.BaseFLAGS):
    lr = 3e-4
    alpha = None   # set to None or 0.0 to enable auto alpha tuning
    gamma = 0.99
    batch_size = 256

    tau = 5e-3
    target_update = 1

    n_grad_iters = 1


class SafeSACTrainer2(nn.Module, BaseTrainer):
    FLAGS = FLAGS

    def __init__(self, policy: NetPolicy, qfns: List[NetQFn], barrier,
                 target_entropy=None, beta=100., *, device='cpu', sampler=None, name='SAC', **kwargs):
        super().__init__()
        self.sampler = sampler
        self.FLAGS = FLAGS.copy().merge(kwargs)
        self.name = name

        self.qfns = nn.ModuleList(qfns)
        self.qfns_target = make_target_network(self.qfns)
        self.beta = beta
        self.barrier = barrier

        self.policy = policy
        self.qfn_opt = Adam(self.qfns.parameters(), self.FLAGS.lr)
        self.policy_opt = Adam(self.policy.parameters(), self.FLAGS.lr)
        self.can_update_policy = False

        self.n_updates = 0
        if self.FLAGS.alpha is None:
            assert target_entropy is not None, '`target_entropy` must be specified when learning alpha'
            self.log_alpha = nn.Parameter(torch.tensor(0.0), True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], self.FLAGS.lr)
            self.auto_entropy = True
            self.target_entropy = target_entropy

            self.log_alpha2 = nn.Parameter(torch.tensor(0.0), True)
            self.alpha_opt2 = torch.optim.Adam([self.log_alpha2], self.FLAGS.lr)
        else:
            self.auto_entropy = False
            self.alpha = torch.tensor(self.FLAGS.alpha)

        self.init_trainer(device=device)

    def get_masked_q(self, states, actions, target=False, flag='no'):
        # barrier = (self.barrier(states, actions) - 1).clamp(min=0.)
        if target:
            qf = self.qfns_target[0](states, actions).min(self.qfns_target[1](states, actions))
        else:
            qf = self.qfns[0](states, actions).min(self.qfns[1](states, actions))
        if flag == 'no':
            return qf
        barrier = self.barrier(states, actions)
        if flag == 'mask':
            return torch.where(barrier > 0, torch.full([len(barrier)], -1000., device=self.device), qf)
        if flag == 'L-mask':
            return torch.where(barrier > 0, torch.full([len(barrier)], -1000., device=self.device) - barrier, qf)
        assert False

    def get_alpha(self):
        if self.auto_entropy:
            return self.log_alpha.exp()
        return self.alpha

    def configure_train_dataloader(self):
        while True:
            yield self.sampler(self.FLAGS.batch_size)

    def training_step(self, batch, batch_idx):

        alpha = self.get_alpha()
        if self.can_update_policy:
            actions, log_prob_actions = sample_with_log_prob(self.policy(batch['next_state']))
            min_qf = self.get_masked_q(batch['next_state'], actions, target=False, flag='L-mask')

            policy_loss = (alpha * log_prob_actions - min_qf).mean()
            self.minimize(self.policy_opt, policy_loss)

            # if we don't optimize the policy, we shouldn't optimize alpha
            if self.auto_entropy:
                alpha_loss = -self.log_alpha * (log_prob_actions.mean() + self.target_entropy).detach()
                self.minimize(self.alpha_opt, alpha_loss)

        polyak_average(self.qfns, self.qfns_target, self.FLAGS.tau)

        with torch.no_grad():
            next_actions, log_prob_actions = sample_with_log_prob(self.policy(batch['next_state']))
            lz.meters['SAC/nll'] += -log_prob_actions.mean().item()
            # next_qfs = [qfn_target(batch['next_state'], next_actions) for qfn_target in self.qfns_target]
            # min_next_qf = torch.min(torch.stack(next_qfs), dim=0)[0]
            min_next_qf = self.get_masked_q(batch['next_state'], next_actions, target=True, flag='no')
            valid_mask = self.barrier(batch['next_state'], next_actions) <= 0
            lz.meters['SAC/q_barrier_mask'] += valid_mask.to(torch.float32).mean()
            qf_ = (batch['reward'] + (1 - batch['done'].float()) * self.FLAGS.gamma *
                   (min_next_qf - alpha * log_prob_actions))

        qfn_losses = torch.stack([F.mse_loss(qfn(batch['state'], batch['action']) * valid_mask, qf_ * valid_mask)
                                  for qfn in self.qfns])
        self.minimize(self.qfn_opt, qfn_losses.sum())

        return {
            'loss': qfn_losses.detach().cpu().numpy().round(6),
            'alpha': self.get_alpha().item(),
        }

    def post_step(self, output):
        if self.n_batches % 1000 == 0:
            loss = output['loss']
            alpha = output['alpha']
            info = lz.meters.purge('SAC/')
            lz.log.debug(f'[{self.name}] # {self.n_batches}: qfn loss = {loss}, alpha = {alpha:.6f}, '
                         f'barrier filter = {info["SAC/q_barrier_mask"]:.6f}, nll = {info["SAC/nll"]:.6f}')
