from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import lunzi as lz

from ..policy import NetPolicy
from ..qfn import NetQFn
from ..trainer import BaseTrainer
from .utils import make_target_network, polyak_average


class FLAGS(lz.BaseFLAGS):
    gamma = 0.99
    lr = 3e-4
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    batch_size = 256

    tau = 0.005


class TD3Trainer(nn.Module, BaseTrainer):
    FLAGS = FLAGS

    def __init__(self, policy: NetPolicy, qfns: List[NetQFn], sampler,
                 *, device='cpu', name='TD3', max_action=1, **kwargs):
        super().__init__()
        self.FLAGS = FLAGS.copy().merge(kwargs)
        self.policy = policy
        self.policy_target = make_target_network(policy)
        self.qfns = nn.ModuleList(qfns)
        self.qfns_target = make_target_network(self.qfns)
        self.sampler = sampler
        self.name = name
        self.max_action = max_action

        self.qfns_opt = torch.optim.Adam(self.qfns.parameters(), FLAGS.lr)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), FLAGS.lr)

        self.init_trainer(device=device)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            next_actions = self.policy_target(batch['next_state'])

            noises = (torch.randn_like(batch['action']) * self.FLAGS.policy_noise)
            noises = noises.clamp(-self.FLAGS.noise_clip, self.FLAGS.noise_clip) * self.max_action
            next_actions = next_actions.add(noises).clamp(-self.max_action, self.max_action)

            next_qfs = [qfn(batch['next_state'], next_actions) for qfn in self.qfns_target]
            min_next_qf, _ = torch.stack(next_qfs).min(dim=0)
            qf_ = (batch['reward'] + (1 - batch['done'].float()) * self.FLAGS.gamma * min_next_qf).detach()

        qfn_losses = torch.stack([F.mse_loss(qfn(batch['state'], batch['action']), qf_) for qfn in self.qfns])
        self.minimize(self.qfns_opt, qfn_losses.sum())

        if self.n_batches % self.FLAGS.policy_freq == 0:
            policy_loss = -self.qfns[0](batch['state'], self.policy(batch['state'])).mean()
            self.minimize(self.policy_opt, policy_loss)

            polyak_average(self.policy, self.policy_target, self.FLAGS.tau)
            polyak_average(self.qfns, self.qfns_target, self.FLAGS.tau)
        return {'loss': qfn_losses.detach().cpu().numpy()}

    def post_step(self, output):
        if self.n_batches % 1000 == 0:
            lz.log.info(f'[{self.name}] # {self.n_batches}: loss = {output["loss"]}')

    def configure_train_dataloader(self):
        while True:
            yield self.sampler(self.FLAGS.batch_size)
