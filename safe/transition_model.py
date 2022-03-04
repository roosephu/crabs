import torch
import torch.nn as nn
import torch.nn.functional as F
import lunzi as lz
import rlz
import pytorch_lightning as pl
import numpy as np


def mixup(batch, alpha=0.2):
    lambda_ = np.random.beta(alpha, alpha)
    batch_size = batch['state'].size(0)
    perm = torch.randperm(batch_size)
    return {
        'state': batch['state'] * lambda_ + batch['state'][perm] * lambda_,
        'action': batch['action'] * lambda_ + batch['action'][perm] * lambda_,
        'next_state': batch['next_state'] * lambda_ + batch['next_state'][perm] * lambda_,
    }


class TransitionModel(pl.LightningModule):

    class FLAGS(lz.BaseFLAGS):
        batch_size = 256
        weight_decay = 0.000075
        lr = 0.001
        mul_std = 0

    def __init__(self, dim_state, normalizer, n_units, *, name=''):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = n_units[0] - dim_state
        self.normalizer = normalizer
        self.net = rlz.MLP(n_units, activation=nn.SiLU)
        self.max_log_std = nn.Parameter(torch.full([dim_state], 0.5), requires_grad=True)
        self.min_log_std = nn.Parameter(torch.full([dim_state], -10.), requires_grad=True)
        self.training_loss = 0.
        self.val_loss = 0.
        self.name = name
        self.mul_std = self.FLAGS.mul_std
        self.automatic_optimization = False

    def forward(self, states, actions, det=True):
        output = self.net(self.normalizer(states), actions)
        mean, log_std = output.split(self.dim_state, dim=-1)
        if self.mul_std:
            mean = mean * self.normalizer.std
        mean = mean + states
        if det:
            return mean
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)
        return torch.distributions.Normal(mean, log_std.exp())

    def get_loss(self, batch, gp=False):
        # batch = mixup(batch)
        batch['state'].requires_grad_()
        predictions: torch.distributions.Normal = self(batch['state'], batch['action'], det=False)
        targets = batch['next_state']
        loss = -predictions.log_prob(targets).mean() + 0.001 * (self.max_log_std - self.min_log_std).mean()
        if gp:
            grad = torch.autograd.grad(loss.sum(), batch['state'], create_graph=True)[0]
            grad_penalty = (grad.norm(dim=-1) - 1).relu().pow(2).sum()
            loss = loss + 10 * grad_penalty
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.FLAGS.lr, weight_decay=self.FLAGS.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, gp=True)
        self.log(f'{self.name}/training_loss', loss.item(), on_step=False, on_epoch=True)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss, opt)
        nn.utils.clip_grad_norm_(self.parameters(), 10)
        opt.step()

        return {
            'loss': loss.item(),
        }

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log(f'{self.name}/val_loss', loss.item(), on_step=False, on_epoch=True)
        return {
            'loss': loss.item(),
        }

    def on_epoch_end(self) -> None:
        print('epoch', self.current_epoch, self.device)

    def test_stability(self, state, policy, horizon):
        states = [state]
        for i in range(horizon):
            action = policy(state)
            state = self(state, action)
            states.append(state)
        states = torch.stack(states)
        breakpoint()
        print(states.norm(dim=-1)[::(horizon - 1) // 10])


class GatedTransitionModel(TransitionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_net = rlz.MLP([self.dim_state + self.dim_action, 256, 256, self.dim_state * 2], activation=nn.SiLU,
                                output_activation=nn.Sigmoid)

    def forward(self, states, actions, det=True):
        nmlz_states = self.normalizer(states)
        reset, update = self.gate_net(nmlz_states, actions).split(self.dim_state, dim=-1)
        output = self.net(nmlz_states * reset, actions)
        mean, log_std = output.split(self.dim_state, dim=-1)
        mean = mean * update + states
        if det:
            return mean
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)
        return torch.distributions.Normal(mean, log_std.exp())
