import torch
import torch.nn as nn
import numpy as np
import lunzi as lz
import pytorch_lightning as pl


class EnsembleModel(pl.LightningModule):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.n_elites = self.n_models
        self.elites = []
        self.recompute_elites()
        self.automatic_optimization = False

    def recompute_elites(self):
        self.elites = list(range(len(self.models)))

    def forward(self, states, actions):
        n = len(states)

        perm = np.random.permutation(n)
        inv_perm = np.argsort(perm)

        next_states = []
        for i, (model_idx, indices) in enumerate(zip(self.elites, np.array_split(perm, len(self.elites)))):
            next_states.append(self.models[model_idx](states[indices], actions[indices]))
        return torch.cat(next_states, dim=0)[inv_perm]

    def get_nlls(self, states, actions, next_states):
        ret = []
        for model in self.models:
            distribution = model(states, actions, det=False)
            nll = -distribution.log_prob(next_states).mean().item()
            ret.append(nll)
        return ret

    def training_step(self, batch, batch_idx):
        total_loss = 0
        for i, model in enumerate(self.models):
            loss = model.get_loss(batch, gp=False)
            total_loss = total_loss + loss
            self.log(f'model/{i}/training_loss', loss.item())

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(total_loss, opt)
        nn.utils.clip_grad_norm_(self.parameters(), 10)
        opt.step()

    def validation_step(self, batch, batch_idx):
        for i, model in enumerate(self.models):
            loss = model.get_loss(batch)
            self.log(f'model/{i}/val_loss', loss.item(), on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)


class FLAGS(lz.BaseFLAGS):
    method = 'max-mean'
    scale = 1.0


class EnsembleUncertainty(nn.Module):
    FLAGS = FLAGS

    def __init__(self, ensemble, barrier):
        super().__init__()
        self.ensemble = ensemble
        self.barrier = barrier
        lz.log.warning(f"[Ensemble Uncertainty]: models = {ensemble.elites}")

    def forward(self, states, actions):
        if FLAGS.method in ['max-mean', 'max-sample_box', 'max-sample_gaussian', 'max-sample_box_corner']:
            all_next_states = []
            for idx in self.ensemble.elites:
                next_distribution = self.ensemble.models[idx](states, actions, det=False)
                mean, std = next_distribution.mean, next_distribution.stddev
                if FLAGS.method == 'max-mean':
                    next_states = mean
                elif FLAGS.method == 'max-sample_box':
                    next_states = mean + std * (torch.rand_like(std) * 2 - 1) * FLAGS.scale
                elif FLAGS.method == 'max-sample_box_corner':
                    next_states = mean + std * (torch.randint_like(std, 0, 1) * 2 - 1) * FLAGS.scale
                elif FLAGS.method == 'max-sample_gaussian':
                    next_states = mean + std * torch.randn_like(std) * FLAGS.scale
                else:
                    assert False, "???"
                all_next_states.append(next_states)
            all_next_states = torch.stack(all_next_states)
            all_nh = self.barrier(all_next_states)
            nh = all_nh.max(dim=0).values
            return nh
        else:
            assert False, "unknown uncertainty estimation!"


@torch.no_grad()
def model_rollout(model, policy, states, n_steps):
    all_states = []
    all_actions = []
    all_next_states = []
    for _ in range(n_steps):
        actions = policy(states)
        next_states = model(states, actions)
        if isinstance(next_states, torch.distributions.Distribution):
            next_states = next_states.sample()

        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)
        states = next_states

        if states.norm() > 1e8:
            break

    all_states = torch.stack(all_states)
    all_actions = torch.stack(all_actions)
    all_next_states = torch.stack(all_next_states)
    return all_states, all_actions, all_next_states
