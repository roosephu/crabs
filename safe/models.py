import numpy as np
import lunzi as lz
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import einops
from torchmetrics import MeanMetric
from .simple_trainer import SimpleTrainer


class FLAGS(lz.BaseFLAGS):

    class model(lz.BaseFLAGS):
        batch = 0
        n_models = 7
        n_elite_models = 5
        early_stop_patience = 5
        max_epochs = 100_000

    log_std_coeff = 0.01
    init_max_log_std = -2.0
    lr = 0.0003
    weight_decay = 0.00075


def mean_meters(meters):
    return np.mean([meter.compute() for meter in meters])


def update_meters(meters, values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    for meter, value in zip(meters, values):
        meter(value)


# Time in training is much longer than inference, so we need to speed up training
# we batch all matmul, so it can be much faster for training.
# It can even improve the speed of inference, if the model is small and pytorch
# overhead is large.
class BatchEnsembleModel(pl.LightningModule):
    FLAGS = FLAGS

    def __init__(self, n_models, n_elites, dim_state, dim_action, normalizer, batch_net):
        super().__init__()
        self.n_models = n_models
        self.n_elites = n_elites
        self.elites = np.arange(n_elites, dtype=np.int64)

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.normalizer = normalizer
        self.batch_net = batch_net
        self.n_grad_iters = 0

        self.max_log_std = nn.Parameter(torch.full([n_models, 1, dim_state + 1], FLAGS.init_max_log_std), requires_grad=True)
        self.min_log_std = nn.Parameter(torch.full([n_models, 1, dim_state + 1], -10.), requires_grad=True)

        self.mse_std = torch.ones(dim_state + 1, requires_grad=False)
        self.avg_train_loss = MeanMetric()
        self.avg_train_nll = [MeanMetric() for _ in range(n_models)]
        self.avg_train_mse = [MeanMetric() for _ in range(n_models)]
        self.avg_val_nll = [MeanMetric() for _ in range(n_models)]
        self.avg_val_mse = [MeanMetric() for _ in range(n_models)]
        self.avg_test_nll = [MeanMetric() for _ in range(n_models)]
        self.avg_test_mse = [MeanMetric() for _ in range(n_models)]

        self.expl_epoch = 0

        self.automatic_optimization = False

    def recompute_elites(self):
        self.elites = np.argsort([meter.compute() for meter in self.avg_val_nll])[:self.n_elites]

    def forward(self, states, actions, det=True):
        inputs = torch.cat([self.normalizer(states), actions], dim=-1)  # [NBI]
        outputs = self.batch_net(inputs)  # [NBO]
        mean, log_std = outputs.split(self.dim_state + 1, dim=-1)  # [NBO], [NBO]
        mean = mean + torch.cat([states, torch.zeros(*states.shape[:-1], 1, device=states.device)], dim=-1)  # [NBO]
        if det:
            return mean
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)
        return torch.distributions.Normal(mean, log_std.exp())  # [NBO]

    def get_metrics(self, batch):
        predictions: torch.distributions.Normal = self(batch['state'], batch['action'], det=False)  # [NBO]
        targets = torch.cat([batch['next_state'], batch['reward'][..., None]], dim=-1)  # [BO]
        nll = -predictions.log_prob(targets).mean([-1, -2])  # [N]
        loss = nll.sum() + FLAGS.log_std_coeff * (self.max_log_std - self.min_log_std).mean()
        # It's important to use `nll.sum()` instead of `nll.mean()`.
        # Adam is scale-invariant to the total loss, but we have weight decay here, so the scale is imporant.

        self.mse_std = self.mse_std.to(targets.device)
        return {
            'nll': nll,
            'loss': loss,
            'mse': ((predictions.mean - targets) / self.mse_std).pow(2).mean(dim=[-1, -2]),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    def on_train_epoch_start(self) -> None:
        for meter in self.avg_train_nll + self.avg_train_mse:
            meter.reset()
        self.avg_train_loss.reset()
        self.n_grad_iters = 0

    def training_step(self, batch):
        self.n_grad_iters += 1
        metrics = self.get_metrics(batch)
        loss = metrics['loss']
        self.avg_train_loss.update(metrics['loss'].item())
        update_meters(self.avg_train_nll, metrics['nll'])
        update_meters(self.avg_train_mse, metrics['mse'])

        opt = self.optimizers(use_pl_optimizer=False)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def on_validation_epoch_start(self) -> None:
        for meter in self.avg_val_mse + self.avg_val_nll:
            meter.reset()

    def validation_step(self, batch):
        metrics = self.get_metrics(batch)
        update_meters(self.avg_val_nll, metrics['nll'])
        update_meters(self.avg_val_mse, metrics['mse'])

    def on_validation_epoch_end(self) -> None:
        wandb.log({
            'model.train.loss': self.avg_train_loss.compute(),
            'model.train.nll': mean_meters(self.avg_train_nll),
            'model.train.mse': mean_meters(self.avg_train_mse),
            'model.val.nll': mean_meters(self.avg_val_nll),
            'model.val.mse': mean_meters(self.avg_val_mse),
            'epoch': self.current_epoch,
        })
        with np.printoptions(precision=6, linewidth=150, sign=' ', floatmode='fixed'):
            val_nll = str(np.array([meter.compute() for meter in self.avg_val_nll]))
            train_nll = str(np.array([meter.compute() for meter in self.avg_train_nll]))
        lz.log.info(
            f"[model] epoch = {self.current_epoch}, "
            f"training loss = {self.avg_train_loss.compute():.6f}, "
            f"training nll = {train_nll}, "
            f"training mse = {mean_meters(self.avg_train_mse):.6f}, "
            f"val nll = {val_nll}, "
            f"val mse = {mean_meters(self.avg_val_mse):.6f}"
        )

    def on_test_epoch_start(self) -> None:
        for meter in self.avg_test_nll + self.avg_test_mse:
            meter.reset()

    def test_step(self, batch):
        metrics = self.get_metrics(batch)
        update_meters(self.avg_test_nll, metrics['nll'])
        update_meters(self.avg_test_mse, metrics['mse'])

    def on_test_epoch_end(self):
        wandb.log({
            'model.train.nll': mean_meters(self.avg_train_nll),
            'model.train.loss': self.avg_train_loss.compute(),
            'model.val.nll': mean_meters(self.avg_val_nll),
            'model.val.mse': mean_meters(self.avg_val_mse),
            'model.test.nll': mean_meters(self.avg_test_nll),
            'model.test.mse': mean_meters(self.avg_test_mse),
            'epoch': self.current_epoch,
            'expl_epoch': self.expl_epoch,
        })
        lz.log.info(
            f"[model] epoch = {self.current_epoch}, "
            f"test nll = {mean_meters(self.avg_test_nll):.6f}, "
            f"test mse = {mean_meters(self.avg_test_mse):.6f}"
        )

    # state dict of a single model
    def single_state_dict(self, index):
        # only parameters in `batch_net` is in ensemble.
        state_dict = self.batch_net.state_dict()
        ret = {key: value[index] if isinstance(value, torch.Tensor) else value for key, value in state_dict.items()}
        ret['max_log_std'] = self.max_log_std[index]
        ret['min_log_std'] = self.min_log_std[index]
        return {k: v.clone() for k, v in ret.items()}

    @torch.no_grad()
    def load_single_state_dict(self, index, single_state_dict):
        state_dict = self.batch_net.state_dict()  # to get a map from key to a Tensor
        state_dict['max_log_std'] = self.max_log_std
        state_dict['min_log_std'] = self.min_log_std
        for key, value in single_state_dict.items():
            state_dict[key][index] = value


class BatchSubDataset:
    def __init__(self, buf, indices, n, batch_size):
        self.indices = indices.copy()
        self.batch_size = batch_size
        self.buf = buf
        self.n = n

    def __iter__(self):
        rng = np.random.default_rng()
        b = self.batch_size
        m = len(self.indices)
        for i in range(m // b):  # drop last batch
            if b == m:
                batch_idx = self.indices[None, :].repeat(self.n, 1)
            else:
                batch_idx = self.indices[rng.integers(m, size=(self.n * b))]
            batch = self.buf.sample(indices=batch_idx)
            batch = {k: v.view(self.n, b, *v.shape[1:]) for k, v in batch.items()}
            yield batch


class BatchModelModule:
    def __init__(self, dim_state, dim_action, batch_net, normalizer, device):

        self.n_models = FLAGS.model.n_models
        self.model = BatchEnsembleModel(
            self.n_models, FLAGS.model.n_elite_models, dim_state, dim_action, normalizer, batch_net)
        self.trainer = SimpleTrainer(self.model, device=device)
        self.normalizer = normalizer
        self.dim_state = dim_state
        self.dim_action = dim_action
        lz.log.info(f"model architecture: \n{batch_net}")

    @lz.timer
    def train(self, buf):
        from collections import namedtuple

        lz.log.info(f"train model")
        self.normalizer.fit(buf['state'])

        n = len(buf)
        perm = np.random.permutation(n)

        val_size = min(int(n * 0.2), 5000)
        train_dataloader = BatchSubDataset(buf, perm[val_size:], self.n_models, 256)
        val_dataloader = BatchSubDataset(buf, perm[:val_size], self.n_models, val_size)

        model, trainer = self.model, self.trainer
        Checkpoint = namedtuple('Checkpoint', 'val_nll, state_dict')
        trainer.validate(dataloader=val_dataloader)
        ckpts = [Checkpoint(model.avg_val_nll[i].compute(), model.single_state_dict(i)) for i in range(self.n_models)]

        n_grad_iters = 0
        since_updated = 0

        while since_updated < FLAGS.model.early_stop_patience:
            trainer.fit(dataloader=train_dataloader)
            trainer.validate(dataloader=val_dataloader)
            n_grad_iters += model.n_grad_iters

            since_updated += 1
            for i in range(self.n_models):
                val_nll = model.avg_val_nll[i].compute()
                if val_nll < ckpts[i].val_nll:
                    ckpts[i] = Checkpoint(val_nll, model.single_state_dict(i))
                    since_updated = 0

        lz.log.info(f"resetting model, val nll = {np.array([ckpt.val_nll for ckpt in ckpts])}")
        for i in range(self.n_models):
            model.load_single_state_dict(i, ckpts[i].state_dict)

        trainer.validate(dataloader=val_dataloader)
        lz.log.debug(f"# grad iters = {n_grad_iters}")
        lz.log.debug(f"max max log std = {model.max_log_std.max().item()}, "
                     f"min min log std = {model.min_log_std.min().item()}")

    def test(self, buf, expl_epoch):
        self.model.expl_epoch = expl_epoch
        n = len(buf)
        test_dataloader = BatchSubDataset(buf, np.arange(n), self.n_models, n)
        self.trainer.test(dataloader=test_dataloader)

    def sample(self, states, actions):
        self.model.eval()
        n = len(states)

        states = einops.repeat(states, "B S -> N B S", n=self.n_models)
        actions = einops.repeat(actions, "B A -> N B A", n=self.n_models)
        predictions = self.model(states, actions, det=False).rsample()  # [NBO]
        indices = np.random.choice(self.model.elites, size=[n])
        return predictions[indices, range(n), :]
