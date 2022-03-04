from typing import Union, Any, Iterator
import torch

import lunzi as lz


class BaseTrainer:  # step-based trainer
    device: Union[str, torch.device]
    n_batches: int
    log_interval: int
    name: str
    train_dataloader: Iterator

    def init_trainer(self, *, device='cpu'):
        self.device = device
        self.n_batches = 0
        self.train_dataloader = self.configure_train_dataloader()
        self.to(device)

    def transfer_to_device(self, batch):
        return lz.utils.rec_map(batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else None)

    def minimize(self, optimizer, loss, create_graph=False, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph, create_graph=create_graph)
        optimizer.step()

    def post_step(self, output):
        pass

    def configure_train_dataloader(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def step(self, batch=None):
        self.n_batches += 1

        if batch is None:
            batch = next(self.train_dataloader)
        batch = self.transfer_to_device(batch)
        output = self.training_step(batch, self.n_batches)
        self.post_step(output)
