import torch
import pytorch_lightning as pl
import lunzi as lz


class SimpleTrainer:

    def __init__(self, model: pl.LightningModule, *, device='cpu'):
        self.model = model
        self.device = device
        self.current_batch = 0
        self.current_epoch = 0
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.model.to(device)
        self.optimizers = self.model.configure_optimizers()
        self.model.trainer = self

    @torch.no_grad()
    def validate(self, *, dataloader=None):
        self.model.eval()
        self.model.on_validation_epoch_start()

        if dataloader is None:
            if self.val_dataloader is not None:
                self.val_dataloader = self.model.val_dataloader()
            dataloader = self.val_dataloader

        for batch in dataloader:
            batch = self.transfer_to_device(batch)
            self.model.validation_step(batch)
        self.model.on_validation_epoch_end()

    def transfer_to_device(self, batch):
        return lz.utils.rec_map(batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else None)

    @torch.no_grad()
    def test(self, *, dataloader=None):
        self.model.eval()
        self.model.on_test_epoch_start()

        if dataloader is None:
            if self.test_dataloader is not None:
                self.test_dataloader = self.model.test_dataloader()
            dataloader = self.test_dataloader

        for batch in dataloader:
            batch = self.transfer_to_device(batch)
            self.model.test_step(batch)
        self.model.on_test_epoch_end()

    def fit(self, *, dataloader=None):
        self.current_epoch += 1
        self.model.train()
        self.model.on_train_epoch_start()

        if dataloader is None:
            if self.train_dataloader is not None:
                self.train_dataloader = self.model.train_dataloader()
            dataloader = self.train_dataloader

        unused = []
        for batch in dataloader:
            unused.append(self.step(batch=batch))
        self.model.on_train_epoch_end(unused=unused)

    def step(self, *, batch=None):
        self.current_batch += 1

        if batch is None:
            batch = next(self.train_dataloader)

        batch = self.transfer_to_device(batch)
        return self.model.training_step(batch)
