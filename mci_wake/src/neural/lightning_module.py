import lightning as light
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from neural.classifier import DiscreteClassifierConfig, DiscreteClassifier, DL_input_data


class DiscreteLightningModule(light.LightningModule):
    def __init__(self, defn: DiscreteClassifier | DiscreteClassifierConfig):
        super().__init__()
        self.save_hyperparameters()
        self.internals: DiscreteClassifier = defn if isinstance(defn, DiscreteClassifier) else DiscreteClassifier(defn)
        self.config = self.internals.config

    def forward(self, x, lengths=None):
        return self.internals.forward_once(x, lengths)

    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        logits = self(x, lengths)
        loss = cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        logits = self(x, lengths)
        loss = cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        def lr_lambda(epoch):
            warmup_epochs = 5
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.9 ** (epoch - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }