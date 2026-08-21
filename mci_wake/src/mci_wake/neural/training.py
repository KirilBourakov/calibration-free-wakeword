from typing import Any, cast

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from numpy import typing as npt
from torch.utils.data import DataLoader

from mci_wake.data.types import EmgDataset
from mci_wake.neural import DiscreteClassifierConfig, DiscreteClassifier, DiscreteLightningModule
from mci_wake.neural.classifier import TrainData, make_data_loader


def train_model(
    train: EmgDataset,
    test: EmgDataset,
    model_config: DiscreteClassifierConfig = DiscreteClassifierConfig(),
    customers: TrainData | None = None,
) -> DiscreteClassifier:
    """Initializes and trains the DiscreteClassifier using the provided datasets.

    Args:
        train: EmgDataset
        test: EmgDataset
        model_config: the model configuration
        customers: Optional list of customer/user IDs used for training.

    Returns:
        DiscreteClassifier: The trained classifier instance.
    """
    if customers is not None:
        model_config.customers = customers

    print("Fitting Discrete Classifier...")
    tr_dl: DataLoader = make_data_loader(train.data, train.labels)
    te_dl: DataLoader = make_data_loader(test.data, test.labels)

    model = DiscreteLightningModule(model_config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        filename="best-model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
        verbose=True,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloaders=tr_dl, val_dataloaders=te_dl)

    return cast(DiscreteLightningModule, trainer.lightning_module).internals
