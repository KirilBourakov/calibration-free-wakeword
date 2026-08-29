import torch

from mci_wake.data import load_raw_data
from mci_wake.data.normalization import Normalize
from mci_wake.data_handler.online import CompatibleOnlineDataHandler
from mci_wake.neural.classifier import DiscreteClassifier, DiscreteClassifierConfig, TrainData
from mci_wake.neural.lightning_module import DiscreteLightningModule
from mci_wake.orchestration.wake_detect import WakeDetect

if __name__ == "__main__":
    # Allowlist custom classes for safe unpickling in PyTorch 2.6+
    torch.serialization.add_safe_globals([DiscreteClassifierConfig, DiscreteClassifier, TrainData])

    print("Computing dataset normalization statistics...")
    emg, adl = load_raw_data()
    normalizer = Normalize.create(emg.combine(adl))

    lightning_model = DiscreteLightningModule.load_from_checkpoint(
        r"D:\Coding\calibration-free-wakeword\mci_wake\scripts\lightning_logs\version_4\checkpoints\best-model-epoch=09-val_acc=0.99.ckpt"
    )
    model1 = lightning_model.internals

    lightning_model = DiscreteLightningModule.load_from_checkpoint(
        r"D:\Coding\calibration-free-wakeword\mci_wake\scripts\lightning_logs\version_5\checkpoints\best-model-epoch=07-val_acc=0.99.ckpt"
    )
    model2 = lightning_model.internals

    discrete = WakeDetect(CompatibleOnlineDataHandler(), 10, 5, [model1, model2], normalize=normalizer)
    discrete.run()
