import torch

from mci_wake.data import load_raw_data
from mci_wake.data_handler.online import CompatibleOnlineDataHandler
from mci_wake.neural.classifier import DiscreteClassifier, DiscreteClassifierConfig, TrainData
from mci_wake.neural.lightning_module import DiscreteLightningModule
from mci_wake.orchestration.wake_detect import WakeDetect
from mci_wake.transform.highpass import HighPassFilter
from mci_wake.transform.rest_normalization import RestNormalizer
from mci_wake.transform.transform import Transform

if __name__ == "__main__":
    # Allowlist custom classes for safe unpickling in PyTorch 2.6+
    torch.serialization.add_safe_globals([DiscreteClassifierConfig, DiscreteClassifier, TrainData])

    print("Computing dataset normalization statistics...")
    emg, adl = load_raw_data()
    transforms = Transform(HighPassFilter(), RestNormalizer())
    transforms.fit(emg)

    lightning_model = DiscreteLightningModule.load_from_checkpoint(
        r"D:\Coding\calibration-free-wakeword\scripts\lightning_logs\version_5\checkpoints\best-model-epoch=07-val_acc=0.99.ckpt"
    )
    model1 = lightning_model.internals

    # lightning_model = DiscreteLightningModule.load_from_checkpoint(
    #     r"D:\Coding\calibration-free-wakeword\mci_wake\scripts\lightning_logs\version_0\checkpoints\best-model-epoch=09-val_acc=0.99.ckpt"
    # )
    # model2 = lightning_model.internals

    discrete = WakeDetect(CompatibleOnlineDataHandler(), 10, 5, [model1], transforms=transforms)
    discrete.run()
