import torch

from mci_wake.data import load_raw_data
from mci_wake.data_handler.online import CompatibleOnlineDataHandler
from mci_wake.model.neural import DiscreteModel, DiscreteClassifierConfig, TrainData
from mci_wake.orchestration.model_chain import ModelChain
from mci_wake.transform.highpass import HighPassFilter
from mci_wake.transform.rest_normalization import RestNormalizer
from mci_wake.transform.transform import Transform
from mci_wake.utils.io import modelpath

if __name__ == "__main__":
    # Allowlist custom classes for safe unpickling in PyTorch 2.6+
    torch.serialization.add_safe_globals([DiscreteClassifierConfig, DiscreteModel, TrainData])

    print("Computing dataset normalization statistics...")
    emg, adl = load_raw_data()
    transforms = Transform(HighPassFilter(), RestNormalizer())
    transforms.fit(emg)

    model1 = DiscreteModel.load_from_checkpoint(modelpath(8))

    # model2 = DiscreteModel.load_from_checkpoint(
    #     r"D:\Coding\calibration-free-wakeword\mci_wake\scripts\lightning_logs\version_0\checkpoints\best-model-epoch=09-val_acc=0.99.ckpt"
    # )

    discrete = ModelChain(CompatibleOnlineDataHandler(), 10, 5, [model1], transforms=transforms)
    discrete.run()

