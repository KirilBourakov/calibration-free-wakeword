import sys
import os
import torch
from pydantic import TypeAdapter

from neural.classifier import DiscreteClassifier, DiscreteClassifierConfig
from neural.io import load
from neural.lightning_module import DiscreteLightningModule

# Add the parent directory of 'scripts' (which contains config.py) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libemg.streamers import myo_streamer
from libemg.data_handler import OnlineDataHandler
from libemg.discrete import DiscreteControl


if __name__ == "__main__":
    # Allowlist custom classes for safe unpickling in PyTorch 2.6+
    torch.serialization.add_safe_globals([DiscreteClassifierConfig, DiscreteClassifier])
    
    adapter = TypeAdapter(DiscreteClassifierConfig)

    # # 2. Save directly to a JSON file
    # with open("../other/models/base/config.json", "wb") as f:
    #     f.write(adapter.dump_json(DiscreteClassifierConfig(), indent=2))

    _, sm = myo_streamer()
    odh = OnlineDataHandler(sm)

    checkpoint_path = r"E:\Programming\Projects\reaserch\new_wakeword\mci_wake\src\lightning_logs\version_0\checkpoints\best-model-epoch=07-val_acc=0.98.ckpt"
    lightning_model = DiscreteLightningModule.load_from_checkpoint(checkpoint_path)
    model = lightning_model.internals
    model.eval()

    discrete = DiscreteControl(odh, 10, 5, model)
    discrete.run()