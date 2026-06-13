import sys
import os
import torch
from pydantic import TypeAdapter

from neural.classifier import DiscreteClassifier, DiscreteClassifierConfig
from neural.io import load

# Add the parent directory of 'scripts' (which contains config.py) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libemg.streamers import myo_streamer
from libemg.data_handler import OnlineDataHandler
from libemg.discrete import DiscreteControl

if __name__ == "__main__":
    adapter = TypeAdapter(DiscreteClassifierConfig)

    # # 2. Save directly to a JSON file
    # with open("../other/models/base/config.json", "wb") as f:
    #     f.write(adapter.dump_json(DiscreteClassifierConfig(), indent=2))

    _, sm = myo_streamer()
    odh = OnlineDataHandler(sm)
    model = load(r"E:\Programming\Projects\reaserch\new_wakeword\mci_wake\other\models\base")
    discrete = DiscreteControl(odh, 10, 5, model)
    model.eval()
    discrete.run()