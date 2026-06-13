import sys
import os
import torch

from neural.classifier import DiscreteClassifier

# Add the parent directory of 'scripts' (which contains config.py) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libemg.streamers import myo_streamer
from libemg.data_handler import OnlineDataHandler
from libemg.discrete import DiscreteControl

if __name__ == "__main__":
    _, sm = myo_streamer()
    odh = OnlineDataHandler(sm)
    weights = torch.load(r'E:\Programming\Projects\reaserch\new_wakeword\mci_wake\other\base.model', map_location=torch.device('cpu'))
    model = DiscreteClassifier((32, 8, 10))
    model.load_state_dict(weights, strict=True)
    model.eval()
    discrete = DiscreteControl(odh, 10, 5, model)
    discrete.run()