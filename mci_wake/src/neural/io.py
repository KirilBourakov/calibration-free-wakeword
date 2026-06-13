from pathlib import Path

import torch
from pydantic import TypeAdapter

from neural.classifier import DiscreteClassifierConfig, DiscreteClassifier

_CONFIG_NAME = 'config.json'
_STATE_DICT_NAME = 'state.pt'

def load(path: str | Path) -> DiscreteClassifier:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    adapt = TypeAdapter(DiscreteClassifierConfig)
    with open(path / _CONFIG_NAME, "rb") as file:
        json_string = file.read()
    config = adapt.validate_json(json_string)

    classifier = DiscreteClassifier(config)
    state_dict = torch.load(path / _STATE_DICT_NAME, map_location='cpu', weights_only=True)
    classifier.load_state_dict(state_dict, strict=True)

    return classifier

def save(classifier: DiscreteClassifier, path: str | Path) -> None:
    path = Path(path)

    adapt = TypeAdapter(DiscreteClassifierConfig)
    with open(path / _CONFIG_NAME, "wb") as file:
        file.write(adapt.dump_json(classifier.config, indent=2))

    torch.save(classifier.state_dict(), path / _STATE_DICT_NAME)


