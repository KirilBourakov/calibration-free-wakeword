import random
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

import torch.nn as nn
from pydantic import Field, TypeAdapter
from pydantic.dataclasses import dataclass
from torch.nn import RNNBase
from torch.utils.data import DataLoader, Dataset

from libemg.utils import get_windows
from mci_wake.data.types import EmgDataset, TrainData
from mci_wake.model.abstract import AbstractModel

_CONFIG_NAME = 'config.json'
_STATE_DICT_NAME = 'state.pt'

@dataclass
class DiscreteClassifierConfig:
    emg_size: tuple[int, int, int] = (32, 8, 10)
    window_size: int = 10
    increment: int = 5
    template_size: int = 250
    temporal_hidden_size: int = 128
    temporal_layers: int = 3
    mlp_layers: list[int] = Field(default_factory=lambda: [128, 64, 32])
    n_classes: int = 6
    type: str = 'GRU'
    conv_kernel_sizes: list[int] = Field(default_factory=lambda: [3, 3, 3])
    conv_out_channels: list[int] = Field(default_factory=lambda: [16, 32, 64])
    lr: float = 1e-3
    customers: TrainData = Field(default_factory=TrainData)
    gestures: list[str] = Field(default_factory=list)

    @property
    def file_name(self):
        return f"ADL_{self.type}"

class DiscreteModel(AbstractModel):
    """
    End-to-end discrete neural model implementing AbstractModel.
    """

    def __init__(
        self,
        config: DiscreteClassifierConfig | None = None,
        net: "_DiscreteClassifierNet | None" = None,
        device: str = 'cpu',
    ):
        self.config = config if config is not None else DiscreteClassifierConfig()
        self.device = device
        self.net = net if net is not None else _DiscreteClassifierNet(self.config)
        self.net.to(self.device)

    def predict(self, data: npt.NDArray[np.float32], device: str | None = None) -> int:
        """Takes raw EMG segment (time, channels) and returns predicted class."""
        tensor = self._prepare_input(data, device=device)
        self.net.eval()
        with torch.no_grad():
            output = self.net.forward_once(tensor)
            pred = output.argmax(dim=1).item()
        return pred

    def predict_logits(self, data: npt.NDArray[np.float32], device: str | None = None) -> torch.Tensor:
        """Returns raw output logits for analysis / visualization."""
        tensor = self._prepare_input(data, device=device)
        self.net.eval()
        with torch.no_grad():
            return self.net.forward_once(tensor)

    def fit(
        self,
        train: EmgDataset,
        test: EmgDataset,
        customers: TrainData | None = None
    ) -> "DiscreteModel":
        """
        Trains the internal neural network using PyTorch Lightning.
        Automatically slices raw EMG trials into sliding windows if not already windowed.
        """
        from mci_wake.model.neural.training import train_model

        # Window data set
        train = self._get_windows(train)
        test = self._get_windows(test)

        # Train
        trained_model = train_model(
            train=train,
            test=test,
            model_config=self.config,
            customers=customers
        )
        self.net = trained_model.net
        self.config = trained_model.config
        return self

    def save(self, path: str | Path) -> None:
        """Saves model configuration and weights to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        adapt = TypeAdapter(DiscreteClassifierConfig)
        with open(path / _CONFIG_NAME, "wb") as file:
            file.write(adapt.dump_json(self.config, indent=2))

        torch.save(self.net.state_dict(), path / _STATE_DICT_NAME)

    @classmethod
    def load(cls, path: str | Path, device: str = 'cpu') -> "DiscreteModel":
        """Loads model from a directory containing config.json and state.pt."""
        path = Path(path)
        adapt = TypeAdapter(DiscreteClassifierConfig)
        with open(path / _CONFIG_NAME, "rb") as file:
            json_string = file.read()
        config = adapt.validate_json(json_string)

        model = DiscreteModel(config=config, device=device)
        state_dict = torch.load(path / _STATE_DICT_NAME, map_location=device, weights_only=True)
        model.net.load_state_dict(state_dict, strict=True)

        return model

    @classmethod
    def load_from_checkpoint(
        cls, ckpt_path: str | Path, device: str = 'cpu', weights_only: bool = False
    ) -> "DiscreteModel":
        """Loads trained weights directly from a PyTorch Lightning .ckpt checkpoint."""
        from mci_wake.model.neural.lightning_module import DiscreteLightningModule

        torch.serialization.add_safe_globals([DiscreteClassifierConfig, cls, TrainData])

        lightning_module = DiscreteLightningModule.load_from_checkpoint(
            str(ckpt_path), map_location=device, weights_only=weights_only
        )
        return cls(
            config=lightning_module.config,
            net=lightning_module.internals,
            device=device,
        )

    def eval(self) -> "DiscreteModel":
        self.net.eval()
        return self

    def set_train_mode(self, mode: bool = True) -> "DiscreteModel":
        self.net.train(mode)
        return self

    @property
    def n_classes(self) -> int:
        return self.config.n_classes

    def _get_windows(self, data: EmgDataset) -> EmgDataset:
        """Extracts sliding windows from an EmgDataset using this model's configuration."""
        windowed_data = [
            get_windows(d, self.config.window_size, self.config.increment).astype(np.float32)
            for d in data.data
        ]
        return EmgDataset(
            data=windowed_data,
            labels=data.labels.copy(),
            subjects=data.subjects.copy(),
            is_normalized=data.is_normalized,
        )

    def _prepare_input(self, data: npt.NDArray[np.float32], device: str | None = None) -> torch.Tensor:
        """Extract sliding windows from ndarray"""
        windows = get_windows(data, self.config.window_size, self.config.increment).astype(np.float32)
        return torch.tensor(windows[None], dtype=torch.float32, device=device or self.device)


class _DiscreteClassifierNet(nn.Module):
    """
    Internal PyTorch neural network module.
    Takes 4D tensor inputs of shape (batch, seq_len, channels, samples) and outputs class logits.
    """

    def __init__(self, config: DiscreteClassifierConfig):
        super().__init__()

        fix_random_seed(0)

        self.config = config
        self.file_name = config.file_name

        dropout = 0.2

        self.conv_layers = nn.ModuleList()
        in_channels = config.emg_size[1]  # Channels in EMG signal
        for i in range(len(config.conv_out_channels)):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=config.conv_out_channels[i],
                    kernel_size=config.conv_kernel_sizes[i],
                    padding='same',
                )
            )
            self.conv_layers.append(nn.BatchNorm1d(config.conv_out_channels[i]))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(kernel_size=2))
            self.conv_layers.append(nn.Dropout(dropout))
            in_channels = config.conv_out_channels[i]

        spoof_emg_input = torch.zeros((1, *config.emg_size))
        conv_out = self.forward_conv(spoof_emg_input)
        conv_out_size = conv_out.shape[-1]

        # Temporal feature extraction
        self.temporal: RNNBase
        if config.type == 'LSTM':
            self.temporal = nn.LSTM(
                conv_out_size,
                config.temporal_hidden_size,
                num_layers=config.temporal_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif config.type == 'BILSTM':
            self.temporal = nn.LSTM(
                conv_out_size,
                config.temporal_hidden_size,
                num_layers=config.temporal_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
        elif config.type == 'RNN':
            self.temporal = nn.RNN(
                conv_out_size,
                config.temporal_hidden_size,
                num_layers=config.temporal_layers,
                batch_first=True,
                dropout=dropout,
                nonlinearity='relu',
            )
        elif config.type == 'GRU':
            self.temporal = nn.GRU(
                conv_out_size,
                config.temporal_hidden_size,
                num_layers=config.temporal_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Invalid selection of model type '{config.type}'.")

        emg_output_shape = self.forward_temporal(conv_out).shape[-1]

        self.initial_layer = nn.Linear(emg_output_shape, config.mlp_layers[0])
        self.layer1 = nn.Linear(config.mlp_layers[0], config.mlp_layers[1])
        self.layer2 = nn.Linear(config.mlp_layers[1], config.mlp_layers[2])
        self.output_layer = nn.Linear(config.mlp_layers[-1], config.n_classes)
        self.relu = nn.ReLU()

    def forward_conv(self, x):
        batch_size, seq_len, channels, samples = x.shape
        x = x.view(batch_size * seq_len, channels, samples)

        for layer in self.conv_layers:
            x = layer(x)

        _, channels_out, samples_out = x.shape
        x = x.view(batch_size, seq_len, channels_out * samples_out)
        return x

    def forward_mlp(self, x):
        out = self.initial_layer(x)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

    def forward_once(self, emg, emg_len=None):
        out = self.forward_conv(emg)
        out = self.forward_temporal(out, emg_len)
        out = self.forward_mlp(out)
        return out

    def forward_temporal(self, emg, lengths=None):
        out, _ = self.temporal(emg)
        if lengths is not None:
            out = torch.stack([s[lengths[i] - 1] for i, s in enumerate(out)])
        else:
            out = out[:, -1, :]
        return out


class DL_input_data(Dataset):
    def __init__(self, windows, classes):
        data, lengths = self.buffer(windows)
        self.data = data
        self.lengths = lengths
        self.classes = torch.tensor(classes, dtype=torch.long)

    def buffer(self, input):
        lengths = torch.tensor(np.array([len(w) for w in input]), dtype=torch.long)
        max_len = max(lengths).item()
        num_channels = input[0].shape[1]
        num_samples = input[0].shape[2] if len(input[0].shape) > 2 else 1
        padded_emg = np.zeros((len(input), max_len, num_channels, num_samples))
        for i, e in enumerate(input):
            padded_emg[i, 0 : e.shape[0], :, :] = e

        return torch.tensor(padded_emg, dtype=torch.float32), lengths

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.classes[idx]
        length = self.lengths[idx]
        return data, label, length

    def __len__(self):
        return self.data.shape[0]


def make_data_loader(windows, classes, batch_size=64, shuffle=True):
    obj = DL_input_data(windows, classes)
    dl = DataLoader(obj, batch_size=batch_size, shuffle=shuffle)
    return dl


def fix_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
