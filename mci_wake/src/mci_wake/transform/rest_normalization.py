import numpy as np
import numpy.typing as npt

from mci_wake.data import EmgDataset
from mci_wake.transform.base import AbstractTransform


class RestNormalizer(AbstractTransform):
    def __init__(self, rest_class_label: int = 0, epsilon: float = 1e-8):
        self.rest_class_label = rest_class_label
        self.epsilon = epsilon
        self.rest_std_ = None

    def fit(self, emg: EmgDataset) -> None:
        """
        Calculates the standard deviation of the resting data PER CHANNEL.
        """
        rest_signals = [d for d, l, s in emg if l == self.rest_class_label]

        if not rest_signals:
            raise ValueError(f"No rest data found for label {self.rest_class_label} in dataset.")

        rest_data = np.concatenate(rest_signals, axis=0)
        self.rest_std_ = np.std(rest_data, axis=0, keepdims=True).astype(np.float32) # (1, channels)

    def __call__(self, emg: EmgDataset | npt.NDArray[np.floating]) -> EmgDataset | npt.NDArray[np.float32]:
        if self.rest_std_ is None:
            raise RuntimeError("RestNormalizer has not been fitted yet. Call fit() first.")

        if isinstance(emg, np.ndarray):
            return (emg / (self.rest_std_ + self.epsilon)).astype(np.float32)

        elif isinstance(emg, EmgDataset):
            new_data = [(d / (self.rest_std_ + self.epsilon)).astype(np.float32) for d in emg.data]
            return EmgDataset(
                data=new_data,
                labels=emg.labels,
                subjects=emg.subjects,
                is_normalized=True
            )

        raise TypeError(f"Unsupported type for RestNormalizer: {type(emg)}")