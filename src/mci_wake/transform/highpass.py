import numpy as np
from scipy.signal import butter, sosfiltfilt

from mci_wake.data import EmgDataset
from mci_wake.transform.base import AbstractTransform
import numpy.typing as npt

class HighPassFilter(AbstractTransform):
    def __init__(self, fs: float = 200, cutoff: float = 20.0, order: int = 4):
        self.fs = fs
        self.cutoff = cutoff
        self.order = order
        self.sos = butter(order, cutoff, btype='highpass', fs=fs, output='sos')

    def fit(self, emg: EmgDataset) -> None:
        pass

    def __call__(self, emg: EmgDataset | npt.NDArray[np.floating]) -> EmgDataset | npt.NDArray[np.floating]:
        if isinstance(emg, np.ndarray):
            return sosfiltfilt(self.sos, emg, axis=0).astype(np.float32)

        elif isinstance(emg, EmgDataset):
            new_data = [sosfiltfilt(self.sos, d, axis=0).astype(np.float32) for d in emg.data]
            return EmgDataset(
                data=new_data,
                labels=emg.labels,
                subjects=emg.subjects,
                is_normalized=emg.is_normalized
            )

        raise TypeError(f"Unsupported type for HighPassFilter: {type(emg)}")