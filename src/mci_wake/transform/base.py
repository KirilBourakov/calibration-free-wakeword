from abc import ABC, abstractmethod
from typing import overload

import numpy as np
import numpy.typing as npt

from mci_wake.data import EmgDataset


class AbstractTransform(ABC):
    @abstractmethod
    def fit(self, emg: EmgDataset) -> None:
        ...

    @overload
    def __call__(self, emg: EmgDataset) -> EmgDataset: ...
    @overload
    def __call__(self, emg: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...
    @abstractmethod
    def __call__(self, emg: EmgDataset | npt.NDArray[np.floating]) -> EmgDataset | npt.NDArray[np.floating]:
        ...