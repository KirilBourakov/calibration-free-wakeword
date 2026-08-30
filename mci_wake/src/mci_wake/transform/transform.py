from abc import ABC
from typing import overload

import numpy as np
from numpy import typing as npt

from mci_wake.data import EmgDataset
from mci_wake.transform.base import AbstractTransform


class Transform(AbstractTransform):
    def __init__(self, *transforms: AbstractTransform):
        self.transforms = transforms


    def fit(self, emg: EmgDataset) -> None:
        for t in self.transforms:
            t.fit(emg)
            emg = t(emg)

    def __call__(self, emg: EmgDataset | npt.NDArray[np.floating]) -> EmgDataset | npt.NDArray[np.floating]:
        for t in self.transforms:
            emg = t(emg)
        return emg