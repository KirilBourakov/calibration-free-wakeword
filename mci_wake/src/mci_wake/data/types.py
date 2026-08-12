import os
from typing import List, Any, Dict

import numpy as np
from numpy import typing as npt

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class EPNData:
    emg: List[Any] = Field(default_factory=list)
    imu: List[Any] = Field(default_factory=list)
    labels: List[Any] = Field(default_factory=list)
    myo_labels: List[Any] = Field(default_factory=list)
    subject_ids: List[int] = Field(default_factory=list)

    def __iter__(self):
        return iter((self.emg, self.imu, self.labels, self.myo_labels, self.subject_ids))


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RawData:
    epn_emg: npt.NDArray[np.object_]
    epn_labels: npt.NDArray[Any]
    epn_subjects: npt.NDArray[Any]
    adl_emg: npt.NDArray[np.object_]
    adl_subjects: npt.NDArray[np.int_]

    def __iter__(self):
        return iter((self.epn_emg, self.epn_labels, self.epn_subjects, self.adl_emg, self.adl_subjects))


gesture_mapping: Dict[str, int] = {'noGesture': 0, 'fist': 1, 'waveIn': 2, 'waveOut': 3, 'open': 4, 'pinch': 5}

# dir setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPN_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "other", "EMG-EPN612"))
ADL_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "other", "DiscoDataset"))
