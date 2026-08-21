import os
from typing import List, Any, Dict, Annotated, overload

import numpy as np
from numpy import typing as npt

from pydantic import ConfigDict, Field, BeforeValidator, PlainSerializer
from pydantic.dataclasses import dataclass

def _validate_ndarray(v: Any) -> npt.NDArray[np.float64]:
    if isinstance(v, np.ndarray):
        return v.astype(np.float64)
    return np.array(v if v is not None else [], dtype=np.float64)


def _serialize_ndarray(v: npt.NDArray[np.float64]) -> list:
    return v.tolist()


PydanticF64Array = Annotated[
    npt.NDArray[np.float64],
    BeforeValidator(_validate_ndarray),
    PlainSerializer(_serialize_ndarray, return_type=list),
]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class EmgDataset:
    data: list[npt.NDArray[np.float32]]  # Each element has shape (T_i, channels)
    labels: npt.NDArray[np.int32]  # Shape: (N,)
    subjects: npt.NDArray[np.int32]  # Shape: (N,)
    is_normalized: bool = False

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter((self.data, self.labels, self.subjects))

    @overload
    def __getitem__(self, idx: int) -> tuple[npt.NDArray[np.float32], np.int32, np.int32]: ...
    @overload
    def __getitem__(self, idx: slice) -> "EmgDataset": ...
    def __getitem__(self, idx: int | slice) -> "EmgDataset | tuple[npt.NDArray[np.float32], np.int32, np.int32]":
        """Allows indexing dataset[0] and slicing dataset[:10]"""
        # slicing
        if isinstance(idx, slice):
            return EmgDataset(
                data=self.data[idx],
                labels=self.labels[idx],
                subjects=self.subjects[idx],
                is_normalized=self.is_normalized
            )
        # single integer
        elif isinstance(idx, int):
            return self.data[idx], self.labels[idx], self.subjects[idx]
        else:
            raise TypeError(f"Invalid argument type: {type(idx)}")

    def combine(self, other: "EmgDataset") -> "EmgDataset":
        assert other.is_normalized == self.is_normalized, "Cannot combine EmgDatasets with different normalized states."
        return EmgDataset(
            data=self.data + other.data,
            labels=np.concatenate((self.labels, other.labels)),
            subjects=np.concatenate((self.subjects, other.subjects)),
            is_normalized=self.is_normalized
        )

@dataclass
class WindowedEmgDataset:
    data: npt.NDArray[np.float32]        # Shape: (N, W, C) contiguous array
    labels: npt.NDArray[np.int64]        # Shape: (N,)
    subjects: npt.NDArray[np.int64]      # Shape: (N,)
    is_normalized: bool = False


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class EmgData:
    data: npt.NDArray[np.floating]
    is_normalized: bool = False

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class EPNData:
    emg: List[EmgData] = Field(default_factory=list)
    imu: List[Any] = Field(default_factory=list)
    labels: List[Any] = Field(default_factory=list)
    myo_labels: List[Any] = Field(default_factory=list)
    subject_ids: List[int] = Field(default_factory=list)

    def __iter__(self):
        return iter((self.emg, self.imu, self.labels, self.myo_labels, self.subject_ids))


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RawData:
    epn_emg: npt.NDArray[EmgData]
    epn_labels: npt.NDArray[Any]
    epn_subjects: npt.NDArray[Any]
    adl_emg: npt.NDArray[EmgData]
    adl_subjects: npt.NDArray[np.int_]

    def __iter__(self):
        return iter((self.epn_emg, self.epn_labels, self.epn_subjects, self.adl_emg, self.adl_subjects))

    def split(self, test_percentage: float) -> tuple["RawData", "RawData"]:
        unique_subjects = np.unique(self.epn_subjects)

        n_test = int(len(unique_subjects) * test_percentage)
        test_subject_ids = list(unique_subjects[:n_test])

        epn_test_mask = np.isin(self.epn_subjects, test_subject_ids)
        epn_train_mask = ~epn_test_mask

        unique_subjects = np.unique(self.adl_subjects)
        n_test = int(len(unique_subjects) * test_percentage)
        test_subject_ids = list(unique_subjects[:n_test])
        adl_test_mask = np.isin(self.adl_subjects, test_subject_ids)
        adl_train_mask = ~adl_test_mask

        test = RawData(
            epn_emg=self.epn_emg[epn_test_mask],
            epn_labels=self.epn_labels[epn_test_mask],
            epn_subjects=self.epn_subjects[epn_test_mask],
            adl_emg=self.adl_emg[adl_test_mask],
            adl_subjects=self.adl_subjects[adl_test_mask]
        )
        train = RawData(
            epn_emg=self.epn_emg[epn_train_mask],
            epn_labels=self.epn_labels[epn_train_mask],
            epn_subjects=self.epn_subjects[epn_train_mask],
            adl_emg=self.adl_emg[adl_train_mask],
            adl_subjects=self.adl_subjects[adl_train_mask]
        )
        return train, test


gesture_mapping: Dict[str, int] = {'noGesture': 0, 'fist': 1, 'waveIn': 2, 'waveOut': 3, 'open': 4, 'pinch': 5}

# dir setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPN_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "other", "EMG-EPN612"))
ADL_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "other", "DiscoDataset"))
