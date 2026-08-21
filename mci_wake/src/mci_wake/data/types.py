import os
from typing import List, Any, Dict, Annotated, overload, Optional

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

    def split(
        self,
        test_percentage: float = 0.1,
        by_subject: bool = True,
        test_subject_ids: Optional[list[int] | npt.NDArray[np.integer]] = None,
        random_seed: Optional[int] = None,
    ) -> tuple["EmgDataset", "EmgDataset"]:
        """Splits the dataset into train and test partitions.

        Args:
            test_percentage: Fraction of data (or subjects) to allocate to the test set. Defaults to 0.1.
            by_subject: If True, splits by unique subject IDs to prevent data leakage. If False, splits by sample count.
            test_subject_ids: Explicit list of subject IDs to hold out for testing. If provided, overrides test_percentage.
            random_seed: Optional seed for reproducible subject/sample selection.

        Returns:
            tuple[EmgDataset, EmgDataset]: (train_dataset, test_dataset)
        """
        if by_subject:
            unique_subjects = np.unique(self.subjects)
            if test_subject_ids is None:
                if random_seed is not None:
                    rng = np.random.default_rng(random_seed)
                    n_test = max(1, int(len(unique_subjects) * test_percentage))
                    test_subject_ids = list(rng.choice(unique_subjects, size=n_test, replace=False))
                else:
                    n_test = max(1, int(len(unique_subjects) * test_percentage))
                    test_subject_ids = list(unique_subjects[:n_test])

            test_mask = np.isin(self.subjects, test_subject_ids)
            train_mask = ~test_mask

            train = EmgDataset(
                data=[d for d, m in zip(self.data, train_mask) if m],
                labels=self.labels[train_mask],
                subjects=self.subjects[train_mask],
                is_normalized=self.is_normalized,
            )
            test = EmgDataset(
                data=[d for d, m in zip(self.data, test_mask) if m],
                labels=self.labels[test_mask],
                subjects=self.subjects[test_mask],
                is_normalized=self.is_normalized,
            )
            return train, test
        else:
            n_train = int(len(self) * (1.0 - test_percentage))
            return self[:n_train], self[n_train:]

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


gesture_mapping: Dict[str, int] = {'noGesture': 0, 'fist': 1, 'waveIn': 2, 'waveOut': 3, 'open': 4, 'pinch': 5}

# dir setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPN_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "other", "EMG-EPN612"))
ADL_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "other", "DiscoDataset"))
