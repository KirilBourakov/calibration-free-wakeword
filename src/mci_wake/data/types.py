import os
import warnings
from dataclasses import replace
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


@dataclass
class TrainData:
    disco: list[str] = Field(default_factory=list)
    emg: list[str] = Field(default_factory=list)



@dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class EmgDataset:
    data: list[npt.NDArray[np.float32]]  # Each element has shape (T_i, channels)
    labels: npt.NDArray[np.int32]  # Shape: (N,)
    subjects: npt.NDArray[np.int32]  # Shape: (N,)
    is_normalized: bool = False

    @staticmethod
    def empty() -> "EmgDataset":
        return EmgDataset(data=[], labels=np.empty(0, dtype=np.int32), subjects=np.empty(0, dtype=np.int32), is_normalized=True)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(zip(self.data, self.labels, self.subjects))

    def permute(self) -> "EmgDataset":
        num_total = len(self.data)
        p = np.random.permutation(num_total)
        return replace(
            self,
            data=[self.data[i] for i in p],
            labels=np.array(self.labels, dtype=np.int32)[p],
            subjects=np.array(self.subjects, dtype=np.int32)[p],
        )

    @overload
    def __getitem__(self, idx: int | np.integer) -> tuple[npt.NDArray[np.float32], np.int32, np.int32]:
        ...
    @overload
    def __getitem__(self, idx: slice | list | npt.NDArray) -> "EmgDataset":
        ...
    def __getitem__(self, idx: int | np.integer | slice | list | npt.NDArray) -> "EmgDataset | tuple[npt.NDArray[np.float32], np.int32, np.int32]":
        # 1.single
        if isinstance(idx, (int, np.integer)):
            return self.data[idx], self.labels[idx], self.subjects[idx]

        # 2. subsets (slice, list, ndarray)
        if isinstance(idx, (slice, list, np.ndarray)):
            if isinstance(idx, slice):
                new_data = self.data[idx]
            elif isinstance(idx, np.ndarray) and idx.dtype == bool:
                new_data = [d for d, m in zip(self.data, idx) if m]
            else:
                new_data = [self.data[i] for i in idx]

            return EmgDataset(
                data=new_data,
                labels=self.labels[idx],
                subjects=self.subjects[idx],
                is_normalized=self.is_normalized
            )

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

    def reduce_proportional(
        self,
        n: int | None = None,
        proportions: tuple[float, ...] | None = None,
        *,
        seed: int | None = None,
    ) -> "EmgDataset":
        """Return a new dataset with (at most) ``n`` samples, dropping the rest."""
        total = len(self.data)
        assert n is not None or proportions is not None, "Provide n, proportions, or both."
        assert n is None or 0 <= n <= total, f"n must be in [0, {total}], got {n}"

        classes, counts = np.unique(self.labels, return_counts=True)

        # 1. Resolve proportions and feasible max
        if proportions is not None and n == total:
            return self
        elif proportions is None:
            props, n_max = counts / total, total
        else:
            props = np.asarray(proportions, dtype=np.float64)
            assert props.shape == classes.shape and (props >= 0).all() and props.sum() > 0, (
                f"Proportions must match {classes.tolist()}, non-negative, and sum > 0"
            )
            props /= props.sum()
            nz = props > 0
            n_max = int(np.min(counts[nz] // props[nz]))

        # 2. Bound n
        if n is not None and n > n_max:
            warnings.warn(
                f"Requested {n} samples with proportions {np.round(props, 4).tolist()}, "
                f"but class counts {dict(zip(classes.tolist(), counts.tolist()))} "
                f"allow at most {n_max}; using {n_max}.",
                stacklevel=2,
            )
        n = min(n if n is not None else n_max, n_max)

        # 3. Largest-remainder quota allocation
        quotas = n * props
        alloc = np.floor(quotas).astype(np.int64)
        remainder = n - int(alloc.sum())
        if remainder > 0:
            alloc[np.argsort(-(quotas % 1), kind="stable")[:remainder]] += 1

        # 4. Sample indices and retain dataset ordering
        rng = np.random.default_rng(seed)
        keep = np.concatenate([
            rng.choice(np.flatnonzero(self.labels == cls), size=k, replace=False)
            for cls, k in zip(classes, alloc)
            if k > 0
        ])
        keep.sort()

        return replace(
            self,
            data=[self.data[i] for i in keep],
            labels=self.labels[keep],
            subjects=self.subjects[keep],
        )

gesture_mapping: Dict[str, int] = {'noGesture': 0, 'fist': 1, 'waveIn': 2, 'waveOut': 3, 'open': 4, 'pinch': 5}

# dir setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EPN_DATA = os.path.join(ROOT_DIR, "data", "EMG-EPN612")
ADL_DATA = os.path.join(ROOT_DIR, "data", "DiscoDataset")
