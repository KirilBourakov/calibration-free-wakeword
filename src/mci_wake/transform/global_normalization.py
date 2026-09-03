from dataclasses import dataclass, replace
from typing import overload, Sequence
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from mci_wake.data.types import PydanticF64Array, EmgDataset
from mci_wake.transform.base import AbstractTransform


class GlobalStatsNormalize(AbstractTransform):
    # Allow ndarray types inside Pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: PydanticF64Array
    std: PydanticF64Array
    eps: float = 1E-3

    def fit(self, emg: EmgDataset) -> None:
        assert not emg.is_normalized, "Cannot fit on normalized data"

        total_count = 0
        global_mean = None
        global_m2 = None

        for chunk, _, _ in emg:
            if chunk.size != 0:
                chunk_count = chunk.shape[0]

                # Per-channel stats
                chunk_mean = np.mean(chunk, axis=0, dtype=np.float64)
                chunk_var = np.var(chunk, axis=0, dtype=np.float64)
                chunk_m2 = chunk_var * chunk_count

                # Chan's algorithm
                if total_count == 0:
                    global_mean = chunk_mean
                    global_m2 = chunk_m2
                    total_count = chunk_count
                else:
                    delta = chunk_mean - global_mean

                    new_count = total_count + chunk_count
                    global_mean = global_mean + delta * (chunk_count / new_count)
                    global_m2 = global_m2 + chunk_m2 + (delta ** 2) * (total_count * chunk_count / new_count)

                    total_count = new_count

        assert total_count != 0 and global_mean is not None and global_m2 is not None

        global_var = global_m2 / total_count
        global_std = np.maximum(np.sqrt(global_var), self.eps)


        self.mean=global_mean.astype(np.float32)
        self.std=global_std.astype(np.float32)


    def __call__(self, emg: EmgDataset | npt.NDArray[np.floating]) -> EmgDataset | npt.NDArray[np.float32]:
        if isinstance(emg, np.ndarray):
            return ((emg - self.mean) / self.std).astype(np.float32)

        if emg.is_normalized:
            return emg

        data: list[npt.NDArray[np.float32]] = []
        for d in emg.data:
            normalized_data = ((d - self.mean) / self.std).astype(np.float32)
            data.append(normalized_data)

        return replace(emg, data=data, is_normalized=True)