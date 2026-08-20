from dataclasses import dataclass
from typing import overload, Sequence
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from mci_wake.data.types import EmgData, PydanticF64Array


class Normalize(BaseModel):
    # Allow ndarray types inside Pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: PydanticF64Array
    std: PydanticF64Array

    @classmethod
    def create(cls, emg: Sequence[EmgData], eps: float = 1e-3) -> "Normalize":
        assert all(not d.is_normalized for d in emg), "Cannot fit on normalized data."

        total_count = 0
        global_mean = None
        global_m2 = None

        for d in emg:
            chunk = d.data
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
        global_std = np.maximum(np.sqrt(global_var), eps)

        return cls(
            mean=global_mean.astype(np.float32),
            std=global_std.astype(np.float32)
        )

    @overload
    def __call__(self, emg: EmgData) -> EmgData: ...
    @overload
    def __call__(self, emg: list[EmgData]) -> list[EmgData]: ...
    def __call__(self, emg: list[EmgData] | EmgData) -> list[EmgData] | EmgData:
        single = isinstance(emg, EmgData)
        items = [emg] if single else emg

        transformed = []
        for d in items:
            if not d.is_normalized:
                # Broadcasting automatically maps the (C,) mean/std arrays across the (T, C) data array
                normalized_data = (d.data - self.mean) / self.std
                transformed.append(EmgData(data=normalized_data, is_normalized=True))
            else:
                transformed.append(EmgData(data=d.data.copy(), is_normalized=True))

        return transformed[0] if single else transformed