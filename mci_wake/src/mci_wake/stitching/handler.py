import random
from typing import Any
import time
import numpy as np
import numpy.typing as npt

from mci_wake.stitching.hanning import stitch


class StitchingDataHandler:
    """
    Simulates an OnlineDataHandler for testing by stitching discrete EMG / ADL datasets
    into a continuous synthetic livestream using constant-power Hanning cross-fading.
    """

    def __init__(
        self,
        emg_data: npt.NDArray[np.object_] | list[npt.NDArray[np.floating]] | None = None,
        emg_labels: npt.NDArray[Any] | list[Any] | None = None,
        adl_data: npt.NDArray[np.object_] | list[npt.NDArray[np.floating]] | None = None,
        sampling_rate: float = 200.0,
        probabilities: tuple[float, float, float] = (0.5, 0.5, 0.0),
    ):
        assert len(probabilities) == 3, "probabilities must be a tuple of 3 floats"
        assert abs(sum(probabilities) - 1.0) < 1e-5, "probabilities must sum to 1.0"

        self.emg_data = emg_data
        self.emg_labels = emg_labels
        self.adl_data = adl_data
        self.sampling_rate = sampling_rate
        self.probabilities = probabilities

        self.start_time: float | None = None
        self.buffer: npt.NDArray[np.floating] = np.zeros((0, 8), dtype=np.float64)
        self.end_idx = 0
        self.reset_idx = 0

        self._stitch_more_data()

    def _get_next_segments(self, count: int = 4) -> list[npt.NDArray[np.floating]]:
        p_adl, p_emg, _ = self.probabilities

        segments: list[npt.NDArray[np.floating]] = []
        for _ in range(count):
            r = random.random()
            if r < p_adl:
                idx = random.randint(0, len(self.adl_data) - 1)
                segments.append(np.asarray(self.adl_data[idx], dtype=np.float64))
            elif r < p_adl + p_emg:
                idx = random.randint(0, len(self.emg_data) - 1)
                segments.append(np.asarray(self.emg_data[idx], dtype=np.float64))
            else:
                # TODO: Implement 3rd segment selection / generation option
                pass

        return segments

    def _stitch_more_data(self) -> None:
        new_segments = self._get_next_segments()
        if not new_segments:
            return
        if len(self.buffer) == 0:
            if len(new_segments) == 1:
                self.buffer = new_segments[0].copy()
            else:
                self.buffer = stitch(new_segments)
        else:
            self.buffer = stitch([self.buffer] + new_segments)

    def update(self) -> int:
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        self.end_idx = int(elapsed * self.sampling_rate)

        while self.end_idx >= len(self.buffer):
            prev_len = len(self.buffer)
            self._stitch_more_data()
            if len(self.buffer) <= prev_len:
                break
        return self.end_idx

    def get_data(
        self, N: int = 0, filter: bool = True
    ) -> tuple[dict[str, npt.NDArray[Any]], dict[str, npt.NDArray[Any]]]:
        """
        Grab data from the synthetic livestream matching OnlineDataHandler interface.

        Parameters
        ----------
        N : int
            Number of samples to grab. If zero, grabs all samples accumulated since last reset.
        filter : bool
            Maintained for OnlineDataHandler interface compatibility.

        Returns
        -------
        val : dict
            Dict with key 'emg' mapping to array of shape (N, n_channels) (newest sample first).
        count : dict
            Dict with key 'emg' mapping to array [[samples_since_reset]].
        """
        self.update()
        samples_since_reset = max(0, self.end_idx - self.reset_idx)
        target_len = N if N > 0 else samples_since_reset
        start_idx = max(0, self.end_idx - target_len)

        data = self.buffer[start_idx:self.end_idx, :][::-1]
        return {"emg": data}, {"emg": np.array([[samples_since_reset]], dtype=int)}

    def reset(self, modality: str | None = None) -> None:
        """
        Reset the sample counter for the specified modality (or all modalities).
        """
        self.update()
        self.reset_idx = self.end_idx
