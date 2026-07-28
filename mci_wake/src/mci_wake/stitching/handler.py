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
    ):
        self.emg_data = emg_data
        self.emg_labels = emg_labels
        self.adl_data = adl_data
        self.sampling_rate = sampling_rate

        self.time_last_call = time.time()
        self.buffer = np.zeros(0)
        self.end_idx = 0

    def update(self) -> int:
        elapsed = time.time() - self.time_last_call
        offset = int(elapsed * self.sampling_rate)
        self.end_idx += offset
        if self.end_idx >= len(self.buffer):
            pass # TODO: stitch
        self.time_last_call = time.time()
        return offset

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
        new = self.update()
        start_idx = 0 if N == 0 else self.end_idx - N
        start_idx = max(0, start_idx)

        data = self.buffer[start_idx:self.end_idx, :][::-1]
        return {"emg": data}, {"emg": np.array([[new]], dtype=int)}

    def reset(self, modality: str | None = None) -> None:
        """
        Reset the sample counter for the specified modality (or all modalities).
        """
        self.time_last_call = time.time()
        self.buffer = np.zeros(0)
        self.end_idx = 0