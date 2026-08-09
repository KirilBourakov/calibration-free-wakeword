from abc import ABC, abstractmethod
import time
from typing import Any
import numpy as np
import numpy.typing as npt


class AbstractDataHandler(ABC):
    """
    Abstract base class for EMG data handlers. Slot in for OnlineDataHandler.
    """
    sampling_rate: float = 200.0

    @abstractmethod
    def get_data(
        self, N: int = 0, filter: bool = True
    ) -> tuple[dict[str, npt.NDArray[Any]], dict[str, npt.NDArray[Any]]]:
        """
        Grab data from the handler matching OnlineDataHandler interface.

        Parameters
        ----------
        N : int
            Number of samples to grab. If zero, grabs all samples accumulated since last reset.
        filter : bool
            Maintained for OnlineDataHandler interface compatibility.

        Returns
        -------
        val : dict
            Dict mapping modalities (e.g. 'emg') to numpy arrays (newest sample first).
        count : dict
            Dict mapping modalities to numpy array [[sample_count_since_reset]].
        """
        pass

    @abstractmethod
    def reset(self, modality: str | None = None) -> None:
        """Reset the sample counter / buffer state for the specified modality or all modalities."""
        pass

    @abstractmethod
    def get_time(self) -> float:
        """Get the time for the data handler."""
        ...

    @abstractmethod
    def on_wake_detected(self, tolerance: float = 0.5) -> None:
        """Hook called when a wake-word / gesture trigger is detected by the orchestrator."""
        pass

    @property
    def is_offline(self) -> bool:
        return False

class OfflineCapableAbstractDataHandler(AbstractDataHandler, ABC):
    @abstractmethod
    def advance(self, samples: int) -> None:
        """Advance the simulation/playback state by a given number of samples (non-realtime mode)."""
        ...

    @property
    def is_offline(self) -> bool:
        raise NotImplementedError()