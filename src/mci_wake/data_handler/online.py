import time
from typing import Any

import numpy as np
from libemg.data_handler import OnlineDataHandler
from libemg.streamers import myo_streamer
from mci_wake.data_handler.abstract import AbstractDataHandler
from mci_wake.data_handler.types import DataHandlerOutput


class CompatibleOnlineDataHandler(OnlineDataHandler, AbstractDataHandler):
    def get_data(self, N: int = 0, filter: bool = True) -> DataHandlerOutput:
        val, count = OnlineDataHandler.get_data(self, N, filter)
        emg = val.get("emg", np.array([]))
        cnt = int(count["emg"][0, 0])
        return DataHandlerOutput(emg=emg, count=cnt)

    def reset(self, modality: str | None = None) -> None:
        OnlineDataHandler.reset(self, modality)

    def __init__(self, shared_memory_items: Any = None, *args: Any, **kwargs: Any) -> None:
        if shared_memory_items is None:
            _, shared_memory_items = myo_streamer()
        super().__init__(shared_memory_items, *args, **kwargs)

    def get_time(self) -> float:
        return time.time()

    def on_wake_detected(self, tolerance: float = 0.5) -> None:
        pass