import time
from typing import Any

from libemg.data_handler import OnlineDataHandler
from numpy import typing as npt

from libemg.streamers import myo_streamer
from mci_wake.data_handler.abstract import AbstractDataHandler


class CompatibleOnlineDataHandler(OnlineDataHandler, AbstractDataHandler):
    def __init__(self, shared_memory_items: Any = None, *args: Any, **kwargs: Any) -> None:
        if shared_memory_items is None:
            _, shared_memory_items = myo_streamer()
        super().__init__(shared_memory_items, *args, **kwargs)

    def get_time(self) -> float:
        return time.time()

    def on_wake_detected(self, tolerance: float = 0.5) -> None:
        pass