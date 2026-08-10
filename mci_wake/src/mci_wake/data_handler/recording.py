from pathlib import Path
from typing import Any

from numpy import typing as npt
from pydantic.dataclasses import dataclass

from mci_wake.data_handler import OfflineCapableAbstractDataHandler

@dataclass
class RecordingFileRegions:
    start: float
    end: float

@dataclass
class RecordingFileContents:
    emg: list[list[float]]
    timestamps: list[float]
    regions: list[RecordingFileRegions]

class RecordingDataHandler(OfflineCapableAbstractDataHandler):
    path: Path

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def advance(self, samples: int) -> None:
        pass

    def get_data(self, N: int = 0, filter: bool = True) -> tuple[
        dict[str, npt.NDArray[Any]], dict[str, npt.NDArray[Any]]]:
        pass

    def reset(self, modality: str | None = None) -> None:
        pass

    def get_time(self) -> float:
        pass

    def on_wake_detected(self, tolerance: float = 0.5) -> None:
        pass
