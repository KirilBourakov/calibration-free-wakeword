from pathlib import Path
from typing import Annotated, Any

import numpy as np
from numpy import typing as npt
from pydantic import BeforeValidator, ConfigDict, PlainSerializer, TypeAdapter
from pydantic.dataclasses import dataclass

from mci_wake.data_handler.abstract import OfflineCapableAbstractDataHandler
from mci_wake.data_handler.types import DataHandlerOutput, RecordingTriggers, TriggerStats



def _validate_ndarray(v: Any) -> npt.NDArray[np.float64]:
    if isinstance(v, np.ndarray):
        return v.astype(np.float64)
    return np.array(v if v is not None else [], dtype=np.float64)


def _serialize_ndarray(v: npt.NDArray[np.float64]) -> list:
    return v.tolist()


PyArray = Annotated[
    npt.NDArray[np.float64],
    BeforeValidator(_validate_ndarray),
    PlainSerializer(_serialize_ndarray, return_type=list),
]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RecordingFileRegions:
    start: float
    end: float


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RecordingFileContents:
    emg: PyArray
    timestamps: PyArray
    regions: list[RecordingFileRegions]


class RecordingDataHandler(OfflineCapableAbstractDataHandler):
    path: Path
    recording: RecordingFileContents
    end_idx: int
    start_idx: int
    over: bool

    def __init__(self, path: Path | str, realtime: bool = False, sampling_rate: float = 200.0):
        self.path = Path(path)
        with open(self.path, "r") as f:
            self.recording = TypeAdapter(RecordingFileContents).validate_json(f.read())

        self.detected = [False] * len(self.recording.regions)
        self.triggers: list[RecordingTriggers] = []

        self.realtime = realtime
        self.sampling_rate = sampling_rate
        self.end_idx = 0
        self.reset_idx = 0
        self.over = True

    @property
    def is_offline(self) -> bool:
        return not self.realtime

    @property
    def is_done(self):
        return self.end_idx == len(self.recording.emg)

    def advance(self, samples: int = 1) -> None:
        self.end_idx = min(len(self.recording.emg), self.end_idx + samples)

    def get_data(
        self, N: int = 0, filter: bool = True
    ) -> DataHandlerOutput:
        samples_since_reset = max(0, self.end_idx - self.reset_idx)
        target_len = N if N > 0 else samples_since_reset
        start_idx = max(0, self.end_idx - target_len)
        data = self.recording.emg[start_idx : self.end_idx][::-1]
        return DataHandlerOutput(emg=data, count=samples_since_reset)

    def reset(self, modality: str | None = None) -> None:
        self.reset_idx = self.end_idx

    def get_time(self) -> float:
        if 0 <= self.end_idx - 1 < len(self.recording.timestamps):
            return float(self.recording.timestamps[self.end_idx - 1])
        return self.end_idx / self.sampling_rate

    def on_wake_detected(self, tolerance: float = 0.5) -> None:
        current_time = self.recording.timestamps[self.end_idx]

        is_fp = True
        for i, (detected, region) in enumerate(zip(self.detected, self.recording.regions)):
            if not detected and region.start <= current_time <= region.end + tolerance:
                self.detected[i] = True
                is_fp = False

        self.triggers.append(RecordingTriggers(
            index=self.end_idx,
            timestamp=self.get_time(),
            is_fp=is_fp,
        ))

    def get_trigger_stats(self, tolerance: float = 0.5) -> TriggerStats:
        tp = sum(1 for r in self.triggers if not r.is_fp)
        fp = sum(1 for r in self.triggers if r.is_fp)
        fn = sum(1 for r in self.detected if not r)
        return TriggerStats(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            total_triggers=len(self.triggers),
            target_regions_count=len(self.recording.regions),
            triggers=self.triggers,
        )