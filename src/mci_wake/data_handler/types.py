from typing import Any
import numpy.typing as npt
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DataHandlerOutput:
    emg: npt.NDArray[Any]
    count: int


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RecordingTriggers:
    index: float
    timestamp: float
    is_fp: bool


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TriggerStats:
    true_positives: int
    false_positives: int
    false_negatives: int
    total_triggers: int
    triggers: list[RecordingTriggers]
    target_regions_count: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    def __str__(self) -> str:
        actual_positives = self.true_positives + self.false_negatives
        detected_positives = self.true_positives + self.false_positives

        return (
            "=== [Trigger Stats] ===\n"
            f"True Positives (TP):   {self.true_positives}\n"
            f"False Positives (FP):  {self.false_positives}\n"
            f"False Negatives (FN):  {self.false_negatives}\n"
            f"Total Triggers:        {self.total_triggers}\n"
            f"Target Regions Count:  {self.target_regions_count}\n"
            "--- Performance Metrics ---\n"
            f"Precision:             {self.precision:.2%} ({self.true_positives}/{detected_positives})\n"
            f"Recall / Sensitivity:  {self.recall:.2%} ({self.true_positives}/{actual_positives})\n"
            f"F1 Score:              {self.f1_score:.4f}\n"
        )