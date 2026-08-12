from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


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

    def __str__(self):
        str = "=== [Trigger Stats] === \n"
        str += f"True Positives:  {self.true_positives}\n"
        str += f"False Positives: {self.false_positives}\n"
        str += f"False Negatives: {self.false_negatives}\n"
        str += f"Total Triggers:  {self.total_triggers}\n"
        return str
