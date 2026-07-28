from dataclasses import dataclass
import random
from typing import Any
import time
import numpy as np
import numpy.typing as npt

from mci_wake.stitching.hanning import stitch
from mci_wake.data.train_utils import gesture_mapping


@dataclass
class TargetRegion:
    start: int
    end: int
    status: str = "pending"  # "pending", "detected", "missed"

class StitchingDataHandler:
    """
    Simulates an OnlineDataHandler for testing by stitching discrete EMG / ADL datasets
    into a continuous synthetic livestream using constant-power Hanning cross-fading.
    """

    def __init__(
        self,
        emg_data: npt.NDArray[np.object_] | list[npt.NDArray[np.floating]],
        emg_labels: npt.NDArray[Any] | list[Any],
        adl_data: npt.NDArray[np.object_] | list[npt.NDArray[np.floating]],
        gestures: list[str] | list[int],
        sampling_rate: float = 200.0,
        probabilities: tuple[float, float, float] = (0.5, 0.5, 0.0),
        realtime: bool = True,
        step_samples: int = 5,
    ):
        assert len(probabilities) == 3, "probabilities must be a tuple of 3 floats"
        assert abs(sum(probabilities) - 1.0) < 1e-5, "probabilities must sum to 1.0"

        self.emg_data = emg_data
        self.emg_labels = emg_labels
        self.adl_data = adl_data
        self.sampling_rate = sampling_rate
        self.probabilities = probabilities
        self.gestures = gestures
        self.realtime = realtime
        self.step_samples = step_samples

        self.gesture_sequence = [
            gesture_mapping[g] if isinstance(g, str) and g in gesture_mapping else int(g)
            for g in gestures
        ]

        # Precompute label indices
        self._label_indices: dict[Any, list[int]] = {}
        for idx, label in enumerate(self.emg_labels):
            self._label_indices.setdefault(label, []).append(idx)

        self.start_time: float | None = None
        self.buffer: npt.NDArray[np.floating] = np.zeros((0, 8), dtype=np.float64)
        self.end_idx = 0
        self.reset_idx = 0

        self.target_regions: list[TargetRegion] = []
        self.triggers: list[dict[str, Any]] = []

        self._stitch_more_data()

    def get_virtual_timestamp(self) -> float:
        if self.realtime:
            return time.time()
        return self.end_idx / self.sampling_rate

    def advance(self, samples: int) -> None:
        self.end_idx += samples
        self.update()

    def update(self) -> int:
        if self.realtime:
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            self.end_idx = int(elapsed * self.sampling_rate)

        while self.end_idx >= len(self.buffer):
            prev_len = len(self.buffer)
            self._stitch_more_data()
            if len(self.buffer) <= prev_len:
                break

        self._check_false_negatives()
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
        if not self.realtime and N > 0 and self.end_idx == self.reset_idx:
            self.end_idx += self.step_samples

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

    def on_wake_detected(self, tolerance=0.5) -> None:
        """
        Called by an orchestrator (e.g. WakeDetect) when a wake detection occurs.
        Tracks true vs. false positives internally.
        A trigger is a true positive if it occurs while being served a positive test case,
        or up to [tolerance] seconds after being served a positive test case.
        """
        self.update()
        current_idx = self.end_idx
        tolerance_samples = int(tolerance * self.sampling_rate)

        is_fp = True
        for region in self.target_regions:
            if region.status == "pending" and region.start <= current_idx <= region.end + tolerance_samples:
                is_fp = False
                region.status = "detected"

        self.triggers.append({
            "sample_idx": current_idx,
            "timestamp": self.get_virtual_timestamp(),
            "is_false_positive": is_fp,
        })

    def get_trigger_stats(self, tolerance=0.5) -> dict[str, Any]:
        self._check_false_negatives(tolerance=tolerance)
        tp = sum(1 for r in self.target_regions if r.status == "detected")
        fp = sum(1 for t in self.triggers if t["is_false_positive"])
        fn = sum(1 for r in self.target_regions if r.status == "missed")
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_triggers": len(self.triggers),
            "target_regions_count": len(self.target_regions),
            "triggers": self.triggers,
        }

    def _check_false_negatives(self, tolerance=0.5) -> None:
        current_idx = self.end_idx
        tolerance_samples = int(tolerance * self.sampling_rate)

        for region in self.target_regions:
            if region.status == "pending" and current_idx > region.end + tolerance_samples:
                region.status = "missed"

    def _stitch_more_data(self) -> None:
        start_len = len(self.buffer)
        new_segments, is_test_case = self._get_next_segments()
        if not new_segments:
            return

        if len(self.buffer) == 0:
            if len(new_segments) == 1:
                self.buffer = new_segments[0].copy()
            else:
                self.buffer = stitch(new_segments)
        else:
            self.buffer = stitch([self.buffer] + new_segments, in_place=True)

        if is_test_case:
            end_len = len(self.buffer)
            self.target_regions.append(TargetRegion(start=start_len, end=end_len))


    def _get_next_segments(self) -> tuple[list[npt.NDArray[np.floating]], bool]:
        p_adl, p_emg, _ = self.probabilities
        segments: list[npt.NDArray[np.floating]] = []
        r = random.random()
        is_test_case = False

        if r < p_adl:
            # empty adl data
            idx = random.randint(0, len(self.adl_data) - 1)
            segments.append(np.asarray(self.adl_data[idx], dtype=np.float64))
        elif r < p_adl + p_emg:
            # some random movement
            if len(self.gesture_sequence) == 1:
                target_g = self.gesture_sequence[0]
                matching = [
                    idx for l, idxs in self._label_indices.items() if l != target_g for idx in idxs
                ]
                if matching:
                    idx = random.choice(matching)
                    segments.append(np.asarray(self.emg_data[idx], dtype=np.float64))
            else:
                idx = random.randint(0, len(self.emg_data) - 1)
                segments.append(np.asarray(self.emg_data[idx], dtype=np.float64))
                if len(self.gesture_sequence) > 1:
                    idx_adl = random.randint(0, len(self.adl_data) - 1)
                    segments.append(np.asarray(self.adl_data[idx_adl], dtype=np.float64))

        else:
            # Test case: Lead with 0-0.25s of no-gesture, followed by gesture sequence with 0-1.25s no-gesture gaps
            # Inserts [empty] [gesture 1] [empty] [gesture 2] ... [gesture n]
            is_test_case = True
            leading_no_g = self._get_no_gesture_segment(max_duration_sec=0.25)
            if leading_no_g is not None and len(leading_no_g) > 0:
                segments.append(leading_no_g)

            for i, g_id in enumerate(self.gesture_sequence):
                matching = self._label_indices.get(g_id, [])
                assert matching, f"No EMG data found matching gesture {g_id}"
                idx = random.choice(matching)
                segments.append(np.asarray(self.emg_data[idx], dtype=np.float64))

                if i < len(self.gesture_sequence) - 1:
                    no_g_seg = self._get_no_gesture_segment(max_duration_sec=1.25)
                    if no_g_seg is not None and len(no_g_seg) > 0:
                        segments.append(no_g_seg)

        return segments, is_test_case

    def _get_no_gesture_segment(self, max_duration_sec: float = 1.25) -> npt.NDArray[np.floating] | None:
        max_samples = int(max_duration_sec * self.sampling_rate)
        if max_samples <= 0:
            return None
        num_samples = random.randint(0, max_samples)
        if num_samples == 0:
            return None

        # Try noGesture (label 0) in emg_data
        matching = self._label_indices.get(0, [])
        assert matching, "Cannot _get_no_gesture_segment: matching is empty"
        rec = np.asarray(self.emg_data[random.choice(matching)], dtype=np.float64)
        if len(rec) >= num_samples:
            start_i = random.randint(0, len(rec) - num_samples)
            return rec[start_i : start_i + num_samples]
        return rec[:num_samples]

