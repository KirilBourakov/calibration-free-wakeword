from dataclasses import dataclass
import random
from typing import Any
import time
import numpy as np
import numpy.typing as npt
import pandas as pd

from mci_wake.data.types import EmgDataset, gesture_mapping
from mci_wake.data_handler.abstract import OfflineCapableAbstractDataHandler
from mci_wake.data_handler.types import DataHandlerOutput, RecordingTriggers, TriggerStats
from mci_wake.stitching.hanning import stitch, stitch_into_buffer


@dataclass
class TargetRegion:
    start: int
    end: int
    status: str = "pending"  # "pending", "detected", "missed"

class StitchingDataHandler(OfflineCapableAbstractDataHandler):
    """
    Simulates an OnlineDataHandler for testing by stitching discrete EMG / ADL datasets
    into a continuous synthetic livestream using constant-power Hanning cross-fading.
    """

    def __init__(
        self,
        emg_data: EmgDataset,
        adl_data: EmgDataset,
        gestures: list[str] | list[int],
        sampling_rate: float = 200.0,
        probabilities: tuple[float, float, float] = (0.5, 0.5, 0.0),
        realtime: bool = True,
        step_samples: int = 5,
        overlap_samples: int = 15,
        initial_capacity: int = 65536,
    ):
        assert len(probabilities) == 3, "probabilities must be a tuple of 3 floats"
        assert abs(sum(probabilities) - 1.0) < 1e-5, "probabilities must sum to 1.0"
        assert emg_data.is_normalized
        assert adl_data.is_normalized

        # Pre-convert datasets to float64 ndarrays to avoid repeated conversions during streaming
        self.emg_data = emg_data
        self.adl_data = adl_data
        self.sampling_rate = sampling_rate
        self.probabilities = probabilities
        self.gestures = gestures
        self.realtime = realtime
        self.step_samples = step_samples
        self.overlap_samples = overlap_samples

        self.gesture_sequence = [
            gesture_mapping[g] if isinstance(g, str) and g in gesture_mapping else int(g)
            for g in gestures
        ]

        # Precompute label indices
        self._label_indices: dict[Any, list[int]] = {}
        for idx, label in enumerate(self.emg_data.labels):
            self._label_indices.setdefault(label, []).append(idx)

        self.start_time: float | None = None

        # Dynamic capacity buffer to avoid quadratic reallocations
        n_channels = 8
        self._buffer: npt.NDArray[np.float64] = np.empty((initial_capacity, n_channels), dtype=np.float64)
        self._buffer_len: int = 0
        self.end_idx = 0
        self.reset_idx = 0

        self.target_regions: list[TargetRegion] = []
        self.triggers: list[RecordingTriggers] = []

        self._stitch_more_data()

    @property
    def buffer(self) -> npt.NDArray[np.floating]:
        """Get the valid portion of the EMG buffer."""
        return self._buffer[:self._buffer_len]

    @buffer.setter
    def buffer(self, val: npt.NDArray[np.floating]) -> None:
        """Set the EMG buffer and update its length."""
        self._buffer = np.asarray(val, dtype=np.float64)
        self._buffer_len = len(self._buffer)

    def get_time(self) -> float:
        if self.realtime:
            return time.time()
        return self.end_idx / self.sampling_rate

    @property
    def is_offline(self):
        return not self.realtime

    def advance(self, samples: int) -> None:
        self.end_idx += samples
        self.update()

    def pregenerate(self, seconds: float) -> None:
        """
        Pre-generate and stitch data into the buffer for the specified duration (in seconds).
        """
        target_samples = int(seconds * self.sampling_rate)
        target_len = self._buffer_len + target_samples
        while self._buffer_len < target_len:
            prev_len = self._buffer_len
            self._stitch_more_data()
            if self._buffer_len <= prev_len:
                break

    def update(self) -> int:
        """Update the stream position and stitch additional data into the buffer if needed."""
        if self.realtime:
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            self.end_idx = int(elapsed * self.sampling_rate)

        while self.end_idx >= self._buffer_len:
            prev_len = self._buffer_len
            self._stitch_more_data()
            if self._buffer_len <= prev_len:
                break

        self._check_false_negatives()
        return self.end_idx

    def get_data(
        self, N: int = 0, filter: bool = True
    ) -> DataHandlerOutput:
        if not self.realtime and N > 0 and self.end_idx == self.reset_idx:
            self.end_idx += self.step_samples

        self.update()
        samples_since_reset = max(0, self.end_idx - self.reset_idx)
        target_len = N if N > 0 else samples_since_reset
        start_idx = max(0, self.end_idx - target_len)

        data = self._buffer[start_idx:self.end_idx, :][::-1]
        return DataHandlerOutput(emg=data, count=samples_since_reset)

    def reset(self, modality: str | None = None) -> None:
        """
        Reset the sample counter for the specified modality (or all modalities).
        """
        self.update()
        self.reset_idx = self.end_idx

    def on_wake_detected(self, tolerance=0.5) -> None:
        """
        Called by an orchestrator when a wake detection occurs. Tracks true vs. false positives internally.
        A trigger is a true positive if it occurs while being served a positive test case, or up to [tolerance] seconds.
        """
        self.update()
        current_idx = self.end_idx
        tolerance_samples = int(tolerance * self.sampling_rate)

        is_fp = True
        for region in self.target_regions:
            if region.status == "pending" and region.start <= current_idx <= region.end + tolerance_samples:
                is_fp = False
                region.status = "detected"

        self.triggers.append(
            RecordingTriggers(
                index=current_idx,
                timestamp=self.get_time(),
                is_fp=is_fp,
            )
        )

    def get_trigger_stats(self, tolerance: float = 0.5) -> TriggerStats:
        self._check_false_negatives(tolerance=tolerance)
        tp = sum(1 for r in self.target_regions if r.status == "detected")
        fp = sum(1 for t in self.triggers if t.is_fp)
        fn = sum(1 for r in self.target_regions if r.status == "missed")
        return TriggerStats(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            total_triggers=len(self.triggers),
            target_regions_count=len(self.target_regions),
            triggers=self.triggers,
        )

    def _check_false_negatives(self, tolerance=0.5) -> None:
        """Mark every passed pending region as missed."""
        current_idx = self.end_idx
        tolerance_samples = int(tolerance * self.sampling_rate)

        for region in self.target_regions:
            if region.status == "pending" and current_idx > region.end + tolerance_samples:
                region.status = "missed"

    def _stitch_more_data(self) -> None:
        """Add more data to buffer."""
        start_len = self._buffer_len
        new_segments, is_test_case = self._get_next_segments()
        if not new_segments:
            return

        self._ensure_capacity(sum(len(s) for s in new_segments))
        for seg in new_segments:
            self._buffer_len = stitch_into_buffer(
                self._buffer,
                self._buffer_len,
                seg,
                overlap_samples=self.overlap_samples,
            )

        if is_test_case:
            end_len = self._buffer_len
            self.target_regions.append(TargetRegion(start=start_len, end=end_len))

    def _get_next_segments(self) -> tuple[list[npt.NDArray[np.float32]], bool]:
        """Get the next set of segments."""
        p_adl, p_emg, _ = self.probabilities
        segments: list[npt.NDArray[np.float32]] = []
        r = random.random()
        is_test_case = False

        if r < p_adl:
            # empty adl data
            idx = random.randint(0, len(self.adl_data) - 1)
            segments.append(self.adl_data.data[idx])
        elif r < p_adl + p_emg:
            # some random movement
            if len(self.gesture_sequence) == 1:
                target_g = self.gesture_sequence[0]
                matching = [
                    idx for l, idxs in self._label_indices.items() if l != target_g for idx in idxs
                ]
                if matching:
                    idx = random.choice(matching)
                    segments.append(self.emg_data.data[idx])
            else:
                idx = random.randint(0, len(self.emg_data) - 1)
                segments.append(self.emg_data.data[idx])
                if len(self.gesture_sequence) > 1:
                    idx_adl = random.randint(0, len(self.adl_data) - 1)
                    segments.append(self.adl_data.data[idx_adl])

        else:
            # Test case: Lead with 0-0.25s of no-gesture, followed by gesture sequence with 0-0.75s no-gesture gaps
            is_test_case = True
            segments = self.get_sequence_segments(self.gesture_sequence)

        return segments, is_test_case

    def get_sequence_segments(
        self, sequence: list[int] | list[str] | None = None
    ) -> list[npt.NDArray[np.float32]]:
        """Builds a list of gesture segments separated by realistic no-gesture gaps."""
        seq =  (
            self.gesture_sequence
            if sequence is None
            else [
                gesture_mapping[g] if isinstance(g, str) and g in gesture_mapping else int(g)
                for g in sequence
            ]
        )
        segments: list[npt.NDArray[np.float32]] = []
        leading_no_g = self._get_no_gesture_segment(max_duration_sec=0.25)
        if leading_no_g is not None and len(leading_no_g) > 0:
            segments.append(leading_no_g)

        for i, g_id in enumerate(seq):
            matching = self._label_indices.get(g_id, [])
            assert matching, f"No EMG data found matching gesture {g_id}"
            idx = random.choice(matching)
            segments.append(self.emg_data.data[idx])

            if i < len(seq) - 1:
                no_g_seg = self._get_no_gesture_segment(max_duration_sec=0.75)
                if no_g_seg is not None and len(no_g_seg) > 0:
                    segments.append(no_g_seg)

        return segments

    def stitch_sequence(
        self, sequence: list[int] | list[str] | None = None
    ) -> npt.NDArray[np.float32]:
        """Stitches a gesture sequence into a single continuous array using constant-power Hanning cross-fading."""
        segments = self.get_sequence_segments(sequence)
        return stitch(segments, overlap_samples=self.overlap_samples).astype(np.float32)

    def generate_positive(self) -> npt.NDArray[np.float32]:
        """Generates a synthetic positive sequence trial."""
        return self.stitch_sequence(self.gesture_sequence)

    def generate_negative(self) -> npt.NDArray[np.float32]:
        """Generates a hard negative sequence trial (reversed order, prefix mismatch, suffix mismatch,
        isolated gestures, other gestures, or ADL noise)."""
        assert len(self.gesture_sequence) > 1, "single gesture sequences not currently supported"
        assert len(self.adl_data) > 0, "ADL data is required"

        all_gestures = list(self._label_indices.keys())
        other_gestures = [
            l for l in self._label_indices.keys()
            if l not in self.gesture_sequence and l != 0 and len(self._label_indices[l]) > 0
        ]
        has_other = len(other_gestures) > 0

        candidates = []

        def add(weight, func):
            candidates.append((weight, func))

        # 1. Easy case: random single gesture
        add(0.15, lambda: self.stitch_sequence([random.choice(all_gestures)]))

        # 2. Rest -> true suffix
        add(0.15, lambda: self.stitch_sequence([0, self.gesture_sequence[-1]]))

        # 3. Wrong-gesture prefix -> true suffix, with a variable-length true suffix
        if has_other:
            def wrong_prefix_case():
                n = len(self.gesture_sequence)
                split = random.randint(1, n - 1)  # how many leading positions to corrupt
                corrupted = [random.choice(other_gestures) for _ in range(split)]
                true_suffix = self.gesture_sequence[split:]
                return self.stitch_sequence(corrupted + true_suffix)
            add(0.15, wrong_prefix_case)

        # 4. True prefix -> [nothing / other]
        def prefix_case():
            neg_seq = self.gesture_sequence[:-1]
            if has_other and random.random() < 0.5:
                neg_seq = neg_seq + [random.choice(other_gestures)]
            return self.stitch_sequence(neg_seq)
        add(0.15, prefix_case)

        # 5. Reversed sequence (skip if palindromic -- would just reproduce the positive)
        reversed_seq = list(reversed(self.gesture_sequence))
        if reversed_seq != self.gesture_sequence:
            add(0.15, lambda: self.stitch_sequence(reversed_seq))

        # 6. Near-miss: exactly one position forced to a genuine "other" gesture,
        # rest drawn from the full label pool (including target labels)
        if has_other:
            def near_miss():
                forced_loc = random.randint(0, len(self.gesture_sequence) - 1)
                neg_seq = [
                    random.choice(other_gestures) if i == forced_loc else random.choice(all_gestures)
                    for i in range(len(self.gesture_sequence))
                ]
                return self.stitch_sequence(neg_seq)

            add(0.15, near_miss)

        # 7. ADL noise
        add(0.20, lambda: self.adl_data.data[random.randint(0, len(self.adl_data) - 1)])

        # 8. Rest/no-gesture fallback
        def rest_case():
            seg = self._get_no_gesture_segment(max_duration_sec=random.random() * 3)
            if seg is not None and len(seg) >= self.overlap_samples:
                return seg
            # fallback only within this branch, not a global catch-all anymore
            return self.stitch_sequence([0, 0])

        add(0.10, rest_case)

        weights = [w for w, _ in candidates]
        fn = random.choices([f for _, f in candidates], weights=weights, k=1)[0]
        return fn()

    def _ensure_capacity(self, needed_additional_samples: int) -> None:
        """Make sure the buffer has at least the needed_additional_samples free"""
        required = self._buffer_len + needed_additional_samples
        if required > len(self._buffer):
            new_capacity = max(len(self._buffer) * 2, required)
            new_buffer = np.empty((new_capacity, self._buffer.shape[1]), dtype=np.float64)
            if self._buffer_len > 0:
                new_buffer[:self._buffer_len] = self._buffer[:self._buffer_len]
            self._buffer = new_buffer

    def _get_no_gesture_segment(self, max_duration_sec: float = 1.25) -> npt.NDArray[np.float32] | None:
        """Extract a random slice of rest/no-gesture EMG data up to max_duration_sec."""
        max_samples = int(max_duration_sec * self.sampling_rate)
        if max_samples <= 0:
            return None
        num_samples = random.randint(0, max_samples)
        if num_samples == 0:
            return None

        # Try noGesture (label 0) in emg_data
        matching = self._label_indices.get(0, [])
        assert matching, "Cannot _get_no_gesture_segment: matching is empty"
        rec = self.emg_data.data[random.choice(matching)]
        if len(rec) >= num_samples:
            start_i = random.randint(0, len(rec) - num_samples)
            return rec[start_i : start_i + num_samples]
        return rec[:num_samples]

    def __str__(self) -> str:
        stitching_regions = [(r.end - r.start) / self.sampling_rate for r in self.target_regions] if self.target_regions else []
        ret = "=== STITCHING HANDLER INFORMATION ===\n"
        ret += str(pd.Series(stitching_regions).describe())
        return ret

