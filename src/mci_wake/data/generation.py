import warnings
from collections import defaultdict
from typing import Optional
import numpy as np
import numpy.typing as npt

from mci_wake.data.types import EmgDataset, gesture_mapping
from mci_wake.stitching.hanning import stitch


def _sample_rest_segment(
    rest_list: list[npt.NDArray[np.float32]],
    duration_range: tuple[float, float],
    sampling_rate: float,
    overlap_samples: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32] | None:
    min_dur, max_dur = duration_range
    min_samples = max(overlap_samples, int(min_dur * sampling_rate))
    max_samples = max(min_samples, int(max_dur * sampling_rate))
    if max_samples <= 0:
        return None
    target_len = int(rng.integers(min_samples, max_samples + 1))
    if not rest_list:
        return np.zeros((target_len, 8), dtype=np.float32)
    rec = rest_list[rng.integers(0, len(rest_list))]
    if len(rec) >= target_len:
        st = int(rng.integers(0, len(rec) - target_len + 1))
        return rec[st : st + target_len]
    elif len(rec) >= overlap_samples:
        return rec
    return None


def _sample_optional_rest(
    rest_list: list[npt.NDArray[np.float32]],
    max_dur: float,
    sampling_rate: float,
    overlap_samples: int,
    rng: np.random.Generator,
    p_include: float = 0.7,
) -> npt.NDArray[np.float32] | None:
    if rng.random() > p_include:
        return None
    return _sample_rest_segment(
        rest_list, (0.05, max_dur), sampling_rate, overlap_samples, rng
    )


def _build_stitched_trial(
    gesture_ids: list[int],
    gesture_dict: dict[int, list[npt.NDArray[np.float32]]],
    rest_list: list[npt.NDArray[np.float32]],
    sampling_rate: float,
    overlap_samples: int,
    rng: np.random.Generator,
    gap_duration_range: tuple[float, float] = (0.1, 0.75),
    lead_trail_rest: bool = True,
) -> npt.NDArray[np.float32]:
    segments: list[npt.NDArray[np.float32]] = []

    if lead_trail_rest:
        leading = _sample_optional_rest(
            rest_list, 0.25, sampling_rate, overlap_samples, rng
        )
        if leading is not None and len(leading) >= overlap_samples:
            segments.append(leading)

    for i, g_id in enumerate(gesture_ids):
        matching = gesture_dict.get(g_id, [])
        if not matching:
            raise ValueError(f"No EMG samples found for gesture ID {g_id}")
        seg = matching[rng.integers(0, len(matching))]
        segments.append(seg)

        if i < len(gesture_ids) - 1:
            gap = _sample_rest_segment(
                rest_list, gap_duration_range, sampling_rate, overlap_samples, rng
            )
            if gap is not None and len(gap) >= overlap_samples:
                segments.append(gap)

    if lead_trail_rest:
        trailing = _sample_optional_rest(
            rest_list, 0.25, sampling_rate, overlap_samples, rng
        )
        if trailing is not None and len(trailing) >= overlap_samples:
            segments.append(trailing)

    if not segments:
        raise ValueError("Cannot stitch an empty list of segments.")

    return stitch(segments, overlap_samples=overlap_samples).astype(np.float32)


def generate_training_data(
    emg: EmgDataset,
    adl: EmgDataset,
    target_sequence: list[str],
    n_sample_per_subject: int = 30,
    positive_sample_precent: float = 0.5,
    sampling_rate: float = 200.0,
    overlap_samples: int = 15,
    random_seed: int | None = 42,
) -> EmgDataset:
    """
    Generates synthetic positive and hard negative sequence trials for given subjects.
    Stitching is done strictly within the same subject to model realistic intra-user dynamics.

    Args:
        emg: EmgDataset containing gesture recordings. Should be normalized.
        target_sequence: List of gesture names defining the positive wake sequence (e.g. ['pinch', 'fist']).
        n_sample_per_subject: Number of samples per subject
        positive_sample_precent: Percentage of samples made positive
        sampling_rate: EMG sampling rate in Hz (default: 200.0).
        adl: Optional EmgDataset containing ADL noise recordings used as additional negatives.
        overlap_samples: Number of samples for raised-cosine cross-fade (default: 15).
        random_seed: Random seed for reproducibility.

    Returns:
        EmgDataset: Dataset of synthesized sequence trials with labels (1=positive, 0=negative)
                    and preserved subject IDs.
    """
    assert len(emg) != 0 and len(adl) != 0
    assert emg.is_normalized, "Emg must be normalized before stitching"
    assert target_sequence, "target_sequence must contain at least one gesture name."

    n_positive_per_subject = int(n_sample_per_subject * positive_sample_precent)
    n_hard_negative_per_subject = n_sample_per_subject - n_positive_per_subject

    target_labels: list[int] = [gesture_mapping[g] for g in target_sequence]

    rng = np.random.default_rng(random_seed)

    # Group recordings by subject and then by gesture label
    # subject_data[subject_id][label] = list of 2D arrays
    subject_data: dict[int, dict[int, list[npt.NDArray[np.float32]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for emg_arr, label, subj in zip(emg.data, emg.labels, emg.subjects):
        subject_data[int(subj)][int(label)].append(
            np.asarray(emg_arr, dtype=np.float32)
        )

    adl_recordings = [
        np.asarray(d, dtype=np.float32)
        for d in adl.data
        if len(d) >= overlap_samples
    ]

    trials_data: list[npt.NDArray[np.float32]] = []
    trials_labels: list[int] = []
    trials_subjects: list[int] = []

    # Get strategies
    non_target_gestures = [g for g in gesture_mapping.values() if g not in target_labels]
    reversed_target = list(reversed(target_labels))
    can_reverse = len(target_labels) > 1 and reversed_target != target_labels
    strategy_weights = np.array([
        0.25 if can_reverse else 0.0,  # 0: Reversed target
        0.25 if non_target_gestures else 0.0,  # 1: Prefix / Suffix
        0.20,  # 2: Single target in rest
        0.15 if non_target_gestures else 0.0,  # 3: Other gestures sequence
        0.15 if adl_recordings else 0.0,  # 4: ADL / rest
    ])
    probs = strategy_weights / strategy_weights.sum()

    # --- Subject Loop ---
    for s_id, gesture_dict in subject_data.items():
        def stitch_trial(seq_in: list[int], lead_trail: bool = False) -> npt.NDArray[np.float32]:
            rest_list = gesture_dict.get(0, [])
            return _build_stitched_trial(
                gesture_ids=seq_in,
                gesture_dict=gesture_dict,
                rest_list=rest_list,
                sampling_rate=sampling_rate,
                overlap_samples=overlap_samples,
                rng=rng,
                lead_trail_rest=lead_trail,
            )

        # 1. Synthesize positive trials
        for _ in range(n_positive_per_subject):
            trials_data.append(stitch_trial(target_labels))
            trials_labels.append(1)
            trials_subjects.append(s_id)

        # 2. Synthesize hard negative trials
        for _ in range(n_hard_negative_per_subject):
            strat = rng.choice(5, p=probs)

            if strat == 0:
                trial_arr = stitch_trial(reversed_target)
            elif strat == 1:
                other_g = non_target_gestures[rng.integers(len(non_target_gestures))]
                seq = [target_labels[0], other_g] if rng.random() < 0.5 else [other_g, target_labels[-1]]
                trial_arr = stitch_trial(seq)
            elif strat == 2:
                single_g = [target_labels[rng.integers(len(target_labels))]]
                trial_arr = stitch_trial(single_g, lead_trail=True)
            elif strat == 3:
                seq = [non_target_gestures[rng.integers(len(non_target_gestures))] for _ in range(len(target_labels))]
                trial_arr = stitch_trial(seq)
            else:
                # ADL
                rec = adl_recordings[rng.integers(len(adl_recordings))]
                t_len = int(rng.integers(int(1.5 * sampling_rate), int(3.0 * sampling_rate)))
                st = int(rng.integers(0, max(1, len(rec) - t_len + 1)))
                trial_arr = rec[st: st + t_len]

            trials_data.append(trial_arr)
            trials_labels.append(0)
            trials_subjects.append(s_id)

    # Shuffle dataset to uniformly interleave positive and negative trials
    num_total = len(trials_data)
    if num_total > 0:
        perm = rng.permutation(num_total)
        shuffled_data = [trials_data[i] for i in perm]
        shuffled_labels = np.array(trials_labels, dtype=np.int32)[perm]
        shuffled_subjects = np.array(trials_subjects, dtype=np.int32)[perm]
    else:
        shuffled_data = []
        shuffled_labels = np.array([], dtype=np.int32)
        shuffled_subjects = np.array([], dtype=np.int32)

    return EmgDataset(
        data=shuffled_data,
        labels=shuffled_labels,
        subjects=shuffled_subjects,
        is_normalized=emg.is_normalized,
    )
