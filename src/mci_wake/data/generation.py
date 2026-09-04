import warnings
from typing import Optional, cast
import numpy as np
import numpy.typing as npt

from mci_wake.data.types import EmgDataset
from mci_wake.data_handler.stitching import StitchingDataHandler


def generate_training_data(
    emg: EmgDataset,
    adl: EmgDataset,
    target_sequence: list[str],
    n_positive_per_subject: int = 15,
    n_hard_negative_per_subject: int = 15,
    sampling_rate: float = 200.0,
    overlap_samples: int = 15,
    random_seed: Optional[int] = 42,
) -> EmgDataset:
    """Generates synthetic sequence trials for each subject using StitchingDataHandler.

    Stitching is done strictly within the same subject to model realistic intra-user dynamics.

    Args:
        emg: EmgDataset containing gesture recordings. Should be normalized.
        adl: ADL noise recordings used as additional negatives.
        target_sequence: List of gesture names defining the positive wake sequence.
        n_positive_per_subject: Number of positive trials synthesized per subject.
        n_hard_negative_per_subject: Number of hard negative trials synthesized per subject.
        sampling_rate: EMG sampling rate in Hz (default: 200.0).
        overlap_samples: Number of samples for raised-cosine cross-fade (default: 15).
        random_seed: Optional seed for reproducibility.

    Returns:
        EmgDataset: Dataset of synthesized sequence trials with labels (1=positive, 0=negative)
                    and preserved subject IDs.
    """
    assert len(emg) != 0 and len(adl) != 0
    assert emg.is_normalized, "Emg must be normalized before stitching"
    assert adl.is_normalized, "ADL must be normalized before stitching"
    assert target_sequence, "target_sequence must contain at least one gesture name."

    if random_seed is not None:
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)

    trials: list[npt.NDArray[np.float32]] = []
    labels: list[int] = []
    subjects: list[int] = []


    for s in np.unique(emg.subjects):
        mask = cast(np.ndarray, cast(object, emg.subjects == s))
        handler = StitchingDataHandler(
            emg_data=emg[mask],
            adl_data=adl,
            gestures=target_sequence,
            sampling_rate=sampling_rate,
            overlap_samples=overlap_samples,
            realtime=False,
        )

        for _ in range(n_positive_per_subject):
            trials.append(handler.generate_positive())
            labels.append(1)
            subjects.append(int(s))

        for _ in range(n_hard_negative_per_subject):
            trials.append(handler.generate_negative())
            labels.append(0)
            subjects.append(int(s))

    num_total = len(trials)
    p = np.random.permutation(num_total)
    shuffled_data = [trials[i] for i in p]
    shuffled_labels = np.array(labels, dtype=np.int32)[p]
    shuffled_subjects = np.array(subjects, dtype=np.int32)[p]

    return EmgDataset(
        data=shuffled_data,
        labels=shuffled_labels,
        subjects=shuffled_subjects,
        is_normalized=emg.is_normalized,
    )
