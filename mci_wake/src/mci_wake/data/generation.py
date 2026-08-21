import numpy as np

import numpy.typing as npt

from mci_wake.data.types import EmgData, EmgDataset


def generate_training_data(
    data: EmgDataset,
    target_sequence: list[str],
    n_positive_per_subject: int = 15,
    n_hard_negative_per_subject: int = 15,
    sampling_rate: float = 200.0
) -> tuple[list[EmgData], list[int]]:
    """
    Generates synthetic positive and hard negative sequence trials for given subjects.
    Stitching is done strictly within the same subject to model realistic intra-user dynamics.
    """
    # need to normalize pre-stitching
    return [], []