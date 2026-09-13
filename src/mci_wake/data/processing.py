from dataclasses import replace
from typing import Union, Any, Optional, Tuple, List, Dict

import numpy as np
from numpy import typing as npt

from mci_wake.data.types import EmgDataset, TrainData



def _parse_subject_id(val: Union[int, str, Any]) -> Optional[int]:
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned.startswith("user"):
            cleaned = cleaned[4:].strip()
        elif cleaned.startswith("s"):
            cleaned = cleaned[1:].strip()
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def filter_training(
    emg_data: EmgDataset,
    adl_data: EmgDataset,
    *train_datas: TrainData,
) -> Tuple[EmgDataset, EmgDataset]:
    """Filters out raw dataset samples corresponding to subjects already present in any of the provided TrainData instances.

    Args:
        emg_data: Combined gesture EMG dataset.
        adl_data: Loaded ADL EMG dataset.
        *train_datas: Variable number of TrainData objects containing used subject IDs in `.emg` and `.disco`.

    Returns:
        Tuple containing filtered (emg_data, adl_data).
    """
    used_emg_subjects: set[int] = set()
    used_disco_subjects: set[int] = set()

    for td in train_datas:
        for s in td.emg:
            parsed = _parse_subject_id(s)
            if parsed is not None:
                used_emg_subjects.add(parsed)
        for s in td.disco:
            parsed = _parse_subject_id(s)
            if parsed is not None:
                used_disco_subjects.add(parsed)

    if used_emg_subjects:
        emg_mask = ~np.isin(emg_data.subjects, list(used_emg_subjects))
        filtered_emg_data = EmgDataset(
            data=[d for d, m in zip(emg_data.data, emg_mask) if m],
            labels=emg_data.labels[emg_mask],
            subjects=emg_data.subjects[emg_mask],
            is_normalized=emg_data.is_normalized,
        )
    else:
        filtered_emg_data = emg_data

    if used_disco_subjects:
        adl_mask = ~np.isin(adl_data.subjects, list(used_disco_subjects))
        filtered_adl_data = EmgDataset(
            data=[d for d, m in zip(adl_data.data, adl_mask) if m],
            labels=adl_data.labels[adl_mask],
            subjects=adl_data.subjects[adl_mask],
            is_normalized=adl_data.is_normalized,
        )
    else:
        filtered_adl_data = adl_data

    print(f"Filtered raw training data using {len(train_datas)} TrainData instances:")
    print(f"  EPN samples remaining: {len(filtered_emg_data)} / {len(emg_data)}")
    print(f"  ADL samples remaining: {len(filtered_adl_data)} / {len(adl_data)}")

    return filtered_emg_data, filtered_adl_data


def preprocess_nm_data(emg: EmgDataset) -> EmgDataset:
    """Randomly clips 'No Motion' segments to introduce variability.
    Args:
        emg: The complete set of EMG data samples.
    Returns:
        npt.NDArray[EmgData]: The EMG data with randomized 'No Motion' segment lengths.
    """
    new: list[npt.NDArray[np.float32]] = []
    for data, label, _ in emg:
        if label == 0:
            clip_length = np.random.randint(150, 351)
            new.append(data[0:clip_length])
        else:
            new.append(data)
    return replace(emg, data=new)

