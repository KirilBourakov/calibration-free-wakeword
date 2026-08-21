import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Sequence

import numpy as np
from numpy import typing as npt

from mci_wake.data.types import EmgData, gesture_mapping, ADL_DATA, EPN_DATA, EmgDataset


def load_raw_data(presplit_adl=True) -> tuple[EmgDataset, EmgDataset]:
    """Loads EPN and ADL datasets"""
    adl = load_disco_adls(ADL_DATA)
    if presplit_adl:
        adl = split_disco_adls(adl)
    epn = load_epn_data(EPN_DATA)

    epn_emg = np.array([EmgData(data=d, is_normalized=epn.is_normalized) for d in epn.data], dtype='object')
    epn_labels = epn.labels
    epn_subjects = epn.subjects

    adl_emg = np.array([EmgData(data=d, is_normalized=adl.is_normalized) for d in adl.data], dtype='object')
    adl_subjects = adl.subjects

    n_subs = len(np.unique(epn_subjects))
    print(f"Loaded {len(epn_emg)} gesture samples from EPN dataset across {n_subs} subjects.")
    print(f"Loaded {len(adl_emg)} ADL noise segments.")

    return epn, adl


def load_epn_data(
    path: Path | str,
    gesture_sets: Sequence[str] = ('trainingSamples', 'testingSamples'),
    subjects: Sequence[int] = tuple(range(1, 307)),
    subject_types: Sequence[str] = ('training', 'testing')
) -> EmgDataset:
    """Loads and collects the EMG dataset from JSON files or a cached pickle."""
    path = Path(path)

    # Check if pkl file exists (to save time)
    if os.path.exists('dataset.pkl') and len(subjects) > 300:
        try:
            with open('dataset.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception:
            print("Failed to load from .pkl file.")
            pass

    emg_list: list[npt.NDArray[np.float32]] = []
    labels_list: list[int] = []
    subjects_list: list[int] = []

    for t in subject_types:
        print("Getting " + t + " subjects...")
        for sub in subjects:
            if sub % 10 == 0:
                print("Subject " + str(sub) + "...")
            user_file = path / (t + 'JSON') / ('user' + str(sub)) / ('user' + str(sub) + '.json')
            if not user_file.exists():
                continue
            with open(user_file, encoding="utf8") as f:
                jd = json.load(f)
                for s in gesture_sets:
                    for sample in jd[s]:
                        e, l = _extract_data(jd[s][sample])
                        if e is not None and l is not None:
                            emg_list.append(e)
                            labels_list.append(l)
                            subjects_list.append(sub)

    result = EmgDataset(
        data=emg_list,
        labels=np.array(labels_list, dtype=np.int32),
        subjects=np.array(subjects_list, dtype=np.int32),
        is_normalized=False,
    )

    # Save dataset as pkl (to save time)
    if len(subjects) > 300:
        with open('dataset.pkl', 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return result


def load_disco_adls(
    path: Path | str,
    min_len: int = 150
) -> EmgDataset:
    """Loads raw ADL dataset from disk, trims EMG channels, and tracks subject IDs.

    Args:
        path: Path to the ADL dataset root directory.
        min_len: Minimum number of rows required to keep a recording.
    Returns:
        EmgDataset: Dataset containing ADL EMG recordings and parallel subject IDs.
    """
    path = Path(path)
    recordings: list[npt.NDArray[np.float32]] = []
    subject_ids: list[int] = []

    for s in range(1, 16):
        sub_path = path / f"S{s}" / "ADL"
        if not sub_path.exists():
            print(f"WARNING: {sub_path} doesn't exist")
            continue

        for file_path in sub_path.glob("*.csv"):
            data = np.loadtxt(file_path, delimiter=',')

            # Guard against 1D arrays and filter out short recordings early
            if data.ndim == 2 and len(data) >= min_len:
                # Keep only the last 8 EMG channels
                recordings.append(np.asarray(data[:, -8:], dtype=np.float32))
                subject_ids.append(s)
            else:
                print(f"WARNING: skipping invalid or too short recording for user {s} ({file_path.name})")

    return EmgDataset(
        data=recordings,
        labels=np.zeros(len(recordings), dtype=np.int32),
        subjects=np.array(subject_ids, dtype=np.int32),
        is_normalized=False,
    )


def split_disco_adls(
    data: EmgDataset,
    window: tuple[int, int] = (150, 400),
    step: int = 50
) -> EmgDataset:
    """Windows continuous EMG recordings into randomized segments while preserving subject IDs.

    Args:
        data: EmgDataset of ADL data.
        window: Tuple of (min_window, max_window) specifying slice lengths.
        step: Step size (stride) between the start of consecutive windows.
    Returns:
        EmgDataset: Windowed ADL dataset with parallel subject IDs.
    """
    min_window, max_window = window

    windows: list[npt.NDArray[np.float32]] = []
    subject_ids: list[int] = []

    for item, s in zip(data.data, data.subjects.tolist()):
        # Secondary safety check in case data is passed directly to the splitter
        if len(item) < min_window:
            continue

        for i in range(0, len(item) - min_window + 1, step):
            max_possible_len = min(max_window, len(item) - i)
            win_len = random.randint(min_window, max_possible_len)

            windows.append(item[i : i + win_len].astype(np.float32))
            subject_ids.append(s)

    return EmgDataset(
        data=windows,
        labels=np.zeros(len(subject_ids), dtype=np.int32),
        subjects=np.array(subject_ids, dtype=np.int32),
        is_normalized=data.is_normalized,
    )

def _extract_data(data: Dict[str, Any]) -> Tuple[Optional[npt.NDArray[np.float32]], Optional[int]]:
    """Extracts EMG and label data from a single sample.

    Args:
        data: A dictionary containing the raw sample data.

    Returns:
        Tuple containing (emg, label) or (None, None) if 'gestureName' is missing.
    """
    if 'gestureName' not in data:
        return None, None

    label = gesture_mapping[data['gestureName']]
    emg = np.transpose([data['emg']['ch' + str(ch)] for ch in range(1, 9)]).astype(np.float32)

    if 'groundTruth' in data:
        ground_truth = np.diff(np.array(data['groundTruth']))
        try:
            start_idx = np.where(ground_truth == 1)[0][0]
        except (IndexError, ValueError):
            start_idx = 0
        try:
            end_idx = np.where(ground_truth == -1)[0][0]
        except (IndexError, ValueError):
            end_idx = len(emg) - 1
    else:
        start_idx = 0
        end_idx = len(emg)

    return emg[start_idx:end_idx], label