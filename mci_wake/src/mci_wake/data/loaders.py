import json
import os
import pickle
import random
from pathlib import Path
from statistics import mode
from typing import Dict, Any, Tuple, Optional, Sequence

import numpy as np
from numpy import typing as npt

from mci_wake.data.types import RawData, gesture_mapping, EPNData, ADL_DATA, EPN_DATA


def load_raw_data(presplit_adl=True) -> RawData:
    """Loads ADL and gesture EMG data from the dataset alongside subject IDs."""
    adl_data, adl_subjects = load_disco_adls(ADL_DATA)
    if presplit_adl:
        adl_data, adl_subjects = split_disco_adls(adl_data, adl_subjects)

    epn = load_epn_data(EPN_DATA)

    epn_emg = np.array(epn.emg, dtype='object')
    epn_labels = np.array(epn.labels)
    epn_subjects = np.array(epn.subject_ids)

    n_subs = len(np.unique(epn_subjects))
    print(f"Loaded {len(epn_emg)} gesture samples from EPN dataset across {n_subs} subjects.")
    print(f"Loaded {len(adl_data)} ADL noise segments.")

    return RawData(
        epn_emg=epn_emg,
        epn_labels=epn_labels,
        epn_subjects=epn_subjects,
        adl_emg=adl_data,
        adl_subjects=adl_subjects,
    )


def extract_data(data: Dict[str, Any]) -> Tuple[Optional[npt.NDArray[Any]], Optional[npt.NDArray[Any]], Optional[int], Optional[int]]:
    """Extracts EMG, IMU, and label data from a single data sample.

    Args:
        data: A dictionary containing the raw sample data.

    Returns:
        Tuple containing:
            - emg: Transposed EMG data for the sample.
            - quat: Transposed quaternion data.
            - label: The mapped gesture label.
            - mode_myo: The most frequent myoDetection label.
            Returns (None, None, None, None) if 'gestureName' is missing.
    """
    emg = np.transpose([data['emg']['ch' + str(ch)] for ch in range(1, 9)])
    quat = np.transpose([data['quaternion'][v] for v in ['w','x','y','z']])
    label = None
    myo_labels = np.array(data['myoDetection'])
    myo_labels = myo_labels[np.where(myo_labels != 0)]
    if len(myo_labels) == 0:
        myo_labels = np.array([0])

    if 'gestureName' in data:
        label = gesture_mapping[data['gestureName']]
    else:
        return None, None, None, None # Half of the test data isn't available

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
    return emg[start_idx:end_idx], quat[int(start_idx * 0.25):int(end_idx*0.25)], label, int(mode(myo_labels))


def load_epn_data(
    path: Path | str,
    gesture_sets: Sequence[str] = ('trainingSamples', 'testingSamples'),
    subjects: Sequence[int] = tuple(range(1, 307)),
    subject_types: Sequence[str] = ('training', 'testing')
) -> EPNData:
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

    result = EPNData()

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
                        e, i, l, ml = extract_data(jd[s][sample])
                        if e is not None:
                            result.emg.append(e)
                            result.imu.append(i)
                            result.labels.append(l)
                            result.myo_labels.append(ml)
                            result.subject_ids.append(sub)

    # Save dataset as pkl (to save time)
    if len(subjects) > 300:
        with open('dataset.pkl', 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return result


def load_disco_adls(
    path: Path | str,
    min_len: int = 150
) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.int_]]:
    """Loads raw ADL dataset from disk, trims EMG channels, and tracks subject IDs.

    Args:
        path: Path to the ADL dataset root directory.
        min_len: Minimum number of rows required to keep a recording.
    Returns:
        tuple: (recordings, subject_ids) where `recordings` is a list of 2D EMG arrays
               and `subject_ids` is a parallel integer array of the subject ID (1-15) for each recording.
    """
    path = Path(path)
    recordings = []
    subject_ids = []

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
                recordings.append(data[:, -8:])
                subject_ids.append(s)
            else:
                print(f"WARNING: skipping invalid or too short recording for user {s} ({file_path.name})")

    return recordings, np.array(subject_ids, dtype=int)


def split_disco_adls(
    recordings: Sequence[npt.NDArray[np.float64]],
    subject_ids: npt.NDArray[np.int_],
    window: tuple[int, int] = (150, 400),
    step: int = 50
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.int_]]:
    """Windows continuous EMG recordings into randomized segments while preserving subject IDs.

    Args:
        recordings: Sequence of 2D arrays containing continuous EMG data.
        subject_ids: Parallel array of subject IDs corresponding to each recording.
        window: Tuple of (min_window, max_window) specifying slice lengths.
        step: Step size (stride) between the start of consecutive windows.
    Returns:
        tuple: (windows, window_subject_ids) as parallel arrays of segmented data and subject labels.
    """
    min_window, max_window = window
    windows = []
    window_subject_ids = []

    for data, s in zip(recordings, subject_ids):
        # Secondary safety check in case data is passed directly to the splitter
        if len(data) < min_window:
            continue

        for i in range(0, len(data) - min_window + 1, step):
            max_possible_len = min(max_window, len(data) - i)
            win_len = random.randint(min_window, max_possible_len)

            windows.append(data[i : i + win_len])
            window_subject_ids.append(s)

    return np.array(windows, dtype='object'), np.array(window_subject_ids, dtype=int)
