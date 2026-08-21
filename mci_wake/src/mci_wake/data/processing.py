from dataclasses import replace
from typing import Union, Any, Optional, Tuple, List, Dict

import numpy as np
from numpy import typing as npt

import libemg
from mci_wake.data.normalization import Normalize
from mci_wake.data.types import EmgData, EmgDataset
from mci_wake.neural.classifier import TrainData


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
    emg_data_all: npt.NDArray[np.object_],
    labels_all: npt.NDArray[Any],
    subject_ids_all: npt.NDArray[Any],
    adl_data: npt.NDArray[np.object_],
    adl_subjects: npt.NDArray[np.int_],
    *train_datas: TrainData,
) -> Tuple[npt.NDArray[np.object_], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.object_], npt.NDArray[np.int_]]:
    """Filters out raw dataset samples corresponding to subjects already present in any of the provided TrainData instances.

    Args:
        emg_data_all: Combined training and testing gesture EMG data.
        labels_all: Combined training and testing gesture labels.
        subject_ids_all: Subject IDs for each gesture sample.
        adl_data: Loaded ADL EMG data.
        adl_subjects: Subject IDs for each ADL sample.
        *train_datas: Variable number of TrainData objects containing used subject IDs in `.emg` and `.disco`.

    Returns:
        Tuple containing filtered (emg_data_all, labels_all, subject_ids_all, adl_data, adl_subjects).
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
        emg_mask = ~np.isin(subject_ids_all, list(used_emg_subjects))
        filtered_emg_data_all = emg_data_all[emg_mask]
        filtered_labels_all = labels_all[emg_mask]
        filtered_subject_ids_all = subject_ids_all[emg_mask]
    else:
        filtered_emg_data_all = emg_data_all
        filtered_labels_all = labels_all
        filtered_subject_ids_all = subject_ids_all

    if used_disco_subjects:
        adl_mask = ~np.isin(adl_subjects, list(used_disco_subjects))
        filtered_adl_data = adl_data[adl_mask]
        filtered_adl_subjects = adl_subjects[adl_mask]
    else:
        filtered_adl_data = adl_data
        filtered_adl_subjects = adl_subjects

    print(f"Filtered raw training data using {len(train_datas)} TrainData instances:")
    print(f"  EPN samples remaining: {len(filtered_emg_data_all)} / {len(emg_data_all)}")
    print(f"  ADL samples remaining: {len(filtered_adl_data)} / {len(adl_data)}")

    return filtered_emg_data_all, filtered_labels_all, filtered_subject_ids_all, filtered_adl_data, filtered_adl_subjects


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

def prepare_datasets(
    emg_data: EmgDataset,
    adl_data: EmgDataset,
    window_size: int,
    increment_size: int,
    train_split: float = 0.95,
    test_split: float = 0.05
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], Normalize]:
    """Extracts features and splits the data into training and testing sets proportionally with safe normalization.

    Args:
        emg_data: The complete set of gesture EMG data samples.
        adl_data: The ADL EMG data samples.
        window_size: The size of the sliding window for feature extraction.
        increment_size: The increment step for the sliding window.
        train_split: The proportion of data to use for training. Defaults to 0.95.
        test_split: The proportion of data to use for testing. Defaults to 0.05.

    Returns:
        Tuple containing (train_emg, train_labels, test_emg, test_labels, normalizer).
    """
    print("Warning: prepare_loso_datasets recommended.")

    n_emg_train = int(len(emg_data) * train_split)
    n_adl_train = int(len(adl_data) * train_split)

    train_emg_raw = emg_data[:n_emg_train]
    test_emg_raw = emg_data[-int(len(emg_data) * test_split):]

    adl_train_raw = adl_data[:n_adl_train]
    adl_test_raw = adl_data[-int(len(adl_data) * test_split):]

    # Fit Normalize strictly on training partition
    normalizer = Normalize.create(train_emg_raw.combine(test_emg_raw))

    # Apply normalizer
    train_emg_norm = normalizer(train_emg_raw)
    adl_train_norm = normalizer(adl_train_raw)
    test_emg_norm = normalizer(test_emg_raw)
    adl_test_norm = normalizer(adl_test_raw)

    # Extract features
    train_emg_feats: npt.NDArray[Any] = get_features(train_emg_norm, window_size, increment_size, None, None)
    test_emg_feats: npt.NDArray[Any] = get_features(test_emg_norm, window_size, increment_size, None, None)
    adl_train_feats: npt.NDArray[Any] = get_features(adl_train_norm, window_size, increment_size, None, None)
    adl_test_feats: npt.NDArray[Any] = get_features(adl_test_norm, window_size, increment_size, None, None)

    train_labels_final: npt.NDArray[Any] = np.hstack([train_labels, np.zeros(len(adl_train_feats))])
    train_emg_final: npt.NDArray[Any] = np.hstack([train_emg_feats, adl_train_feats])
    test_labels_final: npt.NDArray[Any] = np.hstack([test_labels, np.zeros(len(adl_test_feats))])
    test_emg_final: npt.NDArray[Any] = np.hstack([test_emg_feats, adl_test_feats])

    print(f"Final training set: {len(train_emg_final)} samples ({len(train_labels)} gestures + {len(adl_train_feats)} ADL)")
    print(f"Final testing set: {len(test_emg_final)} samples ({len(test_labels)} gestures + {len(adl_test_feats)} ADL)")

    return train_emg_final, train_labels_final, test_emg_final, test_labels_final, normalizer


def prepare_loso_datasets(
    emg_data_all: npt.NDArray[EmgData],
    labels_all: npt.NDArray[Any],
    subject_ids_all: npt.NDArray[Any],
    adl_data: npt.NDArray[EmgData],
    adl_ids: npt.NDArray[np.int64],
    window_size: int,
    increment_size: int,
    test_subject_ids: Optional[List[int]] = None,
    test_subject_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], TrainData, Normalize]:
    """Extracts features and splits data using Leave-One-Subject-Out (LOSO) cross-validation with safe normalization.

    Ensures that test subjects' gesture data is strictly isolated from the training set,
    and that normalization parameters are fitted exclusively on training subjects.

    Args:
        emg_data_all: Complete set of gesture EMG data samples.
        labels_all: Mapped labels for all samples.
        subject_ids_all: Subject/User IDs corresponding to each EMG sample.
        adl_data: ADL noise segments.
        adl_ids: Parallel array of ADL subject IDs.
        window_size: Window size for sliding feature extraction.
        increment_size: Increment step size.
        test_subject_ids: Specific subject IDs to hold out for testing. If None, randomly picks test_subject_ratio of subjects.
        test_subject_ratio: Proportion of unique subjects to allocate to test set if test_subject_ids is None.
        random_seed: Seed for random subject selection.

    Returns:
        Tuple containing (train_emg, train_labels, test_emg, test_labels, train_subject_ids, normalizer).
    """
    unique_subjects = np.unique(subject_ids_all)
    if test_subject_ids is None:
        rng = np.random.default_rng(random_seed)
        n_test = max(1, int(len(unique_subjects) * test_subject_ratio))
        test_subject_ids = list(rng.choice(unique_subjects, size=n_test, replace=False))

    test_mask = np.isin(subject_ids_all, test_subject_ids)
    train_mask = ~test_mask

    train_emg_raw = emg_data_all[train_mask]
    train_labels_raw = labels_all[train_mask]
    test_emg_raw = emg_data_all[test_mask]
    test_labels_raw = labels_all[test_mask]

    # Split ADL noise data proportionally
    n_adl_train = int(len(adl_data) * (1.0 - test_subject_ratio))
    adl_train_raw = adl_data[:n_adl_train]
    adl_test_raw = adl_data[n_adl_train:]

    # Fit Normalize strictly on training partition
    normalizer = Normalize.create(list(train_emg_raw) + list(adl_train_raw))

    # Apply normalizer
    train_emg_norm = normalizer(list(train_emg_raw))
    adl_train_norm = normalizer(list(adl_train_raw))
    test_emg_norm = normalizer(list(test_emg_raw))
    adl_test_norm = normalizer(list(adl_test_raw))

    # Extract features
    train_emg_feats: npt.NDArray[Any] = get_features(train_emg_norm, window_size, increment_size, None, None)
    test_emg_feats: npt.NDArray[Any] = get_features(test_emg_norm, window_size, increment_size, None, None)
    adl_train_feats: npt.NDArray[Any] = get_features(adl_train_norm, window_size, increment_size, None, None)
    adl_test_feats: npt.NDArray[Any] = get_features(adl_test_norm, window_size, increment_size, None, None)

    train_labels_final: npt.NDArray[Any] = np.hstack([train_labels_raw, np.zeros(len(adl_train_feats))])
    train_emg_final: npt.NDArray[Any] = np.hstack([train_emg_feats, adl_train_feats])
    test_labels_final: npt.NDArray[Any] = np.hstack([test_labels_raw, np.zeros(len(adl_test_feats))])
    test_emg_final: npt.NDArray[Any] = np.hstack([test_emg_feats, adl_test_feats])

    data = TrainData()
    for s in np.unique(subject_ids_all[train_mask]):
        item = s.item() if hasattr(s, "item") else s
        data.emg.append(f"user{item}")
    data.disco = [f"S{s}" for s in np.unique(adl_ids[:n_adl_train])]

    print(f"--- LOSO (Leave-One-Subject-Out) Dataset Split ---")
    print(f"Held-out test subject IDs ({len(test_subject_ids)} subjects): {test_subject_ids}")
    print(f"Training EPN subjects ({len(data.emg)} subjects): {data.emg}")
    print(f"Training ADL subjects ({len(data.disco)} subjects): {data.disco}")
    print(f"Final training set: {len(train_emg_final)} samples ({len(train_labels_raw)} gestures + {len(adl_train_feats)} ADL)")
    print(f"Final testing set:  {len(test_emg_final)} samples ({len(test_labels_raw)} gestures + {len(adl_test_feats)} ADL)")

    return train_emg_final, train_labels_final, test_emg_final, test_labels_final, data, normalizer


def get_features(
    data: EmgDataset,
    window_size: int,
    window_inc: int,
    feats: Optional[List[str]],
    feat_dic: Optional[Dict[str, Any]],
    force_normalize: bool = True
) -> npt.NDArray[Any]:
    """Extracts features from the raw EMG data using a sliding window.

    Args:
        data: Raw EMG data samples.
        window_size: Size of the sliding window.
        window_inc: Increment step for the sliding window.
        feats: List of feature names to extract. If None, returns raw windows.
        feat_dic: Optional dictionary for feature extraction parameters.
        force_normalize: Whether EMG signals must be pre-normalized before windowing. Defaults to True.

    Returns:
        npt.NDArray[Any]: Extracted features for each data sample.
    """
    from libemg.feature_extractor import FeatureExtractor
    fe = FeatureExtractor()

    if force_normalize:
        assert all(d.is_normalized for d in data)

    windowed_data = np.array([libemg.utils.get_windows(d, window_size, window_inc) for d in data.data], dtype='object')

    if feats is None:
        return windowed_data

    if feat_dic is not None:
        extracted_feats = np.array([fe.extract_features(feats, d, array=True, feature_dic=feat_dic) for d in windowed_data], dtype='object')
    else:
        extracted_feats = np.array([fe.extract_features(feats, np.array(d, dtype='float'), array=True) for d in windowed_data], dtype='object')

    return np.nan_to_num(extracted_feats, copy=True, nan=0, posinf=0, neginf=0)
