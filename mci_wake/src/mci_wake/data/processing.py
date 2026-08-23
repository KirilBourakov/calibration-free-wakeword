from dataclasses import replace
from typing import Union, Any, Optional, Tuple, List, Dict

import numpy as np
from numpy import typing as npt

from mci_wake.data.normalization import Normalize
from mci_wake.data.types import EmgDataset
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

def prepare_datasets(
    emg_data: EmgDataset,
    adl_data: EmgDataset,
    window_size: int,
    increment_size: int,
    train_split: float = 0.95,
    test_split: float = 0.05
) -> Tuple[EmgDataset, EmgDataset, Normalize]:
    """Extracts features and splits the data into training and testing sets proportionally with safe normalization.

    Args:
        emg_data: The complete set of gesture EMG data samples.
        adl_data: The ADL EMG data samples.
        window_size: The size of the sliding window for feature extraction.
        increment_size: The increment step for the sliding window.
        train_split: The proportion of data to use for training. Defaults to 0.95.
        test_split: The proportion of data to use for testing. Defaults to 0.05.

    Returns:
        Tuple containing (train, test, normalizer)
    """
    print("Warning: prepare_loso_datasets recommended.")

    train_emg_raw, test_emg_raw = emg_data.split(test_percentage=test_split, by_subject=False)
    adl_train_raw, adl_test_raw = adl_data.split(test_percentage=test_split, by_subject=False)

    # Fit Normalize strictly on training partition
    normalizer = Normalize.create(train_emg_raw.combine(adl_train_raw))

    # Apply normalizer
    train_emg_norm = normalizer(train_emg_raw)
    adl_train_norm = normalizer(adl_train_raw)
    test_emg_norm = normalizer(test_emg_raw)
    adl_test_norm = normalizer(adl_test_raw)

    # Extract features
    train_emg_feats = get_features(train_emg_norm, window_size, increment_size)
    test_emg_feats = get_features(test_emg_norm, window_size, increment_size)
    adl_train_feats = get_features(adl_train_norm, window_size, increment_size)
    adl_test_feats = get_features(adl_test_norm, window_size, increment_size)

    train = train_emg_feats.combine(adl_train_feats)
    test = test_emg_feats.combine(adl_test_feats)

    print(f"Final training set: {len(train)} samples ({len(train_emg_feats)} gestures + {len(adl_train_feats)} ADL)")
    print(f"Final testing set: {len(test)} samples ({len(test_emg_feats)} gestures + {len(adl_test_feats)} ADL)")

    return train, test, normalizer

def prepare_loso_datasets(
    emg_data: EmgDataset,
    adl_data: EmgDataset,
    window_size: int,
    increment_size: int,
    test_subject_ids: Optional[List[int]] = None,
    test_subject_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[EmgDataset, EmgDataset, TrainData, Normalize]:
    """Extracts features and splits data using Leave-One-Subject-Out (LOSO) cross-validation with safe normalization.

    Ensures that test subjects' gesture data is strictly isolated from the training set,
    and that normalization parameters are fitted exclusively on training subjects.

    Args:
        emg_data: Complete set of gesture EMG data samples.
        adl_data: ADL noise segments.
        window_size: Window size for sliding feature extraction.
        increment_size: Increment step size.
        test_subject_ids: Specific subject IDs to hold out for testing. If None, randomly picks test_subject_ratio of subjects.
        test_subject_ratio: Proportion of unique subjects to allocate to test set if test_subject_ids is None.
        random_seed: Seed for random subject selection.

    Returns:
        Tuple containing (train, test, train_subject_ids, normalizer).
    """
    train_emg_raw, test_emg_raw = emg_data.split(
        test_percentage=test_subject_ratio,
        by_subject=True,
        test_subject_ids=test_subject_ids,
        random_seed=random_seed,
    )
    adl_train_raw, adl_test_raw = adl_data.split(
        test_percentage=test_subject_ratio,
        by_subject=False,
    )

    # Fit Normalize strictly on training partition
    normalizer = Normalize.create(train_emg_raw.combine(adl_train_raw))

    # Apply normalizer
    train_emg_norm = normalizer(train_emg_raw)
    adl_train_norm = normalizer(adl_train_raw)
    test_emg_norm = normalizer(test_emg_raw)
    adl_test_norm = normalizer(adl_test_raw)

    # Extract features
    train_emg_feats = get_features(train_emg_norm, window_size, increment_size)
    test_emg_feats = get_features(test_emg_norm, window_size, increment_size)
    adl_train_feats = get_features(adl_train_norm, window_size, increment_size)
    adl_test_feats = get_features(adl_test_norm, window_size, increment_size)

    train = train_emg_feats.combine(adl_train_feats)
    test = test_emg_feats.combine(adl_test_feats)

    held_out_subjects = np.unique(test_emg_raw.subjects).tolist()
    data = TrainData()
    for s in np.unique(train_emg_raw.subjects):
        item = s.item() if hasattr(s, "item") else s
        data.emg.append(f"user{item}")
    data.disco = [f"S{s.item() if hasattr(s, 'item') else s}" for s in np.unique(adl_train_raw.subjects)]

    print(f"--- LOSO (Leave-One-Subject-Out) Dataset Split ---")
    print(f"Held-out test subject IDs ({len(held_out_subjects)} subjects): {held_out_subjects}")
    print(f"Training EPN subjects ({len(data.emg)} subjects): {data.emg}")
    print(f"Training ADL subjects ({len(data.disco)} subjects): {data.disco}")
    print(f"Final training set: {len(train)} samples ({len(train_emg_feats)} gestures + {len(adl_train_feats)} ADL)")
    print(f"Final testing set:  {len(test)} samples ({len(test_emg_feats)} gestures + {len(adl_test_feats)} ADL)")

    return train, test, data, normalizer


def get_features(
    data: EmgDataset,
    window_size: int = 10,
    window_inc: int = 5,
    feats: Optional[List[str]] = None,
    feat_dic: Optional[Dict[str, Any]] = None,
    force_normalize: bool = True,
) -> EmgDataset:
    """Extracts sliding windows or engineered features from an EmgDataset.

    Args:
        data: EmgDataset containing raw or normalized EMG recordings.
        window_size: Number of samples in each sliding window. Defaults to 10.
        window_inc: Step size between consecutive sliding windows. Defaults to 5.
        feats: Optional list of feature names to extract. If None, raw windows are returned.
        feat_dic: Optional dictionary of parameters passed to the feature extractor.
        force_normalize: Whether to enforce that the dataset has been normalized. Defaults to True.

    Returns:
        EmgDataset: Dataset containing the windowed/extracted features with preserved labels, subjects, and normalization state.
    """
    if force_normalize and not data.is_normalized:
        raise ValueError("EMG data must be normalized prior to feature extraction.")

    from libemg.utils import get_windows
    from libemg.feature_extractor import FeatureExtractor

    windowed_data = [
        get_windows(d, window_size, window_inc).astype(np.float32)
        for d in data.data
    ]

    if feats is not None:
        fe = FeatureExtractor()
        extracted = [
            fe.extract_features(feats, w, array=True, feature_dic=feat_dic or {})
            for w in windowed_data
        ]
        windowed_data = [
            np.nan_to_num(x, copy=True, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for x in extracted
        ]

    return EmgDataset(
        data=windowed_data,
        labels=data.labels.copy(),
        subjects=data.subjects.copy(),
        is_normalized=data.is_normalized,
    )
