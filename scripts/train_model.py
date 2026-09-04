from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from mci_wake.data import (
    EmgDataset,
    TrainData,
    gesture_mapping,
    get_features,
    load_raw_data,
    preprocess_nm_data,
)
from mci_wake.neural.classifier import DiscreteClassifierConfig
from mci_wake.neural.training import train_model
from mci_wake.transform.highpass import HighPassFilter
from mci_wake.transform.rest_normalization import RestNormalizer
from mci_wake.transform.transform import Transform

def parse_gesture_labels(
    emg: EmgDataset, target_gesture: Optional[str] = None
) -> Tuple[EmgDataset, int]:
    if target_gesture is not None:
        if target_gesture not in gesture_mapping:
            raise ValueError(
                f"Invalid target_gesture '{target_gesture}'. Available options: {list(gesture_mapping.keys())}"
            )
        target_original_label = gesture_mapping[target_gesture]
        emg = replace(emg, labels=np.where(emg.labels == target_original_label, 1, 0))

        print(f"=== Binary Classification Setup ===")
        print(f"Target gesture: '{target_gesture}' (original label {target_original_label}) -> 1")
        print(f"All other gestures + noGesture -> 0")
        print(f"Total positive target samples: {np.sum(emg.labels == 1)}")
        print(f"Total negative samples: {np.sum(emg.labels == 0)}")
        n_classes = 2
    else:
        print("=== Multi-class Classification Setup ===")
        print(f"Training across all {len(gesture_mapping)} gestures + noGesture (6 classes)")
        n_classes = 6

    return emg, n_classes


def main() -> None:
    # Parameters:
    WINDOW_SIZE: int = 10
    INCREMENT_SIZE: int = 5
    TEST_SUBJECT_RATIO: float = 0.1  # Hold out 10% of subjects for unseen test evaluation
    TARGET_GESTURE: str | None = 'fist'

    # 1. Load data
    emg, adl = load_raw_data()

    # 2. Preprocess
    emg = preprocess_nm_data(emg)
    emg, n_classes = parse_gesture_labels(emg, TARGET_GESTURE)

    # 3. Split raw data (LOSO by subject on EMG, sample split on ADL)
    train_emg, test_emg = emg.split(test_percentage=TEST_SUBJECT_RATIO, by_subject=True)
    train_adl, test_adl = adl.split(test_percentage=TEST_SUBJECT_RATIO, by_subject=False)

    # 4. Apply transforms (fit strictly on training data to prevent leakage)
    transforms = Transform(HighPassFilter(), RestNormalizer())
    transforms.fit(train_emg)

    train_emg = transforms(train_emg)
    test_emg = transforms(test_emg)
    train_adl = transforms(train_adl)
    test_adl = transforms(test_adl)

    # 5. Extract sliding window features & combine
    train_emg_feats = get_features(train_emg, WINDOW_SIZE, INCREMENT_SIZE)
    test_emg_feats = get_features(test_emg, WINDOW_SIZE, INCREMENT_SIZE)
    train_adl_feats = get_features(train_adl, WINDOW_SIZE, INCREMENT_SIZE)
    test_adl_feats = get_features(test_adl, WINDOW_SIZE, INCREMENT_SIZE)

    train = train_emg_feats.combine(train_adl_feats)
    test = test_emg_feats.combine(test_adl_feats)

    # 6. Track metadata and summary
    held_out_subjects = np.unique(test_emg.subjects).tolist()
    ids = TrainData(
        emg=[f"user{s.item() if hasattr(s, 'item') else s}" for s in np.unique(train_emg.subjects)],
        disco=[f"S{s.item() if hasattr(s, 'item') else s}" for s in np.unique(train_adl.subjects)],
    )

    print(f"--- LOSO (Leave-One-Subject-Out) Dataset Split ---")
    print(f"Held-out test subject IDs ({len(held_out_subjects)} subjects): {held_out_subjects}")
    print(f"Training EPN subjects ({len(ids.emg)} subjects): {ids.emg}")
    print(f"Training ADL subjects ({len(ids.disco)} subjects): {ids.disco}")
    print(f"Final training set: {len(train)} samples ({len(train_emg_feats)} gestures + {len(train_adl_feats)} ADL)")
    print(f"Final testing set:  {len(test)} samples ({len(test_emg_feats)} gestures + {len(test_adl_feats)} ADL)")

    # 7. Train classifier
    model_config = DiscreteClassifierConfig(
        n_classes=n_classes,
        gestures=[TARGET_GESTURE] if TARGET_GESTURE else list(gesture_mapping.keys()),
    )
    train_model(train, test, model_config, customers=ids)


if __name__ == "__main__":
    main()
