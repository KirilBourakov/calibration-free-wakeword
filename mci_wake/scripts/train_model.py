from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from mci_wake.data import (
    EmgDataset,
    gesture_mapping,
    load_raw_data,
    preprocess_nm_data,
    prepare_loso_datasets,
)
from mci_wake.neural.classifier import DiscreteClassifierConfig
from mci_wake.neural.training import train_model
from mci_wake.transform.highpass import HighPassFilter
from mci_wake.transform.rest_normalization import RestNormalizer
from mci_wake.transform.transform import Transform

# Target gesture to recognize (e.g. 'pinch', 'fist').
# Set to None for multi-class classification (6 classes).
TARGET_GESTURE: Optional[str] = None


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

    # 1. Load data alongside subject IDs
    emg, adl = load_raw_data()
    transforms = Transform(HighPassFilter(), RestNormalizer())
    transforms.fit(emg)

    # 2. Preprocess
    emg = preprocess_nm_data(emg)
    emg, n_classes = parse_gesture_labels(emg, TARGET_GESTURE)

    # 3. Prepare features and splits using LOSO
    emg = transforms(emg)
    adl = transforms(adl)
    train, test, ids = prepare_loso_datasets(
        emg,
        adl,
        WINDOW_SIZE,
        INCREMENT_SIZE,
        test_subject_ratio=TEST_SUBJECT_RATIO,
    )

    # 4. Train classifier
    model_config = DiscreteClassifierConfig(n_classes=n_classes)
    train_model(train, test, model_config, customers=ids)


if __name__ == "__main__":
    main()
