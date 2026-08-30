from dataclasses import replace

import numpy as np

from mci_wake.data import gesture_mapping, load_raw_data, preprocess_nm_data, prepare_loso_datasets
from mci_wake.neural.training import train_model
from mci_wake.neural.classifier import DiscreteClassifierConfig
from mci_wake.transform.highpass import HighPassFilter
from mci_wake.transform.rest_normalization import RestNormalizer
from mci_wake.transform.transform import Transform

# Target gesture to recognize. Everything else will be classified as 'other' (0).
# Available gestures in dataset: 'fist', 'waveIn', 'waveOut', 'open', 'pinch'
TARGET_GESTURE = 'fist'

def main() -> None:
    """Main execution pipeline for training the binary gesture model using Leave-One-Subject-Out (LOSO)."""
    # Parameters:
    WINDOW_SIZE: int = 10 
    INCREMENT_SIZE: int = 5
    TEST_SUBJECT_RATIO: float = 0.1  # Hold out 10% of subjects for unseen test evaluation

    print(f"Target gesture: {TARGET_GESTURE}")
    if TARGET_GESTURE not in gesture_mapping:
        raise ValueError(f"Invalid TARGET_GESTURE '{TARGET_GESTURE}'. Options: {list(gesture_mapping.keys())}")
    
    target_original_label = gesture_mapping[TARGET_GESTURE]

    transforms = Transform(HighPassFilter(), RestNormalizer())


    # 1. Load data alongside subject IDs
    # TODO: data leakage - fitting transforms on test data (minor)
    emg, adl = load_raw_data()
    transforms.fit(emg)
    emg = preprocess_nm_data(emg)
    emg = replace(emg, labels=np.where(emg.labels == target_original_label, 1, 0))
    
    print(f"Mapping details:")
    print(f"  - Target gesture '{TARGET_GESTURE}' (original label {target_original_label}) -> 1")
    print(f"  - All other gestures and noGesture -> 0")
    print(f"  - Total positive target samples: {np.sum(emg.labels == 1)}")
    print(f"  - Total negative samples (other gestures + noGesture): {np.sum(emg.labels == 0)}")

    # 3. Prepare features and splits using LOSO with fitted transforms
    emg = transforms(emg)
    adl = transforms(adl)
    train, test, ids = prepare_loso_datasets(
        emg, adl,
        WINDOW_SIZE,
        INCREMENT_SIZE,
        test_subject_ratio=TEST_SUBJECT_RATIO,
    )

    # 4. Train binary classifier
    model_config = DiscreteClassifierConfig(n_classes=2)
    train_model(train, test, model_config, customers=ids)

if __name__ == "__main__":
    main()
