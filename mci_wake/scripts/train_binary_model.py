import numpy as np

from mci_wake.data.train_utils import (
    gesture_mapping,
    load_raw_data,
    preprocess_nm_data,
    train_model,
    prepare_loso_datasets,
)
from mci_wake.neural.classifier import DiscreteClassifierConfig

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

    # 1. Load data alongside subject IDs
    emg_data_all, labels_all, subject_ids_all, adl_data = load_raw_data()
    binary_labels_all = np.where(labels_all == target_original_label, 1, 0)
    
    print(f"Mapping details:")
    print(f"  - Target gesture '{TARGET_GESTURE}' (original label {target_original_label}) -> 1")
    print(f"  - All other gestures and noGesture -> 0")
    print(f"  - Total positive target samples: {np.sum(binary_labels_all == 1)}")
    print(f"  - Total negative samples (other gestures + noGesture): {np.sum(binary_labels_all == 0)}")

    # 2. Preprocess 'No Motion' (noGesture) data
    emg_data_all = preprocess_nm_data(emg_data_all, labels_all)

    # 3. Prepare features and splits using LOSO
    train_emg, train_labels, test_emg, test_labels = prepare_loso_datasets(
        emg_data_all,
        binary_labels_all,
        subject_ids_all,
        adl_data,
        WINDOW_SIZE,
        INCREMENT_SIZE,
        test_subject_ratio=TEST_SUBJECT_RATIO,
    )

    # 4. Train binary classifier
    model_config = DiscreteClassifierConfig(n_classes=2)
    train_model(train_emg, train_labels, test_emg, test_labels, model_config)

if __name__ == "__main__":
    main()
