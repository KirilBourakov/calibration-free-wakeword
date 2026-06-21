import numpy as np

from train_utils import gesture_mapping, load_raw_data, \
    preprocess_nm_data, train_model, prepare_datasets
from neural.classifier import DiscreteClassifierConfig

# Target gesture to recognize. Everything else will be classified as 'other' (0).
# Available gestures in dataset: 'fist', 'waveIn', 'waveOut', 'open', 'pinch'
TARGET_GESTURE = 'pinch'

def main() -> None:
    """Main execution pipeline for training the binary gesture model."""
    # Parameters:
    WINDOW_SIZE: int = 10 
    INCREMENT_SIZE: int = 5
    TRAIN_SPLIT: float = 0.95
    TEST_SPLIT: float = 0.05

    print(f"Target gesture: {TARGET_GESTURE}")
    if TARGET_GESTURE not in gesture_mapping:
        raise ValueError(f"Invalid TARGET_GESTURE '{TARGET_GESTURE}'. Options: {list(gesture_mapping.keys())}")
    
    target_original_label = gesture_mapping[TARGET_GESTURE]

    # 1. Load data
    emg_data_all, labels_all, adl_data = load_raw_data()
    binary_labels_all = np.where(labels_all == target_original_label, 1, 0)
    
    print(f"Mapping details:")
    print(f"  - Target gesture '{TARGET_GESTURE}' (original label {target_original_label}) -> 1")
    print(f"  - All other gestures and noGesture -> 0")
    print(f"  - Total positive target samples: {np.sum(binary_labels_all == 1)}")
    print(f"  - Total negative samples (other gestures + noGesture): {np.sum(binary_labels_all == 0)}")

    # 3. Preprocess 'No Motion' (noGesture) data
    emg_data_all = preprocess_nm_data(emg_data_all, labels_all)

    # 4. Prepare features and splits
    train_emg, train_labels, test_emg, test_labels = prepare_datasets(
        emg_data_all, binary_labels_all, adl_data, WINDOW_SIZE, INCREMENT_SIZE, TRAIN_SPLIT, TEST_SPLIT
    )

    # 5. Train
    model_config = DiscreteClassifierConfig(n_classes=2)
    train_model(train_emg, train_labels, test_emg, test_labels, model_config)

if __name__ == "__main__":
    main()
