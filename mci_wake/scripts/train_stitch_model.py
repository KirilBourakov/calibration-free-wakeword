import numpy as np

from mci_wake.data import load_raw_data, preprocess_nm_data
from mci_wake.data.generation import generate_training_data
from mci_wake.data.processing import get_features
from mci_wake.neural import DiscreteClassifierConfig, train_model
from mci_wake.neural.classifier import TrainData

TARGET_SEQUENCE = ["pinch", "fist"]
WINDOW_SIZE: int = 10
INCREMENT_SIZE: int = 5
TEST_SUBJECT_RATIO: float = 0.1

def main():
    print(f"Target sequence: {TARGET_SEQUENCE}")

    data = load_raw_data()
    data.epn_emg = preprocess_nm_data(data.epn_emg, data.epn_labels)

    train, test = data.split(TEST_SUBJECT_RATIO)

    training_data, training_labels = generate_training_data(train, TARGET_SEQUENCE)
    test_data, test_labels = generate_training_data(test, TARGET_SEQUENCE)

    print(
        f"Train set: {len(training_data)} samples ({np.sum(training_labels == 1)} positive, {np.sum(training_labels == 0)} negative)")
    print(
        f"Test set:  {len(test_data)} samples ({np.sum(test_labels == 1)} positive, {np.sum(test_labels == 0)} negative)")

    # 5. Extract Sliding Subwindow Features
    train_emg = get_features(training_data, WINDOW_SIZE, INCREMENT_SIZE, None, None, force_normalize=False)
    test_emg = get_features(test_data, WINDOW_SIZE, INCREMENT_SIZE, None, None, force_normalize=False)

    # 6. Train the model using PyTorch Lightning
    model_config = DiscreteClassifierConfig(
        n_classes=2,
        type="GRU",
        temporal_hidden_size=128,
        temporal_layers=3,
        lr=1e-3,
    )

    train_sub_data = TrainData(
        emg=[f"user{s}" for s in np.unique(train.epn_subjects).tolist()],
        disco=[f"S{s}" for s in np.unique(train.adl_subjects).tolist()],
    )

    train_model(train_emg, training_labels, test_emg, test_labels, model_config, customers=train_sub_data)

if __name__ == "__main__":
    main()