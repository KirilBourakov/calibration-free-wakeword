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

    emg, adl  = load_raw_data()
    emg = preprocess_nm_data(emg)

    train, test = emg.combine(adl).split(TEST_SUBJECT_RATIO)

    train_data = generate_training_data(train, TARGET_SEQUENCE)
    test_data = generate_training_data(test, TARGET_SEQUENCE)

    print(
        f"Train set: {len(train_data)} samples ({np.sum(train_data.labels == 1)} positive, {np.sum(train_data.labels == 0)} negative)")
    print(
        f"Test set:  {len(test_data)} samples ({np.sum(test_data.labels == 1)} positive, {np.sum(test_data.labels == 0)} negative)")

    # 5. Extract Sliding Subwindow Features
    train_emg = get_features(train_data, WINDOW_SIZE, INCREMENT_SIZE, force_normalize=False)
    test_emg = get_features(test_data, WINDOW_SIZE, INCREMENT_SIZE, force_normalize=False)

    # 6. Train the model using PyTorch Lightning
    model_config = DiscreteClassifierConfig(
        n_classes=2,
        type="GRU",
        temporal_hidden_size=128,
        temporal_layers=3,
        lr=1e-3,
    )

    train_sub_data = TrainData(
        emg=[f"user{s}" for s in np.unique(train.subjects).tolist()],
        disco=[],
    )

    train_model(train_emg, test_emg, model_config, customers=train_sub_data)

if __name__ == "__main__":
    main()