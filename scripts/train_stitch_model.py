import numpy as np

from mci_wake.data import load_raw_data, preprocess_nm_data
from mci_wake.data.generation import generate_training_data
from mci_wake.data.types import TrainData
from mci_wake.model.lite import MiniRocketModel
from mci_wake.model.neural import DiscreteClassifierConfig, DiscreteModel
from mci_wake.transform.highpass import HighPassFilter
from mci_wake.transform.rest_normalization import RestNormalizer
from mci_wake.transform.transform import Transform

TARGET_SEQUENCE = ["pinch", "fist"]
WINDOW_SIZE: int = 10
INCREMENT_SIZE: int = 5
TEST_SUBJECT_RATIO: float = 0.1

# TODO: customize gap between and retrain?
# Try open (open -> fist -> open)?
# Dario Ferino - main emg person
def main() -> None:
    print(f"Target sequence: {TARGET_SEQUENCE}")

    # 1. Load data
    emg, adl = load_raw_data()
    emg = preprocess_nm_data(emg)

    # 2. Split
    train_emg, test_emg = emg.split(test_percentage=TEST_SUBJECT_RATIO, by_subject=True)
    train_adl, test_adl = adl.split(test_percentage=TEST_SUBJECT_RATIO, by_subject=False)

    # 3. Fit transforms training data
    transforms = Transform(HighPassFilter(), RestNormalizer())
    transforms.fit(train_emg)

    train_emg = transforms(train_emg)
    test_emg = transforms(test_emg)
    train_adl = transforms(train_adl)
    test_adl = transforms(test_adl)

    # 4. Generate synthetic positive and hard negative sequence trials
    train_data = generate_training_data(train_emg, train_adl, TARGET_SEQUENCE, n_positive_per_subject=50, n_hard_negative_per_subject=200)
    test_data = generate_training_data(test_emg, test_adl, TARGET_SEQUENCE)

    print(
        f"Train set: {len(train_data)} samples ({np.sum(train_data.labels == 1)} positive, {np.sum(train_data.labels == 0)} negative)"
    )
    print(
        f"Test set:  {len(test_data)} samples ({np.sum(test_data.labels == 1)} positive, {np.sum(test_data.labels == 0)} negative)"
    )

    # 5. Train the model using PyTorch Lightning (DiscreteModel handles window slicing)
    # model_config = DiscreteClassifierConfig(
    #     n_classes=2,
    #     window_size=10,
    #     increment=5,
    #     type="GRU",
    #     temporal_hidden_size=128,
    #     temporal_layers=3,
    #     lr=1e-3,
    #     gestures=TARGET_SEQUENCE,
    # )
    #
    train_sub_data = TrainData(
        emg=[f"user{s.item() if hasattr(s, 'item') else s}" for s in np.unique(train_emg.subjects).tolist()],
        disco=[f"S{s.item() if hasattr(s, 'item') else s}" for s in np.unique(train_adl.subjects).tolist()],
    )

    # model = DiscreteModel(model_config)
    model = MiniRocketModel()
    model.fit(train_data.reduce_proportional(30000, (.75, .25)), test_data, customers=train_sub_data)
    model.save(r"D:\Coding\calibration-free-wakeword\scripts\models\out.json")


if __name__ == "__main__":
    main()
