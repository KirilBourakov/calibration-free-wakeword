from mci_wake.data import load_raw_data, preprocess_nm_data, prepare_loso_datasets
from mci_wake.neural.training import train_model
from mci_wake.neural.classifier import DiscreteClassifierConfig
from mci_wake.transform.highpass import HighPassFilter
from mci_wake.transform.rest_normalization import RestNormalizer
from mci_wake.transform.transform import Transform


def main() -> None:
    """Main execution pipeline for training the multi-class model using Leave-One-Subject-Out (LOSO) cross-validation."""
    # Parameters:
    WINDOW_SIZE: int = 10 
    INCREMENT_SIZE: int = 5
    TEST_SUBJECT_RATIO: float = 0.1  # Hold out 10% of subjects for unseen test evaluation

    # 1. Load data alongside subject IDs
    # TODO: data leakage - fitting transforms on test data (minor)
    emg, adl  = load_raw_data()
    transforms = Transform(HighPassFilter(), RestNormalizer())
    transforms.fit(emg)
    emg = transforms(emg)
    adl = transforms(adl)

    # 2. Preprocess
    emg = preprocess_nm_data(emg)

    # 3. Prepare features and splits using LOSO
    train, test, train_subject_ids = prepare_loso_datasets(
        emg, adl,
        WINDOW_SIZE,
        INCREMENT_SIZE,
        test_subject_ratio=TEST_SUBJECT_RATIO,
    )

    # 4. Train classifier
    model_config = DiscreteClassifierConfig(n_classes=6)
    train_model(train, test, model_config, customers=train_subject_ids)


if __name__ == "__main__":
    main()
