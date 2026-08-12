from mci_wake.data import load_raw_data, preprocess_nm_data, prepare_loso_datasets
from mci_wake.neural.training import train_model


def main() -> None:
    """Main execution pipeline for training the model using Leave-One-Subject-Out (LOSO) cross-validation."""
    # Parameters:
    WINDOW_SIZE: int = 10 
    INCREMENT_SIZE: int = 5
    TEST_SUBJECT_RATIO: float = 0.1  # Hold out 10% of subjects for unseen test evaluation

    # 1. Load data alongside subject IDs
    emg_data_all, labels_all, subject_ids_all, adl_data, adl_ids = load_raw_data()


    # 2. Preprocess
    emg_data_all = preprocess_nm_data(emg_data_all, labels_all)

    # 3. Prepare features and splits using LOSO
    train_emg, train_labels, test_emg, test_labels, train_subject_ids = prepare_loso_datasets(
        emg_data_all,
        labels_all,
        subject_ids_all,
        adl_data,
        adl_ids,
        WINDOW_SIZE,
        INCREMENT_SIZE,
        test_subject_ratio=TEST_SUBJECT_RATIO,
    )

    # # 4. Train
    # train_model(train_emg, train_labels, test_emg, test_labels, customers=train_subject_ids)

if __name__ == "__main__":
    main()
