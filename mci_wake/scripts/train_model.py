from mci_wake.data.train_utils import preprocess_nm_data, load_raw_data, prepare_loso_datasets, train_model


def main() -> None:
    """Main execution pipeline for training the model using Leave-One-Subject-Out (LOSO) cross-validation."""
    # Parameters:
    WINDOW_SIZE: int = 10 
    INCREMENT_SIZE: int = 5
    TEST_SUBJECT_RATIO: float = 0.1  # Hold out 10% of subjects for unseen test evaluation

    # 1. Load data alongside subject IDs
    emg_data_all, labels_all, subject_ids_all, adl_data = load_raw_data()

    # 2. Preprocess
    emg_data_all = preprocess_nm_data(emg_data_all, labels_all)

    # 3. Prepare features and splits using LOSO
    train_emg, train_labels, test_emg, test_labels = prepare_loso_datasets(
        emg_data_all,
        labels_all,
        subject_ids_all,
        adl_data,
        WINDOW_SIZE,
        INCREMENT_SIZE,
        test_subject_ratio=TEST_SUBJECT_RATIO,
    )

    # 4. Train
    train_model(train_emg, train_labels, test_emg, test_labels)

if __name__ == "__main__":
    main()
