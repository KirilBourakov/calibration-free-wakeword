from train_utils import preprocess_nm_data, load_raw_data, prepare_datasets

def main() -> None:
    """Main execution pipeline for training the model."""
    # Parameters:
    WINDOW_SIZE: int = 10 
    INCREMENT_SIZE: int = 5
    TRAIN_SPLIT: float = 0.95
    TEST_SPLIT: float = 0.05

    # 1. Load data
    emg_data_all, labels_all, adl_data = load_raw_data()

    # 2. Preprocess
    emg_data_all = preprocess_nm_data(emg_data_all, labels_all)

    # 3. Prepare features and splits
    train_emg, train_labels, test_emg, test_labels = prepare_datasets(
        emg_data_all, labels_all, adl_data, WINDOW_SIZE, INCREMENT_SIZE, TRAIN_SPLIT, TEST_SPLIT
    )

    # 4. Train
    train_model(train_emg, train_labels, test_emg, test_labels)

if __name__ == "__main__":
    main()
