import time
from typing import Tuple, Any, cast
import numpy as np
import numpy.typing as npt
from lightning import Trainer

from neural.lightning_module import DiscreteLightningModule
from train_utils import *
from neural.classifier import DiscreteClassifier, make_data_loader, DiscreteClassifierConfig
from torch.utils.data import DataLoader

from train_utils import load_disco_adls, load_epn_data, get_features

EPN_DATA = r"E:\Programming\Projects\reaserch\new_wakeword\mci_wake\other\EMG-EPN612"
ADL_DATA = r"E:\Programming\Projects\reaserch\new_wakeword\mci_wake\other\DiscoDataset"

def load_raw_data() -> Tuple[npt.NDArray[np.object_], npt.NDArray[Any], npt.NDArray[np.object_]]:
    """Loads ADL and gesture EMG data from the dataset.

    Returns:
        Tuple[npt.NDArray[np.object_], npt.NDArray[Any], npt.NDArray[np.object_]]: A tuple containing:
            - emg_data_all: Combined training and testing EMG data.
            - labels_all: Combined training and testing labels.
            - adl_data: Loaded ADL EMG data.
    """
    adl_data: npt.NDArray[np.object_] = load_disco_adls(ADL_DATA)
    # Extract all data 
    emg, imu, labels, myo_labels = load_epn_data(EPN_DATA)
    training_emg = np.array(emg['training'], dtype='object')
    testing_emg = np.array(emg['testing'], dtype='object')
    training_labels = np.array(labels['training'])
    testing_labels = np.array(labels['testing'])

    emg_data_all: npt.NDArray[np.object_] = np.hstack([training_emg, testing_emg])
    labels_all: npt.NDArray[Any] = np.hstack([training_labels, testing_labels])
    return emg_data_all, labels_all, adl_data

def preprocess_nm_data(emg_data_all: npt.NDArray[np.object_], labels_all: npt.NDArray[Any]) -> npt.NDArray[np.object_]:
    """Randomly clips 'No Motion' segments to introduce variability.

    Args:
        emg_data_all: The complete set of EMG data samples.
        labels_all: The corresponding labels for the EMG samples.

    Returns:
        npt.NDArray[np.object_]: The EMG data with randomized 'No Motion' segment lengths.
    """
    nm_idxs = np.where(labels_all == 0)[0] 
    # Resize NM data 
    for i in nm_idxs:
        clip_length = np.random.randint(150, 351)
        emg_data_all[i] = emg_data_all[i][0:clip_length]
    return emg_data_all

def prepare_datasets(
    emg_data_all: npt.NDArray[np.object_], 
    labels_all: npt.NDArray[Any], 
    adl_data: npt.NDArray[np.object_], 
    window_size: int, 
    increment_size: int, 
    train_split: float = 0.95, 
    test_split: float = 0.05
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Extracts features and splits the data into training and testing sets, including ADL data.

    Args:
        emg_data_all: The complete set of gesture EMG data samples.
        labels_all: The corresponding labels for the gesture samples.
        adl_data: The ADL EMG data samples.
        window_size: The size of the sliding window for feature extraction.
        increment_size: The increment step for the sliding window.
        train_split: The proportion of data to use for training. Defaults to 0.95.
        test_split: The proportion of data to use for testing. Defaults to 0.05.

    Returns:
        Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]: A tuple containing:
            - train_emg: Features for the training set.
            - train_labels: Labels for the training set.
            - test_emg: Features for the testing set.
            - test_labels: Labels for the testing set.
    """
    # Extract features
    emg_feats: npt.NDArray[Any] = get_features(emg_data_all, window_size, increment_size, None, None)
    adl_feats: npt.NDArray[Any] = get_features(adl_data, window_size, increment_size, None, None)

    # Split Dataset 
    train_labels = labels_all[0:int(len(labels_all)*train_split)]
    test_labels = labels_all[-int(test_split*len(labels_all)):]
    train_emg = emg_feats[0:int(len(emg_feats)*train_split)]
    test_emg = emg_feats[-int(test_split*len(emg_feats)):]

    # Add ADL data
    adl_train = adl_feats[0:int(len(adl_feats) * train_split)]
    adl_test  = adl_feats[-int(len(adl_feats) * test_split):]
    
    train_labels_final: npt.NDArray[Any] = np.hstack([train_labels, np.zeros(len(adl_train))])
    train_emg_final: npt.NDArray[Any] = np.hstack([train_emg, adl_train])
    test_labels_final: npt.NDArray[Any] = np.hstack([test_labels, np.zeros(len(adl_test))])
    test_emg_final: npt.NDArray[Any] = np.hstack([test_emg, adl_test])
    
    return train_emg_final, train_labels_final, test_emg_final, test_labels_final

def train_model(
    train_emg: npt.NDArray[Any],
    train_labels: npt.NDArray[Any],
    test_emg: npt.NDArray[Any],
    test_labels: npt.NDArray[Any],
    model_config: DiscreteClassifierConfig = DiscreteClassifierConfig()
) -> DiscreteClassifier:
    """Initializes and trains the DiscreteClassifier using the provided datasets.

    Args:
        train_emg: Features for the training set.
        train_labels: Labels for the training set.
        test_emg: Features for the testing set.
        test_labels: Labels for the testing set.
        model_config: the model configuration

    Returns:
        DiscreteClassifier: The trained classifier instance.
    """
    print("Fitting Discrete Classifier...")
    tr_dl: DataLoader = make_data_loader(train_emg, train_labels)
    te_dl: DataLoader = make_data_loader(test_emg, test_labels)

    model = DiscreteLightningModule(model_config)
    trainer = Trainer(max_epochs=10, accelerator="auto", devices="auto")
    trainer.fit(model, train_dataloaders=tr_dl, val_dataloaders=te_dl)

    return cast(DiscreteLightningModule, trainer.lightning_module).internals

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
