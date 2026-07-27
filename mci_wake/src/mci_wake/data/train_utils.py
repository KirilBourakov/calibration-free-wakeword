from pathlib import Path

import numpy as np
from lightning import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

import libemg
from os import walk
import random 
import pickle
import json 
import os 
from typing import Any, List, Dict, Tuple, Optional, Union, cast
import numpy.typing as npt
from statistics import mode

from mci_wake.neural.classifier import make_data_loader, DiscreteClassifierConfig, DiscreteClassifier
from mci_wake.neural.lightning_module import DiscreteLightningModule

gesture_mapping: Dict[str, int] = {'noGesture': 0, 'fist': 1, 'waveIn': 2, 'waveOut': 3, 'open': 4, 'pinch': 5}

# dir setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPN_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "other", "EMG-EPN612"))
ADL_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "other", "DiscoDataset"))


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

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        filename="best-model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
        verbose=True,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloaders=tr_dl, val_dataloaders=te_dl)

    return cast(DiscreteLightningModule, trainer.lightning_module).internals


def load_raw_data() -> Tuple[npt.NDArray[np.object_], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.object_]]:
    """Loads ADL and gesture EMG data from the dataset alongside subject IDs.

    Returns:
        Tuple containing:
            - emg_data_all: Combined training and testing EMG data.
            - labels_all: Combined training and testing labels.
            - subject_ids_all: Combined training and testing subject IDs.
            - adl_data: Loaded ADL EMG data.
    """
    adl_data: npt.NDArray[np.object_] = load_disco_adls(ADL_DATA)
    emg, imu, labels, myo_labels, subject_ids = load_epn_data(EPN_DATA)

    training_emg = np.array(emg['training'], dtype='object')
    testing_emg = np.array(emg['testing'], dtype='object')
    training_labels = np.array(labels['training'])
    testing_labels = np.array(labels['testing'])

    training_subjects = np.array(subject_ids['training'])
    testing_subjects = np.array(subject_ids['testing'])

    emg_data_all: npt.NDArray[np.object_] = np.hstack([training_emg, testing_emg])
    labels_all: npt.NDArray[Any] = np.hstack([training_labels, testing_labels])
    subject_ids_all: npt.NDArray[Any] = np.hstack([training_subjects, testing_subjects])

    n_subs = len(np.unique(subject_ids_all))
    print(f"Loaded {len(emg_data_all)} gesture samples from EPN dataset across {n_subs} subjects.")
    print(f"Loaded {len(adl_data)} ADL noise segments.")

    return emg_data_all, labels_all, subject_ids_all, adl_data


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
    """Extracts features and splits the data into training and testing sets proportionally.

    Args:
        emg_data_all: The complete set of gesture EMG data samples.
        labels_all: The corresponding labels for the gesture samples.
        adl_data: The ADL EMG data samples.
        window_size: The size of the sliding window for feature extraction.
        increment_size: The increment step for the sliding window.
        train_split: The proportion of data to use for training. Defaults to 0.95.
        test_split: The proportion of data to use for testing. Defaults to 0.05.

    Returns:
        Tuple containing (train_emg, train_labels, test_emg, test_labels).
    """
    print("Warning: prepare_loso_datasets recommended.")

    # Extract features
    emg_feats: npt.NDArray[Any] = get_features(emg_data_all, window_size, increment_size, None, None)
    adl_feats: npt.NDArray[Any] = get_features(adl_data, window_size, increment_size, None, None)

    # Split Dataset
    train_labels = labels_all[0:int(len(labels_all) * train_split)]
    test_labels = labels_all[-int(test_split * len(labels_all)):]
    train_emg = emg_feats[0:int(len(emg_feats) * train_split)]
    test_emg = emg_feats[-int(test_split * len(emg_feats)):]

    # Add ADL data
    adl_train = adl_feats[0:int(len(adl_feats) * train_split)]
    adl_test = adl_feats[-int(len(adl_feats) * test_split):]

    train_labels_final: npt.NDArray[Any] = np.hstack([train_labels, np.zeros(len(adl_train))])
    train_emg_final: npt.NDArray[Any] = np.hstack([train_emg, adl_train])
    test_labels_final: npt.NDArray[Any] = np.hstack([test_labels, np.zeros(len(adl_test))])
    test_emg_final: npt.NDArray[Any] = np.hstack([test_emg, adl_test])

    print(f"Final training set: {len(train_emg_final)} samples ({len(train_labels)} gestures + {len(adl_train)} ADL)")
    print(f"Final testing set: {len(test_emg_final)} samples ({len(test_labels)} gestures + {len(adl_test)} ADL)")

    return train_emg_final, train_labels_final, test_emg_final, test_labels_final


def prepare_loso_datasets(
    emg_data_all: npt.NDArray[np.object_],
    labels_all: npt.NDArray[Any],
    subject_ids_all: npt.NDArray[Any],
    adl_data: npt.NDArray[np.object_],
    window_size: int,
    increment_size: int,
    test_subject_ids: Optional[List[int]] = None,
    test_subject_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Extracts features and splits data using Leave-One-Subject-Out (LOSO) cross-validation logic.

    Ensures that test subjects' gesture data is strictly isolated from the training set,
    enabling true evaluation of calibration-free generalization to unseen subjects.

    Args:
        emg_data_all: Complete set of gesture EMG data samples.
        labels_all: Mapped labels for all samples.
        subject_ids_all: Subject/User IDs corresponding to each EMG sample.
        adl_data: ADL noise segments.
        window_size: Window size for sliding feature extraction.
        increment_size: Increment step size.
        test_subject_ids: Specific subject IDs to hold out for testing. If None, randomly picks test_subject_ratio of subjects.
        test_subject_ratio: Proportion of unique subjects to allocate to test set if test_subject_ids is None.
        random_seed: Seed for random subject selection.

    Returns:
        Tuple containing (train_emg, train_labels, test_emg, test_labels).
    """
    unique_subjects = np.unique(subject_ids_all)
    if test_subject_ids is None:
        rng = np.random.default_rng(random_seed)
        n_test = max(1, int(len(unique_subjects) * test_subject_ratio))
        test_subject_ids = list(rng.choice(unique_subjects, size=n_test, replace=False))

    test_mask = np.isin(subject_ids_all, test_subject_ids)
    train_mask = ~test_mask

    train_emg_raw = emg_data_all[train_mask]
    train_labels_raw = labels_all[train_mask]
    test_emg_raw = emg_data_all[test_mask]
    test_labels_raw = labels_all[test_mask]

    # Extract features
    train_emg_feats: npt.NDArray[Any] = get_features(train_emg_raw, window_size, increment_size, None, None)
    test_emg_feats: npt.NDArray[Any] = get_features(test_emg_raw, window_size, increment_size, None, None)
    adl_feats: npt.NDArray[Any] = get_features(adl_data, window_size, increment_size, None, None)

    # Split ADL noise data proportionally
    n_adl_train = int(len(adl_feats) * (1.0 - test_subject_ratio))
    adl_train = adl_feats[:n_adl_train]
    adl_test = adl_feats[n_adl_train:]

    train_labels_final: npt.NDArray[Any] = np.hstack([train_labels_raw, np.zeros(len(adl_train))])
    train_emg_final: npt.NDArray[Any] = np.hstack([train_emg_feats, adl_train])
    test_labels_final: npt.NDArray[Any] = np.hstack([test_labels_raw, np.zeros(len(adl_test))])
    test_emg_final: npt.NDArray[Any] = np.hstack([test_emg_feats, adl_test])

    print(f"--- LOSO (Leave-One-Subject-Out) Dataset Split ---")
    print(f"Held-out test subject IDs ({len(test_subject_ids)} subjects): {test_subject_ids}")
    print(f"Final training set: {len(train_emg_final)} samples ({len(train_labels_raw)} gestures + {len(adl_train)} ADL)")
    print(f"Final testing set:  {len(test_emg_final)} samples ({len(test_labels_raw)} gestures + {len(adl_test)} ADL)")

    return train_emg_final, train_labels_final, test_emg_final, test_labels_final


def extract_data(data: Dict[str, Any]) -> Tuple[Optional[npt.NDArray[Any]], Optional[npt.NDArray[Any]], Optional[int], Optional[int]]:
    """Extracts EMG, IMU, and label data from a single data sample.

    Args:
        data: A dictionary containing the raw sample data.

    Returns:
        Tuple containing:
            - emg: Transposed EMG data for the sample.
            - quat: Transposed quaternion data.
            - label: The mapped gesture label.
            - mode_myo: The most frequent myoDetection label.
            Returns (None, None, None, None) if 'gestureName' is missing.
    """
    emg = np.transpose([data['emg']['ch' + str(ch)] for ch in range(1, 9)])
    quat = np.transpose([data['quaternion'][v] for v in ['w','x','y','z']])
    label = None
    myo_labels = np.array(data['myoDetection'])
    myo_labels = myo_labels[np.where(myo_labels != 0)]
    if len(myo_labels) == 0:
        myo_labels = np.array([0])

    if 'gestureName' in data:
        label = gesture_mapping[data['gestureName']]
    else: 
        return None, None, None, None # Half of the test data isn't available 
    
    if 'groundTruth' in data:
        ground_truth = np.diff(np.array(data['groundTruth']))
        try:
            start_idx = np.where(ground_truth == 1)[0][0]
        except (IndexError, ValueError):
            start_idx = 0
        try:
            end_idx = np.where(ground_truth == -1)[0][0]
        except (IndexError, ValueError):
            end_idx = len(emg) - 1
    else: 
        start_idx = 0
        end_idx = len(emg)
    return emg[start_idx:end_idx], quat[int(start_idx * 0.25):int(end_idx*0.25)], label, int(mode(myo_labels))
    
def load_epn_data(
    path: Path | str,
    gesture_sets: List[str] = ['trainingSamples', 'testingSamples'],
    subjects: List[int] = list(range(1, 307)), 
    subject_types: List[str] = ['training', 'testing']
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[int]]]:
    """Loads and organizes the EMG dataset from JSON files or a cached pickle.

    Args:
        path: path to the root of the dataset
        gesture_sets: List of keys in the JSON to load samples from. Defaults to ['trainingSamples', 'testingSamples'].
        subjects: List of subject IDs to load. Defaults to range(1, 307).
        subject_types: List of directories ('training', 'testing') to search in. Defaults to ['training', 'testing'].

    Returns:
        Tuple of dictionaries for (emg_data, imu_data, labels, myo_labels, subject_ids), 
        each keyed by the subject_types.
    """
    path = Path(path)

    # Check if pkl file exists (to save time)
    if os.path.exists('dataset.pkl') and len(subjects) > 300:
        with open('dataset.pkl', 'rb') as f:
            return tuple(pickle.load(f)) # type: ignore

    emg_data: Dict[str, List[Any]] = {}
    imu_data: Dict[str, List[Any]] = {}
    labels: Dict[str, List[Any]] = {}
    myo_labels: Dict[str, List[Any]] = {}
    subject_ids: Dict[str, List[int]] = {}
    
    # Initialize dictionary:
    for t in subject_types:
        emg_data[t] = []  
        imu_data[t] = []
        labels[t] = []
        myo_labels[t] = []
        subject_ids[t] = []
        
    for t in subject_types:
        print("Getting " + t + " subjects...")
        for sub in subjects:
            if sub % 10 == 0:
                print("Subject " + str(sub) + "...")
            user_file = path / (t + 'JSON') / ('user' + str(sub)) / ('user' + str(sub) + '.json')
            if not user_file.exists():
                continue
            with open(user_file, encoding="utf8") as f:
                jd = json.load(f)
                for s in gesture_sets:
                    for sample in jd[s]: 
                        e,i,l,ml = extract_data(jd[s][sample])
                        if e is not None:
                            emg_data[t].append(e)
                            imu_data[t].append(i)
                            labels[t].append(l)
                            myo_labels[t].append(ml)
                            subject_ids[t].append(sub)

    # Save dataset as pkl (to save time)
    if len(subjects) > 300:
        with open('dataset.pkl', 'wb') as f:
            pickle.dump([emg_data, imu_data, labels, myo_labels, subject_ids], f, protocol=pickle.HIGHEST_PROTOCOL)

    return emg_data, imu_data, labels, myo_labels, subject_ids

def load_disco_adls(path: Path | str) -> npt.NDArray[np.object_]:
    """Loads ADL (Activities of Daily Living) dataset and windows it into segments.

    Args:
        path: path to ADL dataset
    Returns:
        npt.NDArray[np.object_]: An object array containing windowed ADL EMG signals.
    """
    path = Path(path)
    adl_files = []
    for s in range(1, 16):
        sub_path = path / f"S{s}" / "ADL"
        if not sub_path.exists():
            continue
        files = next(walk(sub_path), (None, None, []))[2]
        for f in files:
            adl_files.append(str(sub_path / f))

    adl_data_list = []
    for af in adl_files:
        if '.csv' in af:
            adl_data_list.append(np.loadtxt(af, delimiter=','))
    
    if not adl_data_list:
        return np.array([], dtype='object')
        
    adl_data = np.vstack(adl_data_list)
    adl_data = adl_data[:, -8:]

    adl_data_w = [adl_data[i : i + random.randint(150, 400)] for i in range(0, len(adl_data) - 400, 50)]
    return np.array(adl_data_w, dtype='object')

def get_features(
    data: Union[npt.NDArray[Any], List[Any]], 
    window_size: int, 
    window_inc: int, 
    feats: Optional[List[str]], 
    feat_dic: Optional[Dict[str, Any]]
) -> npt.NDArray[Any]:
    """Extracts features from the raw EMG data using a sliding window.

    Args:
        data: Raw EMG data samples.
        window_size: Size of the sliding window.
        window_inc: Increment step for the sliding window.
        feats: List of feature names to extract. If None, returns raw windows.
        feat_dic: Optional dictionary for feature extraction parameters.

    Returns:
        npt.NDArray[Any]: Extracted features for each data sample.
    """
    from libemg.feature_extractor import FeatureExtractor
    fe = FeatureExtractor()
    windowed_data = np.array([libemg.utils.get_windows(d, window_size, window_inc) for d in data], dtype='object')
    
    if feats is None:
        return windowed_data 
        
    if feat_dic is not None:
        extracted_feats = np.array([fe.extract_features(feats, d, array=True, feature_dic=feat_dic) for d in windowed_data], dtype='object')
    else:
        extracted_feats = np.array([fe.extract_features(feats, np.array(d, dtype='float'), array=True) for d in windowed_data], dtype='object')
        
    return np.nan_to_num(extracted_feats, copy=True, nan=0, posinf=0, neginf=0)
