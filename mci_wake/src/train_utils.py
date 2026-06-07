import numpy as np
import libemg
from os import walk
import random 
import pickle
import json 
import os 
from typing import Any, List, Dict, Tuple, Optional, Union
import numpy.typing as npt
from statistics import mode

gesture_mapping: Dict[str, int] = {'noGesture': 0, 'fist': 1, 'waveIn': 2, 'waveOut': 3, 'open': 4, 'pinch': 5}

def extract_data(data: Dict[str, Any]) -> Tuple[Optional[npt.NDArray[Any]], Optional[npt.NDArray[Any]], Optional[int], Optional[int]]:
    """Extracts EMG, IMU, and label data from a single data sample.

    Args:
        data: A dictionary containing the raw sample data.

    Returns:
        Tuple[Optional[npt.NDArray[Any]], Optional[npt.NDArray[Any]], Optional[int], Optional[int]]: A tuple containing:
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
    
def get_data(
    gesture_sets: List[str] = ['trainingSamples', 'testingSamples'],
    subjects: List[int] = list(range(1, 307)), 
    subject_types: List[str] = ['training', 'testing']
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Loads and organizes the EMG dataset from JSON files or a cached pickle.

    Args:
        gesture_sets: List of keys in the JSON to load samples from. Defaults to ['trainingSamples', 'testingSamples'].
        subjects: List of subject IDs to load. Defaults to range(1, 307).
        subject_types: List of directories ('training', 'testing') to search in. Defaults to ['training', 'testing'].

    Returns:
        Tuple[Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]]]: 
            A tuple of dictionaries for (emg_data, imu_data, labels, myo_labels), 
            each keyed by the subject_types.
    """
    # Check if pkl file exists (to save time)
    if os.path.exists('dataset.pkl') and len(subjects) > 300:
        with open('dataset.pkl', 'rb') as f:
            data = pickle.load(f)
            return data[0], data[1], data[2], data[3]

    emg_data: Dict[str, List[Any]] = {}
    imu_data: Dict[str, List[Any]] = {}
    labels: Dict[str, List[Any]] = {}
    myo_labels: Dict[str, List[Any]] = {}
    
    # Initialize dictionary:
    for t in subject_types:
        emg_data[t] = []  
        imu_data[t] = []
        labels[t] = []
        myo_labels[t] = []
        
    for t in subject_types:
        print("Getting " + t + " subjects...")
        for sub in subjects:
            if sub % 10 == 0:
                print("Subject " + str(sub) + "...")
            f = open('EMG-EPN612/' + t + 'JSON/user' + str(sub) + '/user' + str(sub) + '.json', encoding="utf8")
            jd = json.load(f)
            for s in gesture_sets:
                for sample in jd[s]: 
                    e,i,l,ml = extract_data(jd[s][sample])
                    if e is not None:
                        emg_data[t].append(e)
                        imu_data[t].append(i)
                        labels[t].append(l)
                        myo_labels[t].append(ml)

    # Save dataset as pkl (to save time)
    if len(subjects) > 300:
        with open('dataset.pkl', 'wb') as f:
            pickle.dump([emg_data, imu_data, labels, myo_labels], f, protocol=pickle.HIGHEST_PROTOCOL)

    return emg_data, imu_data, labels, myo_labels

def load_disco_adls() -> npt.NDArray[np.object_]:
    """Loads ADL (Activities of Daily Living) dataset and windows it into segments.

    Returns:
        npt.NDArray[np.object_]: An object array containing windowed ADL EMG signals.
    """
    adl_files = []
    for s in range(1, 16):
        path = 'DiscoDataset/S' + str(s) + '/ADL/'
        files = next(walk(path), (None, None, []))[2]
        for f in files:
            adl_files.append(path + f)

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
