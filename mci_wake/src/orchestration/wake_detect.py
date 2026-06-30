import winsound
import numpy as np
import torch.nn.functional as F
import torch
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows
import pyautogui
import time
import statistics

from libemg.data_handler import OnlineDataHandler
from neural.classifier import DiscreteClassifier

class ModelState:
    def __init__(
        self,
        model: DiscreteClassifier,
        window_size: int,
        increment: int,
        buffer_size: int,
    ):
        assert model.config.n_classes == 2
        self.model = model
        self.window_size = window_size
        self.increment = increment
        self.buffer_size = buffer_size
        self.buffer = []

    def next_step(self, odh: OnlineDataHandler, size: int):
        data, counts = odh.get_data(size)
        emg = data['emg'][::-1]
        feats = self._get_features([emg], None, None)[0]

        # predict
        pred, prob_, output = self.model.predict(feats)
        self.buffer.append(pred)
        mode_pred = statistics.mode(self.buffer[-self.buffer_size:])
        return mode_pred != 0

    def reset(self):
        self.buffer = []

    def _get_features(self, data, feats, feat_dic):
        fe = FeatureExtractor()
        data = np.array([get_windows(d, self.window_size, self.increment) for d in data], dtype='object')
        if feats is None:
            return data
        if feat_dic is not None:
            feats = np.array([fe.extract_features(feats, d, array=True, feature_dic=feat_dic) for d in data],
                             dtype='object')
        else:
            feats = np.array([fe.extract_features(feats, np.array(d, dtype='float'), array=True) for d in data],
                             dtype='object')
        feats = np.nan_to_num(feats, copy=True, nan=0, posinf=0, neginf=0)
        return feats


class WakeDetect:
    """
    Based on the Discrete Control class

    Parameters
    ----------
    odh: OnlineDataHandler
        The online data handler object for streaming EMG data.
    window_size: int
        The window size (in samples) to use for splitting up each template.
    increment: int
        The increment size (in samples) for the sliding window.
    model: torch.nn.Module
        The trained PyTorch model for gesture classification.
    buffer: int, optional
        The size of the prediction buffer to use for mode filtering. Default is 1.
    template_size: int, optional
        The size of each EMG template (in samples). Default is 250 (1.5s for the Myo Armband).
    min_template_size: int, optional
        The minimum number of samples required before starting to make predictions (helps reduce the delay needed between subsequent gestures). Default is 100.
    key_mapping: dict, optional
        A dictionary mapping gesture names to keyboard keys. Default maps 'Close' to 'c', 'Flexion' to 'f', 'Extension' to 'e', 'Open' to 'o', and 'Pinch' to 'p'.
    debug: bool, optional
        If True, enables debug mode with additional print statements. Default is True.
    """

    def __init__(
        self,
        odh: OnlineDataHandler,
        window_size: int,
        increment: int,
        models: list[DiscreteClassifier],
        buffer=5,
        template_size=250,
        min_template_size=150,
        sequence_timeout = 2.0,
        key_mapping: dict[str, str] | None = None,
        debug=True
    ):
        if key_mapping is None:
            key_mapping = {'Close': 'c', 'Flexion': 'f', 'Extension': 'e', 'Open': 'o', 'Pinch': 'p'}

        self.odh = odh
        self.window_size = window_size
        self.increment = increment
        self.buffer_size = buffer
        self.models = [ModelState(m, window_size, increment, buffer) for m in models]
        self.template_size = template_size
        self.min_template_size = min_template_size
        self.sequence_timeout = sequence_timeout
        self.key_mapping = key_mapping
        self.debug = debug

    def run(self):
        """
        Main loop for gesture detection.
        Runs a sliding window over incoming EMG data and makes predictions based on the trained model.
        """
        expected_count = self.min_template_size
        curr_model = 0
        last_step_time = None
        while True:
            # Get and process EMG data
            _, counts = self.odh.get_data(self.window_size)
            if counts['emg'][0][0] >= expected_count:
                # Fetch and reverse
                move = self.models[curr_model].next_step(self.odh, self.template_size)

                if move:
                    self.odh.reset()
                    self.models[curr_model].reset()
                    expected_count = self.min_template_size
                    curr_model += 1
                    last_step_time = time.time()

                    if curr_model == len(self.models):
                        print(f"{str(time.time())} wake detected")
                        winsound.Beep(1000, 250)
                        curr_model = 0
                    else:
                        print(f"{str(time.time())} State transition from {curr_model} to {curr_model + 1}")
                else:
                    expected_count += 10
                    if last_step_time and time.time() - last_step_time > self.sequence_timeout:
                        self.odh.reset()
                        for model in self.models:
                            model.reset()
                        expected_count = self.min_template_size
                        last_step_time = None
                        curr_model = 0
                        print(f"{str(time.time())} reset")

