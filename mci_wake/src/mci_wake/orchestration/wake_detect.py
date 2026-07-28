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
from mci_wake.neural.classifier import DiscreteClassifier
from mci_wake.stitching.handler import StitchingDataHandler
from mci_wake.utils.normalize import safe_znormalize_global

class ModelState:
    def __init__(
        self,
        model: DiscreteClassifier,
        window_size: int,
        increment: int,
        buffer_size: int,
        normalize: bool = True,
    ):
        assert model.config.n_classes == 2
        self.model = model
        self.window_size = window_size
        self.increment = increment
        self.buffer_size = buffer_size
        self.normalize = normalize
        self.buffer = []

    def next_step(self, odh: OnlineDataHandler | StitchingDataHandler, size: int):
        data, counts = odh.get_data(size)
        emg = data['emg'][::-1]
        feats = self._get_features([emg], None, None)[0]

        # predict
        pred, prob_, output = self.model.predict(feats)
        self.buffer.append(pred)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        mode_pred = statistics.mode(self.buffer)
        return mode_pred != 0

    def reset(self):
        self.buffer = []

    def _get_features(self, data, feats, feat_dic):
        fe = FeatureExtractor()
        data = np.array([
            get_windows(
                safe_znormalize_global(d) if (len(d) > 0 and self.normalize) else d,
                self.window_size,
                self.increment
            )
            for d in data
        ], dtype='object')
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
    models: list[DiscreteClassifier]
        The trained PyTorch models for sequence detection.
    buffer: int, optional
        The size of the prediction buffer to use for mode filtering. Default is 5.
    template_size: int, optional
        The size of each EMG template (in samples). Default is 250 (1.5s for the Myo Armband).
    min_template_size: int, optional
        The minimum number of samples required before starting to make predictions. Default is 150.
    key_mapping: dict, optional
        A dictionary mapping gesture names to keyboard keys.
    debug: bool, optional
        If True, enables debug mode with additional print statements. Default is True.
    """

    def __init__(
        self,
        odh: OnlineDataHandler | StitchingDataHandler,
        window_size: int,
        increment: int,
        models: list[DiscreteClassifier],
        buffer=5,
        template_size=250,
        min_template_size=150,
        sequence_timeout = 2.0,
        debug=True,
        normalize=True,
        verbose=True,
        realtime: bool = True,
    ):
        self.odh = odh
        self.window_size = window_size
        self.increment = increment
        self.buffer_size = buffer
        self.verbose = verbose
        self.models = [ModelState(m, window_size, increment, buffer, normalize=normalize) for m in models]
        self.template_size = template_size
        self.min_template_size = min_template_size
        self.sequence_timeout = sequence_timeout
        self.debug = debug
        self.realtime = realtime
        self.running = False

    def stop(self) -> None:
        """Stops the detection loop if running in a background thread."""
        self.running = False

    def _get_timestamp(self) -> float:
        if hasattr(self.odh, "get_virtual_timestamp"):
            return self.odh.get_virtual_timestamp()
        return time.time()

    def run(self, duration_sec: float | None = None):
        """
        Main loop for gesture detection.
        Runs a sliding window over incoming EMG data and makes predictions based on the trained model.
        """
        self.running = True
        expected_count = self.min_template_size
        curr_model = 0
        last_step_time = None
        start_ts = self._get_timestamp()

        while self.running:
            curr_ts = self._get_timestamp()
            if duration_sec is not None and (curr_ts - start_ts) >= duration_sec:
                break

            if not self.realtime and hasattr(self.odh, "advance"):
                self.odh.advance(self.increment)

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
                    last_step_time = self._get_timestamp()

                    if curr_model == len(self.models):
                        if self.verbose:
                            print(f"{str(last_step_time)} wake detected")
                        if self.realtime:
                            winsound.Beep(1000, 250)
                        curr_model = 0
                        if hasattr(self.odh, "on_wake_detected"):
                            self.odh.on_wake_detected()
                    elif self.verbose:
                        print(f"{str(last_step_time)} State transition from {curr_model} to {curr_model + 1}")
                else:
                    expected_count = min(expected_count + 10, self.template_size)
                    if last_step_time and (curr_ts - last_step_time) > self.sequence_timeout:
                        self.odh.reset()
                        for model in self.models:
                            model.reset()
                        expected_count = self.min_template_size
                        last_step_time = None
                        curr_model = 0
                        if self.verbose:
                            print(f"{str(curr_ts)} reset")

            if self.realtime:
                time.sleep(0.005)