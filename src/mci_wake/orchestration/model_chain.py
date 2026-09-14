import winsound
import numpy as np
import time
import statistics

from mci_wake.data_handler.abstract import AbstractDataHandler, OfflineCapableAbstractDataHandler
from mci_wake.model.abstract import AbstractModel
from mci_wake.transform.transform import Transform


class ModelState:
    def __init__(
        self,
        model: AbstractModel,
        buffer_size: int = 5,
        transforms: Transform | None = None,
    ):
        n_classes = getattr(model, "n_classes", getattr(getattr(model, "config", None), "n_classes", None))
        assert n_classes == 2, f"Model must have 2 classes, got {n_classes}"
        self.model = model
        self.buffer_size = buffer_size
        self.transforms = transforms
        self.buffer: list[int] = []

    def next_step(self, odh: AbstractDataHandler, size: int) -> bool:
        dh_out = odh.get_data(size)
        emg = dh_out.emg[::-1]
        data = self.transforms(emg) if (len(emg) > 0 and self.transforms is not None) else emg

        # predict
        pred = self.model.predict(data)
        self.buffer.append(pred)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        mode_pred = statistics.mode(self.buffer)
        return mode_pred != 0

    def reset(self):
        self.buffer = []
        self.model.reset()


class ModelChain:
    """
    Orchestrator that chains multiple models together for sequential gesture/wake detection.

    Parameters
    ----------
    odh: AbstractDataHandler
        The online data handler object for streaming EMG data.
    models: list[AbstractModel]
        The trained models for sequence detection.
    buffer: int, optional
        The size of the prediction buffer to use for mode filtering. Default is 5.
    template_size: int, optional
        The size of each EMG template (in samples) passed to the models. Default is 250 (1.25s for the Myo Armband).
    min_template_size: int, optional
        The minimum number of samples required before starting to make predictions. Default is 150.
    sequence_timeout: float, optional
        Seconds before chained sequence resets. Default is 2.0.
    step_samples: int, optional
        Number of samples to advance the simulation per loop tick in offline mode. Default is 5.
    transforms: Transform | None, optional
        Optional transforms applied to the EMG data before prediction.
    verbose: bool, optional
        If True, prints state transitions. Default is True.
    """

    def __init__(
        self,
        odh: AbstractDataHandler,
        models: list[AbstractModel],
        buffer: int = 5,
        template_size: int = 250,
        min_template_size: int = 150,
        sequence_timeout: float = 2.0,
        step_samples: int = 5,
        transforms: Transform | None = None,
        verbose: bool = True,
    ):
        self.odh = odh
        self.models = [ModelState(m, buffer_size=buffer, transforms=transforms) for m in models]
        self.buffer_size = buffer
        self.template_size = template_size
        self.min_template_size = min_template_size
        self.sequence_timeout = sequence_timeout
        self.step_samples = step_samples
        self.verbose = verbose
        self.running = False

    def stop(self) -> None:
        """Stops the detection loop if running in a background thread."""
        self.running = False

    def run(self, duration_sec: float | None = None):
        """
        Main loop for gesture detection.
        Runs a sliding window over incoming EMG data and makes predictions based on the trained model.
        """
        self.running = True
        expected_count = self.min_template_size
        curr_model = 0
        last_step_time = None
        start_ts = self.odh.get_time()

        while self.running:
            curr_ts = self.odh.get_time()
            if duration_sec is not None and (curr_ts - start_ts) >= duration_sec:
                break

            if self.odh.is_offline:
                assert isinstance(self.odh, OfflineCapableAbstractDataHandler)
                self.odh.advance(self.step_samples)

            # Get and process EMG data
            dh_out = self.odh.get_data(1)
            # offline emg has run out of data
            if dh_out.count >= expected_count:
                # Fetch and reverse
                move = self.models[curr_model].next_step(self.odh, self.template_size)

                if move:
                    self.odh.reset()
                    self.models[curr_model].reset()
                    expected_count = self.min_template_size
                    curr_model += 1
                    last_step_time = self.odh.get_time()

                    if curr_model == len(self.models):
                        if self.verbose:
                            print(f"{str(last_step_time)} wake detected")
                        if not self.odh.is_offline:
                            winsound.Beep(1000, 250)
                        self.odh.on_wake_detected()
                        curr_model = 0
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

            if self.odh.is_offline:
                assert isinstance(self.odh, OfflineCapableAbstractDataHandler)
                if self.odh.is_done:
                    self.running = False

            if not self.odh.is_offline:
                time.sleep(0.005)