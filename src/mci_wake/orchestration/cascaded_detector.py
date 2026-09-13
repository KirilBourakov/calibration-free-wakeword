import time
import winsound
import statistics
import numpy as np
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

from mci_wake.data_handler.abstract import AbstractDataHandler, OfflineCapableAbstractDataHandler
from mci_wake.model.abstract import AbstractModel
from mci_wake.orchestration.model_chain import ModelState
from mci_wake.transform.transform import Transform

class ModelCascade:
    """
    Two-stage cascade orchestrator.

    Parameters
    ----------
    odh: AbstractDataHandler
        The online data handler object for streaming EMG data.
    window_size: int
        The window size (in samples) to use for splitting up each template.
    increment: int
        The increment size (in samples) for the sliding window.
    light_model: AbstractModel
        Fast model that scans the stream continuously (must have 2 classes).
    heavy_model: AbstractModel
        Accurate model invoked only to verify candidates (must have 2 classes).
    buffer: int, optional
        The size of the prediction buffer to use for mode filtering.
    template_size: int, optional
        The size of each EMG template (in samples).
    min_template_size: int, optional
        The minimum number of samples required before starting predictions. Default is 150.
    verification_steps: int, optional
        Number of heavy-model attempts to confirm a candidate before rejection.
    reject_cooldown: float, optional
        Seconds to ignore light-model triggers after a rejection.
    debug: bool, optional
        If True, enables debug mode. Default is True.
    transforms: Transform | None, optional
        Optional transforms applied before feature extraction.
    verbose: bool, optional
        If True, prints state transitions. Default is True.
    """

    def __init__(
        self,
        odh: AbstractDataHandler,
        window_size: int,
        increment: int,
        light_model: AbstractModel,
        heavy_model: AbstractModel,
        buffer=5,
        template_size=250,
        min_template_size=150,
        verification_steps=3,
        reject_cooldown=0.25,
        debug=True,
        transforms: Transform | None = None,
        verbose=True,
    ):
        assert verification_steps >= 1, "Verification steps must be >= 1."

        self.odh = odh
        self.window_size = window_size
        self.increment = increment
        self.buffer_size = buffer
        self.verbose = verbose

        self.light = ModelState(light_model, window_size, increment, buffer, transforms=transforms)
        self.heavy = ModelState(heavy_model, window_size, increment, buffer, transforms=transforms)

        self.template_size = template_size
        self.min_template_size = min_template_size
        self.verification_steps = verification_steps
        self.reject_cooldown = reject_cooldown
        self.debug = debug
        self.running = False

    def stop(self) -> None:
        """Stops the detection loop if running in a background thread."""
        self.running = False

    def run(self, duration_sec: float | None = None):
        """
        Main loop. Scanning phase: light model watches the stream.
        Verification phase: heavy model must confirm within `verification_steps` attempts.
        """
        self.running = True
        expected_count = self.min_template_size
        verifying = False
        steps_left = 0
        cooldown_until = None
        start_ts = self.odh.get_time()

        while self.running:
            curr_ts = self.odh.get_time()
            if duration_sec is not None and (curr_ts - start_ts) >= duration_sec:
                break

            if self.odh.is_offline:
                assert isinstance(self.odh, OfflineCapableAbstractDataHandler)
                self.odh.advance(self.increment)

            dh_out = self.odh.get_data(self.window_size)

            if dh_out.count >= expected_count:
                # --- SCANNING: light model watches the stream ---
                if not verifying and (cooldown_until is None or curr_ts >= cooldown_until):
                    candidate = self.light.next_step(self.odh, self.template_size)
                    if candidate:
                        verifying = True
                        steps_left = self.verification_steps
                        self.heavy.reset()
                        if self.verbose:
                            print(f"{str(curr_ts)} candidate detected, verifying ({steps_left} attempts)")
                    else:
                        expected_count = min(expected_count + 10, self.template_size)

                # --- VERIFYING: heavy model checks the same buffered window ---
                # (first attempt runs immediately, on the exact data that triggered the light model)
                if verifying:
                    verified = self.heavy.next_step(self.odh, self.template_size)
                    steps_left -= 1

                    if verified:
                        if self.verbose:
                            print(f"{str(curr_ts)} wake detected")
                        self.odh.reset()
                        self.light.reset()
                        self.heavy.reset()
                        expected_count = self.min_template_size
                        verifying = False
                        steps_left = 0

                        if not self.odh.is_offline:
                            winsound.Beep(1000, 250)
                        self.odh.on_wake_detected()

                    elif steps_left <= 0:
                        if self.verbose:
                            print(f"{str(curr_ts)} candidate rejected by heavy model")
                        self.odh.reset()
                        self.light.reset()
                        self.heavy.reset()
                        expected_count = self.min_template_size
                        verifying = False
                        steps_left = 0
                        cooldown_until = (curr_ts + self.reject_cooldown) if self.reject_cooldown > 0 else None

            if self.odh.is_offline:
                assert isinstance(self.odh, OfflineCapableAbstractDataHandler)
                if self.odh.is_done:
                    self.running = False

            if not self.odh.is_offline:
                time.sleep(0.005)