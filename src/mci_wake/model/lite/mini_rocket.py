from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from libemg.utils import get_windows
from sklearn.linear_model import LogisticRegression
from sktime.transformations.rocket import MiniRocketMultivariate

from mci_wake.data.types import EmgDataset, TrainData
from mci_wake.model.abstract import AbstractModel


@dataclass
class MiniRocketConfig:
    window_size: int = 250
    increment: int = 25
    num_kernels: int = 84
    decision_threshold: float = 0.3
    window_agg: str = "mean"  # "mean" or "max"
    class_weight: str = "balanced"
    max_iter: int = 1000
    n_classes: int = 2
    gestures: list[str] = field(default_factory=list)
    customers: TrainData = field(default_factory=TrainData)
    seed: int = 0


class MiniRocketModel(AbstractModel):
    def __init__(self, config: MiniRocketConfig | None = None):
        self.config = config or MiniRocketConfig()
        self.trf = MiniRocketMultivariate(num_kernels=self.config.num_kernels, random_state=self.config.seed)
        self.clf = LogisticRegression(
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.seed,
        )
        self._fitted = False

    @property
    def n_classes(self) -> int:
        return self.config.n_classes

    def _get_windows(self, arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if len(arr) < self.config.window_size:
            arr = np.pad(arr, ((self.config.window_size - len(arr), 0), (0, 0)))
        return get_windows(arr, self.config.window_size, self.config.increment).astype(np.float32) # type: ignore // libemg incorrect typing here

    def fit(self, train: EmgDataset, test: EmgDataset, customers: TrainData | None = None) -> "MiniRocketModel":
        if customers is not None:
            self.config.customers = customers

        windows = [self._get_windows(trial) for trial in train.data]
        windowed_signals = np.concatenate(windows, axis=0)
        window_labels = np.concatenate([[label] * len(w) for w, label in zip(windows, train.labels)])

        rocket_features = self.trf.fit_transform(windowed_signals)
        self.clf.fit(rocket_features, window_labels)
        self.trf.transform(windowed_signals[:1])  # Numba JIT warm-up
        self._fitted = True
        return self

    def predict(self, data: npt.NDArray[np.float32], **kwargs: Any) -> int:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() before predicting.")
        windows = self._get_windows(data)
        rocket_features = self.trf.transform(windows)
        proba = self.clf.predict_proba(rocket_features)[:, 1]
        score = proba.max() if self.config.window_agg == "max" else proba.mean()
        return int(score > self.config.decision_threshold)

    def save(self, path: str | Path) -> None:
        joblib.dump({"config": self.config, "trf": self.trf, "clf": self.clf, "fitted": self._fitted}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MiniRocketModel":
        bundle = joblib.load(path)
        model = cls(config=bundle["config"])
        model.trf, model.clf, model._fitted = bundle["trf"], bundle["clf"], bundle["fitted"]
        model.trf.transform(np.zeros((1, 8, model.config.window_size), dtype=np.float32))  # Warm-up
        return model