import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sktime.transformations.rocket import MiniRocketMultivariate

from mci_wake.model import AbstractModel


# Draft version; non functional
class MiniRocketModel(AbstractModel):
    """
    Light model: sktime MiniRocketMultivariate transform + linear head.

    Stateless at inference, so reset() is a no-op and the cascade's
    reset discipline costs nothing.
    """

    def __init__(
        self,
        num_kernels: int = 512,           # sktime default -> 512 PPV features
        decision_threshold: float = 0.3,  # < 0.5 biases toward recall (stage-1 role)
        window_agg: str = "mean",         # how to combine windows in one template
        class_weight="balanced",          # rest windows will dominate your data
        seed: int = 0,
    ):
        self._n_classes = 2
        self.num_kernels = num_kernels
        self.decision_threshold = decision_threshold
        self.window_agg = window_agg
        self.class_weight = class_weight
        self.seed = seed

        self.trf = MiniRocketMultivariate(num_kernels=num_kernels, random_state=seed)
        self.clf = LogisticRegression(max_iter=1000, class_weight=class_weight)
        self._fitted = False

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MiniRocketModel":
        """
        X: (n_windows, n_channels, n_timepoints) float32 — raw windows, channels first.
        y: (n_windows,) int — 0 = rest/negative, 1 = gesture/positive.
        """
        X = self._check_X(X)
        if X.shape[0] < 10:
            raise ValueError(
                f"MiniRocket quantile binning needs >= 10 training windows, got {X.shape[0]}"
            )
        self.n_channels, self.n_timepoints = X.shape[1], X.shape[2]

        Xt = self.trf.fit_transform(X)      # (n_windows, num_kernels), PPV in [0, 1]
        self.clf.fit(Xt, y)

        # numba JIT warm-up
        self.trf.transform(X[:2])
        self._fitted = True
        return self

    def predict_proba(self, data) -> np.ndarray:
        """Per-window probabilities, shape (n_windows, 2)."""
        if not self._fitted:
            raise RuntimeError("Call fit() or load() before predicting")
        X = self._check_X(data)
        return self.clf.predict_proba(self.trf.transform(X))

    def predict(self, data, **kwargs) -> int:
        proba = self.predict_proba(data)
        # template contains several overlapping windows -> aggregate, then threshold
        p1 = proba[:, 1].mean() if self.window_agg == "mean" else proba[:, 1].max()
        return int(p1 > self.decision_threshold)

    def reset(self) -> None:
        pass  # stateless

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "MiniRocketModel":
        model = joblib.load(path)
        model.trf.transform(np.zeros((2, model.n_channels, model.n_timepoints), dtype=np.float32))
        return model  # warm up numba after reload too

    def _check_X(self, data) -> np.ndarray:
        X = np.asarray(data, dtype=np.float32)
        if X.ndim == 2:  # single window -> batch of 1
            X = X[None]
        if X.ndim != 3:
            raise ValueError(f"expected 2D/3D window array, got shape {X.shape}")
        # orient to (n, channels, time) using dims memorized at fit time —
        # covers both get_windows layouts: (n, t, c) and (n, c, t)
        if X.shape[1] != self.n_channels:
            X = np.ascontiguousarray(X.transpose(0, 2, 1))
        if X.shape[1:] != (self.n_channels, self.n_timepoints):
            raise ValueError(
                f"windows {X.shape} don't match fitted "
                f"({self.n_channels}, {self.n_timepoints})"
            )
        return X