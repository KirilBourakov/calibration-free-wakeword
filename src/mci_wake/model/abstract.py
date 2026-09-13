from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


from mci_wake.data.types import EmgDataset, TrainData


class AbstractModel(ABC):
    @property
    @abstractmethod
    def n_classes(self) -> int:
        """The number of classes the model predicts."""
        ...

    @abstractmethod
    def fit(
        self,
        train: EmgDataset,
        test: EmgDataset,
        customers: TrainData | None = None,
    ) -> "AbstractModel":
        """
        Train the model on the provided datasets.

        Returns
        -------
        self : AbstractModel
            The trained model instance.
        """
        ...

    @abstractmethod
    def predict(self, data: npt.NDArray[np.float32], **kwargs: Any) -> int:
        """
        Make a discrete class prediction on the provided EMG data segment.

        Parameters
        ----------
        data : npt.NDArray[np.float32]
            Input raw or transformed EMG segment (shape: (timepoints, channels)).
        **kwargs : Any
            Backend-specific inference arguments (e.g. device for PyTorch models).

        Returns
        -------
        int
            The predicted class index.
        """
        ...

    def reset(self) -> None:
        """
        Reset internal streaming/temporal state of the model.
        Default implementation is a no-op for stateless models.
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save model weights and configuration to the specified path or directory.
        """
        ...

    def eval(self) -> "AbstractModel":
        """
        Set the model to evaluation/inference mode.
        Default implementation is a no-op for non-neural models.
        """
        return self

    def set_train_mode(self, mode: bool = True) -> "AbstractModel":
        """
        Set the model to training mode.
        Default implementation is a no-op for non-neural models.
        """
        return self

