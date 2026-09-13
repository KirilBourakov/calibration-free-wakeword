from abc import ABC, abstractmethod
from typing import Any


class AbstractModel(ABC):
    @property
    @abstractmethod
    def n_classes(self) -> int:
        """The number of classes the model predicts."""
        ...

    @abstractmethod
    def predict(self, data: Any, **kwargs: Any) -> int:
        """
        Make a prediction on the provided windowed features / EMG data.

        Parameters
        ----------
        data : Any
            Input data (e.g., numpy array or tensor representing windowed EMG features).
        **kwargs : Any
            Additional backend-specific arguments (e.g. device for PyTorch models).

        Returns
        -------
        int
            The predicted class index.
        """
        ...

    def reset(self) -> None:
        """
        Reset internal state of the model.
        """
        pass
