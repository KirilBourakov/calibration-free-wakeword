import numpy as np
import numpy.typing as npt


def safe_znormalize_global(
    data: npt.NDArray[np.floating],
    eps: float = 1e-3
) -> npt.NDArray[np.floating]:
    """
    Applies safe Z-score normalization (mean 0, variance 1) with a noise-floor clamp
    to prevent zero-division and baseline noise amplification in biosignals.

    Args:
        data: Input NumPy array (e.g., shape (T, 8) for raw 8-channel EMG).
        axis: The axis or axes along which to compute mean and standard deviation.
              Defaults to 0 (normalizing each channel independently over time).
        eps: The noise-floor clamp. If the signal's standard deviation falls below
             this value, the denominator is clamped to eps.

    Returns:
        The normalized array with the exact same shape as the input.
    """
    data_float = np.asarray(data, dtype=np.float32)
    mean = np.mean(data_float)
    std = np.std(data_float)
    safe_std = np.maximum(std, eps)
    return (data_float - mean) / safe_std