import numpy as np
import numpy.typing as npt

from mci_wake.data.types import EmgData


def stitch_into_buffer(
    buffer: npt.NDArray[np.floating],
    buffer_len: int,
    segment: EmgData,
    overlap_samples: int = 15,
) -> int:
    """
    Stitches a segment into a preallocated buffer using constant-power
    (raised-cosine / Hanning) cross-fading at the boundary seam.

    Args:
        buffer: Destination buffer array with sufficient capacity.
        buffer_len: Current number of valid samples in the buffer.
        segment: New segment to stitch.  Must be normalized.
        overlap_samples: Maximum number of overlap samples for cross-fading.

    Returns:
        New buffer length after stitching.
    """
    assert segment.is_normalized

    seg_len = len(segment.data)
    if buffer_len == 0:
        buffer[:seg_len] = segment.data
        return seg_len

    overlap = min(buffer_len, seg_len, overlap_samples)
    assert overlap > 0, f"Invalid size: overlap sample: {overlap_samples}, buffer_len: {buffer_len}, seg_len: {seg_len}"

    theta = np.linspace(0, np.pi / 2, overlap)
    if segment.data.ndim > 1:
        w_out = (np.cos(theta) ** 2)[:, None]
        w_in = (np.sin(theta) ** 2)[:, None]
    else:
        w_out = np.cos(theta) ** 2
        w_in = np.sin(theta) ** 2

    seam_out = buffer[buffer_len - overlap : buffer_len]
    seam_in = segment.data[:overlap]
    buffer[buffer_len - overlap : buffer_len] = (seam_out * w_out) + (seam_in * w_in)

    rem_len = seg_len - overlap
    buffer[buffer_len : buffer_len + rem_len] = segment.data[overlap:]
    return buffer_len + rem_len


def stitch(
    data: list[EmgData],
    overlap_samples: int = 15,
) -> npt.NDArray[np.floating]:
    """
    Stitches a list of 2D EMG arrays into a single continuous time series
    using a constant-power (raised-cosine / Hanning) cross-fade.

    Args:
        data: List of npt.NDArray[np.floating] of shape (T, 8) to be stitched. Must be normalized
        overlap_samples: Number of timesteps to cross-fade across the boundary.
                         Defaults to 15 samples (75 ms at a 200 Hz sampling rate).

    Returns:
        A single stitched array of shape (T_total, 8).
    """
    assert data, "No data provided to stitch"

    total_len = len(data[0].data)
    for arr in data[1:]:
        overlap = min(total_len, len(arr.data), overlap_samples)
        assert overlap > 0, f"Invalid size: overlap sample: {overlap_samples}, result: {total_len}, next_arr: {len(arr.data)}"
        total_len += len(arr.data) - overlap

    shape = (total_len,) + data[0].data.shape[1:]
    result = np.empty(shape, dtype=data[0].data.dtype)

    curr_len = 0
    for arr in data:
        curr_len = stitch_into_buffer(result, curr_len, arr, overlap_samples=overlap_samples)

    return result

