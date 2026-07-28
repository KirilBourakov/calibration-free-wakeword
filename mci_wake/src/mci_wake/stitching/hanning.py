import numpy as np
import numpy.typing as npt


def stitch(
    data: list[npt.NDArray[np.floating]],
    overlap_samples: int = 15,
    in_place: bool = False,
) -> npt.NDArray[np.floating]:
    """
    Stitches a list of 2D EMG arrays into a single continuous time series
    using a constant-power (raised-cosine / Hanning) cross-fade.

    Args:
        data: List of npt.NDArray[np.floating] of shape (T, 8) to be stitched.
        overlap_samples: Number of timesteps to cross-fade across the boundary.
                         Defaults to 15 samples (75 ms at a 200 Hz sampling rate).
        in_place: If True, mutates the seam of data[0] in place to avoid unneeded array copying.

    Returns:
        A single stitched array of shape (T_total, 8).
    """
    assert data, "No data provided to stitch"
    assert len(data) > 1, "Must have at least 2 samples to stitch"

    result = data[0] if in_place else data[0].copy()
    for next_arr in data[1:]:
        overlap = min(len(result), len(next_arr), overlap_samples)
        assert overlap > 0, f"Invalid size: overlap sample: {overlap_samples}, result: {len(result)}, next_arr: {len(next_arr)}"

        # 1. Generate constant-power Hanning weights from 0 to pi/2
        theta = np.linspace(0, np.pi / 2, overlap)

        # Reshape to (overlap, 1) so weights broadcast across all 8 EMG channels
        w_out = (np.cos(theta) ** 2)[:, None]  # Fades from 1.0 down to 0.0
        w_in = (np.sin(theta) ** 2)[:, None]  # Fades from 0.0 up to 1.0

        # 2. Extract overlapping boundary slices
        seam_out = result[-overlap:]
        seam_in = next_arr[:overlap]

        # 3. Apply the linear superposition cross-fade
        blended_seam = (seam_out * w_out) + (seam_in * w_in)

        # 4. Construct the new continuous array: [before seam] -> [blended seam] -> [after seam]
        result[-overlap:] = blended_seam
        result = np.vstack([result, next_arr[overlap:]])

    return result
