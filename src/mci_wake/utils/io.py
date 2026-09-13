import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
LIGHTNING_LOGS_DIR = ROOT_DIR / "scripts" / "lightning_logs"


def modelpath(version: int) -> Path:
    """Returns the path to the best model checkpoint for a given lightning_logs version.

    Args:
        version: The version number of the run (e.g. 9 for 'version_9').

    Returns:
        Path: Path to the best model checkpoint (.ckpt).

    Raises:
        FileNotFoundError: If the version folder or best checkpoint cannot be found.
    """
    ckpt_dir = LIGHTNING_LOGS_DIR / f"version_{version}" / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found at '{ckpt_dir}'.")

    # Find best model checkpoints (e.g., best-model-epoch=07-val_acc=1.00.ckpt)
    candidates = list(ckpt_dir.glob("best-model*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in '{ckpt_dir}'.")
    if len(candidates) > 1:
        raise FileNotFoundError(f"More than one checkpoint found in '{ckpt_dir}' that claim to be best-model.")

    return candidates[0]
