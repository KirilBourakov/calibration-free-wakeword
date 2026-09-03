# Calibration-Free EMG Wake-Word Detection (`mci_wake`)

Calibration-free wake-word detection using surface electromyography (sEMG).

## Setup & Installation

1. **Clone the repository and submodules:**
   ```bash
   git clone --recurse-submodules https://github.com/KirilBourakov/calibration-free-wakeword.git
   cd calibration-free-wakeword
   ```

2. **Install dependencies in editable mode:**
   ```bash
   # Install local libemg submodule
   pip install -e ./libemg

   # Install mci_wake package in editable mode
   pip install -e .
   ```

## Running Scripts

All executable workflows live in the `scripts/` directory:

- `python scripts/train_model.py`: Train classifier on EMG datasets.
- `python scripts/record.py`: Live EMG recording session.
- `python scripts/run_discrete.py`: Run discrete gesture recognition.
- `python scripts/run_base_model_stitching.py`: Offline evaluation with stitching.

## References

Historical reference implementations stored under `reference/`:
- **1DMCI**: 1D continuous control ([eeddy/1DMCI](https://github.com/eeddy/1DMCI))
- **DiscreteMCI**: Discrete gesture detection without calibration ([eeddy/DiscreteMCI](https://github.com/eeddy/DiscreteMCI))
