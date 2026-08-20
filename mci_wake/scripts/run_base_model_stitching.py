"""
Run Base Model across StitchingDataHandler output and plot results.

Highlights TargetRegions (synthetic wake sequence test cases) in red.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure workspace src is on Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mci_wake.data import gesture_mapping, load_raw_data
from mci_wake.neural.classifier import DiscreteClassifier, DiscreteClassifierConfig
from mci_wake.neural.io import load
from mci_wake.data_handler.stitching import StitchingDataHandler
from libemg.utils import get_windows

GESTURE_NAMES = {v: k for k, v in gesture_mapping.items()}

DEFAULT_MODEL_PATH = ROOT_DIR / "other" / "models" / "base"

gestures = ['pinch']
probabilities = (0.8, 0, 0.2)
duration_sec = 300
template_size = 250  # Template buffer size matching wake_detect.py
window_size = 10     # Subwindow size for get_windows matching wake_detect.py
increment = 5        # Subwindow & streaming step size matching wake_detect.py

def main():
    # Safe unpickling globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([DiscreteClassifierConfig, DiscreteClassifier])

    model_path = Path(DEFAULT_MODEL_PATH)
    print(f"Loading base model from: {model_path}")
    base_model = load(model_path)
    base_model.eval()

    # 2. Load EMG & ADL dataset
    print("Loading raw EMG and ADL datasets...")
    emg_data_all, labels_all, _, adl_data, _ = load_raw_data()

    # 3. Instantiate StitchingDataHandler
    print(f"Initializing StitchingDataHandler with gestures={gestures}, probabilities={probabilities}...")
    handler = StitchingDataHandler(
        emg_data=emg_data_all,
        emg_labels=labels_all,
        adl_data=adl_data,
        gestures=gestures,
        probabilities=probabilities,
        realtime=False,
    )

    # Advance stream to desired duration
    target_samples = int(duration_sec * handler.sampling_rate)
    print(f"Generating {duration_sec} simulated seconds ({target_samples} samples)...")
    handler.advance(target_samples)

    emg_buffer = handler.buffer
    fs = handler.sampling_rate
    num_samples = len(emg_buffer)
    actual_duration = num_samples / fs
    print(f"Stitching completed: {num_samples} total samples ({actual_duration:.2f}s).")
    print(f"Generated {len(handler.target_regions)} target regions.")

    # 4. Run Base Model sliding window inference over StitchingDataHandler output
    print(f"Running base model sliding window inference (template_size={template_size}, window_size={window_size}, increment={increment})...")
    time_points = []
    raw_outputs = []  # shape: (N_steps, n_classes)

    for sample_idx in range(template_size, num_samples, increment):
        window_raw = emg_buffer[sample_idx - template_size : sample_idx]
        feats = get_windows(window_raw, window_size, increment)
        
        # Predict logits
        _, _, output_logits = base_model.predict(feats)
        logits = output_logits.detach().cpu().numpy()[0]

        time_sec = sample_idx / fs
        time_points.append(time_sec)
        raw_outputs.append(logits)

    time_points = np.array(time_points)
    raw_outputs = np.array(raw_outputs)  # shape (N, n_classes)

    # 5. Plot EMG datastream & Raw Model Outputs with highlighted target regions in Red
    print("Generating visualization plot...")
    fig, (ax_emg, ax_out) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # Time axis for continuous raw EMG
    emg_time = np.arange(num_samples) / fs

    # --- Plot 1: 8-Channel EMG Signal ---
    channel_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]
    offset_spacing = np.std(emg_buffer) * 6.0 if np.std(emg_buffer) > 0 else 1.0
    for ch in range(8):
        offset = ch * offset_spacing
        ax_emg.plot(
            emg_time,
            emg_buffer[:, ch] + offset,
            color=channel_colors[ch],
            linewidth=0.8,
            alpha=0.85,
        )

    ax_emg.set_ylabel("EMG Channels")
    ax_emg.set_title("EMG Datastream")
    ax_emg.grid(True, linestyle="--", alpha=0.5)

    # --- Plot 2: Raw Model Outputs ---
    class_colors = ["#7f7f7f", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    n_classes = base_model.config.n_classes
    for cls_idx in range(n_classes):
        cls_name = GESTURE_NAMES.get(cls_idx, f"Class {cls_idx}")
        ax_out.plot(
            time_points,
            raw_outputs[:, cls_idx],
            label=cls_name,
            color=class_colors[cls_idx % len(class_colors)],
            linewidth=1.5,
        )

    ax_out.set_ylabel("Raw Model Output")
    ax_out.set_xlabel("Time (s)")
    ax_out.set_title("Raw Model Outputs")
    ax_out.legend(loc="upper right", ncol=3, fontsize=9)
    ax_out.grid(True, linestyle="--", alpha=0.5)

    # --- Highlight Target Regions in RED ---
    for region in handler.target_regions:
        start_sec = region.start / fs
        end_sec = region.end / fs
        
        ax_emg.axvspan(start_sec, end_sec, color="red", alpha=0.3)
        ax_out.axvspan(start_sec, end_sec, color="red", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    main()
