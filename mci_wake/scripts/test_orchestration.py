import sys
import os
import time
import argparse
import threading
import torch

from mci_wake.data.train_utils import filter_training, load_raw_data
from mci_wake.neural.classifier import DiscreteClassifier, DiscreteClassifierConfig
from mci_wake.neural.lightning_module import DiscreteLightningModule
from mci_wake.orchestration.wake_detect import WakeDetect
from mci_wake.stitching.handler import StitchingDataHandler


def get_models() -> tuple[DiscreteClassifier, ...]:
    model1 = DiscreteLightningModule.load_from_checkpoint(
        r"D:\Coding\calibration-free-wakeword\mci_wake\scripts\lightning_logs\version_2\checkpoints\best-model-epoch=05-val_acc=0.98.ckpt"
    )
    model2 = DiscreteLightningModule.load_from_checkpoint(
        r"D:\Coding\calibration-free-wakeword\mci_wake\scripts\lightning_logs\version_3\checkpoints\best-model-epoch=09-val_acc=0.99.ckpt"
    )
    return model1.internals, model2.internals


def print_stats(handler: StitchingDataHandler):
    stats = handler.get_trigger_stats()
    print("\n=== [Trigger Stats] ===")
    print(f"True Positives:  {stats['true_positives']}")
    print(f"False Positives: {stats['false_positives']}")
    print(f"False Negatives: {stats['false_negatives']}")
    print(f"Total Triggers:  {stats['total_triggers']}")
    print(f"Target Regions:  {stats['target_regions_count']}")
    print("=======================\n")


def main():
    parser = argparse.ArgumentParser(description="Test wake word orchestration.")
    parser.add_argument("--realtime", action="store_true", help="Run in real-time wall-clock mode.")
    parser.add_argument("--duration", type=float, default=300.0, help="Duration of simulation in seconds.")
    args = parser.parse_args()

    # Allowlist custom classes for safe unpickling in PyTorch 2.6+
    torch.serialization.add_safe_globals([DiscreteClassifierConfig, DiscreteClassifier])

    models = get_models()
    emg_data_all, labels_all, subject_ids_all, adl_data, adl_ids = filter_training(
        *load_raw_data(), # *[m.config.customers for m in models]
    )

    gestures = ["pinch", "fist"]
    realtime = args.realtime
    handler = StitchingDataHandler(
        emg_data=emg_data_all,
        emg_labels=labels_all,
        adl_data=adl_data,
        gestures=gestures,
        probabilities=(0.4, 0.4, 0.2),
        realtime=realtime,
    )

    if realtime:
        def print_stats_periodically():
            while True:
                time.sleep(30.0)
                print_stats(handler)

        stats_thread = threading.Thread(target=print_stats_periodically, daemon=True)
        stats_thread.start()

        discrete = WakeDetect(handler, 10, 5, list(models), normalize=False, realtime=True)
        discrete.run()
    else:
        print(f"Running fast simulation for {args.duration} simulated seconds...")
        start_t = time.time()
        discrete = WakeDetect(handler, 10, 5, list(models), normalize=False, realtime=False, verbose=False)
        discrete.run(duration_sec=args.duration)
        elapsed = time.time() - start_t
        print(f"Simulation completed in {elapsed:.2f} seconds wall-clock time!")
        print_stats(handler)


if __name__ == "__main__":
    main()