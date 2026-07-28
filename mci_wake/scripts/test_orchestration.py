import sys
import os
import time
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


def print_stats_periodically(handler: StitchingDataHandler, interval: float = 30.0):
    while True:
        time.sleep(interval)
        stats = handler.get_trigger_stats()
        print("\n=== [Trigger Stats (every 30s)] ===")
        print(f"True Positives:  {stats['true_positives']}")
        print(f"False Positives: {stats['false_positives']}")
        print(f"False Negatives: {stats['false_negatives']}")
        print(f"Total Triggers:  {stats['total_triggers']}")
        print(f"Target Regions:  {stats['target_regions_count']}")
        print("===================================\n")


def main():
    # Allowlist custom classes for safe unpickling in PyTorch 2.6+
    torch.serialization.add_safe_globals([DiscreteClassifierConfig, DiscreteClassifier])

    models = get_models()
    emg_data_all, labels_all, subject_ids_all, adl_data, adl_ids = filter_training(
        *load_raw_data(), *[m.config.customers for m in models]
    )

    gestures = ["pinch", "fist"]
    handler = StitchingDataHandler(
        emg_data=emg_data_all,
        emg_labels=labels_all,
        adl_data=adl_data,
        gestures=gestures,
        probabilities=(0.4, 0.4, 0.2),
    )

    stats_thread = threading.Thread(
        target=print_stats_periodically, args=(handler, 30.0), daemon=True
    )
    stats_thread.start()

    discrete = WakeDetect(handler, 10, 5, list(models), normalize=False)
    discrete.run()


if __name__ == "__main__":
    main()