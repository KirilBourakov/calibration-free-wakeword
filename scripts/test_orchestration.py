import time
import argparse
import threading
import torch

from mci_wake.data import filter_training, load_raw_data
from mci_wake.model.neural import DiscreteModel, DiscreteClassifierConfig, TrainData
from mci_wake.orchestration.model_chain import ModelChain
from mci_wake.data_handler.stitching import StitchingDataHandler
from mci_wake.transform.highpass import HighPassFilter
from mci_wake.transform.rest_normalization import RestNormalizer
from mci_wake.transform.transform import Transform
from mci_wake.utils.io import modelpath


def get_models() -> tuple[DiscreteModel, ...]:
    model1 = DiscreteModel.load_from_checkpoint(modelpath(8))
    return (model1,)


def main():
    parser = argparse.ArgumentParser(description="Test wake word orchestration.")
    parser.add_argument("--realtime", action="store_true", help="Run in real-time wall-clock mode.")
    parser.add_argument("--duration", type=float, default=300.0, help="Duration of simulation in seconds.")
    args = parser.parse_args()

    # Allowlist custom classes for safe unpickling in PyTorch 2.6+
    torch.serialization.add_safe_globals([DiscreteClassifierConfig, DiscreteModel, TrainData])


    models = get_models()
    emg, adl = filter_training(
        *load_raw_data(), # *[m.config.customers for m in models]
    )

    transforms = Transform(HighPassFilter(), RestNormalizer())
    transforms.fit(emg)
    emg_data_norm = transforms(emg)
    adl_data_norm = transforms(adl)

    gestures = ["waveIn", "waveOut"]
    realtime = args.realtime
    handler = StitchingDataHandler(
        emg_data=emg_data_norm,
        adl_data=adl_data_norm,
        gestures=gestures,
        probabilities=(0.4, 0.4, 0.2),
        realtime=realtime,
    )
    # handler = RecordingDataHandler(path=r"D:\Coding\calibration-free-wakeword\mci_wake\recordings\pinchfirst\shake1.json")

    if realtime:
        def print_stats_periodically():
            while True:
                time.sleep(30.0)
                print(handler.get_trigger_stats())

        stats_thread = threading.Thread(target=print_stats_periodically, daemon=True)
        stats_thread.start()

        discrete = ModelChain(handler, 10, 5, list(models), transforms=None)
        discrete.run()
    else:
        print(f"Running fast simulation for {args.duration} simulated seconds...")
        start_t = time.time()
        discrete = ModelChain(handler, 10, 5, list(models), transforms=None, verbose=False)
        discrete.run(duration_sec=args.duration)
        elapsed = time.time() - start_t
        print(f"Simulation completed in {elapsed:.2f} seconds wall-clock time!")
        print(handler.get_trigger_stats())


if __name__ == "__main__":
    main()