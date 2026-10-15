"""Measure how many ADL activities, and how many minutes of each, suppress false triggers.

Run from the repository root with ``python -m exp.adl_data_variation``.
Gesture data is fixed at maximum. ADL participants are split into three disjoint groups
(reshuffled every repeat): a training pool, validation, and test. Each run fully
retrains the model with ADL from ALL pool participants, covering the first ``k``
activities of a per-repeat shuffled order, ``minutes`` of each activity per participant
(nested across levels). Validation ADL is cut to the same activities and minutes, so
early stopping never sees more ADL than training does. Test participants are always
scored on every activity, so false triggers are reported separately for activities the
model trained on ("seen") and never saw ("unseen"). Level (0, 0) trains with no ADL.
"""

import argparse
import itertools
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning import seed_everything

from gesture_data_variation import (
    ADL_STEP,
    ADL_WINDOW,
    GESTURE_INCREMENT,
    GESTURE_WINDOW,
    GESTURES,
    load_epn,
    write_row,
)
from mci_wake.data import EmgDataset, TrainData, preprocess_nm_data
from mci_wake.data.loaders import load_disco_adls, split_disco_adls
from mci_wake.data.types import ADL_ACTIVITY_MAPPING, ADL_DATA
from mci_wake.model.neural import DiscreteClassifierConfig, DiscreteModel
from mci_wake.model.neural.classifier import make_data_loader
from mci_wake.transform.highpass import HighPassFilter
from mci_wake.transform.rest_normalization import RestNormalizer
from mci_wake.transform.transform import Transform

MINUTES = (.5, 1, 2, 10)  # of each activity, per participant
ACTIVITY_FRACTIONS = (0.25, 0.5, 0.75, 1.0)  # of the available activities
SAMPLE_RATE = 200  # Hz (Myo armband)
WINDOWS_PER_MINUTE = 60 * SAMPLE_RATE // ADL_STEP  # windows advance by ADL_STEP samples
WINDOWS_PER_HOUR = 60 * WINDOWS_PER_MINUTE
HOLDOUT_FRACTION = 0.15  # of ADL participants, for each of validation and test

ACTIVITY_NAMES = {activity_id: name for name, activity_id in ADL_ACTIVITY_MAPPING.items()}
ACTIVITY_IDS = sorted(ACTIVITY_NAMES)
ACTIVITY_COUNTS = sorted({max(1, round(len(ACTIVITY_IDS) * f)) for f in ACTIVITY_FRACTIONS})


def prepare_data(seed: int):
    """Split gestures 80/10/10 by participant and normalize everything.

    Returns (train_emg, val_emg, test_emg, all_adl, train_gesture_people).
    ADL participants are split per repeat in main(), so ADL is returned whole.
    """
    emg = preprocess_nm_data(load_epn())
    train_emg, held_emg = emg.split(test_percentage=0.2, by_subject=True, random_seed=seed)
    val_emg, test_emg = held_emg.split(test_percentage=0.5, by_subject=True, random_seed=seed + 1)

    random.seed(seed)
    adl = split_disco_adls(
        load_disco_adls(ADL_DATA), window=(ADL_WINDOW, ADL_WINDOW), step=ADL_STEP
    )

    transform = Transform(HighPassFilter(), RestNormalizer())
    transform.fit(train_emg)  # fit on gesture training data only
    return (
        transform(train_emg),
        transform(val_emg),
        transform(test_emg),
        transform(adl),
        sorted(int(s) for s in np.unique(train_emg.subjects)),
    )


def shuffled_windows(adl, people) -> dict[tuple[int, int], np.ndarray]:
    """Shuffled window indices per (person, activity); prefixes give nested budgets."""
    return {
        (p, a): np.random.permutation(np.flatnonzero((adl.subjects == p) & (adl.activity_labels == a)))
        for p in people
        for a in ACTIVITY_IDS
    }


def take_minutes(adl, windows, people, activities, minutes):
    """``minutes`` of each listed activity from each person (less if they have less)."""
    n = round(minutes * WINDOWS_PER_MINUTE)
    return adl[np.concatenate([windows[p, a][:n] for p in people for a in activities])]


def predict(model: DiscreteModel, dataset: EmgDataset) -> np.ndarray:
    """Predicted class for each example."""
    windowed = model._get_windows(dataset)
    assert len(windowed.data) == len(dataset), "windowing changed the number of examples"
    loader = make_data_loader(windowed.data, windowed.labels, batch_size=128, shuffle=False)
    net = model.net.eval()
    device = next(net.parameters()).device
    with torch.no_grad():
        return torch.cat(
            [net.forward_once(x.to(device), n.to(device)).argmax(1).cpu() for x, _, n in loader]
        ).numpy()


def score_adl(preds: np.ndarray, adl, seen: list[int], prefix: str) -> dict[str, float]:
    """False triggers (any non-noGesture prediction): overall, per hour, per activity."""
    fp = preds != 0
    nan = float("nan")

    def rate(mask) -> float:
        return float(fp[mask].mean()) if mask.any() else nan

    seen_mask = np.isin(adl.activity_labels, seen)
    out = {
        f"{prefix}_windows": len(preds),
        f"{prefix}_hours": len(preds) / WINDOWS_PER_HOUR,
        f"{prefix}_fp_count": int(fp.sum()),
        f"{prefix}_fp_rate": rate(np.ones_like(fp)),
        f"{prefix}_fp_per_hour": rate(np.ones_like(fp)) * WINDOWS_PER_HOUR,
        f"{prefix}_seen_fp_rate": rate(seen_mask),
        f"{prefix}_unseen_fp_rate": rate(~seen_mask),
    }
    for activity in ACTIVITY_IDS:
        mask = adl.activity_labels == activity
        out[f"{prefix}_{ACTIVITY_NAMES[activity]}_windows"] = int(mask.sum())
        out[f"{prefix}_{ACTIVITY_NAMES[activity]}_fp_rate"] = rate(mask)
    for name, label in GESTURES:
        if label:
            out[f"{prefix}_triggered_{name}"] = int((preds == label).sum())
    return out


def score_gestures(model: DiscreteModel, test_emg: EmgDataset) -> dict[str, float]:
    """Accuracy plus per-class recall and precision on held-out gesture participants."""
    preds, labels = predict(model, test_emg), test_emg.labels
    out = {"gesture_test_accuracy": float((preds == labels).mean())}
    for name, label in GESTURES:
        hits = int(((preds == label) & (labels == label)).sum())
        out[f"gesture_test_{name}_recall"] = hits / max(int((labels == label).sum()), 1)
        out[f"gesture_test_{name}_precision"] = hits / max(int((preds == label).sum()), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--strategy",
        choices=["grid", "minutes", "activities"],
        default="grid",
        help="grid: activities x minutes; minutes: all activities, vary minutes; "
        "activities: max minutes, vary activity count",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"exp/results/adl_data_variation_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )
    args = parser.parse_args()

    train_emg, val_emg, test_emg, adl, gesture_people = prepare_data(args.seed)
    adl_people = [int(s) for s in np.unique(adl.subjects)]
    n_holdout = max(1, round(HOLDOUT_FRACTION * len(adl_people)))
    pool_size = len(adl_people) - 2 * n_holdout
    assert pool_size >= 1, "ADL split leaves no training participants"

    levels = {
        "grid": list(itertools.product(ACTIVITY_COUNTS, MINUTES)),
        "minutes": [(ACTIVITY_COUNTS[-1], m) for m in MINUTES],
        "activities": [(c, MINUTES[-1]) for c in ACTIVITY_COUNTS],
    }[args.strategy]
    levels = [(0, 0)] + levels  # baseline: no ADL
    print(f"ADL participants: pool={pool_size}, val={n_holdout}, test={n_holdout}")
    print(f"Activities: {len(ACTIVITY_IDS)}; counts {ACTIVITY_COUNTS}; minutes {MINUTES}")
    print(f"{len(levels)} training runs per repeat; results: {args.output}")

    for repeat in range(args.repeats):
        seed_everything(args.seed + repeat)  # also seeds numpy for the shuffles below
        order = [int(p) for p in np.random.permutation(adl_people)]
        pool, val_people, test_people = order[:pool_size], order[pool_size:-n_holdout], order[-n_holdout:]
        activity_order = [int(a) for a in np.random.permutation(ACTIVITY_IDS)]

        pool_windows = shuffled_windows(adl, pool)
        val_windows = shuffled_windows(adl, val_people)
        test_adl = adl[np.isin(adl.subjects, test_people)]
        assert len(test_adl), "no test ADL windows"

        for n_activities, minutes in levels:
            seen = activity_order[:n_activities]
            train_dataset, val_dataset, train_windows = train_emg, val_emg, 0
            if minutes:
                train_adl = take_minutes(adl, pool_windows, pool, seen, minutes)
                val_adl = take_minutes(adl, val_windows, val_people, seen, minutes)
                train_dataset, val_dataset = train_emg.combine(train_adl), val_emg.combine(val_adl)
                train_windows = len(train_adl)

            # Same seed at every level, so levels differ only in their data.
            seed_everything(args.seed + repeat, workers=True)
            config = DiscreteClassifierConfig.default_lite().create(
                n_classes=len(GESTURES),
                window_size=GESTURE_WINDOW,
                increment=GESTURE_INCREMENT,
                gestures=[name for name, label in GESTURES if label],
            )
            model = DiscreteModel(config)
            model.fit(
                train_dataset,
                val_dataset,
                customers=TrainData(
                    emg=[f"user{s}" for s in gesture_people],
                    disco=[f"S{s}" for s in pool] if minutes else [],
                ),
            )

            row = {
                "repeat": repeat,
                "adl_activity_count": len(seen),
                "adl_minutes_per_activity": minutes,
                "train_adl_minutes": train_windows * ADL_STEP / SAMPLE_RATE / 60,
                "train_activities": ";".join(ACTIVITY_NAMES[a] for a in seen),
                "train_adl_participant_ids": ";".join(map(str, pool if minutes else [])),
                "test_adl_participant_ids": ";".join(map(str, test_people)),
                **score_adl(predict(model, test_adl), test_adl, seen, "test_adl"),
                **score_gestures(model, test_emg),
            }
            write_row(args.output, row)
            print(
                f"repeat={repeat} activities={len(seen)} min/activity={minutes} "
                f"(used {row['train_adl_minutes']:.1f} min) | test ADL FPs/hour="
                f"{row['test_adl_fp_per_hour']:.1f} (seen {row['test_adl_seen_fp_rate']:.2%}, "
                f"unseen {row['test_adl_unseen_fp_rate']:.2%}) | "
                f"gesture acc={row['gesture_test_accuracy']:.2%}"
            )


if __name__ == "__main__":
    main()