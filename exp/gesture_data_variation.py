"""Measure how target-gesture data (participants x trials each) affects transfer.

Run from the repository root with ``python -m exp.gesture_data_variation``.
For each target gesture, a model is trained without it and frozen. A new output
layer is then trained on cached features using ``trials`` examples of the target
from each of ``people`` participants. Early stopping sees the same target budget
(same trials per person, taken from validation participants), so validation never
provides more target data than training does.
"""

import argparse
import copy
import csv
import itertools
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning import seed_everything
from torch.nn import functional as F

from mci_wake.data import EmgDataset, TrainData, gesture_mapping, preprocess_nm_data
from mci_wake.data.loaders import load_disco_adls, load_epn_data, split_disco_adls
from mci_wake.data.types import ADL_DATA, EPN_DATA
from mci_wake.model.neural import DiscreteClassifierConfig, DiscreteModel
from mci_wake.model.neural.classifier import make_data_loader
from mci_wake.transform.highpass import HighPassFilter
from mci_wake.transform.rest_normalization import RestNormalizer
from mci_wake.transform.transform import Transform

PEOPLE = (10, 20, 30, 40)
TRIALS = (5, 10, 15, 20, 25)  # each EPN participant has 25 examples per gesture
EPN_IDS_PER_COHORT = 306
GESTURE_WINDOW, GESTURE_INCREMENT = 10, 5
ADL_WINDOW, ADL_STEP = 150, 100
HEAD_BATCH, HEAD_EPOCHS, HEAD_PATIENCE = 256, 100, 12

# [(name, label)] sorted by label; noGesture is label 0.
GESTURES = sorted(gesture_mapping.items(), key=lambda item: item[1])


def load_epn() -> EmgDataset:
    """Load trainingSamples from both EPN cohorts, offsetting the second cohort's IDs.

    Both cohorts use IDs 1-306. Subjects are requested in two chunks because the
    loader ignores its filters (using a full-dataset cache) for requests over 300 IDs.
    """
    cohorts = []
    for offset, cohort in ((0, "training"), (EPN_IDS_PER_COHORT, "testing")):
        chunks = [
            load_epn_data(
                EPN_DATA,
                gesture_sets=("trainingSamples",),
                subjects=tuple(ids),
                subject_types=(cohort,),
            )
            for ids in (range(1, 301), range(301, 307))
        ]
        data = chunks[0].combine(chunks[1])
        cohorts.append(
            EmgDataset(
                data=data.data,
                labels=data.labels,
                subjects=data.subjects + offset,
                is_normalized=data.is_normalized,
            )
        )
    return cohorts[0].combine(cohorts[1])


def prepare_data(seed: int):
    """Split gestures and ADLs by participant (80/10/10), then normalize.

    Returns (train, val, test, n_test_gesture_rows, train_gesture_people,
    val_gesture_people, train_adl_people); each split is gestures followed by ADLs.
    """
    emg = preprocess_nm_data(load_epn())
    train_emg, held_emg = emg.split(test_percentage=0.2, by_subject=True, random_seed=seed)
    val_emg, test_emg = held_emg.split(test_percentage=0.5, by_subject=True, random_seed=seed + 1)

    random.seed(seed)
    adl = split_disco_adls(
        load_disco_adls(ADL_DATA), window=(ADL_WINDOW, ADL_WINDOW), step=ADL_STEP
    )
    people = sorted(int(s) for s in np.unique(adl.subjects))
    n = max(1, round(0.2 * len(people)))
    adl_people = people[:-2 * n], people[-2 * n:-n], people[-n:]
    assert adl_people[0], "ADL split leaves no training participants"
    train_adl, val_adl, test_adl = (adl[np.isin(adl.subjects, p)] for p in adl_people)

    # Fit preprocessing on gesture training data only.
    transform = Transform(HighPassFilter(), RestNormalizer())
    transform.fit(train_emg)
    t_train_emg, t_val_emg, t_test_emg, t_train_adl, t_val_adl, t_test_adl = (
        transform(d) for d in (train_emg, val_emg, test_emg, train_adl, val_adl, test_adl)
    )
    return (
        t_train_emg.combine(t_train_adl),
        t_val_emg.combine(t_val_adl),
        t_test_emg.combine(t_test_adl),
        len(t_test_emg),
        sorted(int(s) for s in np.unique(train_emg.subjects)),
        sorted(int(s) for s in np.unique(val_emg.subjects)),
        adl_people[0],
    )


def remap_labels(dataset: EmgDataset, label_map: dict[int, int]) -> EmgDataset:
    """Keep only labels in label_map, renumbered to the mapped class indices."""
    dataset = dataset[np.isin(dataset.labels, list(label_map))]
    return EmgDataset(
        data=dataset.data,
        labels=np.array([label_map[int(l)] for l in dataset.labels], dtype=np.int32),
        subjects=dataset.subjects,
        is_normalized=dataset.is_normalized,
    )


def encode(model: DiscreteModel, dataset: EmgDataset) -> torch.Tensor:
    """Frozen-network features (through the last hidden layer) for each example."""
    windowed = model._get_windows(dataset)
    loader = make_data_loader(windowed.data, windowed.labels, batch_size=128, shuffle=False)
    net = model.net.eval()
    device = next(net.parameters()).device

    batches = []
    with torch.no_grad():
        for inputs, _labels, lengths in loader:
            h = net.forward_temporal(net.forward_conv(inputs.to(device)), lengths.to(device))
            for layer in (net.initial_layer, net.layer1, net.layer2):
                h = net.relu(layer(h))
            batches.append(h.cpu())
    features = torch.cat(batches)
    assert len(features) == len(dataset), "windowing changed the number of examples"
    return features


def macro_recall(head: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean recall over the classes present in y."""
    with torch.no_grad():
        pred = head(x).argmax(1)
    return float(np.mean([(pred[y == c] == c).float().mean().item() for c in y.unique()]))


def train_head(model, x, y, x_val, y_val, seed) -> tuple[int, float]:
    """Freeze the network, train a new output layer; return (best epoch, best val macro recall)."""
    net = model.net
    device = next(net.parameters()).device
    n_classes = len(gesture_mapping)
    torch.manual_seed(seed)

    for p in net.parameters():
        p.requires_grad_(False)
    model.config = model.config.create(
        n_classes=n_classes, gestures=[name for name, label in GESTURES if label]
    )
    net.config = model.config
    head = torch.nn.Linear(net.config.mlp_layers[-1], n_classes).to(device)
    net.output_layer = head

    x, y, x_val, y_val = (t.to(device) for t in (x, y, x_val, y_val))
    # Each represented class gets equal total weight in the loss.
    counts = torch.bincount(y, minlength=n_classes).float()
    present = counts > 0
    weights = torch.where(
        present, len(y) / (present.sum() * counts.clamp(min=1)), torch.zeros_like(counts)
    )

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    shuffle = torch.Generator().manual_seed(seed)
    best_score, best_epoch, best_state, stale = -1.0, 0, copy.deepcopy(head.state_dict()), 0
    for epoch in range(1, HEAD_EPOCHS + 1):
        for idx in torch.randperm(len(y), generator=shuffle).to(device).split(HEAD_BATCH):
            optimizer.zero_grad()
            F.cross_entropy(head(x[idx]), y[idx], weight=weights).backward()
            optimizer.step()

        score = macro_recall(head, x_val, y_val)
        if score > best_score + 1e-6:
            best_score, best_epoch, best_state, stale = score, epoch, copy.deepcopy(head.state_dict()), 0
        else:
            stale += 1
            if stale >= HEAD_PATIENCE:
                break

    head.load_state_dict(best_state)
    return best_epoch, best_score


def score_model(head, features, labels, target, adl_mask) -> dict[str, float]:
    """Target recall/precision/F1, per-class recall, and false activations."""
    with torch.no_grad():
        pred = head(features.to(next(head.parameters()).device)).argmax(1).cpu().numpy()
    hit = pred == labels
    is_target = labels == target

    true_positives = int((hit & is_target).sum())
    precision = true_positives / max(int((pred == target).sum()), 1)
    recall = true_positives / int(is_target.sum())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    recalls = {name: float(hit[labels == label].mean()) for name, label in GESTURES}
    rest = (labels == 0) & ~adl_mask  # EPN rest windows only; ADLs are scored separately
    return {
        "target_recall": recall,
        "target_precision": precision,
        "target_f1": f1,
        "other_gestures_macro_recall": float(
            np.mean([recalls[n] for n, l in GESTURES if l not in (0, target)])
        ),
        "no_gesture_false_positive_rate": float((pred[rest] != 0).mean()),
        "adl_false_positive_rate": float((pred[adl_mask] != 0).mean()),
        "test_accuracy": float(hit.mean()),
        **{f"test_{name}_recall": r for name, r in recalls.items()},
    }


def target_pool(data: EmgDataset, people: list[int], target: int, rng) -> dict[int, np.ndarray]:
    """Per-person shuffled indices of target examples (slicing gives nested quotas)."""
    return {
        p: rng.permutation(np.flatnonzero((data.subjects == p) & (data.labels == target)))
        for p in people
    }


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gestures", help="comma-separated target gestures (default: all)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"exp/results/gesture_data_variation_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )
    args = parser.parse_args()

    valid = [name for name, label in GESTURES if label]
    wanted = [g.strip() for g in args.gestures.split(",")] if args.gestures else valid
    if not set(wanted) <= set(valid):
        parser.error(f"--gestures must be chosen from: {', '.join(valid)}")
    targets = [(name, label) for name, label in GESTURES if name in wanted]

    train, val, test, n_test_emg, train_people, val_people, train_adl_people = prepare_data(args.seed)

    # Fail now rather than after training a base model.
    if max(PEOPLE) > len(train_people):
        parser.error(f"Need {max(PEOPLE)} training participants, have {len(train_people)}.")
    for split in (train, val):
        per_person = Counter(zip(split.subjects.tolist(), split.labels.tolist()))
        if min(n for (_, label), n in per_person.items() if label) < max(TRIALS):
            parser.error(f"Some participants have fewer than {max(TRIALS)} examples of a gesture.")
    print(f"Participants (train/val): {len(train_people)}/{len(val_people)}; results: {args.output}")

    test_labels = test.labels.astype(np.int64)
    test_adl_mask = np.arange(len(test)) >= n_test_emg
    y_train, y_val = (torch.as_tensor(d.labels.astype(np.int64)) for d in (train, val))
    customers = TrainData(
        emg=[f"user{s}" for s in train_people], disco=[f"S{s}" for s in train_adl_people]
    )

    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        train_order = rng.permutation(train_people).tolist()
        val_order = rng.permutation(val_people).tolist()

        for target_name, target in targets:
            # Base model: trained on every gesture except the target (plus ADLs).
            base_seed = args.seed + repeat * 10000 + target
            seed_everything(base_seed, workers=True)
            others = [(name, label) for name, label in GESTURES if label != target]
            label_map = {label: i for i, (_, label) in enumerate(others)}
            config = DiscreteClassifierConfig.default_lite().create(
                n_classes=len(others),
                window_size=GESTURE_WINDOW,
                increment=GESTURE_INCREMENT,
                gestures=[name for name, label in others if label],
            )
            model = DiscreteModel(config)
            model.fit(remap_labels(train, label_map), remap_labels(val, label_map), customers=customers)
            x_train, x_val, x_test = (encode(model, d) for d in (train, val, test))

            pick = np.random.default_rng(base_seed)
            train_pool = target_pool(train, train_order, target, pick)
            val_pool = target_pool(val, val_order, target, pick)
            train_rest = np.flatnonzero(train.labels != target)
            val_rest = np.flatnonzero(val.labels != target)

            for people, trials in itertools.product(PEOPLE, TRIALS):
                train_target = np.concatenate([train_pool[p][:trials] for p in train_order[:people]])
                val_target = np.concatenate([val_pool[p][:trials] for p in val_order[:people]])
                train_idx = np.concatenate([train_rest, train_target])
                val_idx = np.concatenate([val_rest, val_target])

                seed = base_seed + 100 * people + trials
                epoch, val_recall = train_head(
                    model, x_train[train_idx], y_train[train_idx], x_val[val_idx], y_val[val_idx], seed
                )
                row = {
                    "repeat": repeat,
                    "target_gesture": target_name,
                    "participant_count": people,
                    "target_trials_per_participant": trials,
                    "target_train_trials": len(train_target),
                    "validation_target_trials": len(val_target),
                    "validation_macro_recall": val_recall,
                    "head_best_epoch": epoch,
                    "seed": seed,
                    **score_model(model.net.output_layer, x_test, test_labels, target, test_adl_mask),
                }
                write_row(args.output, row)
                print(
                    f"repeat={repeat} target={target_name} people={people} trials={trials} "
                    f"(train {len(train_target)}, val {len(val_target)}) recall={row['target_recall']:.3f}"
                )


if __name__ == "__main__":
    main()