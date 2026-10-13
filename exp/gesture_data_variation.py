"""Measure how target-gesture trials per participant affect transfer.

Run from the repository root with ``python -m exp.gesture_data_variation``.
The experiment freezes a model trained without one gesture, then trains its
output layer with a fixed number of examples of that gesture from each of an
increasing number of participants.
"""

import argparse
import csv
import random
from dataclasses import dataclass
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


# Each EPN participant has 25 training examples per gesture.
PARTICIPANT_COUNTS = (10, 20, 30, 40)
TRIALS_PER_PARTICIPANT = (5, 10, 15, 20, 25)

GESTURE_WINDOW_SIZE = 10
GESTURE_WINDOW_INCREMENT = 5
ADL_WINDOW_SIZE = 150
ADL_WINDOW_STEP = 100

HEAD_BATCH_SIZE = 256
HEAD_MAX_EPOCHS = 100
HEAD_EARLY_STOPPING_PATIENCE = 12


@dataclass(frozen=True)
class ExperimentData:
    """Participant-disjoint data splits used by the experiment."""

    train_emg: EmgDataset
    validation_emg: EmgDataset
    test_emg: EmgDataset
    train_adl: EmgDataset
    validation_adl: EmgDataset
    test_adl: EmgDataset
    train_adl_participants: list[int]
    validation_adl_participants: list[int]
    test_adl_participants: list[int]


def combine_datasets(datasets: list[EmgDataset]) -> EmgDataset:
    """Combine datasets in order, returning an empty dataset for no inputs."""
    if not datasets:
        return EmgDataset.empty()
    result = datasets[0]
    for dataset in datasets[1:]:
        result = result.combine(dataset)
    return result


def load_epn_training_examples() -> EmgDataset:
    """Load only trainingSamples, keeping the two EPN cohorts distinct.

    EPN's trainingJSON and testingJSON folders both use participant IDs 1–306.
    They are separate cohorts, so the second cohort's IDs are shifted by 306.
    Loading only trainingSamples gives each participant 25 examples per class.
    The source loader uses its full-dataset cache for requests over 300 IDs, so
    each cohort is loaded in two smaller chunks to ensure these filters apply.
    """
    cohort_datasets: list[EmgDataset] = []
    subject_chunks = (tuple(range(1, 301)), tuple(range(301, 307)))

    for cohort_name in ("training", "testing"):
        chunks = [
            load_epn_data(
                EPN_DATA,
                gesture_sets=("trainingSamples",),
                subjects=subject_ids,
                subject_types=(cohort_name,),
            )
            for subject_ids in subject_chunks
        ]
        cohort_datasets.append(combine_datasets(chunks))

    first_cohort, second_cohort = cohort_datasets
    second_cohort = EmgDataset(
        data=second_cohort.data,
        labels=second_cohort.labels,
        subjects=second_cohort.subjects + int(np.max(first_cohort.subjects)),
        is_normalized=second_cohort.is_normalized,
    )
    return first_cohort.combine(second_cohort)


def split_adl_by_participant(seed: int) -> tuple[
    EmgDataset, EmgDataset, EmgDataset, list[int], list[int], list[int]
]:
    """Load/window ADLs and assign whole participants to fixed splits."""
    random.seed(seed)
    raw_adl = load_disco_adls(ADL_DATA)
    adl = split_disco_adls(
        raw_adl,
        window=(ADL_WINDOW_SIZE, ADL_WINDOW_SIZE),
        step=ADL_WINDOW_STEP,
    )

    participants = sorted(int(subject) for subject in np.unique(adl.subjects))
    if len(participants) < 3:
        raise ValueError("At least three ADL participants are required for the splits.")

    test_count = max(1, round(len(participants) * 0.2))
    validation_count = max(1, round(len(participants) * 0.2))
    test_participants = participants[-test_count:]
    validation_participants = participants[-(test_count + validation_count):-test_count]
    train_participants = participants[:-(test_count + validation_count)]
    if not train_participants:
        raise ValueError("The ADL split leaves no training participants.")

    train_adl = adl[np.isin(adl.subjects, train_participants)]
    validation_adl = adl[np.isin(adl.subjects, validation_participants)]
    test_adl = adl[np.isin(adl.subjects, test_participants)]
    return (
        train_adl,
        validation_adl,
        test_adl,
        train_participants,
        validation_participants,
        test_participants,
    )


def prepare_data(seed: int) -> ExperimentData:
    """Load, split by participant, and normalize gesture and ADL data."""
    emg = preprocess_nm_data(load_epn_training_examples())
    train_emg, held_emg = emg.split(
        test_percentage=0.2,
        by_subject=True,
        random_seed=seed,
    )
    validation_emg, test_emg = held_emg.split(
        test_percentage=0.5,
        by_subject=True,
        random_seed=seed + 1,
    )

    (
        train_adl,
        validation_adl,
        test_adl,
        train_adl_participants,
        validation_adl_participants,
        test_adl_participants,
    ) = split_adl_by_participant(seed)

    # Fit preprocessing only on gesture training data, matching model training.
    transforms = Transform(HighPassFilter(), RestNormalizer())
    transforms.fit(train_emg)

    return ExperimentData(
        train_emg=transforms(train_emg),
        validation_emg=transforms(validation_emg),
        test_emg=transforms(test_emg),
        train_adl=transforms(train_adl),
        validation_adl=transforms(validation_adl),
        test_adl=transforms(test_adl),
        train_adl_participants=train_adl_participants,
        validation_adl_participants=validation_adl_participants,
        test_adl_participants=test_adl_participants,
    )


def remap_labels(dataset: EmgDataset, label_map: dict[int, int]) -> EmgDataset:
    """Keep selected labels and map them to the requested class indices."""
    mask = np.isin(dataset.labels, list(label_map))
    selected = dataset[mask]
    labels = np.asarray(
        [label_map[int(label)] for label in selected.labels],
        dtype=np.int32,
    )
    return EmgDataset(
        data=selected.data,
        labels=labels,
        subjects=selected.subjects,
        is_normalized=selected.is_normalized,
    )


def encode_dataset(model: DiscreteModel, dataset: EmgDataset) -> torch.Tensor:
    """Cache the frozen model's features for each input example."""
    windowed = model._get_windows(dataset)
    loader = make_data_loader(
        windowed.data,
        windowed.labels,
        batch_size=128,
        shuffle=False,
    )
    network = model.net
    device = next(network.parameters()).device
    network.eval()

    batches: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, _labels, lengths in loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            features = network.forward_conv(inputs)
            features = network.forward_temporal(features, lengths)
            features = network.relu(network.initial_layer(features))
            features = network.relu(network.layer1(features))
            features = network.relu(network.layer2(features))
            batches.append(features.cpu())
    return torch.cat(batches, dim=0)


def balanced_class_weights(labels: torch.Tensor, class_count: int) -> torch.Tensor:
    """Give each represented class equal total weight in the loss."""
    counts = torch.bincount(labels, minlength=class_count).to(torch.float32)
    represented = counts > 0
    weights = torch.zeros(class_count, dtype=torch.float32)
    weights[represented] = len(labels) / (
        int(represented.sum()) * counts[represented]
    )
    return weights


def macro_recall(
    head: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute mean recall over the classes present in the labels."""
    device = next(head.parameters()).device
    with torch.no_grad():
        predictions = head(features.to(device)).argmax(dim=1).cpu()

    recalls = []
    for label in torch.unique(labels):
        mask = labels == label
        recalls.append(float((predictions[mask] == labels[mask]).float().mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def train_output_layer(
    model: DiscreteModel,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    seed: int,
) -> tuple[int, float]:
    """Freeze the representation and train a new six-class output layer."""
    network = model.net
    for parameter in network.parameters():
        parameter.requires_grad_(False)

    device = next(network.parameters()).device
    torch.manual_seed(seed)
    model.config = model.config.create(
        n_classes=len(gesture_mapping),
        gestures=[name for name, label in sorted(gesture_mapping.items(), key=lambda x: x[1]) if label],
    )
    network.config = model.config
    network.output_layer = torch.nn.Linear(
        network.config.mlp_layers[-1],
        len(gesture_mapping),
    ).to(device)
    head = network.output_layer
    for parameter in head.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    weights = balanced_class_weights(train_labels, len(gesture_mapping)).to(device)
    order_generator = torch.Generator(device="cpu").manual_seed(seed)
    best_score = -1.0
    best_epoch = 0
    best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    stale_epochs = 0

    for epoch in range(1, HEAD_MAX_EPOCHS + 1):
        head.train()
        order = torch.randperm(len(train_labels), generator=order_generator)
        for start in range(0, len(order), HEAD_BATCH_SIZE):
            indices = order[start:start + HEAD_BATCH_SIZE]
            inputs = train_features[indices].to(device)
            labels = train_labels[indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(inputs), labels, weight=weights)
            loss.backward()
            optimizer.step()

        head.eval()
        score = macro_recall(head, validation_features, validation_labels)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in head.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= HEAD_EARLY_STOPPING_PATIENCE:
                break

    head.load_state_dict(best_state)
    head.eval()
    return best_epoch, best_score


def score_model(
    model: DiscreteModel,
    features: torch.Tensor,
    labels: np.ndarray,
    target_label: int,
    adl_mask: np.ndarray,
) -> dict[str, int | float]:
    """Report target recall/precision and false activations on other inputs."""
    head = model.net.output_layer
    device = next(head.parameters()).device
    with torch.no_grad():
        predictions = head(features.to(device)).argmax(dim=1).cpu().numpy()

    target_mask = labels == target_label
    target_hits = int(np.sum(predictions[target_mask] == target_label))
    target_count = int(target_mask.sum())
    target_predictions = predictions == target_label
    target_false_positives = int(np.sum(target_predictions & ~target_mask))
    target_precision = (
        float(target_hits / int(target_predictions.sum()))
        if target_predictions.any()
        else 0.0
    )
    target_recall = float(target_hits / target_count) if target_count else float("nan")
    target_f1 = (
        2 * target_precision * target_recall / (target_precision + target_recall)
        if target_precision + target_recall
        else 0.0
    )

    class_results: dict[str, int | float] = {}
    other_gesture_recalls = []
    all_gesture_hits = 0
    all_gesture_count = 0
    for name, label in gesture_mapping.items():
        mask = labels == label
        count = int(mask.sum())
        hits = int(np.sum(predictions[mask] == label))
        class_results[f"test_{name}_correct"] = hits
        class_results[f"test_{name}_samples"] = count
        class_results[f"test_{name}_recall"] = float(hits / count) if count else float("nan")
        if label != gesture_mapping["noGesture"]:
            all_gesture_hits += hits
            all_gesture_count += count
            if label != target_label and count:
                other_gesture_recalls.append(hits / count)

    no_gesture_mask = labels == gesture_mapping["noGesture"]
    no_gesture_false_positives = int(np.sum(predictions[no_gesture_mask] != 0))
    no_gesture_count = int(no_gesture_mask.sum())
    adl_false_positives = int(np.sum(predictions[adl_mask] != 0))
    adl_count = int(adl_mask.sum())
    gesture_recall = all_gesture_hits / all_gesture_count if all_gesture_count else float("nan")

    return {
        **class_results,
        "target_recall": target_recall,
        "target_precision": target_precision,
        "target_f1": float(target_f1),
        "target_false_positives": target_false_positives,
        "target_test_samples": target_count,
        "other_gestures_macro_recall": (
            float(np.mean(other_gesture_recalls)) if other_gesture_recalls else float("nan")
        ),
        "gesture_false_negative_rate": 1.0 - gesture_recall,
        "no_gesture_false_positives": no_gesture_false_positives,
        "no_gesture_samples": no_gesture_count,
        "no_gesture_false_positive_rate": (
            no_gesture_false_positives / no_gesture_count if no_gesture_count else float("nan")
        ),
        "adl_false_positives": adl_false_positives,
        "adl_clips": adl_count,
        "adl_false_positive_rate": (
            adl_false_positives / adl_count if adl_count else float("nan")
        ),
        "test_accuracy": float(np.mean(predictions == labels)),
    }


def write_result(path: Path, row: dict[str, object]) -> None:
    """Append one result row, writing the CSV header when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def parse_integer_list(value: str, argument: str, *, allow_zero: bool) -> list[int]:
    try:
        numbers = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{argument} must be comma-separated integers") from error
    minimum = 0 if allow_zero else 1
    if not numbers or numbers[0] < minimum:
        requirement = "non-negative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(f"{argument} must contain {requirement} integers")
    return numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--participant-counts",
        default=",".join(map(str, PARTICIPANT_COUNTS)),
        help="Number of target-gesture contributors to evaluate.",
    )
    parser.add_argument(
        "--trials-per-participant",
        default=",".join(map(str, TRIALS_PER_PARTICIPANT)),
        help="Target trials taken from each selected participant.",
    )
    parser.add_argument(
        "--gestures",
        help="Comma-separated target gestures; default runs every gesture.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    try:
        participant_counts = parse_integer_list(
            args.participant_counts,
            "--participant-counts",
            allow_zero=False,
        )
        trials_per_participant = parse_integer_list(
            args.trials_per_participant,
            "--trials-per-participant",
            allow_zero=False,
        )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if trials_per_participant[-1] > 25:
        parser.error("--trials-per-participant cannot exceed the 25 available examples per person")

    requested_gestures = None
    if args.gestures:
        requested_gestures = [name.strip() for name in args.gestures.split(",") if name.strip()]
        valid_gestures = {name for name, label in gesture_mapping.items() if label != 0}
        if not requested_gestures or not set(requested_gestures).issubset(valid_gestures):
            parser.error(f"--gestures must be selected from: {', '.join(sorted(valid_gestures))}")

    output = args.output or Path("exp/results") / (
        f"gesture_data_variation_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    data = prepare_data(args.seed)

    train_participants = sorted(int(value) for value in np.unique(data.train_emg.subjects))
    validation_participants = sorted(
        int(value) for value in np.unique(data.validation_emg.subjects)
    )
    test_participants = sorted(int(value) for value in np.unique(data.test_emg.subjects))
    if participant_counts[-1] > len(train_participants):
        parser.error(
            f"Requested {participant_counts[-1]} target contributors, but only "
            f"{len(train_participants)} are in the training split."
        )

    print(f"Training gesture participants: {len(train_participants)}")
    print(f"Validation gesture participants: {len(validation_participants)}")
    print(f"Test gesture participants: {len(test_participants)}")
    print(f"Participant counts: {participant_counts}")
    print(f"Target trials per participant: {trials_per_participant}")
    print(f"Repeats: {args.repeats}")
    print(f"Results: {output}")

    # All non-target gesture and ADL examples stay fixed. Only target examples
    # are varied; each selected participant contributes the same quota.
    train_all = data.train_emg.combine(data.train_adl)
    validation_all = data.validation_emg.combine(data.validation_adl)
    test_all = data.test_emg.combine(data.test_adl)
    test_adl_mask = np.concatenate(
        [
            np.zeros(len(data.test_emg), dtype=bool),
            np.ones(len(data.test_adl), dtype=bool),
        ]
    )
    train_labels = torch.as_tensor(train_all.labels.astype(np.int64))
    validation_labels = torch.as_tensor(validation_all.labels.astype(np.int64))
    test_labels = test_all.labels.astype(np.int64)
    ordered_gestures = sorted(gesture_mapping.items(), key=lambda item: item[1])
    targets = [(name, label) for name, label in ordered_gestures if label != 0]
    if requested_gestures:
        targets = [target for target in targets if target[0] in requested_gestures]

    for repeat in range(args.repeats):
        participant_rng = np.random.default_rng(args.seed + repeat)
        participant_order = [
            int(value) for value in participant_rng.permutation(train_participants)
        ]

        for target_name, target_label in targets:
            other_gestures = [
                (name, label) for name, label in ordered_gestures if label != target_label
            ]
            base_label_map = {
                old_label: new_label
                for new_label, (_name, old_label) in enumerate(other_gestures)
            }
            base_train = remap_labels(train_all, base_label_map)
            base_validation = remap_labels(validation_all, base_label_map)
            customers = TrainData(
                emg=[f"user{subject}" for subject in train_participants],
                disco=[f"S{subject}" for subject in data.train_adl_participants],
            )

            base_seed = args.seed + repeat * 10000 + target_label
            seed_everything(base_seed, workers=True)
            base_config = DiscreteClassifierConfig.default_lite().create(
                n_classes=len(base_label_map),
                window_size=GESTURE_WINDOW_SIZE,
                increment=GESTURE_WINDOW_INCREMENT,
                gestures=[name for name, _label in other_gestures if name != "noGesture"],
            )
            model = DiscreteModel(base_config)
            model.fit(base_train, base_validation, customers=customers)

            train_features = encode_dataset(model, train_all)
            validation_features = encode_dataset(model, validation_all)
            test_features = encode_dataset(model, test_all)

            for participant_count in participant_counts:
                selected_participants = participant_order[:participant_count]
                target_rng = np.random.default_rng(
                    args.seed + repeat * 100000 + target_label
                )
                indices_by_participant: dict[int, list[int]] = {}
                for participant in selected_participants:
                    indices = np.flatnonzero(
                        (train_all.subjects == participant)
                        & (train_all.labels == target_label)
                    )
                    if len(indices) < trials_per_participant[-1]:
                        raise ValueError(
                            f"Participant {participant} has {len(indices)} examples for "
                            f"{target_name}; {trials_per_participant[-1]} are required."
                        )
                    indices_by_participant[participant] = target_rng.permutation(
                        indices
                    ).astype(int).tolist()

                for quota in trials_per_participant:
                    selected_target = {
                        index
                        for indices in indices_by_participant.values()
                        for index in indices[:quota]
                    }
                    training_indices = [
                        index
                        for index, label in enumerate(train_all.labels)
                        if int(label) != target_label or index in selected_target
                    ]
                    epoch, validation_recall = train_output_layer(
                        model,
                        train_features[training_indices],
                        train_labels[training_indices],
                        validation_features,
                        validation_labels,
                        seed=args.seed + repeat * 100000 + target_label * 1000 + participant_count,
                    )
                    metrics = score_model(
                        model,
                        test_features,
                        test_labels,
                        target_label,
                        test_adl_mask,
                    )
                    row: dict[str, object] = {
                        "repeat": repeat,
                        "participant_count": participant_count,
                        "target_participants": ";".join(map(str, selected_participants)),
                        "validation_participants": ";".join(map(str, validation_participants)),
                        "test_participants": ";".join(map(str, test_participants)),
                        "target_gesture": target_name,
                        "target_trials_per_participant": quota,
                        "target_train_trials": len(selected_target),
                        "head_non_target_trials": int(
                            np.sum(train_all.labels != target_label)
                        ),
                        "head_total_trials": len(training_indices),
                        "validation_macro_recall": validation_recall,
                        "head_best_epoch": epoch,
                        "seed": args.seed + repeat * 100000 + target_label * 1000 + participant_count,
                        **metrics,
                    }
                    write_result(output, row)
                    print(
                        f"repeat={repeat}; target={target_name}; "
                        f"people={participant_count}; trials/person={quota}; "
                        f"total target trials={len(selected_target)}; "
                        f"recall={metrics['target_recall']:.3f}"
                    )


if __name__ == "__main__":
    main()
