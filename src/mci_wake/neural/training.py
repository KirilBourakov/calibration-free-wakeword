import os
from dataclasses import fields
from datetime import datetime
from typing import Any, cast

import numpy as np
import yaml
from torch.utils.data import DataLoader

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from mci_wake.data.types import EmgDataset, gesture_mapping
from mci_wake.neural.classifier import (
    DiscreteClassifier,
    DiscreteClassifierConfig,
    TrainData,
    make_data_loader,
)
from mci_wake.neural.lightning_module import DiscreteLightningModule


def train_model(
    train: EmgDataset,
    test: EmgDataset,
    model_config: DiscreteClassifierConfig = DiscreteClassifierConfig(),
    customers: TrainData | None = None,
) -> DiscreteClassifier:
    """Initializes and trains the DiscreteClassifier using the provided datasets.

    Args:
        train: EmgDataset
        test: EmgDataset
        model_config: the model configuration
        customers: Optional list of customer/user IDs used for training.

    Returns:
        DiscreteClassifier: The trained classifier instance.
    """
    if customers is not None:
        model_config.customers = customers

    print("Fitting Discrete Classifier...")
    tr_dl: DataLoader = make_data_loader(train.data, train.labels)
    te_dl: DataLoader = make_data_loader(test.data, test.labels)

    model = DiscreteLightningModule(model_config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        filename="best-model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
        verbose=True,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
    )

    # Save training metadata after training completes so metrics & checkpoints are populated
    save_dir = trainer.log_dir or checkpoint_callback.dirpath or "."
    os.makedirs(save_dir, exist_ok=True)

    metadata_text, metadata_dict = _generate_training_metadata(
        train=train,
        test=test,
        model_config=model_config,
        trainer=trainer,
        checkpoint_callback=checkpoint_callback,
    )

    # 1. Save human-readable text
    metadata_path = os.path.join(save_dir, "training_metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(metadata_text)
    print(f"Training metadata saved to: {metadata_path}")

    # 2. Save structured YAML for experiment tracking/tools
    yaml_path = os.path.join(save_dir, "metadata.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata_dict, f, sort_keys=False)
    print(f"Structured metadata saved to: {yaml_path}")

    trainer.fit(model, train_dataloaders=tr_dl, val_dataloaders=te_dl)



    return cast(DiscreteLightningModule, trainer.lightning_module).internals


def _generate_training_metadata(
    train: EmgDataset,
    test: EmgDataset,
    model_config: DiscreteClassifierConfig,
    trainer: Trainer,
    checkpoint_callback: ModelCheckpoint,
) -> tuple[str, dict]:
    """Generates a detailed summary of training metadata and metrics.

    Returns:
        A tuple containing the formatted text report and the raw structured dictionary.
    """
    md = _collect_metadata(train, test, model_config, trainer, checkpoint_callback)
    return _render_text(md), md


# =====================================================================
# Private Helper Functions
# =====================================================================

def _collect_metadata(
    train: EmgDataset,
    test: EmgDataset,
    model_config: DiscreteClassifierConfig,
    trainer: Trainer,
    ckpt: ModelCheckpoint,
) -> dict:
    """Extracts all training metadata into a structured dictionary."""
    best_score = None
    if ckpt.best_model_score is not None:
        val = ckpt.best_model_score.item() if hasattr(ckpt.best_model_score, "item") else ckpt.best_model_score
        best_score = float(val)

    return {
        "training_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": {
            "classification_mode": "Binary" if model_config.n_classes == 2 else "Multi-class",
            "n_classes": model_config.n_classes,
            "target_gestures": _resolve_gestures(model_config),
        },
        "dataset": {
            "train": _dataset_info(train),
            "validation": _dataset_info(test),
        },
        "model": _model_config_dict(model_config),
        "training": {"max_epochs": trainer.max_epochs},
        "results": {
            "best_checkpoint_path": ckpt.best_model_path,
            "best_monitor_score": best_score,
            "final_metrics": _format_metrics(trainer.callback_metrics),
        },
    }


def _render_text(md: dict, width: int = 60) -> str:
    """Renders the metadata dictionary into a human-readable string report."""

    def section(title: str, items: dict) -> list[str]:
        lines = [f"--- {title} ---"]
        for k, v in items.items():
            if isinstance(v, dict):
                # Flatten nested dicts (like metrics or class distribution) for text view
                nested_str = ", ".join(f"{k2}: {v2}" for k2, v2 in v.items())
                lines.append(f"  {k}: {nested_str}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")
        return lines

    lines = [
        "=" * width,
        "TRAINING METADATA REPORT".center(width),
        "=" * width,
        f"Training Date / Time: {md['training_datetime']}",
        "",
        *section("Task & Gestures", md["task"]),
        *section("Train Dataset", md["dataset"]["train"]),
        *section("Validation Dataset", md["dataset"]["validation"]),
        *section("Model & Hyperparameters", {**md["model"], **md["training"]}),
        *section("Results & Checkpoints", md["results"]),
        "=" * width,
    ]
    return "\n".join(lines) + "\n"


def _dataset_info(ds: EmgDataset) -> dict:
    """Extracts standard dataset statistics."""
    labels, counts = np.unique(ds.labels, return_counts=True)
    subjects = sorted({int(s) for s in ds.subjects}) if len(ds.subjects) > 0 else []
    return {
        "n_samples": len(ds),
        "class_distribution": dict(zip(labels.tolist(), counts.tolist())),
        "n_subjects": len(subjects),
        "subject_ids": subjects,
        "is_normalized": bool(ds.is_normalized),
    }


def _model_config_dict(cfg: DiscreteClassifierConfig) -> dict:
    """Dynamically pulls all fields from the dataclass config."""
    skip = {"customers"}  # Fields you don't want printed
    return {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name not in skip}


def _resolve_gestures(model_config: DiscreteClassifierConfig) -> str:
    """Resolves the target gesture string based on config."""
    if model_config.gestures:
        return ", ".join(model_config.gestures)
    elif model_config.n_classes == 2:
        return "Binary (0: Background / Other gestures, 1: Target Gesture)"
    else:
        return ", ".join([f"{k} ({v})" for k, v in gesture_mapping.items()])


def _format_metrics(metrics) -> dict[str, float | str]:
    """Safely formats PyTorch/Lightning metric tensors into native Python types."""
    out = {}
    for k, v in metrics.items():
        if hasattr(v, "item"):
            v = v.item()
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = str(v)
    return out