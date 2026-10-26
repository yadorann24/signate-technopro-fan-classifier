"""Shared training loop for the competition's two-class image models."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

CHECKPOINT_FORMAT = "fan-classifier-checkpoint-v1"
CLASS_MAPPING = {"0": "not-hold", "1": "hold"}


def validate_training_csv_sha256(value: str | None) -> None:
    """Accept a SHA-256 hex digest or absent provenance in older unit fixtures."""
    if value is not None and (
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError("training_csv_sha256 must be a lowercase SHA-256 hex digest.")


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    f1: float
    sample_count: int


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    metrics_path: Path
    metadata_path: Path
    best_epoch: int
    best_score: float
    oof_path: Path | None = None


def seed_everything(seed: int) -> None:
    """Call before model construction as well as before each fold's training."""
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "auto"
        else torch.device(device)
    )
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu, cuda, cuda:N, and auto.")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return resolved


def _check_logits(logits: torch.Tensor, batch_size: int) -> None:
    if logits.shape != (batch_size, 2):
        raise ValueError(f"Expected logits of shape ({batch_size}, 2), got {tuple(logits.shape)}.")
    if not torch.isfinite(logits).all():
        raise ValueError("The model produced non-finite logits.")


def _run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    sample_count = correct_count = true_positive = false_positive = false_negative = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            raw_labels = batch["holding_label"]
            if raw_labels.ndim != 1 or not ((raw_labels == 0) | (raw_labels == 1)).all():
                raise ValueError("holding_label must be a one-dimensional batch of 0/1 labels.")
            labels = raw_labels.to(device=device, dtype=torch.long)
            batch_size = labels.numel()
            if batch_size == 0:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                _check_logits(logits, batch_size)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("The model produced a non-finite loss.")
            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            predictions = logits.detach().argmax(dim=1)
            loss_sum += loss.detach().item() * batch_size
            correct_count += (predictions == labels).sum().item()
            true_positive += ((predictions == 1) & (labels == 1)).sum().item()
            false_positive += ((predictions == 1) & (labels == 0)).sum().item()
            false_negative += ((predictions == 0) & (labels == 1)).sum().item()
            sample_count += batch_size
    if sample_count == 0:
        raise ValueError("A training/validation loader must contain at least one sample.")
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return EpochMetrics(
        loss=loss_sum / sample_count,
        accuracy=correct_count / sample_count,
        f1=2 * true_positive / f1_denominator if f1_denominator else 0.0,
        sample_count=sample_count,
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    device="auto",
    scaler=None,
) -> EpochMetrics:
    return _run_epoch(
        model, loader, device=resolve_device(device), optimizer=optimizer, scaler=scaler
    )


def evaluate(model: nn.Module, loader: Iterable, *, device="auto") -> EpochMetrics:
    """Return sample-weighted loss/accuracy and positive-class binary F1."""
    return _run_epoch(model, loader, device=resolve_device(device))


def fit_model(
    model: nn.Module,
    train_loader: Iterable,
    validation_loader: Iterable,
    *,
    model_name: str,
    image_size: int,
    fold: int,
    output_directory: str | Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.001,
    scheduler_t_max: int | None = None,
    scheduler_eta_min: float = 1e-6,
    selection_metric: str = "loss",
    device: str = "auto",
    seed: int = 42,
    num_folds: int | None = None,
    training_csv_sha256: str | None = None,
    config: Mapping[str, Any] | None = None,
    fold_manifest: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train one fold; existing run directories are never overwritten.

    The checkpoint contains inference weights and provenance, not optimizer state
    for resuming an interrupted run. Dataset-dependent artifacts stay in the run
    directory and must not be published.
    """
    from .normalization import restore_normalization

    normalization = restore_normalization((config or {}).get("normalization"))
    validate_training_csv_sha256(training_csv_sha256)
    if type(fold) is not int or epochs < 1 or image_size < 1 or fold < 0:
        raise ValueError("epochs/image_size must be positive and fold must be nonnegative.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    if not math.isfinite(scheduler_eta_min) or not 0 <= scheduler_eta_min <= learning_rate:
        raise ValueError("scheduler_eta_min must lie between zero and learning_rate.")
    if selection_metric not in {"loss", "accuracy", "f1"}:
        raise ValueError("selection_metric must be loss, accuracy, or f1.")
    if num_folds is not None and (num_folds < 2 or fold >= num_folds):
        raise ValueError("fold must be smaller than num_folds (at least 2).")
    if scheduler_t_max is None:
        scheduler_t_max = epochs
    if scheduler_t_max < 1:
        raise ValueError("scheduler_t_max must be positive.")
    selected_device = resolve_device(device)
    seed_everything(seed)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_directory / "best.pt"
    metrics_path = output_directory / "metrics.csv"
    metadata_path = output_directory / "training.json"
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "model_name": model_name,
        "image_size": image_size,
        "fold": fold,
        "num_folds": num_folds,
        "seed": seed,
        "training_csv_sha256": training_csv_sha256,
        "num_classes": 2,
        "class_mapping": CLASS_MAPPING,
        "positive_class_index": 1,
        "selection_metric": selection_metric,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler_t_max": scheduler_t_max,
        "scheduler_eta_min": scheduler_eta_min,
        "device": str(selected_device),
        "torch_version": str(torch.__version__),
        "config": dict(config or {}),
        "normalization": normalization.to_dict(),
        "status": "running",
    }

    def save_metadata() -> None:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    save_metadata()
    (output_directory / "fold_manifest.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "num_folds": num_folds,
                "seed": seed,
                "split": dict(fold_manifest or {}),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    best_score = math.inf if selection_metric == "loss" else -math.inf
    best_epoch = 0
    oof_path = None
    fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_f1",
        "train_samples",
        "validation_loss",
        "validation_accuracy",
        "validation_f1",
        "validation_samples",
    ]
    try:
        model.to(selected_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_t_max, eta_min=scheduler_eta_min
        )
        scaler = torch.amp.GradScaler("cuda", enabled=selected_device.type == "cuda")
        with metrics_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                current_learning_rate = optimizer.param_groups[0]["lr"]
                training_metrics = train_one_epoch(
                    model, train_loader, optimizer, device=selected_device, scaler=scaler
                )
                validation_metrics = evaluate(model, validation_loader, device=selected_device)
                score = getattr(validation_metrics, selection_metric)
                improved = score < best_score if selection_metric == "loss" else score > best_score
                if improved:
                    checkpoint = {
                        "metadata": {
                            **metadata,
                            "status": "checkpoint",
                            "best_epoch": epoch,
                            "best_score": score,
                            "validation_metrics": asdict(validation_metrics),
                        },
                        "state_dict": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                    }
                    temporary_checkpoint = output_directory / ".best.pt.tmp"
                    try:
                        torch.save(checkpoint, temporary_checkpoint)
                        temporary_checkpoint.replace(checkpoint_path)
                        # Record only the best weights successfully persisted.
                        best_score, best_epoch = score, epoch
                    finally:
                        # CUDA weights were copied to CPU only for serialization.
                        # Release them before the next epoch or checkpoint copy.
                        del checkpoint
                writer.writerow(
                    {
                        "epoch": epoch,
                        "learning_rate": current_learning_rate,
                        "train_loss": training_metrics.loss,
                        "train_accuracy": training_metrics.accuracy,
                        "train_f1": training_metrics.f1,
                        "train_samples": training_metrics.sample_count,
                        "validation_loss": validation_metrics.loss,
                        "validation_accuracy": validation_metrics.accuracy,
                        "validation_f1": validation_metrics.f1,
                        "validation_samples": validation_metrics.sample_count,
                    }
                )
                stream.flush()
                print(
                    f"fold={fold} epoch={epoch}/{epochs} train_loss={training_metrics.loss:.6f} validation_loss={validation_metrics.loss:.6f} validation_accuracy={validation_metrics.accuracy:.6f} validation_f1={validation_metrics.f1:.6f}",
                    flush=True,
                )
                scheduler.step()
        if fold_manifest is not None:
            from .oof import save_fold_oof

            # Free optimizer references before restoring the persisted best weights.
            del scheduler, optimizer, scaler
            oof_path = save_fold_oof(
                model,
                validation_loader,
                checkpoint_path,
                split=fold_manifest,
                device=selected_device,
            )
        metadata.update(status="complete", best_epoch=best_epoch, best_score=best_score)
    except Exception:
        metadata.update(
            status="failed", best_epoch=best_epoch, best_score=best_score if best_epoch else None
        )
        save_metadata()
        raise
    save_metadata()
    return TrainingResult(
        checkpoint_path, metrics_path, metadata_path, best_epoch, best_score, oof_path
    )
