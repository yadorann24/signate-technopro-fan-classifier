"""Checkpoint inference and image-ID-aligned probability averaging.

CSV ensembling needs only the standard library. Fold probabilities are averaged
within each model before the per-model CSVs are combined with equal weight.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .normalization import NormalizationConfig, restore_normalization

PROBABILITY_COLUMNS = ["image_filename", "holding_probability"]


@dataclass(frozen=True)
class PredictionResult:
    probability_path: Path
    submission_path: Path
    metadata_path: Path


def _check_probability(value: Any, image_filename: str) -> float:
    if not isinstance(image_filename, str) or not image_filename.strip():
        raise ValueError("image_filename must be a nonempty string.")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability for {image_filename!r} must be finite and within [0, 1].")
    return probability


def _canonical_config(config: Mapping[str, Any] | None) -> str | None:
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint training config must be a JSON object.")
    try:
        # Tuple/list normalization matches configurations read from run.json.
        return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint training config must contain valid JSON values.") from error


def read_probabilities(path: str | Path) -> dict[str, float]:
    predictions: dict[str, float] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != PROBABILITY_COLUMNS:
            raise ValueError(f"{path}: expected CSV columns {PROBABILITY_COLUMNS}.")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV row.")
            image_filename = row["image_filename"]
            if image_filename in predictions:
                raise ValueError(f"{path}: duplicate image_filename {image_filename!r}.")
            predictions[image_filename] = _check_probability(
                row["holding_probability"], image_filename
            )
    if not predictions:
        raise ValueError(f"{path}: the probability CSV is empty.")
    return predictions


def _average_predictions(members: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not members or not members[0]:
        raise ValueError("At least one nonempty set of predictions is required.")
    image_ids = set(members[0])
    for member in members:
        if set(member) != image_ids:
            raise ValueError("Prediction image IDs differ; refusing to average misaligned inputs.")
    return {
        image_id: math.fsum(_check_probability(member[image_id], image_id) for member in members)
        / len(members)
        for image_id in members[0]
    }


def write_prediction_files(
    predictions: Mapping[str, float],
    output_directory: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PredictionResult:
    if not predictions:
        raise ValueError("Cannot write empty predictions.")
    checked = {
        name: _check_probability(probability, name) for name, probability in predictions.items()
    }
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    probability_path = output_directory / "probabilities.csv"
    submission_path = output_directory / "submission.csv"
    metadata_path = output_directory / "prediction.json"
    with probability_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(PROBABILITY_COLUMNS)
        writer.writerows(checked.items())
    with submission_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        # Emit class 1 only above 0.5, with no header in the submission CSV.
        writer.writerows((name, int(probability > 0.5)) for name, probability in checked.items())
    metadata_path.write_text(
        json.dumps(
            {
                **dict(metadata or {}),
                "sample_count": len(checked),
                "threshold": 0.5,
                "threshold_operator": ">",
                "positive_class_index": 1,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return PredictionResult(probability_path, submission_path, metadata_path)


def ensemble_probability_files(
    input_paths: Sequence[str | Path], output_directory: str | Path
) -> PredictionResult:
    """Average model probabilities after exact image-ID matching.

    Sidecars produced by this package identify model members. Duplicate members
    are rejected when that provenance is available; arbitrary external CSVs do
    not carry enough information to determine their model identities.
    """
    paths = [Path(path).resolve() for path in input_paths]
    if len(paths) < 2:
        raise ValueError("Model ensembling requires at least two probability CSV files.")
    if len(set(paths)) != len(paths):
        raise ValueError("The same probability file cannot be included twice.")
    predictions = []
    known_models: set[str] = set()
    for path in paths:
        predictions.append(read_probabilities(path))
        sidecar = path.parent / "prediction.json"
        if sidecar.exists():
            provenance = json.loads(sidecar.read_text(encoding="utf-8"))
            if provenance.get("operation") == "ensemble":
                raise ValueError(
                    "Use per-model probability CSVs, not previously averaged ensembles."
                )
            model_name = provenance.get("model_name")
            if model_name is not None:
                if not isinstance(model_name, str):
                    raise ValueError("Model identity in prediction metadata must be a string.")
                if model_name in known_models:
                    raise ValueError(
                        f"Model {model_name!r} appears more than once in the ensemble."
                    )
                known_models.add(model_name)
    averaged = _average_predictions(predictions)
    return write_prediction_files(
        averaged,
        output_directory,
        metadata={
            "operation": "ensemble",
            "aggregation": "arithmetic_mean",
            "input_paths": [str(path) for path in paths],
            "known_model_names": sorted(known_models),
        },
    )


def predict_checkpoints(
    checkpoint_paths: Sequence[str | Path],
    loader_factory: Callable[[int, NormalizationConfig], Iterable],
    *,
    output_directory: str | Path,
    device: str = "auto",
    expected_folds: Sequence[int] | None = None,
    expected_model: str | None = None,
    expected_training_csv_sha256: str | None = None,
    expected_config: Mapping[str, Any] | None = None,
) -> PredictionResult:
    """Load compatible folds of one model and average positive-class probability.

    Only this project's structured checkpoints are accepted. The model is
    created with pretrained=False, so inference cannot fetch pretrained weights.
    loader_factory receives image_size and the exact saved normalization. Older
    checkpoints without normalization retain their original ImageNet settings.
    """
    import torch

    from .engine import (
        CHECKPOINT_FORMAT,
        CLASS_MAPPING,
        _check_logits,
        resolve_device,
        validate_training_csv_sha256,
    )
    from .registry import build_model, get_model_spec

    expected_model_alias = (
        get_model_spec(expected_model).alias if expected_model is not None else None
    )
    validate_training_csv_sha256(expected_training_csv_sha256)
    canonical_expected_config = _canonical_config(expected_config)

    paths = [Path(path).resolve() for path in checkpoint_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide at least one checkpoint without duplicate paths.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained fold checkpoint: {path}")
    if expected_folds is not None and (
        not expected_folds
        or len(set(expected_folds)) != len(expected_folds)
        or any(type(fold) is not int or fold < 0 for fold in expected_folds)
    ):
        raise ValueError("expected_folds must contain distinct nonnegative fold indices.")
    if Path(output_directory).exists():
        raise FileExistsError(f"Output directory already exists: {output_directory}")
    selected_device = resolve_device(device)
    members: list[dict[str, float]] = []
    seen_folds: set[int] = set()
    model_name = None
    common_signature = None
    for path in paths:
        # weights_only avoids executing arbitrary pickle payloads in checkpoints.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (
            not isinstance(checkpoint, dict)
            or not isinstance(checkpoint.get("metadata"), dict)
            or not isinstance(checkpoint.get("state_dict"), dict)
        ):
            raise ValueError(f"{path}: not a structured project checkpoint.")
        metadata = checkpoint["metadata"]
        if metadata.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: unsupported checkpoint format.")
        if (
            metadata.get("num_classes") != 2
            or metadata.get("positive_class_index") != 1
            or metadata.get("class_mapping") != CLASS_MAPPING
        ):
            raise ValueError(f"{path}: incompatible class labels in checkpoint.")
        checkpoint_model_name = metadata.get("model_name")
        if not isinstance(checkpoint_model_name, str) or not checkpoint_model_name:
            raise ValueError(f"{path}: model_name is missing or invalid in checkpoint metadata.")
        spec = get_model_spec(checkpoint_model_name)
        if expected_model_alias is not None and spec.alias != expected_model_alias:
            raise ValueError(f"{path}: checkpoint model does not match the expected trained run.")
        image_size = metadata.get("image_size")
        if type(image_size) is not int or image_size != spec.image_size:
            raise ValueError(f"{path}: image size does not match the registered model.")
        fold = metadata.get("fold")
        num_folds = metadata.get("num_folds")
        if (
            type(fold) is not int
            or fold < 0
            or (
                num_folds is not None
                and (type(num_folds) is not int or num_folds < 2 or fold >= num_folds)
            )
        ):
            raise ValueError(f"{path}: invalid fold metadata.")
        training_csv_sha256 = metadata.get("training_csv_sha256")
        validate_training_csv_sha256(training_csv_sha256)
        if (
            expected_training_csv_sha256 is not None
            and training_csv_sha256 != expected_training_csv_sha256
        ):
            raise ValueError(
                f"{path}: checkpoint training CSV hash does not match the expected trained run."
            )
        canonical_config = _canonical_config(metadata.get("config"))
        if canonical_expected_config is not None and canonical_config != canonical_expected_config:
            raise ValueError(
                f"{path}: checkpoint training config does not match the expected trained run."
            )
        saved_normalization = (metadata.get("config") or {}).get("normalization")
        normalization = restore_normalization(saved_normalization)
        if "normalization" in metadata:
            recorded_normalization = restore_normalization(metadata["normalization"])
            if recorded_normalization != normalization:
                raise ValueError("Checkpoint normalization conflicts with its training config")
        signature = (
            spec.alias,
            image_size,
            num_folds,
            metadata.get("seed"),
            training_csv_sha256,
            canonical_config,
            normalization,
        )
        if common_signature is not None and signature != common_signature:
            raise ValueError(
                "Checkpoints must share model, image size, split count, split seed, training CSV hash, and training config."
            )
        common_signature = signature
        model_name = spec.alias
        if fold in seen_folds:
            raise ValueError(f"Fold {fold} appears more than once.")
        seen_folds.add(fold)
        model = build_model(spec.alias, pretrained=False, num_classes=2)
        # Inference has no optimizer references: assigning mapped tensors avoids
        # making another full copy of the large EVA checkpoint in host memory.
        model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
        del checkpoint
        model.to(selected_device)
        model.eval()
        predictions: dict[str, float] = {}
        with torch.inference_mode():
            for batch in loader_factory(image_size, normalization):
                images = batch["image"].to(device=selected_device, dtype=torch.float32)
                filenames = batch["image_filename"]
                if isinstance(filenames, str) or len(filenames) != len(images):
                    raise ValueError(
                        "The inference batch must contain one image_filename per image."
                    )
                with torch.amp.autocast(
                    device_type=selected_device.type, enabled=selected_device.type == "cuda"
                ):
                    logits = model(images)
                    _check_logits(logits, len(images))
                probabilities = logits.float().softmax(dim=1)[:, 1].cpu().tolist()
                for filename, probability in zip(filenames, probabilities):
                    if filename in predictions:
                        raise ValueError(f"Duplicate inference image_filename: {filename!r}.")
                    predictions[filename] = _check_probability(probability, filename)
        if not predictions:
            raise ValueError("The inference loader contains no images.")
        members.append(predictions)
        del model
    if expected_folds is not None and seen_folds != set(expected_folds):
        raise ValueError(
            f"Expected trained folds {sorted(expected_folds)}, found {sorted(seen_folds)}."
        )
    return write_prediction_files(
        _average_predictions(members),
        output_directory,
        metadata={
            "operation": "predict",
            "aggregation": "fold_arithmetic_mean",
            "model_name": model_name,
            "folds": sorted(seen_folds),
            "checkpoints": [str(path) for path in paths],
            "normalization": normalization.to_dict(),
        },
    )
