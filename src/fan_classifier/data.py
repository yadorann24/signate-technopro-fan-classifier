"""Private image manifests, explicit stratification, and image loading.

This module does not download data or infer missing location/brightness labels.
CSV validation stays usable without installing the training dependencies.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ImageRecord:
    """One image with the task's label and optional stratification category."""

    image_filename: str
    image_path: Path
    holding_label: int | None = None
    stratification_label: str | None = None
    fold_index: int | None = None
    split_seed: int | None = None
    split_count: int | None = None


_FOLD_COLUMNS = ("fold_index", "split_seed", "split_count")


def _read_rows(csv_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or any(not field or field != field.strip() for field in fields):
            raise ValueError("CSV requires a nonempty header without surrounding whitespace.")
        if len(fields) != len(set(fields)):
            raise ValueError("CSV contains duplicate column names.")
        if "image_filename" not in fields:
            raise ValueError("CSV requires the image_filename column.")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV row {row_number} has a different width from its header.")
            rows.append(row)
    if not rows:
        raise ValueError("CSV must contain at least one image.")
    return fields, rows


def _column_value(row: dict[str, str], canonical: str, legacy: str) -> str:
    names = [name for name in (canonical, legacy) if name in row]
    if not names:
        raise ValueError(f"Training CSV requires {canonical} (legacy alias: {legacy}).")
    values = [row[name].strip() for name in names]
    if len(set(values)) != 1:
        raise ValueError(f"Conflicting values in {canonical} and {legacy} columns.")
    return values[0]


def _validate_filename(filename: str) -> None:
    path = PurePosixPath(filename)
    if (
        not filename
        or filename != filename.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in filename)
        or "\\" in filename
        or ":" in filename
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != filename
        or filename == "."
    ):
        raise ValueError("image_filename must be a normalized relative path within image_root.")


def _validate_stratification_label(label: str) -> None:
    if (
        not label
        or label.lower() in {"nan", "none", "null", "na", "n/a", "inf", "+inf", "-inf"}
        or any(ord(character) < 32 or ord(character) == 127 for character in label)
    ):
        raise ValueError("stratification_label must be a nonempty, valid category.")


def _read_fold_values(row: dict[str, str]) -> tuple[int, int, int]:
    values = []
    for name in _FOLD_COLUMNS:
        value = row[name].strip()
        if not value or not value.isascii() or not value.isdecimal():
            raise ValueError(f"{name} must be a nonnegative integer.")
        values.append(int(value))
    fold_index, split_seed, split_count = values
    if split_count < 2:
        raise ValueError("split_count must be at least 2.")
    if not 0 <= split_seed < 2**32:
        raise ValueError("split_seed must be between 0 and 2**32 - 1.")
    if not 0 <= fold_index < split_count:
        raise ValueError("fold_index must be between 0 and split_count - 1.")
    return fold_index, split_seed, split_count


def _read_records(
    csv_path: str | Path, image_root: str | Path, *, training: bool
) -> list[ImageRecord]:
    fields, rows = _read_rows(csv_path)
    if training:
        for canonical, legacy in (("holding_label", "check"), ("stratification_label", "new_fold")):
            if canonical not in fields and legacy not in fields:
                raise ValueError(f"Training CSV requires {canonical} (legacy alias: {legacy}).")
    has_stored_folds = training and any(name in fields for name in _FOLD_COLUMNS)
    if has_stored_folds and not all(name in fields for name in _FOLD_COLUMNS):
        raise ValueError(
            "Stored folds require all three columns: fold_index, split_seed, split_count."
        )
    root = Path(image_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("image_root must be a directory.")
    records: list[ImageRecord] = []
    seen_filenames: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        filename = row["image_filename"]
        try:
            _validate_filename(filename)
            if filename in seen_filenames:
                raise ValueError("Duplicate image_filename; one row per image is required.")
            seen_filenames.add(filename)
            holding_label = None
            stratification_label = None
            fold_index = split_seed = split_count = None
            if training:
                label_value = _column_value(row, "holding_label", "check")
                if label_value not in {"0", "1"}:
                    raise ValueError("holding_label must be 0 (not-hold) or 1 (hold).")
                holding_label = int(label_value)
                stratification_label = _column_value(row, "stratification_label", "new_fold")
                _validate_stratification_label(stratification_label)
                if has_stored_folds:
                    fold_index, split_seed, split_count = _read_fold_values(row)
                image_path = root / ("hold" if holding_label == 1 else "not-hold") / filename
            else:
                image_path = root / filename
            resolved_path = image_path.resolve()
            if not resolved_path.is_relative_to(root):
                raise ValueError("Image path or symlink points outside image_root.")
            if not resolved_path.is_file():
                raise FileNotFoundError(f"Image does not exist: {image_path}")
        except ValueError as error:
            raise ValueError(f"CSV row {row_number}: {error}") from error
        records.append(
            ImageRecord(
                filename,
                resolved_path,
                holding_label,
                stratification_label,
                fold_index,
                split_seed,
                split_count,
            )
        )
    return records


def read_training_records(csv_path: str | Path, image_root: str | Path) -> list[ImageRecord]:
    """Read a private training CSV and verify all image paths before training.

    Canonical columns: image_filename, holding_label, stratification_label.
    Legacy aliases check and new_fold remain supported. new_fold is a category
    for StratifiedKFold, not a preassigned fold number. Labels map to hold/ and
    not-hold/ directories below image_root. Optional fold_index, split_seed and
    split_count columns must be supplied together to preserve prepared splits.
    """
    return _read_records(csv_path, image_root, training=True)


def read_inference_records(csv_path: str | Path, image_root: str | Path) -> list[ImageRecord]:
    """Read images directly below image_root, preserving CSV order and IDs."""
    return _read_records(csv_path, image_root, training=False)


def scan_labeled_images(image_root: str | Path) -> list[ImageRecord]:
    """Collect private training images from hold/ and not-hold/, without a CSV.

    IDs are paths relative to their class folder, sorted globally. Identical IDs
    across class folders are rejected, since manifests require unique image IDs.
    No image pixels are read here; the feature extractor validates decoding.
    """
    root = Path(image_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("image_root must be a directory.")
    records = []
    seen_filenames: set[str] = set()
    for class_name, holding_label in (("not-hold", 0), ("hold", 1)):
        class_directory = root / class_name
        if not class_directory.is_dir():
            raise ValueError(f"Missing training class directory: {class_name}/.")
        resolved_class = class_directory.resolve()
        if not resolved_class.is_relative_to(root):
            raise ValueError(f"Class directory {class_name}/ points outside image_root.")
        class_count = 0
        for image_path in sorted(class_directory.rglob("*")):
            resolved_path = image_path.resolve()
            if not resolved_path.is_relative_to(resolved_class):
                raise ValueError("Image path or symlink points outside its class directory.")
            if image_path.is_symlink() and image_path.is_dir():
                raise ValueError(
                    "Symlinked image subdirectories are not supported; use real folders."
                )
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if not image_path.is_file():
                if image_path.is_symlink():
                    raise ValueError(
                        f"Image symlink has no readable file target: {image_path.name}"
                    )
                continue
            filename = image_path.relative_to(class_directory).as_posix()
            _validate_filename(filename)
            if filename in seen_filenames:
                raise ValueError(
                    "Duplicate image_filename across class folders; IDs must be unique."
                )
            seen_filenames.add(filename)
            records.append(ImageRecord(filename, resolved_path, holding_label))
            class_count += 1
        if not class_count:
            raise ValueError(
                f"Training class {class_name}/ contains no .jpg, .jpeg or .png images."
            )
    return sorted(records, key=lambda record: record.image_filename)


def split_records(
    records: Sequence[ImageRecord], *, fold_count: int, fold_index: int, seed: int
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Reproduce stratified splitting from supplied location/brightness categories.

    Prepared fold assignments are retained exactly after validating metadata and
    balance. Without assignments, StratifiedKFold creates them from categories.
    Samples in a category can appear in different folds. This is not GroupKFold
    and does not establish subject, video, or location independence.
    """
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError("fold_count must be an integer of at least 2.")
    if (
        isinstance(fold_index, bool)
        or not isinstance(fold_index, int)
        or not 0 <= fold_index < fold_count
    ):
        raise ValueError("fold_index must be between 0 and fold_count - 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    if not records:
        raise ValueError("Cannot split an empty manifest.")
    if len({record.image_filename for record in records}) != len(records):
        raise ValueError("Cannot split duplicate image_filename values.")
    labels: list[str] = []
    for record in records:
        if record.holding_label not in (0, 1) or record.stratification_label is None:
            raise ValueError("Every training record needs holding_label and stratification_label.")
        _validate_stratification_label(record.stratification_label)
        labels.append(record.stratification_label)
    category_counts = Counter(labels)
    insufficient = sum(count < fold_count for count in category_counts.values())
    if insufficient:
        smallest = min(category_counts.values())
        raise ValueError(
            f"{insufficient} stratification categories contain fewer than {fold_count} images "
            f"(minimum: {smallest}). Revise the private stratification manifest or fold_count. "
            "No fallback to label-only splitting is performed."
        )
    has_stored_folds = any(
        getattr(record, name) is not None for record in records for name in _FOLD_COLUMNS
    )
    if has_stored_folds:
        fold_counts: Counter[int] = Counter()
        strata_fold_counts: dict[str, Counter[int]] = {}
        for record in records:
            values = tuple(getattr(record, name) for name in _FOLD_COLUMNS)
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ValueError("Every record must contain complete integer stored fold metadata.")
            if record.split_seed != seed or record.split_count != fold_count:
                raise ValueError(
                    "Stored split_seed/split_count do not match the requested seed/fold_count. "
                    "Use matching settings or explicitly prepare a new manifest."
                )
            if not 0 <= record.fold_index < fold_count:
                raise ValueError("Stored fold_index must be between 0 and fold_count - 1.")
            fold_counts[record.fold_index] += 1
            strata_fold_counts.setdefault(record.stratification_label, Counter())[
                record.fold_index
            ] += 1
        if set(fold_counts) != set(range(fold_count)):
            raise ValueError("Stored assignments must include every fold_index.")
        for category, counts in strata_fold_counts.items():
            sizes = [counts[index] for index in range(fold_count)]
            if max(sizes) - min(sizes) > 1:
                raise ValueError(
                    f"Stored folds are not balanced within stratification category {category}."
                )
        if max(fold_counts.values()) - min(fold_counts.values()) > 1:
            raise ValueError("Stored folds have unbalanced total image counts.")
        return (
            [record for record in records if record.fold_index != fold_index],
            [record for record in records if record.fold_index == fold_index],
        )
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as error:
        raise RuntimeError(
            "Splitting requires scikit-learn; install the training dependencies."
        ) from error
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    for index, (training_indices, validation_indices) in enumerate(
        splitter.split(range(len(records)), labels)
    ):
        if index == fold_index:
            return (
                [records[int(i)] for i in training_indices],
                [records[int(i)] for i in validation_indices],
            )
    raise RuntimeError("Requested fold was not produced.")


class FanHoldingDataset:
    """A PyTorch map-style dataset with explicit IDs and holding labels.

    Dataset implements __len__/__getitem__ without a torch import at module load.
    Use build_transforms() for model input. Without a transform, items are RGB
    float tensors in CHW order scaled to [0, 1], without ImageNet normalization.
    """

    def __init__(
        self, records: Sequence[ImageRecord], transform: Callable[..., dict[str, Any]] | None = None
    ) -> None:
        self.records = tuple(records)
        self.transform = transform
        label_presence = {record.holding_label is not None for record in self.records}
        if len(label_presence) > 1:
            raise ValueError("Dataset cannot mix labeled and unlabeled images.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import cv2

        record = self.records[index]
        bgr_image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise ValueError(f"Image could not be decoded: {record.image_path}")
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=rgb_image)["image"]
        else:
            import torch

            image = torch.from_numpy(rgb_image).permute(2, 0, 1).float().div(255.0)
        sample: dict[str, Any] = {"image": image, "image_filename": record.image_filename}
        if record.holding_label is not None:
            sample["holding_label"] = record.holding_label
        return sample
