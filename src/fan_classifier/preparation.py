"""Prepare private label × scene × brightness folds from labeled training images."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Sequence

from .config import resolve_training_config
from .data import ImageRecord, _column_value, _read_rows, _validate_filename, scan_labeled_images
from .stratification import (
    build_triple_stratification,
    cluster_scene_features,
    quantile_brightness_thresholds,
)


@dataclass(frozen=True)
class PreparationResult:
    training_csv: Path
    summary_path: Path


def read_scene_clusters(csv_path: str | Path, records: Sequence[ImageRecord]):
    """Align previously computed scene clusters by exact image ID, never row order."""
    import numpy as np

    fields, rows = _read_rows(csv_path)
    if "scene_cluster" not in fields and "cluster" not in fields:
        raise ValueError("Cluster CSV requires scene_cluster (legacy alias: cluster).")
    by_filename: dict[str, int] = {}
    for row in rows:
        filename = row["image_filename"]
        _validate_filename(filename)
        if filename in by_filename:
            raise ValueError("Cluster CSV contains duplicate image_filename values.")
        value = _column_value(row, "scene_cluster", "cluster")
        if not value.isascii() or not value.isdecimal():
            raise ValueError("scene_cluster must contain nonnegative integers.")
        try:
            cluster = int(value)
        except ValueError as error:
            raise ValueError("scene_cluster must contain nonnegative integers.") from error
        if not 0 <= cluster < 2**31:
            raise ValueError("scene_cluster must be an integer in [0, 2**31).")
        by_filename[filename] = cluster
    expected = {record.image_filename for record in records}
    if set(by_filename) != expected:
        raise ValueError(
            "Cluster CSV image IDs must exactly match these training images "
            f"(missing={len(expected - set(by_filename))}, "
            f"extra={len(set(by_filename) - expected)})."
        )
    return np.asarray([by_filename[r.image_filename] for r in records], dtype=np.int64)


def _brightness_means(records: Sequence[ImageRecord]):
    import cv2
    import numpy as np

    means = np.empty(len(records), dtype=np.float64)
    for index, record in enumerate(records):
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Image could not be decoded: {record.image_path}")
        # A mean over all three channels is invariant to RGB/BGR channel order.
        means[index] = image.mean(dtype=np.float64)
    return means


def _environment() -> dict:
    packages = {}
    for package in ("numpy", "scikit-learn", "opencv-python-headless", "tensorflow-cpu", "keras"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def prepare_stratification(
    image_root: str | Path,
    output_directory: str | Path,
    *,
    model_name: str,
    brightness_bins: int | None = None,
    brightness_thresholds: Sequence[float] | None = None,
    num_folds: int | None = None,
    seed: int | None = None,
    cluster_seed: int = 42,
    pca_components: int = 50,
    cluster_count: int = 3,
    batch_size: int = 16,
    clusters_csv: str | Path | None = None,
) -> PreparationResult:
    """Write a directly trainable CSV and inspectable, private preparation metadata.

    Default brightness thresholds are quantiles of these training images' mean
    brightness, targeting three similarly sized groups. Equal values remain in
    the same group. Explicit thresholds are an alternative. The split seed and fold
    count come from the chosen classifier; clustering uses a separate seed.
    An existing cluster CSV avoids repeated VGG/PCA/K-means computation when
    preparing a different classifier's folds. All generated files are private.
    """
    import numpy as np

    config = resolve_training_config(model_name, seed=seed, num_folds=num_folds)
    if brightness_thresholds is not None and brightness_bins is not None:
        raise ValueError("Choose brightness_bins or brightness_thresholds, not both.")
    if brightness_thresholds is None:
        bin_count = 3 if brightness_bins is None else brightness_bins
        if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 2:
            raise ValueError("brightness_bins must be an integer of at least 2.")
        thresholds = None
        brightness_strategy = "quantile"
    else:
        values = np.asarray(brightness_thresholds)
        if (
            values.ndim != 1
            or not values.size
            or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 255))
            or np.any(np.diff(values.astype(float)) <= 0)
        ):
            raise ValueError(
                "brightness_thresholds must be strictly increasing finite values in [0, 255]."
            )
        thresholds = tuple(float(value) for value in values)
        bin_count = len(thresholds) + 1
        brightness_strategy = "explicit_thresholds"
    for value, name, minimum, maximum in (
        (cluster_seed, "cluster_seed", 0, 2**32 - 1),
        (pca_components, "pca_components", 1, 2**31 - 1),
        (cluster_count, "cluster_count", 2, 2**31 - 1),
        (batch_size, "batch_size", 1, 2**31 - 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    output = Path(output_directory).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Preparation output already exists: {output}")
    records = scan_labeled_images(image_root)
    if brightness_strategy == "quantile" and bin_count > len(records):
        raise ValueError("brightness_bins cannot exceed the number of training images.")
    if clusters_csv is None:
        if pca_components > min(len(records), 25088):
            raise ValueError("pca_components exceeds min(image count, 25088); explicitly lower it.")
        if cluster_count > len(records):
            raise ValueError("cluster_count cannot exceed the number of training images.")
        scenes = None
    else:
        scenes = read_scene_clusters(clusters_csv, records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-strata-", dir=output.parent) as temporary:
        staging = Path(temporary)
        # Measure the actual training distribution before expensive feature
        # extraction, so impossible brightness partitions fail immediately.
        brightness = _brightness_means(records)
        if brightness_strategy == "quantile":
            thresholds = quantile_brightness_thresholds(brightness, bin_count)
            brightness_reference = {
                "source": "training_images",
                "image_count": len(records),
                "minimum": float(np.min(brightness)),
                "maximum": float(np.max(brightness)),
                "quantiles": [index / bin_count for index in range(1, bin_count)],
                "quantile_method": "linear",
            }
        else:
            brightness_reference = None
        if clusters_csv is None:
            from .features import extract_vgg16_features

            print(f"Extracting VGG16 features for {len(records)} training images...", flush=True)
            extracted_brightness = extract_vgg16_features(
                records, staging / "features.npy", batch_size=batch_size, seed=cluster_seed
            )
            if not np.array_equal(brightness, extracted_brightness):
                raise ValueError("Training image brightness changed during feature extraction.")
            features = np.load(staging / "features.npy", mmap_mode="r", allow_pickle=False)
            try:
                clustered = cluster_scene_features(
                    features,
                    pca_components=pca_components,
                    cluster_count=cluster_count,
                    seed=cluster_seed,
                )
            finally:
                features._mmap.close()
            scenes = clustered.scene_clusters
            np.savez_compressed(
                staging / "clustering.npz",
                scene_clusters=scenes,
                reduced_features=clustered.reduced_features,
                pca_mean=clustered.pca_mean,
                pca_components=clustered.pca_components,
                pca_explained_variance_ratio=clustered.pca_explained_variance_ratio,
                cluster_centers=clustered.cluster_centers,
            )
            clustering_metadata = {
                "mode": "vgg16_pca_kmeans",
                "feature_model": "keras.applications.VGG16",
                "weights": "imagenet",
                "include_top": False,
                "input_size": 224,
                "feature_dimension": 25088,
                "channel_order": "OpenCV BGR converted to RGB before Keras preprocess_input",
                "batch_size": batch_size,
                "pca_components": pca_components,
                "pca_solver": "randomized",
                "cluster_count": cluster_count,
                "kmeans_n_init": 10,
                "cluster_seed": cluster_seed,
                "feature_file": "features.npy",
                "clustering_file": "clustering.npz",
            }
        else:
            clustering_metadata = {
                "mode": "existing_cluster_csv",
                "cluster_count": len(np.unique(scenes)),
                "source_csv": str(Path(clusters_csv).resolve()),
                "source_csv_sha256": hashlib.sha256(Path(clusters_csv).read_bytes()).hexdigest(),
            }
        stratification = build_triple_stratification(
            [record.holding_label for record in records],
            scenes,
            brightness,
            brightness_thresholds=thresholds,
            fold_count=config.num_folds,
            seed=config.seed,
        )
        csv_path = staging / "train_strata.csv"
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "image_filename",
                    "holding_label",
                    "scene_cluster",
                    "brightness_mean",
                    "brightness_bin",
                    "stratification_label",
                    "fold_index",
                    "split_seed",
                    "split_count",
                ]
            )
            for index, record in enumerate(records):
                writer.writerow(
                    [
                        record.image_filename,
                        record.holding_label,
                        int(scenes[index]),
                        float(brightness[index]),
                        int(stratification.brightness_bins[index]),
                        stratification.stratification_labels[index],
                        int(stratification.fold_indices[index]),
                        config.seed,
                        config.num_folds,
                    ]
                )
        summary = {
            "format": "fan-classifier-stratification-v1",
            "model_name": config.model_name,
            "image_root": str(Path(image_root).expanduser().resolve()),
            "input_scope": "labeled training images only; no test images",
            "clustering": clustering_metadata,
            "brightness_strategy": brightness_strategy,
            "brightness_bin_count": bin_count,
            "brightness_reference": brightness_reference,
            "brightness_definition": "mean of all decoded RGB pixels before resize or normalization, [0, 255]",
            "training_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "environment": _environment(),
            **stratification.summary,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Reserve a new destination exclusively, so existing preparation outputs
        # cannot be overwritten, including when two jobs finish concurrently.
        output.mkdir(exist_ok=False)
        try:
            for path in staging.iterdir():
                path.rename(output / path.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return PreparationResult(output / "train_strata.csv", output / "summary.json")
