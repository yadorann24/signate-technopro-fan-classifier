"""Scene proxies and explicit label × scene × brightness stratification.

Scene clusters are visual groups, not verified shooting-location labels. PCA
and K-means are fitted to training images only, across all future folds. This
balances the split; it does not create an independent-location evaluation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence


@dataclass(frozen=True)
class SceneClustering:
    """Numeric clustering outputs; callers may store these privately as NPZ."""

    scene_clusters: Any
    reduced_features: Any
    pca_mean: Any
    pca_components: Any
    pca_explained_variance_ratio: Any
    cluster_centers: Any


@dataclass(frozen=True)
class TripleStratification:
    """Per-image split columns in input order and JSON-compatible counts."""

    stratification_labels: tuple[str, ...]
    brightness_means: Any
    brightness_bins: Any
    scene_clusters: Any
    fold_indices: Any
    summary: dict[str, Any]


def _integer(value: int, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        limit = f" through {maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{name} must be an integer from {minimum}{limit}.")
    return int(value)


def _brightness_vector(values: Any) -> Any:
    import numpy as np

    brightness = np.asarray(values)
    if (
        brightness.ndim != 1
        or not brightness.size
        or brightness.dtype.kind not in "iuf"
        or not np.isfinite(brightness).all()
        or np.any((brightness < 0) | (brightness > 255))
        or any(isinstance(value, (bool, np.bool_)) for value in values)
    ):
        raise ValueError("brightness_means must contain finite numeric values in [0, 255].")
    return brightness


def quantile_brightness_thresholds(
    brightness_means: Any, bin_count: int
) -> tuple[float, ...]:
    """Derive approximately equal-count bins from measured image brightness.

    Linear quantiles use every supplied image once. Equal brightness values
    remain in the same bin, so ties can prevent exactly equal image counts.
    If ties produce duplicate boundaries or empty bins, fail instead of silently
    reducing bin_count. Exact boundary values enter the higher bin.
    """
    import numpy as np

    brightness = _brightness_vector(brightness_means)
    bin_count = _integer(bin_count, "bin_count", minimum=2, maximum=len(brightness))
    quantiles = np.arange(1, bin_count, dtype=float) / bin_count
    thresholds = np.quantile(brightness, quantiles, method="linear")
    bin_counts = np.bincount(
        np.searchsorted(thresholds, brightness, side="right"), minlength=bin_count
    )
    if np.any(np.diff(thresholds) <= 0) or np.any(bin_counts == 0):
        raise ValueError(
            f"Cannot form {bin_count} nonempty brightness bins using quantiles while "
            "keeping equal brightness values together: duplicate thresholds or "
            f"empty bins (image counts={bin_counts.tolist()}). Lower brightness_bins "
            "or supply explicit brightness_thresholds; no bins were merged."
        )
    return tuple(float(value) for value in thresholds)


def cluster_scene_features(
    features: Any, *, pca_components: int = 50, cluster_count: int = 3, seed: int = 42
) -> SceneClustering:
    """Compress image features with PCA, then form visual groups with K-means.

    Defaults are 50 PCA components and three clusters. The randomized PCA
    solver and K-means n_init=10 use an explicit seed for repeatable fitting
    within the same numerical environment.
    No automatic reduction of component/cluster counts is performed.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    pca_components = _integer(pca_components, "pca_components", minimum=1)
    cluster_count = _integer(cluster_count, "cluster_count", minimum=2)
    seed = _integer(seed, "seed", minimum=0, maximum=2**32 - 1)
    feature_matrix = np.asarray(features)
    if (
        feature_matrix.ndim != 2
        or feature_matrix.shape[0] < 2
        or feature_matrix.shape[1] < 1
        or feature_matrix.dtype.kind not in "iuf"
        or not np.isfinite(feature_matrix).all()
    ):
        raise ValueError("features must be a finite numeric matrix with at least two images.")
    if pca_components > min(feature_matrix.shape):
        raise ValueError(
            f"pca_components={pca_components} exceeds min(image count, feature count)="
            f"{min(feature_matrix.shape)}. Supply more images or explicitly lower "
            "pca_components; no automatic dimension reduction is performed."
        )
    if cluster_count > len(feature_matrix):
        raise ValueError("cluster_count cannot exceed the number of training images.")
    # Stop as soon as enough distinct rows are found, avoiding a second full
    # feature-matrix copy. Duplicate images can acquire tiny SVD roundoff
    # differences; these must not be interpreted as distinct visual groups.
    distinct_features = []
    for row in feature_matrix:
        if not any(np.array_equal(row, previous) for previous in distinct_features):
            distinct_features.append(row)
            if len(distinct_features) == cluster_count:
                break
    if len(distinct_features) < cluster_count:
        if len(distinct_features) == 1:
            raise ValueError("All image features are identical; scene clustering is undefined.")
        raise ValueError(
            "Fewer distinct image feature rows than cluster_count. Inspect duplicate "
            "images or explicitly reduce cluster_count."
        )
    pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=seed)
    reduced_features = pca.fit_transform(feature_matrix)
    kmeans = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10)
    scene_clusters = kmeans.fit_predict(reduced_features)
    if len(np.unique(scene_clusters)) != cluster_count:
        raise ValueError(
            "Fewer distinct scene clusters were produced than cluster_count. "
            "Inspect duplicate features or explicitly change the clustering settings."
        )
    return SceneClustering(
        scene_clusters=scene_clusters,
        reduced_features=reduced_features,
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        cluster_centers=kmeans.cluster_centers_,
    )


def _integer_vector(values: Any, name: str, *, binary: bool = False) -> Any:
    import numpy as np

    array = np.asarray(values)
    if array.ndim != 1 or not array.size or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty one-dimensional integer sequence.")
    if binary:
        if not np.isin(array, [0, 1]).all():
            raise ValueError("holding_labels must be 0 (not-hold) or 1 (hold).")
    elif np.any(array < 0):
        raise ValueError(f"{name} must contain nonnegative integers.")
    return array


def build_triple_stratification(
    holding_labels: Sequence[int],
    scene_clusters: Any,
    brightness_means: Any,
    *,
    brightness_thresholds: Sequence[float],
    fold_count: int,
    seed: int = 42,
) -> TripleStratification:
    """Assign all folds using explicit brightness thresholds in the 0–255 scale.

    A value equal to a threshold belongs to the higher bin. Thresholds must be
    supplied by the caller, either fitted to training brightness or set manually.
    Each observed label/scene/brightness combination needs at least fold_count
    images. Sparse combinations fail with counts instead of changing the split.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    fold_count = _integer(fold_count, "fold_count", minimum=2)
    seed = _integer(seed, "seed", minimum=0, maximum=2**32 - 1)
    labels = _integer_vector(holding_labels, "holding_labels", binary=True)
    scenes = _integer_vector(scene_clusters, "scene_clusters")
    brightness = _brightness_vector(brightness_means)
    if len(labels) != len(scenes) or len(labels) != len(brightness):
        raise ValueError(
            "holding_labels, scene_clusters, and brightness_means must have identical "
            "image counts and input order."
        )
    thresholds = np.asarray(brightness_thresholds)
    if (
        thresholds.ndim != 1
        or not thresholds.size
        or thresholds.dtype.kind not in "iuf"
        or not np.isfinite(thresholds).all()
        or np.any((thresholds < 0) | (thresholds > 255))
        or np.any(np.diff(thresholds.astype(float)) <= 0)
    ):
        raise ValueError(
            "brightness_thresholds must be an explicit, nonempty, strictly increasing "
            "sequence of distinct finite values in [0, 255]."
        )
    brightness_bins = np.searchsorted(thresholds, brightness, side="right")
    strata = tuple(
        f"label={int(label)}|scene={int(scene)}|brightness={int(brightness_bin)}"
        for label, scene, brightness_bin in zip(labels, scenes, brightness_bins)
    )
    counts = Counter(strata)
    rare_counts = {category: count for category, count in sorted(counts.items()) if count < fold_count}
    if rare_counts:
        details = ", ".join(f"{category}: {count}" for category, count in rare_counts.items())
        raise ValueError(
            f"Each triple stratum requires at least fold_count={fold_count} images. "
            f"Sparse strata: {details}. Inspect the training data, explicitly revise "
            "brightness thresholds/cluster_count, or lower fold_count. No fallback "
            "merging or label-only splitting is performed."
        )
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    fold_indices = np.full(len(labels), -1, dtype=np.int64)
    fold_summaries: dict[str, dict[str, Any]] = {}
    for fold_index, (_, validation_indices) in enumerate(splitter.split(np.zeros(len(labels)), strata)):
        fold_indices[validation_indices] = fold_index
        fold_summaries[str(fold_index)] = {
            "image_count": len(validation_indices),
            "holding_label_counts": dict(
                sorted(Counter(str(int(labels[index])) for index in validation_indices).items())
            ),
            "brightness_bin_counts": {
                str(index): int(count)
                for index, count in enumerate(
                    np.bincount(brightness_bins[validation_indices], minlength=len(thresholds) + 1)
                )
            },
            "stratification_counts": dict(
                sorted(Counter(strata[index] for index in validation_indices).items())
            ),
        }
    return TripleStratification(
        stratification_labels=strata,
        brightness_means=brightness.astype(float, copy=True),
        brightness_bins=brightness_bins,
        scene_clusters=scenes.copy(),
        fold_indices=fold_indices,
        summary={
            "image_count": len(labels),
            "fold_count": fold_count,
            "split_seed": seed,
            "brightness_thresholds": thresholds.astype(float).tolist(),
            "brightness_threshold_boundary": "equal values enter the higher bin",
            "brightness_bin_counts": {
                str(index): int(count)
                for index, count in enumerate(
                    np.bincount(brightness_bins, minlength=len(thresholds) + 1)
                )
            },
            "holding_label_counts": dict(sorted(Counter(str(int(value)) for value in labels).items())),
            "stratification_counts": dict(sorted(counts.items())),
            "folds": fold_summaries,
        },
    )
