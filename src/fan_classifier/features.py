"""VGG16 image features and mean pixel brightness for private training images.

The convolutional features use ImageNet VGG16 without
the classification head, 224 x 224 input, and flattened 7 x 7 x 512 output.
OpenCV BGR images are converted to RGB before Keras preprocess_input so that
channel reordering and ImageNet mean subtraction receive the expected input.

Only one image batch is held in memory. Features are written to a private
NumPy file; callers must exclude this data-derived artifact from publication.
TensorFlow is loaded only when constructing the feature extractor.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .data import ImageRecord

if TYPE_CHECKING:
    import numpy as np


VGG16_INPUT_SIZE = 224
VGG16_FEATURE_DIMENSION = 7 * 7 * 512


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")


def build_vgg16_feature_model(*, seed: int = 42, weights: str | None = "imagenet") -> Any:
    """Build a frozen VGG16 feature extractor; weights=None allows offline checks.

    ImageNet weights are downloaded by Keras on first use. The seed and
    deterministic TensorFlow operations make repeated runs reproducible in
    the same software and hardware environment.
    """
    _validate_seed(seed)
    if weights not in ("imagenet", None):
        raise ValueError("weights must be 'imagenet' or None.")
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "Feature extraction requires TensorFlow; install the stratification extra "
            "with pip install '.[stratification]'."
        ) from error
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = tf.keras.applications.VGG16(
        weights=weights,
        include_top=False,
        input_shape=(VGG16_INPUT_SIZE, VGG16_INPUT_SIZE, 3),
    )
    model.trainable = False
    if model.output_shape[1:] != (7, 7, 512):
        raise RuntimeError("VGG16 must produce a 7 x 7 x 512 convolutional feature map.")
    return model


def extract_vgg16_features(
    records: Sequence[ImageRecord],
    feature_path: str | Path,
    *,
    batch_size: int = 16,
    seed: int = 42,
    feature_model: Any | None = None,
    preprocess: Callable[[Any], Any] | None = None,
) -> np.ndarray:
    """Write features in record order and return native-resolution mean brightness.

    Every record must have a binary holding_label. stratification_label is not
    required because these features are used to derive it. Brightness is the
    arithmetic mean of all decoded RGB pixel values, on the [0, 255] scale,
    before resizing or VGG preprocessing, to measure brightness independently
    of the model's input transformation.

    A complete float32 .npy feature matrix is published only after all batches
    succeed. Existing files are never overwritten; failed partial files are
    removed. The final short batch is included. Supply feature_model and
    preprocess together for offline tests; a model must accept training=False
    and return a numeric batch with a consistent, nonempty feature dimension.
    """
    import cv2
    import numpy as np

    _validate_seed(seed)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    records = tuple(records)
    if not records:
        raise ValueError("Feature extraction requires at least one image.")
    if len({record.image_filename for record in records}) != len(records):
        raise ValueError("Duplicate image_filename; one feature row per image is required.")
    for record in records:
        if isinstance(record.holding_label, bool) or record.holding_label not in (0, 1):
            raise ValueError("Every feature record requires holding_label 0 or 1.")
        if not Path(record.image_path).is_file():
            raise FileNotFoundError(f"Image does not exist: {record.image_path}")
    if (feature_model is None) != (preprocess is None):
        raise ValueError("Supply feature_model and preprocess together.")

    feature_path = Path(feature_path).expanduser()
    if feature_path.suffix != ".npy":
        raise ValueError("feature_path must have the .npy extension.")
    if feature_path.exists() or feature_path.is_symlink():
        raise FileExistsError(f"Feature output already exists: {feature_path}")
    use_vgg16 = feature_model is None
    if use_vgg16:
        feature_model = build_vgg16_feature_model(seed=seed)
        from tensorflow.keras.applications.vgg16 import preprocess_input

        preprocess = preprocess_input

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    mean_brightness = np.empty(len(records), dtype=np.float64)
    feature_matrix = None
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{feature_path.name}.",
            suffix=".partial",
            dir=feature_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            images = np.empty(
                (len(batch_records), VGG16_INPUT_SIZE, VGG16_INPUT_SIZE, 3), dtype=np.float32
            )
            for offset, record in enumerate(batch_records):
                bgr_image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
                if bgr_image is None or bgr_image.size == 0:
                    raise ValueError(f"Image could not be decoded: {record.image_path}")
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                mean_brightness[start + offset] = rgb_image.mean(dtype=np.float64)
                images[offset] = cv2.resize(
                    rgb_image,
                    (VGG16_INPUT_SIZE, VGG16_INPUT_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )

            processed_images = np.asarray(preprocess(images), dtype=np.float32)
            if processed_images.shape != images.shape or not np.isfinite(processed_images).all():
                raise ValueError("VGG preprocessing must preserve image shape and finite values.")
            features = np.asarray(feature_model(processed_images, training=False), dtype=np.float32)
            if features.ndim < 2 or features.shape[0] != len(batch_records) or features.size == 0:
                raise ValueError("Feature model must return one nonempty feature row per image.")
            features = features.reshape(len(batch_records), -1)
            if not np.isfinite(features).all():
                raise ValueError("Feature model returned nonfinite values.")
            if not np.any(features != 0, axis=1).all():
                raise ValueError("Feature model returned an all-zero feature row.")
            if use_vgg16 and features.shape[1] != VGG16_FEATURE_DIMENSION:
                raise ValueError("VGG16 must return 25088 flattened features per image.")
            if feature_matrix is None:
                feature_matrix = np.lib.format.open_memmap(
                    temporary_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(records), features.shape[1]),
                )
            elif features.shape[1] != feature_matrix.shape[1]:
                raise ValueError("Feature dimension changed between batches.")
            feature_matrix[start : start + len(batch_records)] = features

        feature_matrix.flush()
        # A hard link publishes the complete file and fails if the target was
        # concurrently created, unlike replace(), which would overwrite it.
        os.link(temporary_path, feature_path)
    finally:
        if feature_matrix is not None:
            feature_matrix._mmap.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return mean_brightness
