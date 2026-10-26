"""Aspect-ratio preserving preprocessing and configurable training augmentation."""

from __future__ import annotations

import math
from typing import Any

from .normalization import IMAGENET_MEAN, IMAGENET_STD, validate_channels


def build_transforms(
    image_size: int,
    *,
    training: bool,
    seed: int | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    horizontal_flip_probability: float = 0.5,
    vertical_flip_probability: float = 0.0,
    gamma_probability: float = 0.3,
    gamma_limit: tuple[int, int] = (80, 120),
    shift_limit: float = 0.1,
    scale_limit: float = 0.1,
    rotate_limit: float = 15,
    geometric_probability: float = 0.5,
) -> Any:
    """Build RGB transforms; validation/inference never use random augmentation.

    Training augmentation settings can be supplied from the model registry.
    Geometric transforms use reflected borders; the final square padding is
    black. Albumentations expects RGB values in [0, 255].
    """
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    mean, std = validate_channels(mean, std)
    for name, probability in (
        ("horizontal_flip_probability", horizontal_flip_probability),
        ("vertical_flip_probability", vertical_flip_probability),
        ("gamma_probability", gamma_probability),
        ("geometric_probability", geometric_probability),
    ):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if len(gamma_limit) != 2 or not 0 < gamma_limit[0] <= gamma_limit[1]:
        raise ValueError("gamma_limit must contain two positive, ordered limits.")
    for name, limit, upper in (
        ("shift_limit", shift_limit, 1),
        ("scale_limit", scale_limit, 1),
        ("rotate_limit", rotate_limit, 180),
    ):
        if not math.isfinite(limit) or not 0 <= limit <= upper:
            raise ValueError(f"{name} must be between 0 and {upper}.")
    if scale_limit == 1:
        raise ValueError("scale_limit must be less than 1 to retain a positive scale.")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be None or an integer between 0 and 2**32 - 1.")
    try:
        import albumentations as A
        import cv2
        from albumentations.pytorch import ToTensorV2
    except ImportError as error:
        raise RuntimeError("Image transforms require the training dependencies.") from error

    operations = []
    if training:
        operations.append(A.HorizontalFlip(p=horizontal_flip_probability))
        if vertical_flip_probability:
            operations.append(A.VerticalFlip(p=vertical_flip_probability))
        operations.extend(
            [
                A.RandomGamma(gamma_limit=gamma_limit, p=gamma_probability),
                A.ShiftScaleRotate(
                    shift_limit=shift_limit,
                    scale_limit=scale_limit,
                    rotate_limit=rotate_limit,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=geometric_probability,
                ),
            ]
        )
    operations.extend(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
                position="center",
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    return A.Compose(operations, seed=seed)
