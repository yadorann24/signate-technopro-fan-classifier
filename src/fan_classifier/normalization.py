"""Resolve new training normalization and restore saved values without downloads."""

from dataclasses import asdict, dataclass
import math
from typing import Mapping

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TIMM_VERSION = "1.0.11"


@dataclass(frozen=True)
class NormalizationConfig:
    mode: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def validate_channels(mean, std) -> tuple[tuple, tuple]:
    for name, values in (("mean", mean), ("std", std)):
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (name == "std" and value <= 0)
                for value in values
            )
        ):
            raise ValueError(f"normalization {name} requires three finite RGB values (std > 0)")
    return tuple(mean), tuple(std)


def imagenet_normalization() -> NormalizationConfig:
    return NormalizationConfig("imagenet", IMAGENET_MEAN, IMAGENET_STD, "imagenet")


def restore_normalization(value: Mapping | None) -> NormalizationConfig:
    """Missing legacy settings mean ImageNet; never consult new model defaults."""
    if value is None:
        return imagenet_normalization()
    if not isinstance(value, Mapping) or set(value) != {"mode", "mean", "std", "source"}:
        raise ValueError("Saved normalization requires mode, mean, std, and source")
    mode = value["mode"]
    if mode not in {"pretrained", "imagenet", "custom"}:
        raise ValueError("normalization mode must be pretrained, imagenet, or custom")
    mean, std = validate_channels(value["mean"], value["std"])
    if not isinstance(value["source"], str) or not value["source"]:
        raise ValueError("normalization source must be a nonempty string")
    result = NormalizationConfig(mode, mean, std, value["source"])
    if mode == "imagenet" and result != imagenet_normalization():
        raise ValueError("imagenet normalization must use the ImageNet mean/std and source")
    return result


def resolve_normalization(model_name: str, value=None) -> NormalizationConfig:
    """Resolve explicitly selected new-run settings using the pinned timm registry."""
    from .registry import get_model_spec

    if value is None:
        value = {"mode": "pretrained"}
    if not isinstance(value, Mapping) or set(value) - {"mode", "mean", "std", "source"}:
        raise ValueError("normalization must be an object containing mode and optional mean/std")
    mode = value.get("mode")
    if mode == "pretrained":
        import timm
        from timm.models import get_pretrained_cfg

        if timm.__version__ != TIMM_VERSION:
            raise ValueError(f"pretrained normalization requires timm=={TIMM_VERSION}")
        spec = get_model_spec(model_name)
        pretrained = get_pretrained_cfg(spec.timm_name)
        mean, std = validate_channels(pretrained.mean, pretrained.std)
        result = NormalizationConfig(mode, mean, std, f"timm=={TIMM_VERSION}:{spec.timm_name}")
    elif mode == "imagenet":
        result = imagenet_normalization()
    elif mode == "custom":
        mean, std = validate_channels(value.get("mean"), value.get("std"))
        result = NormalizationConfig(mode, mean, std, value.get("source", "custom"))
    else:
        raise ValueError("normalization mode must be pretrained, imagenet, or custom")
    for key in ("mean", "std", "source"):
        if key in value:
            if key in {"mean", "std"} and not isinstance(value[key], (list, tuple)):
                raise ValueError(f"normalization {key} must contain three RGB values")
            supplied = tuple(value[key]) if key in {"mean", "std"} else value[key]
            if supplied != getattr(result, key):
                raise ValueError(f"normalization {key} conflicts with mode={mode}; use custom")
    return restore_normalization(result.to_dict())
