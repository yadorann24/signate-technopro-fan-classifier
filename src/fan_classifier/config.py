"""Validated, serializable settings for a reproducible run."""

from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path

from .registry import AugmentationConfig, get_model_spec
from .normalization import NormalizationConfig, resolve_normalization, restore_normalization


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    image_size: int
    num_folds: int
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    min_learning_rate: float
    selection_metric: str
    augmentation: AugmentationConfig
    num_workers: int = 0
    weight_decay: float = 0.001
    pretrained: bool = True
    normalization: NormalizationConfig | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "TrainingConfig":
        spec = get_model_spec(self.model_name)
        if type(self.image_size) is not int or self.image_size != spec.image_size:
            raise ValueError("image_size is determined by the model; choose a matching model name")
        for key in ("epochs", "batch_size", "num_folds"):
            value = getattr(self, key)
            if type(value) is not int or value < (2 if key == "num_folds" else 1):
                raise ValueError(f"{key} must be a positive integer (num_folds >= 2)")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for key in ("learning_rate", "min_learning_rate", "weight_decay"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a finite number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
        if self.learning_rate == 0 or self.min_learning_rate > self.learning_rate:
            raise ValueError(
                "require 0 <= min_learning_rate <= learning_rate and learning_rate > 0"
            )
        if self.selection_metric not in {"loss", "accuracy", "f1"}:
            raise ValueError("selection_metric must be loss, accuracy, or f1")
        if not isinstance(self.pretrained, bool):
            raise ValueError("pretrained must be a boolean")
        if not isinstance(self.normalization, NormalizationConfig):
            raise ValueError("normalization must be resolved before training")
        restore_normalization(self.normalization.to_dict())
        for key in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
            "gamma_probability",
            "geometric_probability",
        ):
            _validate_number(getattr(self.augmentation, key), key, 0, 1)
        _validate_number(self.augmentation.shift_limit, "shift_limit", 0, 1)
        _validate_number(self.augmentation.scale_limit, "scale_limit", 0, 0.99)
        _validate_number(self.augmentation.rotate_limit, "rotate_limit", 0, 180)
        gamma = self.augmentation.gamma_limit
        if not isinstance(gamma, (list, tuple)) or len(gamma) != 2:
            raise ValueError("gamma_limit must contain two positive integers")
        if any(type(v) is not int or v <= 0 for v in gamma) or gamma[0] > gamma[1]:
            raise ValueError("gamma_limit must contain ordered positive integers")
        return self


def _validate_number(value, name: str, lower: float, upper: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not lower <= value <= upper
    ):
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")


def resolve_training_config(
    selected_model: str, *, config_path=None, restore_saved: bool = False, **overrides
) -> TrainingConfig:
    """Apply JSON overrides, then explicit CLI overrides, to a model's defaults."""
    spec = get_model_spec(selected_model)
    defaults = spec.defaults
    config = TrainingConfig(
        model_name=spec.alias,
        image_size=spec.image_size,
        num_folds=spec.num_folds,
        seed=spec.seed,
        epochs=defaults.epochs,
        batch_size=defaults.batch_size,
        learning_rate=defaults.learning_rate,
        min_learning_rate=defaults.min_learning_rate,
        selection_metric=defaults.selection_metric,
        augmentation=defaults.augmentation,
    )
    values = {}
    if config_path is not None:
        values = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Training config must be a JSON object")
    values.update({key: value for key, value in overrides.items() if value is not None})
    allowed = {f.name for f in fields(TrainingConfig)}
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
    if "model_name" in values:
        if get_model_spec(values["model_name"]).alias != spec.alias:
            raise ValueError("Config model_name conflicts with --model")
        values["model_name"] = spec.alias
    if "augmentation" in values:
        augmentation = values["augmentation"]
        if not isinstance(augmentation, dict):
            raise ValueError("augmentation must be a JSON object")
        allowed_augmentation = {f.name for f in fields(AugmentationConfig)}
        unknown_augmentation = augmentation.keys() - allowed_augmentation
        if unknown_augmentation:
            raise ValueError(f"Unknown augmentation keys: {sorted(unknown_augmentation)}")
        values["augmentation"] = replace(config.augmentation, **augmentation)
    normalization = values.get("normalization")
    values["normalization"] = (
        restore_normalization(normalization)
        if restore_saved
        else resolve_normalization(spec.alias, normalization)
    )
    return replace(config, **values).validate()
