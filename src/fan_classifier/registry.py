"""Model identifiers and per-model defaults, independent of the training loop.

Each model's defaults apply to every fold. Settings can be inspected through
describe-model and overridden through the configuration and CLI interfaces.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.0
    gamma_probability: float = 0.3
    gamma_limit: tuple[int, int] = (80, 120)
    shift_limit: float = 0.1
    scale_limit: float = 0.1
    rotate_limit: float = 15.0
    geometric_probability: float = 0.5


@dataclass(frozen=True)
class TrainingDefaults:
    epochs: int
    batch_size: int
    learning_rate: float
    min_learning_rate: float
    selection_metric: str
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    timm_name: str
    image_size: int
    num_folds: int
    seed: int
    defaults: TrainingDefaults


_NARROW_GAMMA = AugmentationConfig(gamma_limit=(85, 115))
_SPECS = (
    ModelSpec(
        "eva02_large_448",
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        448,
        3,
        4489,
        TrainingDefaults(11, 8, 2e-5, 2e-6, "accuracy", _NARROW_GAMMA),
    ),
    ModelSpec(
        "eva_giant_560",
        "eva_giant_patch14_560.m30m_ft_in22k_in1k",
        560,
        3,
        5600,
        TrainingDefaults(
            15,
            6,
            5e-6,
            5e-7,
            "f1",
            AugmentationConfig(
                vertical_flip_probability=0.5,
                gamma_probability=0.5,
                geometric_probability=0.75,
            ),
        ),
    ),
    ModelSpec(
        "eva_giant_336",
        "eva_giant_patch14_336.m30m_ft_in22k_in1k",
        336,
        5,
        3370,
        TrainingDefaults(15, 16, 5e-6, 5e-7, "loss"),
    ),
    ModelSpec(
        "eva_large_336",
        "eva_large_patch14_336.in22k_ft_in22k_in1k",
        336,
        5,
        3333333,
        TrainingDefaults(15, 32, 1e-5, 1e-6, "loss"),
    ),
    ModelSpec(
        "eva_giant_clip_224",
        "eva_giant_patch14_224.clip_ft_in1k",
        224,
        5,
        3369,
        TrainingDefaults(15, 12, 1e-5, 1e-6, "accuracy", _NARROW_GAMMA),
    ),
    ModelSpec(
        "eva_large_196",
        "eva_large_patch14_196.in22k_ft_in22k_in1k",
        196,
        5,
        3369,
        TrainingDefaults(15, 32, 2e-5, 2e-6, "accuracy", _NARROW_GAMMA),
    ),
)
_BY_NAME = {name: spec for spec in _SPECS for name in (spec.alias, spec.timm_name)}


def list_model_specs() -> tuple[ModelSpec, ...]:
    """Return the six registered models without importing Torch."""
    return _SPECS


def get_model_spec(model_name: str) -> ModelSpec:
    """Resolve a readable alias or an exact timm model identifier."""
    try:
        return _BY_NAME[model_name]
    except (KeyError, TypeError):
        choices = ", ".join(spec.alias for spec in _SPECS)
        raise ValueError(f"Unknown model {model_name!r}. Available models: {choices}") from None


def build_model(model_name: str, *, pretrained: bool = True, num_classes: int = 2):
    """Build a timm classifier selected by name.

    pretrained=True downloads public backbone weights if they are not cached.
    Prediction uses pretrained=False and restores a private training checkpoint.
    No competition-trained weights are distributed with this project.
    """
    spec = get_model_spec(model_name)
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if not isinstance(pretrained, bool):
        raise ValueError("pretrained must be a boolean")
    import timm

    return timm.create_model(spec.timm_name, pretrained=pretrained, num_classes=num_classes)
