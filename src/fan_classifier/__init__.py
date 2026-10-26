"""Model selection and image classification for the fan-holding task."""

from .registry import build_model, get_model_spec, list_model_specs

__all__ = ["build_model", "get_model_spec", "list_model_specs"]
__version__ = "0.1.0"
