"""Check pinned model metadata and legacy behavior without downloading weights."""

import unittest
from unittest.mock import patch

from fan_classifier.config import resolve_training_config
from fan_classifier.normalization import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    imagenet_normalization,
    resolve_normalization,
    restore_normalization,
)
from fan_classifier.registry import list_model_specs


class NormalizationTests(unittest.TestCase):
    def test_all_registered_models_use_real_pinned_timm_metadata(self):
        clip = {"eva02_large_448", "eva_large_336", "eva_giant_clip_224", "eva_large_196"}
        with patch("timm.models._hub.hf_hub_download", side_effect=AssertionError("no downloads")):
            for spec in list_model_specs():
                with self.subTest(model=spec.alias):
                    config = resolve_training_config(spec.alias)
                    normalization = config.normalization
                    self.assertEqual(normalization.mode, "pretrained")
                    if spec.alias in clip:
                        self.assertEqual(normalization.mean, (0.48145466, 0.4578275, 0.40821073))
                        self.assertEqual(normalization.std, (0.26862954, 0.26130258, 0.27577711))
                    else:
                        self.assertEqual(normalization.mean, IMAGENET_MEAN)
                        self.assertEqual(normalization.std, IMAGENET_STD)
                    self.assertEqual(restore_normalization(normalization.to_dict()), normalization)

    def test_legacy_config_never_resolves_new_pretrained_defaults(self):
        with patch("fan_classifier.config.resolve_normalization", side_effect=AssertionError):
            config = resolve_training_config("eva_large_196", restore_saved=True)
        self.assertEqual(config.normalization, imagenet_normalization())

    def test_saved_custom_values_are_restored_without_model_registry_resolution(self):
        saved = {
            "mode": "pretrained",
            "mean": [0.1, 0.2, 0.3],
            "std": [0.2, 0.3, 0.4],
            "source": "recorded-experiment",
        }
        with patch("timm.models.get_pretrained_cfg", side_effect=AssertionError):
            config = resolve_training_config(
                "eva_large_196", restore_saved=True, normalization=saved
            )
        self.assertEqual(config.normalization.mean, (0.1, 0.2, 0.3))

    def test_explicit_imagenet_and_custom_new_experiments(self):
        self.assertEqual(
            resolve_normalization("eva_large_196", {"mode": "imagenet"}), imagenet_normalization()
        )
        custom = resolve_normalization(
            "eva_large_196", {"mode": "custom", "mean": [0.1, 0.2, 0.3], "std": [1, 1, 1]}
        )
        self.assertEqual(custom.mean, (0.1, 0.2, 0.3))
        for value in (
            {"mode": "typo"},
            {"mode": "custom", "mean": [1, 2, 3], "std": [0, 1, 1]},
            {"mode": "custom", "mean": [float("nan"), 1, 1], "std": [1, 1, 1]},
            {"mode": "imagenet", "mean": [0, 0, 0]},
            {"mode": "pretrained", "std": [1, 1, 1]},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_normalization("eva_large_196", value)

    def test_transform_applies_exact_saved_rgb_values(self):
        import numpy as np
        from fan_classifier.transforms import build_transforms

        image = np.full((4, 4, 3), 255, dtype=np.uint8)
        transform = build_transforms(4, training=False, mean=(0.1, 0.2, 0.3), std=(1, 2, 4))
        tensor = transform(image=image)["image"]
        np.testing.assert_allclose(tensor[:, 0, 0], [0.9, 0.4, 0.175], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
