import json
from pathlib import Path
import tempfile
import unittest

from fan_classifier.config import resolve_training_config


class ConfigTests(unittest.TestCase):
    def test_registry_controls_resolution_and_training_defaults(self):
        config = resolve_training_config("eva02_large_448")
        self.assertEqual(config.image_size, 448)
        self.assertEqual(config.selection_metric, "accuracy")
        self.assertEqual(config.num_folds, 3)

    def test_cli_overrides_json_and_partial_augmentation_preserves_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({"batch_size": 4, "augmentation": {"gamma_probability": 0}}))
            config = resolve_training_config("eva_giant_560", config_path=path, batch_size=2)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.augmentation.gamma_probability, 0)
        self.assertEqual(config.augmentation.vertical_flip_probability, 0.5)

    def test_invalid_or_misspelled_settings_fail_early(self):
        cases = [
            {"batchsize": 2},
            {"num_folds": 1},
            {"image_size": 1},
            {"learning_rate": float("nan")},
            {"learning_rate": -1},
            {"seed": -1},
            {"epochs": True},
            {"min_learning_rate": 1.0},
            {"augmentation": {"gamma_probability": 2}},
            {"augmentation": {"gamma_limit": [120, 80]}},
            {"model_name": "eva_giant_560"},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_training_config("eva_large_196", **overrides)


if __name__ == "__main__":
    unittest.main()
