"""Synthetic CSVs only: no competition data or downloaded model weights."""

import csv
import tempfile
import unittest
from pathlib import Path

from fan_classifier.prediction import (
    ensemble_probability_files,
    read_probabilities,
    write_prediction_files,
)


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.addCleanup(self.temporary_directory.cleanup)

    def write_member(self, name, predictions, model_name=None):
        return write_prediction_files(
            predictions,
            self.root / name,
            metadata={"operation": "predict", "model_name": model_name},
        )

    def test_ensemble_aligns_by_id_and_preserves_first_order(self):
        first = self.write_member("a", {"synthetic_a.png": 0.1, "synthetic_b.png": 0.8}, "model_a")
        second = self.write_member("b", {"synthetic_b.png": 0.6, "synthetic_a.png": 0.5}, "model_b")
        result = ensemble_probability_files(
            [first.probability_path, second.probability_path], self.root / "ensemble"
        )
        probabilities = read_probabilities(result.probability_path)
        self.assertEqual(list(probabilities), ["synthetic_a.png", "synthetic_b.png"])
        self.assertAlmostEqual(probabilities["synthetic_a.png"], 0.3)
        self.assertAlmostEqual(probabilities["synthetic_b.png"], 0.7)
        with result.submission_path.open(newline="") as stream:
            self.assertEqual(
                list(csv.reader(stream)), [["synthetic_a.png", "0"], ["synthetic_b.png", "1"]]
            )

    def test_threshold_is_strictly_greater_than_half(self):
        result = self.write_member("a", {"synthetic.png": 0.5})
        self.assertEqual(result.submission_path.read_text().strip(), "synthetic.png,0")

    def test_mismatched_ids_are_rejected_without_creating_output(self):
        first = self.write_member("a", {"a.png": 0.2})
        second = self.write_member("b", {"b.png": 0.4})
        with self.assertRaisesRegex(ValueError, "image IDs differ"):
            ensemble_probability_files(
                [first.probability_path, second.probability_path], self.root / "out"
            )
        self.assertFalse((self.root / "out").exists())

    def test_invalid_values_and_duplicate_ids_are_rejected(self):
        for value in ["nan", "inf", "-0.1", "1.1", "not-a-probability"]:
            with self.subTest(value=value):
                source = self.root / "invalid.csv"
                source.write_text(f"image_filename,holding_probability\na.png,{value}\n")
                with self.assertRaises(ValueError):
                    read_probabilities(source)
        source.write_text("image_filename,holding_probability\na.png,0.1\na.png,0.2\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_probabilities(source)

    def test_same_input_and_same_model_are_rejected(self):
        first = self.write_member("a", {"a.png": 0.2}, "same_model")
        second = self.write_member("b", {"a.png": 0.4}, "same_model")
        with self.assertRaisesRegex(ValueError, "same probability file"):
            ensemble_probability_files(
                [first.probability_path, first.probability_path], self.root / "out"
            )
        with self.assertRaisesRegex(ValueError, "appears more than once"):
            ensemble_probability_files(
                [first.probability_path, second.probability_path], self.root / "out"
            )

    def test_existing_output_is_never_overwritten(self):
        first = self.write_member("a", {"a.png": 0.2})
        with self.assertRaises(FileExistsError):
            self.write_member("a", {"a.png": 0.9})
        self.assertEqual(read_probabilities(first.probability_path), {"a.png": 0.2})

    def test_nested_ensemble_is_rejected(self):
        first = self.write_member("a", {"a.png": 0.2})
        second = self.write_member("b", {"a.png": 0.4})
        result = ensemble_probability_files(
            [first.probability_path, second.probability_path], self.root / "ensemble"
        )
        with self.assertRaisesRegex(ValueError, "previously averaged"):
            ensemble_probability_files(
                [first.probability_path, result.probability_path], self.root / "nested"
            )


if __name__ == "__main__":
    unittest.main()
