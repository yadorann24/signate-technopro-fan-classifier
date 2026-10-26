"""Manifest checks use invented filenames and temporary, empty placeholders."""

import csv
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fan_classifier.data import (
    FanHoldingDataset,
    ImageRecord,
    read_inference_records,
    read_training_records,
    scan_labeled_images,
    split_records,
)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        (self.images / "hold").mkdir(parents=True)
        (self.images / "not-hold").mkdir()
        for relative in ("hold/positive.png", "not-hold/negative.png", "unlabeled.png"):
            (self.images / relative).touch()

    def write_csv(self, fields, rows):
        path = self.root / "manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
        return path

    def test_canonical_and_legacy_training_schemas_agree(self):
        rows = [("positive.png", "1", "light_a"), ("negative.png", "0", "dark_b")]
        canonical = read_training_records(
            self.write_csv(["image_filename", "holding_label", "stratification_label"], rows),
            self.images,
        )
        legacy = read_training_records(
            self.write_csv(["image_filename", "check", "new_fold"], rows), self.images
        )
        self.assertEqual(canonical, legacy)
        self.assertEqual([record.holding_label for record in legacy], [1, 0])
        self.assertEqual(legacy[0].image_path.parent.name, "hold")
        self.assertEqual(legacy[1].image_path.parent.name, "not-hold")

    def test_inference_preserves_order_and_has_no_label(self):
        (self.images / "second.png").touch()
        records = read_inference_records(
            self.write_csv(["image_filename"], [("second.png",), ("unlabeled.png",)]), self.images
        )
        self.assertEqual(
            [record.image_filename for record in records], ["second.png", "unlabeled.png"]
        )
        self.assertTrue(all(record.holding_label is None for record in records))

    def test_requires_explicit_stratification(self):
        path = self.write_csv(["image_filename", "holding_label"], [("positive.png", "1")])
        with self.assertRaisesRegex(ValueError, "stratification_label"):
            read_training_records(path, self.images)

    def test_rejects_conflicting_legacy_alias(self):
        path = self.write_csv(
            ["image_filename", "holding_label", "check", "stratification_label"],
            [("positive.png", "1", "0", "bright")],
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            read_training_records(path, self.images)

    def test_rejects_duplicate_ids(self):
        path = self.write_csv(["image_filename"], [("unlabeled.png",), ("unlabeled.png",)])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_inference_records(path, self.images)

    def test_rejects_unsafe_filenames(self):
        for filename in (
            "../escape.png",
            "/escape.png",
            "hold/../unlabeled.png",
            "C:\\escape.png",
            "./unlabeled.png",
            "",
            " bad.png",
        ):
            with self.subTest(filename=filename):
                path = self.write_csv(["image_filename"], [(filename,)])
                with self.assertRaises(ValueError):
                    read_inference_records(path, self.images)

    def test_rejects_symlink_outside_image_root(self):
        outside = self.root / "outside.png"
        outside.touch()
        (self.images / "escape.png").symlink_to(outside)
        path = self.write_csv(["image_filename"], [("escape.png",)])
        with self.assertRaisesRegex(ValueError, "outside"):
            read_inference_records(path, self.images)

    def test_rejects_missing_image_before_training(self):
        path = self.write_csv(["image_filename"], [("missing.png",)])
        with self.assertRaises(FileNotFoundError):
            read_inference_records(path, self.images)

    def test_rejects_nonbinary_labels_and_missing_categories(self):
        for label, category in (
            ("2", "bright"),
            ("1.0", "bright"),
            ("yes", "bright"),
            ("1", ""),
            ("1", "NaN"),
        ):
            with self.subTest(label=label, category=category):
                path = self.write_csv(
                    ["image_filename", "holding_label", "stratification_label"],
                    [("positive.png", label, category)],
                )
                with self.assertRaises(ValueError):
                    read_training_records(path, self.images)

    def test_rejects_duplicate_headers_and_row_width_mismatch(self):
        for fields, rows in (
            (["image_filename", "image_filename"], [("unlabeled.png", "unlabeled.png")]),
            (["image_filename"], [("unlabeled.png", "extra")]),
            (["image_filename", "unused"], [("unlabeled.png",)]),
        ):
            with self.subTest(fields=fields, rows=rows):
                with self.assertRaises(ValueError):
                    read_inference_records(self.write_csv(fields, rows), self.images)

    def test_rejects_empty_manifest(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            read_inference_records(self.write_csv(["image_filename"], []), self.images)

    def test_insufficient_stratification_fails_without_fallback(self):
        records = [ImageRecord("synthetic.png", Path("synthetic.png"), 1, "rare_category")]
        with self.assertRaisesRegex(ValueError, "No fallback"):
            split_records(records, fold_count=3, fold_index=0, seed=42)

    def test_invalid_fold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fold_index"):
            split_records([], fold_count=3, fold_index=3, seed=42)

    def test_dataset_rejects_mixed_label_presence(self):
        records = [ImageRecord("a.png", Path("a.png"), 1), ImageRecord("b.png", Path("b.png"))]
        with self.assertRaisesRegex(ValueError, "mix"):
            FanHoldingDataset(records)

    def test_read_training_manifest_preserves_all_stored_fold_columns(self):
        records = read_training_records(
            self.write_csv(
                [
                    "image_filename",
                    "holding_label",
                    "stratification_label",
                    "fold_index",
                    "split_seed",
                    "split_count",
                ],
                [("positive.png", "1", "bright", "2", "42", "3")],
            ),
            self.images,
        )
        self.assertEqual(
            (records[0].fold_index, records[0].split_seed, records[0].split_count), (2, 42, 3)
        )

    def test_stored_fold_columns_must_be_complete_and_valid(self):
        base_fields = ["image_filename", "holding_label", "stratification_label"]
        base_row = ["positive.png", "1", "bright"]
        for columns, values in (
            (["fold_index"], ["0"]),
            (["split_seed", "split_count"], ["42", "3"]),
            (["fold_index", "split_seed", "split_count"], ["", "42", "3"]),
            (["fold_index", "split_seed", "split_count"], ["1.0", "42", "3"]),
            (["fold_index", "split_seed", "split_count"], ["3", "42", "3"]),
            (["fold_index", "split_seed", "split_count"], ["0", "-1", "3"]),
            (["fold_index", "split_seed", "split_count"], ["0", str(2**32), "3"]),
            (["fold_index", "split_seed", "split_count"], ["0", "42", "1"]),
        ):
            with self.subTest(columns=columns, values=values):
                with self.assertRaises(ValueError):
                    read_training_records(
                        self.write_csv(base_fields + columns, [base_row + values]), self.images
                    )

    def test_scan_labeled_images_has_stable_unique_ids_and_directory_labels(self):
        (self.images / "hold" / "nested").mkdir()
        (self.images / "hold" / "nested" / "example.JPG").touch()
        (self.images / "not-hold" / "another.JPEG").touch()
        (self.images / "not-hold" / "ignored.txt").touch()
        records = scan_labeled_images(self.images)
        self.assertEqual(
            [record.image_filename for record in records],
            ["another.JPEG", "negative.png", "nested/example.JPG", "positive.png"],
        )
        self.assertEqual([record.holding_label for record in records], [0, 0, 1, 1])
        self.assertTrue(all(record.stratification_label is None for record in records))

    def test_scan_rejects_duplicate_ids_across_classes(self):
        (self.images / "hold" / "negative.png").touch()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            scan_labeled_images(self.images)

    def test_scan_rejects_missing_or_empty_class(self):
        (self.images / "hold" / "positive.png").unlink()
        with self.assertRaisesRegex(ValueError, "contains no"):
            scan_labeled_images(self.images)
        (self.images / "hold").rmdir()
        with self.assertRaisesRegex(ValueError, "Missing"):
            scan_labeled_images(self.images)

    def test_scan_rejects_unsafe_names_and_external_symlinks(self):
        invalid = self.images / "hold" / "bad:name.png"
        invalid.touch()
        with self.assertRaisesRegex(ValueError, "normalized relative path"):
            scan_labeled_images(self.images)
        invalid.unlink()
        outside = self.root / "external.png"
        outside.touch()
        (self.images / "hold" / "escape.png").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "outside"):
            scan_labeled_images(self.images)

    def test_scan_rejects_external_class_directory_symlink(self):
        (self.images / "hold" / "positive.png").unlink()
        (self.images / "hold").rmdir()
        external_class = self.root / "external_class"
        external_class.mkdir()
        (external_class / "outside.png").touch()
        (self.images / "hold").symlink_to(external_class, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            scan_labeled_images(self.images)

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn"), "training dependencies are not installed"
    )
    def test_folds_are_disjoint_exhaustive_and_reproducible(self):
        records = [
            ImageRecord(
                f"synthetic_{label}_{index}.png", Path("unused.png"), label, f"category_{label}"
            )
            for label in (0, 1)
            for index in range(6)
        ]
        validation_ids = []
        for fold in range(3):
            training, validation = split_records(records, fold_count=3, fold_index=fold, seed=42)
            repeated = split_records(records, fold_count=3, fold_index=fold, seed=42)
            self.assertEqual((training, validation), repeated)
            train_ids = {record.image_filename for record in training}
            valid_ids = {record.image_filename for record in validation}
            self.assertFalse(train_ids & valid_ids)
            self.assertEqual(len(train_ids | valid_ids), len(records))
            self.assertEqual(sorted(record.holding_label for record in validation), [0, 0, 1, 1])
            validation_ids.extend(valid_ids)
        self.assertEqual(len(validation_ids), len(set(validation_ids)))
        self.assertEqual(len(validation_ids), len(records))


class StoredFoldTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            ImageRecord(
                f"synthetic_{label}_{index}.png",
                Path("unused.png"),
                label,
                f"category_{label}",
                index % 3,
                42,
                3,
            )
            for label in (0, 1)
            for index in range(6)
        ]

    def test_stored_folds_are_retained_exactly_with_input_order(self):
        records = list(reversed(self.records))
        for fold_index in range(3):
            training, validation = split_records(
                records, fold_count=3, fold_index=fold_index, seed=42
            )
            self.assertEqual(
                training, [record for record in records if record.fold_index != fold_index]
            )
            self.assertEqual(
                validation, [record for record in records if record.fold_index == fold_index]
            )

    def test_stored_metadata_must_match_requested_settings(self):
        for changes in ({"split_seed": 43}, {"split_count": 4}):
            with self.subTest(changes=changes):
                records = [replace(record, **changes) for record in self.records]
                with self.assertRaisesRegex(ValueError, "do not match"):
                    split_records(records, fold_count=3, fold_index=0, seed=42)

    def test_stored_metadata_cannot_be_missing_partial_or_noninteger(self):
        for changes in (
            {"fold_index": None, "split_seed": None, "split_count": None},
            {"split_seed": None},
            {"fold_index": True},
            {"fold_index": 1.0},
        ):
            with self.subTest(changes=changes):
                records = [replace(self.records[0], **changes), *self.records[1:]]
                with self.assertRaisesRegex(ValueError, "complete integer"):
                    split_records(records, fold_count=3, fold_index=0, seed=42)

    def test_stored_fold_indices_must_be_in_range_and_exhaustive(self):
        records = [replace(self.records[0], fold_index=3), *self.records[1:]]
        with self.assertRaisesRegex(ValueError, "Stored fold_index"):
            split_records(records, fold_count=3, fold_index=0, seed=42)
        records = [replace(record, fold_index=record.fold_index % 2) for record in self.records]
        with self.assertRaisesRegex(ValueError, "every fold_index"):
            split_records(records, fold_count=3, fold_index=0, seed=42)

    def test_stored_assignments_must_balance_each_stratum(self):
        records = [replace(self.records[0], fold_index=1), *self.records[1:]]
        with self.assertRaisesRegex(ValueError, "within stratification category"):
            split_records(records, fold_count=3, fold_index=0, seed=42)

    def test_balanced_strata_cannot_have_unbalanced_total_fold_sizes(self):
        records = [
            ImageRecord(f"{label}_{index}.png", Path("unused"), label, str(label), fold, 42, 3)
            for label in (0, 1)
            for index, fold in enumerate((0, 0, 1, 2))
        ]
        with self.assertRaisesRegex(ValueError, "unbalanced total"):
            split_records(records, fold_count=3, fold_index=0, seed=42)


if __name__ == "__main__":
    unittest.main()
