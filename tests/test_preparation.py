"""Synthetic preparation-to-training checks; no distributed images or weights."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from fan_classifier.cli import main
from fan_classifier.data import read_training_records, scan_labeled_images, split_records
from fan_classifier.preparation import prepare_stratification, read_scene_clusters


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.cluster_csv = self.root / "clusters.csv"
        rows = []
        for label in (0, 1):
            directory = self.images / ("hold" if label else "not-hold")
            directory.mkdir(parents=True)
            for scene in (0, 1):
                for brightness in (40, 50, 60, 90, 150, 200):
                    for repeat in (0, 1):
                        filename = f"synthetic_{label}_{scene}_{brightness}_{repeat}.png"
                        cv2.imwrite(
                            str(directory / filename), np.full((11, 7, 3), brightness, np.uint8)
                        )
                        rows.append([filename, scene])
        # Reverse input rows to catch accidental positional joins.
        self.write_clusters(list(reversed(rows)))

    def write_clusters(self, rows, fields=("image_filename", "cluster")):
        with self.cluster_csv.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    def prepare(self, name="prepared", **overrides):
        options = dict(model_name="eva_large_196", num_folds=2, clusters_csv=self.cluster_csv)
        options.update(overrides)
        return prepare_stratification(self.images, self.root / name, **options)

    def test_training_quantiles_balance_image_counts_and_preserve_folds(self):
        result = self.prepare()
        summary = json.loads(result.summary_path.read_text())
        self.assertEqual(summary["brightness_strategy"], "quantile")
        np.testing.assert_allclose(summary["brightness_thresholds"], [56 + 2 / 3, 110])
        self.assertEqual(summary["brightness_bin_counts"], {"0": 16, "1": 16, "2": 16})
        self.assertEqual(summary["brightness_bin_count"], 3)
        self.assertEqual(
            summary["brightness_reference"],
            {
                "source": "training_images",
                "image_count": 48,
                "minimum": 40.0,
                "maximum": 200.0,
                "quantiles": [1 / 3, 2 / 3],
                "quantile_method": "linear",
            },
        )
        self.assertEqual(summary["split_seed"], 3369)
        self.assertEqual(summary["clustering"]["mode"], "existing_cluster_csv")
        self.assertEqual(
            summary["training_csv_sha256"],
            hashlib.sha256(result.training_csv.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            {path.name for path in result.training_csv.parent.iterdir()},
            {"train_strata.csv", "summary.json"},
        )
        records = read_training_records(result.training_csv, self.images)
        self.assertEqual(len(records), 48)
        for fold in (0, 1):
            training, validation = split_records(records, fold_count=2, fold_index=fold, seed=3369)
            self.assertEqual(len(validation), 24)
            self.assertEqual({r.fold_index for r in validation}, {fold})
            self.assertTrue(
                {r.image_filename for r in validation}.isdisjoint(
                    r.image_filename for r in training
                )
            )
            self.assertEqual(len({r.stratification_label for r in validation}), 12)
            self.assertEqual(
                summary["folds"][str(fold)]["brightness_bin_counts"], {"0": 8, "1": 8, "2": 8}
            )
        with result.training_csv.open() as stream:
            for row in csv.DictReader(stream):
                self.assertEqual(
                    int(row["scene_cluster"]), int(row["image_filename"].split("_")[2])
                )
                brightness = float(row["brightness_mean"])
                expected_bin = 0 if brightness in (40, 50) else 1 if brightness in (60, 90) else 2
                self.assertEqual(int(row["brightness_bin"]), expected_bin)
        repeated = self.prepare("repeated")
        self.assertEqual(result.training_csv.read_bytes(), repeated.training_csv.read_bytes())
        with self.assertRaisesRegex(ValueError, "seed"):
            split_records(records, fold_count=2, fold_index=0, seed=42)

    def test_custom_bin_count_and_manual_thresholds(self):
        result = self.prepare(brightness_bins=5)
        self.assertEqual(
            json.loads(result.summary_path.read_text())["brightness_thresholds"],
            [50, 60, 90, 150],
        )
        manual = self.prepare("manual", brightness_thresholds=[100, 200])
        summary = json.loads(manual.summary_path.read_text())
        self.assertEqual(summary["brightness_strategy"], "explicit_thresholds")
        self.assertIsNone(summary["brightness_reference"])
        with manual.training_csv.open() as stream:
            for row in csv.DictReader(stream):
                if float(row["brightness_mean"]) == 200:
                    self.assertEqual(row["brightness_bin"], "2")

    def test_quantiles_follow_training_measurements_and_ignore_other_images(self):
        first = json.loads(self.prepare().summary_path.read_text())
        # Unlabeled images are outside the two training class directories.
        unrelated = self.images / "test"
        unrelated.mkdir()
        for index in range(12):
            cv2.imwrite(str(unrelated / f"test_{index}.png"), np.zeros((8, 8, 3), np.uint8))
        unchanged = json.loads(self.prepare("unrelated").summary_path.read_text())
        self.assertEqual(first["brightness_thresholds"], unchanged["brightness_thresholds"])
        for record in scan_labeled_images(self.images):
            pixels = cv2.imread(str(record.image_path))
            cv2.imwrite(str(record.image_path), pixels + 10)
        shifted = json.loads(self.prepare("shifted").summary_path.read_text())
        np.testing.assert_allclose(
            shifted["brightness_thresholds"], np.asarray(first["brightness_thresholds"]) + 10
        )
        self.assertEqual(shifted["brightness_bin_counts"], first["brightness_bin_counts"])
        self.assertEqual(shifted["brightness_reference"]["minimum"], 50)

    def test_impossible_quantiles_fail_before_vgg_and_leave_no_outputs(self):
        for record in scan_labeled_images(self.images):
            cv2.imwrite(str(record.image_path), np.full((8, 8, 3), 100, np.uint8))
        with patch("fan_classifier.features.extract_vgg16_features") as extractor:
            with self.assertRaisesRegex(ValueError, "brightness|quantile"):
                self.prepare(clusters_csv=None, pca_components=2)
            extractor.assert_not_called()
        self.assertFalse((self.root / "prepared").exists())
        self.assertEqual(list(self.root.glob(".prepare-strata-*")), [])

    def test_conflicting_or_invalid_configuration_fails_before_extraction(self):
        for overrides in (
            {"brightness_bins": 3, "brightness_thresholds": [85, 170]},
            {"brightness_bins": 1},
            {"brightness_bins": True},
            {"brightness_thresholds": [170, 85]},
            {"brightness_thresholds": [float("nan")]},
            {"brightness_thresholds": []},
            {"cluster_seed": -1},
            {"batch_size": 0},
            {"pca_components": 0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.prepare(**overrides)
            self.assertFalse((self.root / "prepared").exists())

    def test_id_mismatch_duplicates_invalid_clusters_rejected(self):
        records = scan_labeled_images(self.images)
        rows = [[record.image_filename, 0] for record in records]
        for invalid in (
            rows[:-1],
            rows + [["extra.png", 0]],
            rows + [rows[0]],
            [[rows[0][0], "1.0"], *rows[1:]],
            [[rows[0][0], -1], *rows[1:]],
        ):
            with self.subTest(rows=invalid):
                self.write_clusters(invalid)
                with self.assertRaises(ValueError):
                    read_scene_clusters(self.cluster_csv, records)
        self.write_clusters(
            [[r.image_filename, 0, 1] for r in records],
            ("image_filename", "cluster", "scene_cluster"),
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            read_scene_clusters(self.cluster_csv, records)

    def test_sparse_strata_and_corrupt_images_leave_no_outputs(self):
        with self.assertRaisesRegex(ValueError, "Sparse strata"):
            self.prepare(num_folds=5)
        self.assertFalse((self.root / "prepared").exists())
        self.assertEqual(list(self.root.glob(".prepare-strata-*")), [])
        next((self.images / "hold").glob("*.png")).write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "decoded"):
            self.prepare()
        self.assertFalse((self.root / "prepared").exists())

    def test_existing_output_never_overwritten(self):
        output = self.root / "prepared"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("unchanged")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "unchanged")

    def test_feature_mode_runs_real_pca_and_stores_numeric_artifacts(self):
        def synthetic_features(records, path, **kwargs):
            features = np.asarray(
                [
                    [1.0 + int(r.image_filename.split("_")[2]) * 100.0, i / 100]
                    for i, r in enumerate(records)
                ],
                dtype=np.float32,
            )
            np.save(path, features, allow_pickle=False)
            return np.asarray([cv2.imread(str(r.image_path)).mean() for r in records])

        with (
            patch("fan_classifier.features.extract_vgg16_features", side_effect=synthetic_features),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = self.prepare(clusters_csv=None, pca_components=2, cluster_count=2)
        summary = json.loads(result.summary_path.read_text())
        self.assertEqual(summary["clustering"]["mode"], "vgg16_pca_kmeans")
        self.assertEqual(summary["clustering"]["cluster_seed"], 42)
        self.assertEqual(summary["clustering"]["pca_components"], 2)
        with np.load(result.training_csv.parent / "clustering.npz", allow_pickle=False) as saved:
            self.assertEqual(saved["pca_components"].shape, (2, 2))
            self.assertEqual(len(np.unique(saved["scene_clusters"])), 2)
        self.assertEqual(len(read_training_records(result.training_csv, self.images)), 48)

    def test_cli_preparation_flows_into_real_training(self):
        from torch import nn

        def tiny_model(*args, **kwargs):
            return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))

        prepared = self.root / "prepared"
        arguments = [
            "prepare-strata",
            "--model",
            "eva_large_196",
            "--image-root",
            str(self.images),
            "--output-dir",
            str(prepared),
            "--clusters-csv",
            str(self.cluster_csv),
            "--brightness-bins",
            "2",
            "--num-folds",
            "2",
        ]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(arguments), 0)
            self.assertEqual(main(arguments), 2)
            with patch("fan_classifier.cli.build_model", side_effect=tiny_model):
                self.assertEqual(
                    main(
                        [
                            "train",
                            "--model",
                            "eva_large_196",
                            "--image-root",
                            str(self.images),
                            "--train-csv",
                            str(prepared / "train_strata.csv"),
                            "--output-dir",
                            str(self.root / "run"),
                            "--num-folds",
                            "2",
                            "--folds",
                            "0",
                            "--epochs",
                            "1",
                            "--batch-size",
                            "4",
                            "--device",
                            "cpu",
                            "--no-pretrained",
                        ]
                    ),
                    0,
                )
        actual = json.loads((self.root / "run" / "fold_0" / "fold_manifest.json").read_text())
        records = read_training_records(prepared / "train_strata.csv", self.images)
        expected = {r.image_filename for r in records if r.fold_index == 0}
        self.assertEqual(set(actual["split"]["validation_image_filenames"]), expected)


if __name__ == "__main__":
    unittest.main()
