"""Synthetic held-out partitions: verify best weights and reject contaminated evaluation."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fan_classifier.engine import EpochMetrics, fit_model
from fan_classifier.oof import evaluate_oof_runs, file_sha256


class OOFTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_saved_oof_restores_best_epoch_and_contains_only_held_out_images(self):
        model = torch.nn.Linear(2, 2)
        epoch_count = 0

        def train_epoch(model, loader, optimizer, **kwargs):
            nonlocal epoch_count
            epoch_count += 1
            with torch.no_grad():
                model.weight.zero_()
                model.bias.copy_(torch.tensor([0.0, 2.0] if epoch_count == 1 else [2.0, 0.0]))
            optimizer.step()
            return EpochMetrics(1, 0.5, 0.5, 1)

        validation = [
            {
                "image": torch.zeros(1, 2),
                "holding_label": torch.tensor([1]),
                "image_filename": ["held_out.png"],
            }
        ]
        with (
            patch("fan_classifier.engine.train_one_epoch", side_effect=train_epoch),
            patch(
                "fan_classifier.engine.evaluate",
                side_effect=[
                    EpochMetrics(1, 1, 1, 1),
                    EpochMetrics(2, 0, 0, 1),
                ],
            ),
        ):
            result = fit_model(
                model,
                [],
                validation,
                model_name="synthetic",
                image_size=2,
                fold=0,
                num_folds=2,
                output_directory=self.root / "fold_0",
                epochs=2,
                learning_rate=0.01,
                device="cpu",
                training_csv_sha256="a" * 64,
                fold_manifest={
                    "training_image_filenames": ["train.png"],
                    "validation_image_filenames": ["held_out.png"],
                    "validation_targets": {"held_out.png": 1},
                },
            )
        with result.oof_path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["image_filename"] for row in rows], ["held_out.png"])
        self.assertEqual(rows[0]["holding_label"], "1")
        self.assertEqual(rows[0]["fold_index"], "0")
        self.assertAlmostEqual(
            float(rows[0]["holding_probability"]),
            torch.tensor([0.0, 2.0]).softmax(0)[1].item(),
            places=6,
        )
        metadata = json.loads((result.oof_path.parent / "oof.json").read_text())
        self.assertEqual(metadata["best_epoch"], 1)
        self.assertEqual(metadata["checkpoint_sha256"], file_sha256(result.checkpoint_path))

    def test_non_held_out_loader_marks_training_failed_instead_of_saving_oof(self):
        batches = [
            {
                "image": torch.eye(2),
                "holding_label": torch.tensor([0, 1]),
                "image_filename": ["train.png", "held_out.png"],
            }
        ]
        with self.assertRaisesRegex(ValueError, "non-held-out"):
            fit_model(
                torch.nn.Linear(2, 2),
                batches,
                batches,
                model_name="synthetic",
                image_size=2,
                fold=0,
                num_folds=2,
                output_directory=self.root / "bad_fold",
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                fold_manifest={
                    "training_image_filenames": ["train.png"],
                    "validation_image_filenames": ["held_out.png"],
                },
            )
        self.assertFalse((self.root / "bad_fold" / "oof.csv").exists())
        metadata = json.loads((self.root / "bad_fold" / "training.json").read_text())
        self.assertEqual(metadata["status"], "failed")

    def make_run(self, name, probabilities, *, folds=(0, 0, 1, 1), targets=(0, 1, 0, 1)):
        run = self.root / name
        run.mkdir()
        names = [f"synthetic_{index}.png" for index in range(4)]
        config = {"model_name": name, "num_folds": 2, "seed": 42}
        manifest = {
            "format": "fan-classifier-run-v1",
            "config": config,
            "training_csv_sha256": "a" * 64,
            "sample_count": 4,
        }
        (run / "run.json").write_text(json.dumps(manifest))
        for fold in range(2):
            directory = run / f"fold_{fold}"
            directory.mkdir()
            valid = [index for index, assigned in enumerate(folds) if assigned == fold]
            split = {
                "training_image_filenames": [names[i] for i in range(4) if i not in valid],
                "validation_image_filenames": [names[i] for i in valid],
                "validation_targets": {names[i]: targets[i] for i in valid},
            }
            (directory / "fold_manifest.json").write_text(
                json.dumps({"fold": fold, "num_folds": 2, "seed": 42, "split": split})
            )
            metadata = {
                "status": "complete",
                "fold": fold,
                "best_epoch": 1,
                "model_name": name,
                "config": config,
                "training_csv_sha256": "a" * 64,
            }
            (directory / "training.json").write_text(json.dumps(metadata))
            checkpoint_path = directory / "best.pt"
            # Evaluation streams bytes to verify provenance; it does not load weights.
            checkpoint_path.write_bytes(f"synthetic checkpoint {name} fold {fold}".encode())
            csv_path = directory / "oof.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["image_filename", "holding_label", "fold_index", "holding_probability"]
                )
                writer.writerows(
                    (names[i], targets[i], fold, probabilities[i]) for i in reversed(valid)
                )
            (directory / "oof.json").write_text(
                json.dumps(
                    {
                        **metadata,
                        "format": "fan-classifier-oof-v1",
                        "sample_count": len(valid),
                        "oof_sha256": file_sha256(csv_path),
                        "checkpoint_sha256": file_sha256(checkpoint_path),
                    }
                )
            )
        return run

    def test_ensemble_aligns_ids_and_reports_different_fold_assignments(self):
        first = self.make_run("first", (0.1, 0.8, 0.6, 0.4))
        second = self.make_run("second", (0.2, 0.9, 0.1, 0.9), folds=(1, 0, 1, 0))
        report_path = evaluate_oof_runs([first, second], self.root / "comparison")
        report = json.loads(report_path.read_text())
        self.assertFalse(report["same_fold_assignments"])
        self.assertEqual(report["members"][0]["metrics"]["accuracy"], 0.5)
        self.assertEqual(report["members"][1]["metrics"]["accuracy"], 1)
        self.assertEqual(report["ensemble"]["metrics"]["accuracy"], 1)
        for index, run in enumerate((first, second)):
            for fold in range(2):
                provenance = report["members"][index]["fold_artifacts"][str(fold)]
                directory = run / f"fold_{fold}"
                self.assertEqual(provenance["best_epoch"], 1)
                self.assertEqual(provenance["checkpoint_path"], str(directory / "best.pt"))
                self.assertEqual(
                    provenance["checkpoint_sha256"], file_sha256(directory / "best.pt")
                )
                self.assertEqual(provenance["oof_sha256"], file_sha256(directory / "oof.csv"))
        with (report_path.parent / "probabilities.csv").open() as stream:
            rows = {row["image_filename"]: row for row in csv.DictReader(stream)}
        self.assertAlmostEqual(float(rows["synthetic_2.png"]["holding_probability"]), 0.35)
        with (report_path.parent / "errors.csv").open() as stream:
            errors = list(csv.DictReader(stream))
        self.assertEqual(len(errors), 2)
        self.assertEqual(
            {row["image_filename"] for row in errors}, {"synthetic_2.png", "synthetic_3.png"}
        )

    def test_partial_run_and_training_validation_overlap_are_rejected(self):
        first = self.make_run("first", (0.1, 0.8, 0.6, 0.4))
        (first / "fold_1" / "oof.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate_oof_runs([first], self.root / "partial")
        self.assertFalse((self.root / "partial").exists())
        second = self.make_run("second", (0.1, 0.8, 0.6, 0.4))
        path = second / "fold_0" / "fold_manifest.json"
        metadata = json.loads(path.read_text())
        metadata["split"]["training_image_filenames"].append("synthetic_0.png")
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate_oof_runs([second], self.root / "overlap")

    def test_changed_oof_content_targets_and_duplicate_runs_are_rejected(self):
        first = self.make_run("first", (0.1, 0.8, 0.6, 0.4))
        second = self.make_run("second", (0.1, 0.8, 0.6, 0.4), targets=(1, 1, 0, 1))
        with self.assertRaisesRegex(ValueError, "matching targets"):
            evaluate_oof_runs([first, second], self.root / "labels")
        with self.assertRaisesRegex(ValueError, "distinct run"):
            evaluate_oof_runs([first, first], self.root / "duplicates")
        path = first / "fold_0" / "oof.csv"
        path.write_text(path.read_text().replace("0.8", "0.1"))
        with self.assertRaisesRegex(ValueError, "content hash"):
            evaluate_oof_runs([first], self.root / "modified")

    def test_changed_or_missing_best_checkpoint_is_rejected_before_writing_report(self):
        for state in ("changed", "missing"):
            with self.subTest(state=state):
                run = self.make_run(state, (0.1, 0.8, 0.6, 0.4))
                checkpoint_path = run / "fold_0" / "best.pt"
                if state == "changed":
                    checkpoint_path.write_bytes(b"different synthetic checkpoint")
                    expected_error = self.assertRaisesRegex(ValueError, "checkpoint content hash")
                else:
                    checkpoint_path.unlink()
                    expected_error = self.assertRaises(FileNotFoundError)
                output_directory = self.root / f"report_{state}"
                with expected_error:
                    evaluate_oof_runs([run], output_directory)
                self.assertFalse(output_directory.exists())


if __name__ == "__main__":
    unittest.main()
