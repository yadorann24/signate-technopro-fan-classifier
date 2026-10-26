"""Exercise the real CLI on newly generated images, using a tiny trainable model."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fan_classifier.cli import main


class CommandTests(unittest.TestCase):
    def test_model_discovery_and_errors(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["list-models"]), 0)
        self.assertEqual(len(output.getvalue().splitlines()), 6)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["describe-model", "--model", "eva_large_196"]), 0)
        self.assertEqual(json.loads(output.getvalue())["image_size"], 196)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["describe-model", "--model", "unknown"]), 2)

    def test_train_save_predict_and_partial_fold_contract(self):
        import cv2
        import numpy as np
        import torch
        from torch import nn

        def tiny_model(*args, **kwargs):
            return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            for directory in (images / "hold", images / "not-hold", root / "test"):
                directory.mkdir(parents=True)
            train_csv, test_csv = root / "train.csv", root / "test.csv"
            with train_csv.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image_filename", "holding_label", "stratification_label"])
                for label in (0, 1):
                    for index in range(4):
                        filename = f"synthetic_{label}_{index}.png"
                        image = np.full((20, 8, 3), 30 + label * 170 + index, dtype=np.uint8)
                        cv2.imwrite(
                            str(images / ("hold" if label else "not-hold") / filename), image
                        )
                        writer.writerow([filename, label, f"synthetic_class_{label}"])
            with test_csv.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image_filename"])
                for name in ("synthetic_b.png", "synthetic_a.png"):
                    cv2.imwrite(str(root / "test" / name), np.full((17, 9, 3), 90, dtype=np.uint8))
                    writer.writerow([name])
            train_args = [
                "train",
                "--model",
                "eva_large_196",
                "--train-csv",
                str(train_csv),
                "--image-root",
                str(images),
                "--output-dir",
                str(root / "run"),
                "--epochs",
                "1",
                "--batch-size",
                "3",
                "--num-folds",
                "2",
                "--device",
                "cpu",
                "--no-pretrained",
            ]
            predict_args = [
                "predict",
                "--model",
                "eva_large_196",
                "--run-dir",
                str(root / "run"),
                "--test-csv",
                str(test_csv),
                "--image-root",
                str(root / "test"),
                "--output-dir",
                str(root / "prediction"),
                "--device",
                "cpu",
            ]
            with (
                patch("fan_classifier.cli.build_model", side_effect=tiny_model),
                patch("fan_classifier.registry.build_model", side_effect=tiny_model),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(train_args + ["--folds", "0"]), 0)
                self.assertEqual(
                    main(predict_args), 2, "all-fold prediction must reject a missing fold"
                )
                self.assertFalse((root / "prediction").exists())
                self.assertEqual(main(train_args + ["--folds", "1"]), 0)
                training_path = root / "run" / "fold_1" / "training.json"
                completed_metadata = json.loads(training_path.read_text())
                training_path.write_text(json.dumps({**completed_metadata, "status": "failed"}))
                self.assertEqual(
                    main(predict_args), 2, "failed training must not masquerade as complete"
                )
                self.assertFalse((root / "prediction").exists())
                training_path.write_text(json.dumps(completed_metadata))
                self.assertEqual(main(predict_args), 0)
                self.assertEqual(
                    main(train_args + ["--folds", "0"]), 2, "must not overwrite trained fold"
                )
                self.assertEqual(main(predict_args), 2, "must not overwrite predictions")
            with (root / "prediction" / "probabilities.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [r["image_filename"] for r in rows], ["synthetic_b.png", "synthetic_a.png"]
            )
            self.assertTrue(all(0 <= float(r["holding_probability"]) <= 1 for r in rows))
            self.assertEqual(
                len((root / "prediction" / "submission.csv").read_text().splitlines()), 2
            )
            run = json.loads((root / "run" / "run.json").read_text())
            self.assertEqual(run["config"]["model_name"], "eva_large_196")
            self.assertEqual(run["config"]["num_folds"], 2)
            self.assertEqual(len(run["training_csv_sha256"]), 64)
            checkpoint = torch.load(root / "run" / "fold_0" / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["metadata"]["num_classes"], 2)
            self.assertEqual(checkpoint["metadata"]["normalization"]["mode"], "pretrained")
            self.assertEqual(
                checkpoint["metadata"]["normalization"],
                checkpoint["metadata"]["config"]["normalization"],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "evaluate-oof",
                            "--run-dir",
                            str(root / "run"),
                            "--output-dir",
                            str(root / "evaluation"),
                        ]
                    ),
                    0,
                )
            report = json.loads((root / "evaluation" / "evaluation.json").read_text())
            self.assertEqual(report["members"][0]["metrics"]["sample_count"], 8)
            for fold in range(2):
                with (root / "run" / f"fold_{fold}" / "oof.csv").open() as stream:
                    oof = list(csv.DictReader(stream))
                split = json.loads(
                    (root / "run" / f"fold_{fold}" / "fold_manifest.json").read_text()
                )["split"]
                self.assertEqual(
                    {r["image_filename"] for r in oof}, set(split["validation_image_filenames"])
                )
                self.assertFalse(
                    {r["image_filename"] for r in oof} & set(split["training_image_filenames"])
                )
            # Recreate the legacy serialization shape to exercise the complete
            # run.json -> checkpoint comparison -> inference transform contract.
            import shutil
            from fan_classifier.normalization import IMAGENET_MEAN, IMAGENET_STD
            from fan_classifier.transforms import build_transforms

            legacy = root / "legacy"
            shutil.copytree(root / "run", legacy)
            legacy_manifest = json.loads((legacy / "run.json").read_text())
            legacy_manifest["config"].pop("normalization")
            (legacy / "run.json").write_text(json.dumps(legacy_manifest))
            for fold in range(2):
                path = legacy / f"fold_{fold}" / "best.pt"
                payload = torch.load(path, weights_only=True)
                payload["metadata"].pop("normalization")
                payload["metadata"]["config"].pop("normalization")
                torch.save(payload, path)
            legacy_args = [str(legacy) if arg == str(root / "run") else arg for arg in predict_args]
            legacy_args = [
                str(root / "legacy_prediction") if arg == str(root / "prediction") else arg
                for arg in legacy_args
            ]
            with (
                patch("fan_classifier.registry.build_model", side_effect=tiny_model),
                patch("fan_classifier.config.resolve_normalization", side_effect=AssertionError),
                patch(
                    "fan_classifier.transforms.build_transforms", wraps=build_transforms
                ) as transform,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(legacy_args), 0)
            self.assertEqual(transform.call_count, 2)
            for call in transform.call_args_list:
                self.assertEqual(call.kwargs["mean"], IMAGENET_MEAN)
                self.assertEqual(call.kwargs["std"], IMAGENET_STD)


if __name__ == "__main__":
    unittest.main()
