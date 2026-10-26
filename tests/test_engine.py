"""Exercise the training/inference contract with tiny local torch modules."""

import json
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from fan_classifier.engine import (
        CLASS_MAPPING,
        CHECKPOINT_FORMAT,
        EpochMetrics,
        evaluate,
        fit_model,
    )
    from fan_classifier.prediction import predict_checkpoints, read_probabilities


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.addCleanup(self.temporary_directory.cleanup)

    def test_evaluation_uses_eval_mode_and_sample_weighted_metrics(self):
        class FixedLogits(torch.nn.Module):
            def forward(self, images):
                self.asserted_training = self.training
                return images

        model = FixedLogits()
        batches = [
            {
                "image": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
                "holding_label": torch.tensor([0, 1]),
            },
            {"image": torch.tensor([[0.0, 4.0]]), "holding_label": torch.tensor([0])},
        ]
        metrics = evaluate(model, batches, device="cpu")
        expected_loss = torch.nn.functional.cross_entropy(
            torch.cat([b["image"] for b in batches]), torch.tensor([0, 1, 0])
        ).item()
        self.assertFalse(model.asserted_training)
        self.assertEqual(metrics.sample_count, 3)
        self.assertAlmostEqual(metrics.accuracy, 2 / 3)
        self.assertAlmostEqual(metrics.f1, 2 / 3)
        self.assertAlmostEqual(metrics.loss, expected_loss, places=6)

    def test_fit_saves_loadable_checkpoint_and_records_selected_metric(self):
        model = torch.nn.Linear(2, 2)
        batches = [
            {"image": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "holding_label": torch.tensor([0, 1])}
        ]
        result = fit_model(
            model,
            batches,
            batches,
            model_name="synthetic_model",
            image_size=2,
            fold=0,
            num_folds=2,
            output_directory=self.root / "fit",
            epochs=2,
            learning_rate=0.01,
            selection_metric="accuracy",
            device="cpu",
            training_csv_sha256="a" * 64,
        )
        checkpoint = torch.load(result.checkpoint_path, weights_only=True)
        self.assertEqual(checkpoint["metadata"]["selection_metric"], "accuracy")
        self.assertEqual(checkpoint["metadata"]["class_mapping"], CLASS_MAPPING)
        self.assertEqual(checkpoint["metadata"]["fold"], 0)
        self.assertEqual(checkpoint["metadata"]["training_csv_sha256"], "a" * 64)
        self.assertEqual(json.loads(result.metadata_path.read_text())["status"], "complete")
        self.assertEqual(len(result.metrics_path.read_text().splitlines()), 3)
        restored = torch.nn.Linear(2, 2)
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        with self.assertRaises(FileExistsError):
            fit_model(
                model,
                batches,
                batches,
                model_name="synthetic_model",
                image_size=2,
                fold=0,
                output_directory=self.root / "fit",
                epochs=1,
                learning_rate=0.01,
                device="cpu",
            )

    @staticmethod
    def track_checkpoint_copies(model):
        """Make state.cpu() allocate a separate tensor, as CUDA-to-CPU copying does."""
        original_state_dict = model.state_dict
        snapshots = []
        live_copy_counts = []

        class DeviceStateTensor:
            def __init__(self, tensor):
                self.tensor = tensor

            def detach(self):
                return self

            def cpu(self):
                snapshot = self.tensor.detach().clone()
                snapshots.append(weakref.ref(snapshot))
                live_copy_counts.append(sum(ref() is not None for ref in snapshots))
                return snapshot

        def copied_state_dict():
            return {
                name: DeviceStateTensor(tensor) for name, tensor in original_state_dict().items()
            }

        return copied_state_dict, snapshots, live_copy_counts

    def test_checkpoint_copies_are_released_between_epochs_and_improvements(self):
        model = torch.nn.Linear(2, 2)
        copied_state_dict, snapshots, live_copy_counts = self.track_checkpoint_copies(model)
        live_copies_at_epoch_start = []

        def train_epoch(model, loader, optimizer, **kwargs):
            live_copies_at_epoch_start.append(sum(ref() is not None for ref in snapshots))
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(len(live_copies_at_epoch_start))
            optimizer.step()
            return EpochMetrics(loss=1.0, accuracy=0.5, f1=0.5, sample_count=2)

        with (
            patch.object(model, "state_dict", new=copied_state_dict),
            patch("fan_classifier.engine.train_one_epoch", side_effect=train_epoch),
            patch(
                "fan_classifier.engine.evaluate",
                side_effect=[
                    EpochMetrics(loss=loss, accuracy=0.5, f1=0.5, sample_count=2)
                    for loss in (3.0, 2.0, 1.0)
                ],
            ),
        ):
            result = fit_model(
                model,
                [],
                [],
                model_name="synthetic_model",
                image_size=2,
                fold=0,
                output_directory=self.root / "bounded_copies",
                epochs=3,
                learning_rate=0.01,
                device="cpu",
            )

        self.assertEqual(live_copies_at_epoch_start, [0, 0, 0])
        self.assertEqual(len(snapshots), 6)
        self.assertLessEqual(max(live_copy_counts), 2)
        self.assertTrue(all(ref() is None for ref in snapshots))
        checkpoint = torch.load(result.checkpoint_path, weights_only=True)
        self.assertEqual(checkpoint["metadata"]["best_epoch"], 3)
        restored = torch.nn.Linear(2, 2)
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        for parameter in restored.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 3.0))

    def test_checkpoint_failures_record_only_successfully_saved_best(self):
        for boundary in ("save", "replace"):
            for failed_attempt in (1, 2):
                with self.subTest(boundary=boundary, failed_attempt=failed_attempt):
                    model = torch.nn.Linear(2, 2)
                    batches = [{"image": torch.eye(2), "holding_label": torch.tensor([0, 1])}]
                    original_persist = torch.save if boundary == "save" else Path.replace
                    persistence_count = 0

                    def fail_persistence(*args, **kwargs):
                        nonlocal persistence_count
                        persistence_count += 1
                        if persistence_count == failed_attempt:
                            if boundary == "save":
                                Path(args[1]).write_bytes(b"interrupted checkpoint")
                            raise OSError(f"synthetic {boundary} failure")
                        return original_persist(*args, **kwargs)

                    target = (
                        "fan_classifier.engine.torch.save"
                        if boundary == "save"
                        else "fan_classifier.engine.Path.replace"
                    )
                    run_directory = self.root / f"failed_{boundary}_{failed_attempt}"
                    with (
                        patch(target, new=fail_persistence),
                        patch(
                            "fan_classifier.engine.evaluate",
                            side_effect=[
                                EpochMetrics(loss=loss, accuracy=0.5, f1=0.5, sample_count=2)
                                for loss in (2.0, 1.0)
                            ],
                        ),
                        self.assertRaisesRegex(OSError, f"synthetic {boundary} failure"),
                    ):
                        fit_model(
                            model,
                            batches,
                            batches,
                            model_name="synthetic_model",
                            image_size=2,
                            fold=0,
                            output_directory=run_directory,
                            epochs=2,
                            learning_rate=0.01,
                            device="cpu",
                        )

                    self.assertEqual(persistence_count, failed_attempt)
                    metadata = json.loads((run_directory / "training.json").read_text())
                    self.assertEqual(metadata["status"], "failed")
                    checkpoint_path = run_directory / "best.pt"
                    if failed_attempt == 1:
                        self.assertFalse(checkpoint_path.exists())
                        self.assertEqual(metadata["best_epoch"], 0)
                        self.assertIsNone(metadata["best_score"])
                    else:
                        checkpoint = torch.load(checkpoint_path, weights_only=True)
                        self.assertEqual(checkpoint["metadata"]["best_epoch"], 1)
                        self.assertEqual(checkpoint["metadata"]["best_score"], 2.0)
                        for key in ("best_epoch", "best_score"):
                            self.assertEqual(metadata[key], checkpoint["metadata"][key])
                        restored = torch.nn.Linear(2, 2)
                        restored.load_state_dict(checkpoint["state_dict"], strict=True)

    def make_checkpoint(self, fold, *, state_dict=None, metadata_overrides=None):
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
            model.bias.zero_()
        path = self.root / f"fold_{fold}.pt"
        metadata = {
            "format": CHECKPOINT_FORMAT,
            "model_name": "synthetic_model",
            "image_size": 2,
            "fold": fold,
            "num_folds": 2,
            "seed": 42,
            "num_classes": 2,
            "positive_class_index": 1,
            "class_mapping": CLASS_MAPPING,
        }
        metadata.update(metadata_overrides or {})
        torch.save(
            {
                "metadata": metadata,
                "state_dict": model.state_dict() if state_dict is None else state_dict,
            },
            path,
        )
        return path

    @staticmethod
    def loader_factory(image_size, normalization):
        if image_size != 2:
            raise AssertionError("Incorrect inference image size")
        from fan_classifier.normalization import imagenet_normalization

        if normalization != imagenet_normalization():
            raise AssertionError("Legacy checkpoint preprocessing must remain ImageNet")
        return [
            {
                "image": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                "image_filename": ["synthetic_a.png", "synthetic_b.png"],
            }
        ]

    def test_checkpoint_inference_never_downloads_weights_and_checks_folds(self):
        paths = [self.make_checkpoint(0), self.make_checkpoint(1)]
        spec = SimpleNamespace(alias="synthetic_model", image_size=2)
        with (
            patch("fan_classifier.registry.get_model_spec", return_value=spec),
            patch(
                "fan_classifier.registry.build_model",
                side_effect=lambda *args, **kwargs: torch.nn.Linear(2, 2),
            ) as factory,
        ):
            result = predict_checkpoints(
                paths,
                self.loader_factory,
                output_directory=self.root / "predict",
                device="cpu",
                expected_folds=[0, 1],
            )
            probabilities = read_probabilities(result.probability_path)
            self.assertGreater(probabilities["synthetic_a.png"], 0.5)
            self.assertLess(probabilities["synthetic_b.png"], 0.5)
            for call in factory.call_args_list:
                self.assertFalse(call.kwargs["pretrained"])
            with self.assertRaisesRegex(ValueError, "Expected trained folds"):
                predict_checkpoints(
                    paths[:1],
                    self.loader_factory,
                    output_directory=self.root / "missing_fold",
                    device="cpu",
                    expected_folds=[0, 1],
                )
            self.assertFalse((self.root / "missing_fold").exists())

    def test_checkpoint_state_dict_shape_is_validated(self):
        path = self.make_checkpoint(
            0, state_dict={"weight": torch.zeros(3, 2), "bias": torch.zeros(3)}
        )
        spec = SimpleNamespace(alias="synthetic_model", image_size=2)
        with (
            patch("fan_classifier.registry.get_model_spec", return_value=spec),
            patch("fan_classifier.registry.build_model", return_value=torch.nn.Linear(2, 2)),
        ):
            with self.assertRaises(RuntimeError):
                predict_checkpoints(
                    [path],
                    self.loader_factory,
                    output_directory=self.root / "invalid",
                    device="cpu",
                    expected_folds=[0],
                )

    def test_inference_consumes_saved_normalization_without_resolving_timm_defaults(self):
        from fan_classifier.normalization import imagenet_normalization, restore_normalization

        normalization = {
            "mode": "custom",
            "mean": [0.1, 0.2, 0.3],
            "std": [0.2, 0.3, 0.4],
            "source": "custom",
        }
        path = self.make_checkpoint(
            0,
            metadata_overrides={
                "config": {"normalization": normalization},
                "normalization": normalization,
            },
        )
        observed = []

        def loader(image_size, saved):
            observed.append(saved)
            return self.loader_factory(image_size, imagenet_normalization())

        spec = SimpleNamespace(alias="synthetic_model", image_size=2)
        with (
            patch("fan_classifier.registry.get_model_spec", return_value=spec),
            patch("fan_classifier.registry.build_model", return_value=torch.nn.Linear(2, 2)),
            patch("fan_classifier.normalization.resolve_normalization", side_effect=AssertionError),
        ):
            result = predict_checkpoints(
                [path],
                loader,
                output_directory=self.root / "custom_norm",
                device="cpu",
            )
        self.assertEqual(observed, [restore_normalization(normalization)])
        self.assertEqual(
            json.loads(result.metadata_path.read_text())["normalization"], normalization
        )

    def test_conflicting_checkpoint_normalization_is_rejected_before_model_creation(self):
        from fan_classifier.normalization import imagenet_normalization

        path = self.make_checkpoint(
            0,
            metadata_overrides={
                "config": {"normalization": imagenet_normalization().to_dict()},
                "normalization": {
                    "mode": "custom",
                    "mean": [0, 0, 0],
                    "std": [1, 1, 1],
                    "source": "custom",
                },
            },
        )
        spec = SimpleNamespace(alias="synthetic_model", image_size=2)
        with (
            patch("fan_classifier.registry.get_model_spec", return_value=spec),
            patch("fan_classifier.registry.build_model") as factory,
            self.assertRaisesRegex(ValueError, "normalization conflicts"),
        ):
            predict_checkpoints(
                [path],
                self.loader_factory,
                output_directory=self.root / "conflict",
                device="cpu",
            )
        factory.assert_not_called()

    def test_checkpoints_must_match_expected_model_and_dataset(self):
        path = self.make_checkpoint(0, metadata_overrides={"training_csv_sha256": "a" * 64})

        def lookup(model_name):
            return SimpleNamespace(alias=model_name, image_size=2)

        with (
            patch("fan_classifier.registry.get_model_spec", side_effect=lookup),
            patch("fan_classifier.registry.build_model") as factory,
        ):
            with self.assertRaisesRegex(ValueError, "checkpoint model"):
                predict_checkpoints(
                    [path],
                    self.loader_factory,
                    output_directory=self.root / "wrong_model",
                    device="cpu",
                    expected_model="different_model",
                )
            with self.assertRaisesRegex(ValueError, "training CSV hash"):
                predict_checkpoints(
                    [path],
                    self.loader_factory,
                    output_directory=self.root / "wrong_data",
                    device="cpu",
                    expected_model="synthetic_model",
                    expected_training_csv_sha256="b" * 64,
                )
            factory.assert_not_called()

    def test_checkpoint_folds_cannot_mix_training_datasets(self):
        paths = [
            self.make_checkpoint(0, metadata_overrides={"training_csv_sha256": "a" * 64}),
            self.make_checkpoint(1, metadata_overrides={"training_csv_sha256": "b" * 64}),
        ]
        spec = SimpleNamespace(alias="synthetic_model", image_size=2)
        with (
            patch("fan_classifier.registry.get_model_spec", return_value=spec),
            patch(
                "fan_classifier.registry.build_model",
                side_effect=lambda *args, **kwargs: torch.nn.Linear(2, 2),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "training CSV hash"):
                predict_checkpoints(
                    paths, self.loader_factory, output_directory=self.root / "mixed", device="cpu"
                )
        self.assertFalse((self.root / "mixed").exists())

    def test_config_comparison_normalizes_json_and_rejects_different_runs(self):
        config = {"epochs": 2, "augmentation": {"gamma_limit": [80, 120]}}
        path = self.make_checkpoint(0, metadata_overrides={"config": config})
        spec = SimpleNamespace(alias="synthetic_model", image_size=2)
        with (
            patch("fan_classifier.registry.get_model_spec", return_value=spec),
            patch(
                "fan_classifier.registry.build_model",
                side_effect=lambda *args, **kwargs: torch.nn.Linear(2, 2),
            ) as factory,
        ):
            with self.assertRaisesRegex(ValueError, "training config"):
                predict_checkpoints(
                    [path],
                    self.loader_factory,
                    output_directory=self.root / "wrong_config",
                    device="cpu",
                    expected_config={**config, "epochs": 3},
                )
            factory.assert_not_called()
            result = predict_checkpoints(
                [path],
                self.loader_factory,
                output_directory=self.root / "same_config",
                device="cpu",
                expected_config={"augmentation": {"gamma_limit": (80, 120)}, "epochs": 2},
            )
            self.assertTrue(result.probability_path.is_file())
            second = self.make_checkpoint(1, metadata_overrides={"config": {**config, "epochs": 3}})
            with self.assertRaisesRegex(ValueError, "training config"):
                predict_checkpoints(
                    [path, second],
                    self.loader_factory,
                    output_directory=self.root / "mixed_config",
                    device="cpu",
                )

    def test_malformed_metadata_has_actionable_validation_errors(self):
        spec = SimpleNamespace(alias="synthetic_model", image_size=2)
        invalid_metadata = [
            ({"model_name": None}, "model_name"),
            ({"fold": True}, "fold metadata"),
            ({"num_folds": True}, "fold metadata"),
            ({"training_csv_sha256": "invalid"}, "SHA-256"),
            ({"config": "invalid"}, "config"),
        ]
        with (
            patch("fan_classifier.registry.get_model_spec", return_value=spec),
            patch("fan_classifier.registry.build_model") as factory,
        ):
            for overrides, message in invalid_metadata:
                with self.subTest(overrides=overrides):
                    path = self.make_checkpoint(0, metadata_overrides=overrides)
                    with self.assertRaisesRegex(ValueError, message):
                        predict_checkpoints(
                            [path],
                            self.loader_factory,
                            output_directory=self.root / "invalid",
                            device="cpu",
                        )
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
