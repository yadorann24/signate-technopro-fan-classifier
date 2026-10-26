"""Image checks use only generated arrays, never competition images."""

import importlib.util
import unittest

from fan_classifier.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms


class TransformArgumentTests(unittest.TestCase):
    def test_rejects_invalid_image_size_without_loading_libraries(self):
        for size in (0, -1, True, 16.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                build_transforms(size, training=False)

    def test_rejects_invalid_augmentation(self):
        for kwargs in (
            {"horizontal_flip_probability": 2},
            {"vertical_flip_probability": -1},
            {"gamma_limit": (120, 80)},
            {"scale_limit": 1},
            {"shift_limit": -0.1},
            {"rotate_limit": float("nan")},
            {"seed": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_transforms(32, training=True, **kwargs)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("albumentations", "cv2", "numpy", "torch")),
    "training dependencies are not installed",
)
class SyntheticImageTests(unittest.TestCase):
    def test_seed_reproduces_a_sequence_of_training_augmentations(self):
        import numpy as np
        import torch

        image = np.arange(15 * 30 * 3, dtype=np.uint8).reshape(15, 30, 3)
        first = build_transforms(32, training=True, seed=123, vertical_flip_probability=0.5)
        second = build_transforms(32, training=True, seed=123, vertical_flip_probability=0.5)
        for _ in range(5):
            self.assertTrue(torch.equal(first(image=image)["image"], second(image=image)["image"]))

    def test_validation_keeps_aspect_ratio_and_black_padding(self):
        import numpy as np
        import torch

        image = np.full((10, 20, 3), 255, dtype=np.uint8)
        transform = build_transforms(40, training=False)
        tensor = transform(image=image)["image"]
        self.assertEqual(tuple(tensor.shape), (3, 40, 40))
        self.assertEqual(tensor.dtype, torch.float32)
        recovered = tensor.numpy().transpose(1, 2, 0) * np.array(IMAGENET_STD) + np.array(
            IMAGENET_MEAN
        )
        np.testing.assert_allclose(recovered[:10], 0, atol=1e-6)
        np.testing.assert_allclose(recovered[10:30], 1, atol=1e-6)
        np.testing.assert_allclose(recovered[30:], 0, atol=1e-6)
        self.assertTrue(torch.equal(tensor, transform(image=image)["image"]))

    def test_dataset_converts_bgr_to_rgb_and_retains_label_and_id(self):
        import tempfile
        from pathlib import Path

        import cv2
        import numpy as np

        from fan_classifier.data import FanHoldingDataset, ImageRecord

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic_red.png"
            bgr = np.zeros((8, 8, 3), dtype=np.uint8)
            bgr[:, :, 2] = 255
            self.assertTrue(cv2.imwrite(str(path), bgr))
            record = ImageRecord("synthetic_red.png", path, 1, "synthetic_category")
            sample = FanHoldingDataset([record], build_transforms(8, training=False))[0]
            self.assertEqual(sample["holding_label"], 1)
            self.assertEqual(sample["image_filename"], "synthetic_red.png")
            self.assertAlmostEqual(
                float(sample["image"][0, 0, 0]), (1 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], places=5
            )
            self.assertAlmostEqual(
                float(sample["image"][2, 0, 0]), -IMAGENET_MEAN[2] / IMAGENET_STD[2], places=5
            )

    def test_corrupt_image_fails_with_clear_error(self):
        import tempfile
        from pathlib import Path

        from fan_classifier.data import FanHoldingDataset, ImageRecord

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corrupt.png"
            path.write_text("synthetic invalid image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decoded"):
                FanHoldingDataset([ImageRecord(path.name, path)])[0]


if __name__ == "__main__":
    unittest.main()
