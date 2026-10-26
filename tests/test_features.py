"""Feature extraction checks use generated images and never download weights."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from fan_classifier.data import ImageRecord
from fan_classifier.features import build_vgg16_feature_model, extract_vgg16_features


HAS_IMAGE_DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("numpy", "cv2"))


@unittest.skipUnless(HAS_IMAGE_DEPENDENCIES, "image dependencies are not installed")
class FeatureExtractionTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.feature_path = self.root / "private" / "features.npy"
        self.records = []
        self.colors = [(210, 80, 10), (12, 40, 170), (50, 90, 120), (5, 10, 20), (100, 70, 30)]
        # Deliberately nonalphabetical IDs expose accidental record sorting.
        for name, color in zip(("z", "b", "x", "a", "q"), self.colors):
            rgb_image = np.full((3, 7, 3), color, dtype=np.uint8)
            self.records.append(self.write_image(f"{name}.png", rgb_image))

    def write_image(self, filename, rgb_image):
        import cv2

        path = self.root / filename
        self.assertTrue(cv2.imwrite(str(path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)))
        return ImageRecord(filename, path, holding_label=1)

    @staticmethod
    def identity(images):
        return images

    @staticmethod
    def channel_means(images, *, training):
        if training:
            raise AssertionError("Feature inference must use training=False.")
        return images.mean(axis=(1, 2))[:, None, None, :]

    def extract(self, *, records=None, feature_model=None, preprocess=None, **options):
        return extract_vgg16_features(
            self.records if records is None else records,
            self.feature_path,
            feature_model=self.channel_means if feature_model is None else feature_model,
            preprocess=self.identity if preprocess is None else preprocess,
            **options,
        )

    def assert_no_partial_output(self):
        self.assertFalse(self.feature_path.exists())
        if self.feature_path.parent.exists():
            self.assertEqual(list(self.feature_path.parent.iterdir()), [])

    def test_rgb_preprocessing_order_brightness_and_final_partial_batch(self):
        batch_sizes = []

        def preprocess(rgb_batch):
            batch_sizes.append(len(rgb_batch))
            return rgb_batch / 255.0

        brightness = self.extract(batch_size=2, preprocess=preprocess)
        features = self.np.load(self.feature_path, mmap_mode="r", allow_pickle=False)
        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertEqual(features.shape, (5, 3))
        self.assertEqual(features.dtype, self.np.float32)
        self.np.testing.assert_allclose(features, self.np.array(self.colors) / 255, atol=0.0005)
        self.np.testing.assert_allclose(brightness, self.np.array(self.colors).mean(axis=1))

    def test_brightness_uses_original_pixel_values_before_resize_and_preprocess(self):
        rgb_image = self.np.zeros((3, 7, 3), dtype=self.np.uint8)
        rgb_image[0, 0] = [255, 80, 5]
        record = self.write_image("uneven.png", rgb_image)
        brightness = self.extract(records=[record], preprocess=lambda image: image + 100)
        self.assertAlmostEqual(float(brightness[0]), (255 + 80 + 5) / (3 * 7 * 3))

    def test_duplicate_image_ids_are_rejected_before_feature_extraction(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.extract(records=[self.records[0], self.records[0]])
        self.assert_no_partial_output()

    def test_corrupt_later_image_removes_partial_features(self):
        self.records[2].image_path.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            self.extract(batch_size=2)
        self.assert_no_partial_output()

    def test_nonfinite_and_zero_features_are_rejected(self):
        for fill_value, message in ((float("nan"), "nonfinite"), (0.0, "all-zero")):
            with self.subTest(fill_value=fill_value):

                def invalid_model(images, *, training):
                    return self.np.full((len(images), 3), fill_value)

                with self.assertRaisesRegex(ValueError, message):
                    self.extract(feature_model=invalid_model)
                self.assert_no_partial_output()

    def test_missing_feature_rows_and_changed_dimensions_are_rejected(self):
        def missing_row(images, *, training):
            return self.np.ones((len(images) - 1, 3))

        with self.assertRaisesRegex(ValueError, "one nonempty feature row"):
            self.extract(feature_model=missing_row)
        self.assert_no_partial_output()
        batch_count = 0

        def changing_dimension(images, *, training):
            nonlocal batch_count
            batch_count += 1
            return self.np.ones((len(images), batch_count))

        with self.assertRaisesRegex(ValueError, "dimension changed"):
            self.extract(batch_size=2, feature_model=changing_dimension)
        self.assert_no_partial_output()

    def test_model_failure_removes_partial_output(self):
        batches = 0

        def failing_model(images, *, training):
            nonlocal batches
            batches += 1
            if batches > 1:
                raise RuntimeError("synthetic failure")
            return self.channel_means(images, training=training)

        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            self.extract(batch_size=2, feature_model=failing_model)
        self.assert_no_partial_output()

    def test_existing_output_is_preserved(self):
        self.feature_path.parent.mkdir()
        self.feature_path.write_bytes(b"existing output")
        with self.assertRaises(FileExistsError):
            self.extract()
        self.assertEqual(self.feature_path.read_bytes(), b"existing output")

    def test_preprocessor_must_preserve_batch_shape(self):
        with self.assertRaisesRegex(ValueError, "preserve image shape"):
            self.extract(preprocess=lambda images: images[0])
        self.assert_no_partial_output()

    def test_empty_unlabeled_and_invalid_options_are_rejected(self):
        unlabeled = ImageRecord("z.png", self.records[0].image_path)
        for options in (
            {"records": []},
            {"records": [unlabeled]},
            {"batch_size": 0},
            {"batch_size": True},
            {"seed": -1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.extract(**options)
        self.assert_no_partial_output()

    def test_injected_model_requires_matching_preprocessor(self):
        with self.assertRaisesRegex(ValueError, "together"):
            extract_vgg16_features(
                self.records, self.feature_path, feature_model=self.channel_means
            )
        self.assert_no_partial_output()


class VGG16ArchitectureTests(unittest.TestCase):
    @unittest.skipUnless(
        HAS_IMAGE_DEPENDENCIES and importlib.util.find_spec("tensorflow"),
        "optional TensorFlow or image dependencies not installed",
    )
    def test_offline_vgg16_outputs_25088_features_per_image(self):
        import cv2
        import numpy as np
        from tensorflow.keras.applications.vgg16 import preprocess_input

        model = build_vgg16_feature_model(weights=None)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 7, 7, 512))
        self.assertFalse(model.trainable)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "synthetic.png"
            # OpenCV writes BGR; the RGB image is [210, 80, 10].
            self.assertTrue(
                cv2.imwrite(str(image_path), np.full((8, 11, 3), [10, 80, 210], dtype=np.uint8))
            )
            brightness = extract_vgg16_features(
                [ImageRecord("synthetic.png", image_path, holding_label=1)],
                root / "features.npy",
                feature_model=model,
                preprocess=preprocess_input,
            )
            features = np.load(root / "features.npy", allow_pickle=False)
            self.assertEqual(features.shape, (1, 25088))
            self.assertTrue(np.isfinite(features).all())
            self.assertAlmostEqual(float(brightness[0]), 100.0)
            # Independently construct Keras' BGR mean subtraction to check the
            # actual OpenCV -> RGB -> Keras path, using no pretrained weights.
            keras_input = np.full(
                (1, 224, 224, 3), [10 - 103.939, 80 - 116.779, 210 - 123.68], dtype=np.float32
            )
            expected = np.asarray(model(keras_input, training=False)).reshape(1, -1)
            np.testing.assert_allclose(features, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
