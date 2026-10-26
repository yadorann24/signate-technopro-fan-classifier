"""Triple-stratified split checks use numeric synthetic inputs only."""

import importlib.util
import json
import unittest
from collections import Counter

from fan_classifier.stratification import (
    build_triple_stratification,
    cluster_scene_features,
    quantile_brightness_thresholds,
)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
class BrightnessThresholdTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np

    def counts(self, brightness, thresholds):
        return self.np.bincount(
            self.np.searchsorted(thresholds, brightness, side="right"),
            minlength=len(thresholds) + 1,
        ).tolist()

    def test_skewed_distribution_produces_three_equal_image_counts(self):
        brightness = [0, 1, 2, 3, 4, 5, 100, 150, 250]
        thresholds = quantile_brightness_thresholds(brightness, 3)
        self.np.testing.assert_allclose(thresholds, [8 / 3, 110 / 3])
        self.assertEqual(self.counts(brightness, thresholds), [3, 3, 3])

    def test_nondivisible_image_count_differs_by_at_most_one(self):
        brightness = list(range(10))
        thresholds = quantile_brightness_thresholds(brightness, 3)
        self.assertEqual(thresholds, (3.0, 6.0))
        self.assertEqual(self.counts(brightness, thresholds), [3, 3, 4])

    def test_ties_remain_together_with_uneven_nonempty_bins(self):
        brightness = [10, 20, 20, 20, 30, 40]
        thresholds = quantile_brightness_thresholds(brightness, 3)
        self.np.testing.assert_allclose(thresholds, [20, 70 / 3])
        self.assertEqual(self.counts(brightness, thresholds), [1, 3, 2])

    def test_duplicate_or_empty_quantile_bins_fail_without_merging(self):
        for brightness in ([10, 10, 10, 10, 20, 30], [10, 10, 20, 20], [42] * 6):
            with self.subTest(brightness=brightness):
                with self.assertRaisesRegex(ValueError, "Lower brightness_bins.*no bins were merged"):
                    quantile_brightness_thresholds(brightness, 3)

    def test_bin_count_must_be_an_integer_bounded_by_image_count(self):
        for count in (0, 1, -1, 7, 3.0, True, "3", None):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "bin_count"):
                    quantile_brightness_thresholds([0, 10, 20, 30, 40, 50], count)
        with self.assertRaisesRegex(ValueError, "bin_count"):
            quantile_brightness_thresholds([10], 2)

    def test_continuous_brightness_does_not_impose_a_256_bin_limit(self):
        brightness = self.np.linspace(0, 255, 300)
        thresholds = quantile_brightness_thresholds(brightness, 300)
        self.assertEqual(len(thresholds), 299)
        self.assertEqual(self.counts(brightness, thresholds), [1] * 300)

    def test_thresholds_do_not_depend_on_image_order(self):
        brightness = [0, 1, 2, 3, 4, 5, 100, 150, 250]
        self.assertEqual(
            quantile_brightness_thresholds(brightness, 3),
            quantile_brightness_thresholds(brightness[::-1], 3),
        )

    def test_invalid_brightness_values_are_rejected(self):
        for brightness in (
            [], [float("nan"), 1], [float("inf"), 1], [-1, 1], [1, 256],
            [True, False], [1, True, 3], ["1", "2"], [[1], [2]], None,
        ):
            with self.subTest(brightness=brightness):
                with self.assertRaisesRegex(ValueError, "brightness_means"):
                    quantile_brightness_thresholds(brightness, 2)


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("sklearn"),
    "training dependencies are not installed",
)
class StratificationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.labels = [label for label in (0, 1) for _ in range(12)]
        self.scenes = [scene for _ in (0, 1) for scene in (0, 1) for _ in range(6)]
        self.brightness = [value for _ in range(4) for value in (0, 99, 99, 100, 150, 255)]

    def build(self, **overrides):
        arguments = dict(
            holding_labels=self.labels,
            scene_clusters=self.scenes,
            brightness_means=self.brightness,
            brightness_thresholds=[100],
            fold_count=3,
            seed=42,
        )
        arguments.update(overrides)
        return build_triple_stratification(**arguments)

    def test_triple_categories_threshold_boundaries_and_summary(self):
        result = self.build()
        self.assertEqual(result.stratification_labels[0], "label=0|scene=0|brightness=0")
        self.assertEqual(result.stratification_labels[3], "label=0|scene=0|brightness=1")
        self.np.testing.assert_array_equal(result.brightness_bins, [0, 0, 0, 1, 1, 1] * 4)
        self.assertEqual(result.summary["holding_label_counts"], {"0": 12, "1": 12})
        self.assertEqual(result.summary["brightness_bin_counts"], {"0": 12, "1": 12})
        self.assertEqual(set(result.summary["stratification_counts"].values()), {3})
        json.dumps(result.summary, allow_nan=False)

    def test_fold_assignments_are_disjoint_exhaustive_and_balanced(self):
        result = self.build()
        all_validation_indices = []
        for fold in range(3):
            training = set(self.np.flatnonzero(result.fold_indices != fold).tolist())
            validation = set(self.np.flatnonzero(result.fold_indices == fold).tolist())
            self.assertFalse(training & validation)
            self.assertEqual(training | validation, set(range(24)))
            all_validation_indices.extend(validation)
            self.assertEqual(Counter(self.labels[index] for index in validation), {0: 4, 1: 4})
            self.assertEqual(
                set(Counter(result.stratification_labels[index] for index in validation).values()),
                {1},
            )
            self.assertEqual(result.summary["folds"][str(fold)]["image_count"], 8)
            self.assertEqual(
                result.summary["folds"][str(fold)]["brightness_bin_counts"], {"0": 4, "1": 4}
            )
        self.assertEqual(sorted(all_validation_indices), list(range(24)))

    def test_repeated_seed_reproduces_assignments(self):
        self.np.testing.assert_array_equal(self.build().fold_indices, self.build().fold_indices)
        self.assertFalse(
            self.np.array_equal(self.build(seed=0).fold_indices, self.build(seed=42).fold_indices)
        )

    def test_sparse_strata_fail_with_counts_without_fallback(self):
        with self.assertRaisesRegex(ValueError, "label=0\\|scene=0\\|brightness=0: 3.*No fallback"):
            self.build(fold_count=4)

    def test_thresholds_must_be_explicit_sorted_unique_finite(self):
        for thresholds in ([], [100, 50], [100, 100], [-1], [256], [float("nan")], [True]):
            with self.subTest(thresholds=thresholds):
                with self.assertRaisesRegex(ValueError, "brightness_thresholds"):
                    self.build(brightness_thresholds=thresholds)

    def test_multiple_thresholds_and_endpoint_bins(self):
        result = build_triple_stratification(
            [0, 0, 0, 0, 0, 0], [0] * 6, [0, 0, 100, 100, 255, 255],
            brightness_thresholds=[0, 100, 255], fold_count=2,
        )
        self.np.testing.assert_array_equal(result.brightness_bins, [1, 1, 2, 2, 3, 3])
        self.assertEqual(result.summary["brightness_bin_counts"], {"0": 0, "1": 2, "2": 2, "3": 2})
        for fold in result.summary["folds"].values():
            self.assertEqual(fold["brightness_bin_counts"], {"0": 0, "1": 1, "2": 1, "3": 1})

    def test_quantile_boundary_values_enter_the_higher_bin(self):
        brightness = [10, 10, 20, 20, 20, 20, 20, 20, 30, 30, 40, 40]
        result = build_triple_stratification(
            [0] * 12,
            [0] * 12,
            brightness,
            brightness_thresholds=quantile_brightness_thresholds(brightness, 3),
            fold_count=2,
        )
        self.np.testing.assert_array_equal(result.brightness_bins, [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])

    def test_rejects_misaligned_counts_and_invalid_numeric_values(self):
        invalid = (
            {"scene_clusters": self.scenes[:-1]},
            {"brightness_means": self.brightness[:-1]},
            {"holding_labels": [2] * 24},
            {"holding_labels": [True] * 24},
            {"holding_labels": [0.0] * 24},
            {"scene_clusters": [-1] * 24},
            {"scene_clusters": ["0"] * 24},
            {"brightness_means": [float("inf")] * 24},
            {"brightness_means": [-0.1] * 24},
            {"brightness_means": [255.1] * 24},
            {"brightness_means": [[1]] * 24},
            {"fold_count": 1},
            {"fold_count": True},
            {"seed": -1},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.build(**kwargs)

    def test_clustering_is_deterministic_and_exposes_numeric_transform(self):
        features = self.np.random.default_rng(7).normal(size=(30, 8))
        first = cluster_scene_features(features, pca_components=4, cluster_count=3, seed=42)
        second = cluster_scene_features(features, pca_components=4, cluster_count=3, seed=42)
        self.np.testing.assert_array_equal(first.scene_clusters, second.scene_clusters)
        self.np.testing.assert_allclose(first.reduced_features, second.reduced_features)
        self.assertEqual(first.pca_components.shape, (4, 8))
        self.assertEqual(first.cluster_centers.shape, (3, 4))
        self.assertEqual(set(first.scene_clusters.tolist()), {0, 1, 2})
        self.np.testing.assert_allclose(
            (features - first.pca_mean) @ first.pca_components.T,
            first.reduced_features,
            atol=1e-10,
        )

    def test_pca_sample_and_dimension_limits_are_explicit(self):
        features = self.np.random.default_rng(7).normal(size=(6, 4))
        with self.assertRaisesRegex(ValueError, "no automatic dimension reduction"):
            cluster_scene_features(features, pca_components=5)
        with self.assertRaisesRegex(ValueError, "no automatic dimension reduction"):
            cluster_scene_features(features.T, pca_components=5)
        with self.assertRaisesRegex(ValueError, "cluster_count"):
            cluster_scene_features(features, pca_components=2, cluster_count=7)

    def test_identical_features_cannot_claim_scene_groups(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            cluster_scene_features(self.np.ones((8, 4)), pca_components=2)

    def test_collapsed_clusters_fail_instead_of_silently_reducing_count(self):
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        features = self.np.asarray([[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            with self.assertRaisesRegex(ValueError, "Fewer distinct image feature rows"):
                cluster_scene_features(features, pca_components=1, cluster_count=3)

    def test_invalid_feature_matrices_are_rejected(self):
        for features in ([1, 2], [[float("nan")], [0]], [[True], [False]], [[1]], [[], []]):
            with self.subTest(features=features):
                with self.assertRaises(ValueError):
                    cluster_scene_features(features, pca_components=1, cluster_count=2)


if __name__ == "__main__":
    unittest.main()
