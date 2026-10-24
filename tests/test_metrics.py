import math
import unittest

import numpy as np
from sklearn import metrics as skm

import ml_metrics as mm


class RegressionMetricsTest(unittest.TestCase):
    def test_rmse_by_hand(self):
        # ошибки 1 и 3: sqrt((1 + 9) / 2)
        self.assertAlmostEqual(mm.compute("rmse", [0, 0], [1, 3]), math.sqrt(5))

    def test_mae_mse_r2_match_sklearn(self):
        rng = np.random.default_rng(0)
        y, p = rng.normal(size=50), rng.normal(size=50)
        self.assertAlmostEqual(mm.compute("mae", y, p), skm.mean_absolute_error(y, p))
        self.assertAlmostEqual(mm.compute("mse", y, p), skm.mean_squared_error(y, p))
        self.assertAlmostEqual(mm.compute("r2", y, p), skm.r2_score(y, p))

    def test_column_vector_predictions_are_accepted(self):
        self.assertAlmostEqual(mm.compute("rmse", [0, 0], [[1], [3]]), math.sqrt(5))

    def test_nan_and_inf_are_rejected(self):
        with self.assertRaises(mm.MetricError):
            mm.compute("rmse", [1, 2], [1, float("nan")])
        with self.assertRaises(mm.MetricError):
            mm.compute("rmse", [1, 2], [1, float("inf")])
        with self.assertRaises(mm.MetricError):
            mm.compute("rmse", [1, float("nan")], [1, 2])

    def test_length_mismatch_and_empty(self):
        with self.assertRaises(mm.MetricError):
            mm.compute("rmse", [1, 2, 3], [1, 2])
        with self.assertRaises(mm.MetricError):
            mm.compute("rmse", [], [])


class ClassificationMetricsTest(unittest.TestCase):
    y_true = ["a", "a", "b", "b", "c", "c", "c", "a"]
    y_pred = ["a", "b", "b", "b", "c", "a", "c", "a"]

    def test_f1_macro_matches_platform_formula(self):
        expected = skm.f1_score(self.y_true, self.y_pred, average="macro")
        self.assertAlmostEqual(mm.compute("f1_macro", self.y_true, self.y_pred), expected)

    def test_other_label_metrics_match_sklearn(self):
        self.assertAlmostEqual(
            mm.compute("accuracy", self.y_true, self.y_pred),
            skm.accuracy_score(self.y_true, self.y_pred),
        )
        self.assertAlmostEqual(
            mm.compute("f1_weighted", self.y_true, self.y_pred),
            skm.f1_score(self.y_true, self.y_pred, average="weighted"),
        )
        self.assertAlmostEqual(
            mm.compute("balanced_accuracy", self.y_true, self.y_pred),
            skm.balanced_accuracy_score(self.y_true, self.y_pred),
        )

    def test_integer_labels_work(self):
        self.assertAlmostEqual(mm.compute("accuracy", [0, 1, 1], [0, 1, 0]), 2 / 3)

    def test_missing_labels_are_rejected(self):
        with self.assertRaises(mm.MetricError):
            mm.compute("f1_macro", ["a", None], ["a", "a"])

    def test_logloss_with_labels_missing_from_fold(self):
        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        value = mm.compute("logloss", ["a", "b"], proba, labels=["a", "b", "c"])
        self.assertAlmostEqual(value, -(math.log(0.7) + math.log(0.8)) / 2)

    def test_logloss_binary_vector(self):
        value = mm.compute("logloss", [0, 1], [0.2, 0.9], labels=[0, 1])
        self.assertAlmostEqual(value, -(math.log(0.8) + math.log(0.9)) / 2)

    def test_roc_auc_binary_and_multiclass(self):
        self.assertAlmostEqual(mm.compute("roc_auc", [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        rng = np.random.default_rng(1)
        y = np.array(["x", "y", "z"] * 20)
        proba = rng.dirichlet(np.ones(3), size=60)
        expected = skm.roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=["x", "y", "z"])
        self.assertAlmostEqual(mm.compute("roc_auc", y, proba, labels=["x", "y", "z"]), expected)

    def test_average_precision_binary_and_multiclass(self):
        y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
        self.assertAlmostEqual(mm.compute("average_precision", y, p), skm.average_precision_score(y, p))
        self.assertEqual(mm.get_metric("PR-AUC").name, "average_precision")
        rng = np.random.default_rng(2)
        yy = np.array(["x", "y", "z"] * 20)
        proba = rng.dirichlet(np.ones(3), size=60)
        expected = np.mean([skm.average_precision_score(yy == c, proba[:, k]) for k, c in enumerate("xyz")])
        self.assertAlmostEqual(mm.compute("average_precision", yy, proba, labels=["x", "y", "z"]), expected)

    def test_roc_auc_column_count_mismatch(self):
        with self.assertRaises(mm.MetricError):
            mm.compute("roc_auc", ["a", "b", "c"], np.full((3, 2), 0.5))

    def test_nan_probabilities_are_rejected(self):
        with self.assertRaises(mm.MetricError):
            mm.compute("logloss", [0, 1], [0.5, float("nan")], labels=[0, 1])


class RegistryTest(unittest.TestCase):
    def test_aliases_resolve_to_one_metric(self):
        for alias in ("RMSE", "rmse", "Root Mean Squared Error"):
            self.assertEqual(mm.get_metric(alias).name, "rmse")
        for alias in ("Macro F1", "macro_f1", "F1-macro", "f1_macro"):
            self.assertEqual(mm.get_metric(alias).name, "f1_macro")
        self.assertEqual(mm.get_metric("LogLoss").name, "logloss")
        self.assertEqual(mm.get_metric("AUC").name, "roc_auc")

    def test_unknown_metric_is_an_explicit_error(self):
        with self.assertRaises(mm.MetricError) as ctx:
            mm.get_metric("ndcg@10")
        self.assertIn("rmse", str(ctx.exception))

    def test_direction(self):
        self.assertFalse(mm.get_metric("rmse").higher_is_better)
        self.assertTrue(mm.get_metric("f1_macro").higher_is_better)
        self.assertFalse(mm.get_metric("logloss").higher_is_better)

    def test_is_better_respects_direction_and_nan(self):
        self.assertTrue(mm.is_better("rmse", 1.0, 2.0))
        self.assertFalse(mm.is_better("rmse", 2.0, 1.0))
        self.assertTrue(mm.is_better("f1_macro", 0.9, 0.8))
        self.assertFalse(mm.is_better("f1_macro", 0.8, 0.8))
        self.assertFalse(mm.is_better("rmse", float("nan"), 2.0))
        self.assertTrue(mm.is_better("rmse", 2.0, float("nan")))

    def test_is_better_with_missing_scores(self):
        # ревью: score неудачного эксперимента равен None, и is_better падал с TypeError
        self.assertFalse(mm.is_better("rmse", None, 2.0))
        self.assertTrue(mm.is_better("rmse", 2.0, None))
        self.assertFalse(mm.is_better("f1_macro", None, None))
        self.assertTrue(mm.is_better("f1_macro", 0.5, float("nan")))

    def test_worst_value(self):
        self.assertEqual(mm.worst_value("rmse"), np.inf)
        self.assertEqual(mm.worst_value("f1_macro"), -np.inf)
        self.assertTrue(mm.is_better("rmse", 1e9, mm.worst_value("rmse")))


if __name__ == "__main__":
    unittest.main()
