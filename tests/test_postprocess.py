import unittest

import numpy as np
from sklearn.metrics import f1_score

import ml_postprocess as pp


def biased_problem(n=6000, seed=0, shift=3.0):
    """Три класса, редкий последний; shift — насколько модель занижает его вероятность."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.6, 0.35, 0.05])
    logits = rng.normal(size=(n, 3)) * 1.0
    logits[np.arange(n), y] += 1.6
    logits[:, 2] -= shift
    proba = np.exp(logits)
    return proba / proba.sum(axis=1, keepdims=True), y


class MacroF1Test(unittest.TestCase):
    def test_matches_sklearn_including_absent_classes(self):
        rng = np.random.default_rng(1)
        y, pred = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
        self.assertAlmostEqual(pp.macro_f1_codes(y, pred, 4), f1_score(y, pred, average="macro"))
        # класс 4 не встречается ни в y, ни в pred: sklearn его не учитывает, мы тоже
        self.assertAlmostEqual(pp.macro_f1_codes(y, pred, 6), f1_score(y, pred, average="macro"))

    def test_perfect_and_worst(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        self.assertEqual(pp.macro_f1_codes(y, y, 3), 1.0)
        self.assertEqual(pp.macro_f1_codes(y, (y + 1) % 3, 3), 0.0)


class ClassWeightsTest(unittest.TestCase):
    def test_weights_raise_macro_f1_on_fit_data(self):
        proba, y = biased_problem()
        result = pp.fit_class_weights(proba, y)
        self.assertGreater(result["score_after"], result["score_before"] + 0.05)
        self.assertGreater(result["weights"][2], 1.0)  # редкому классу вес повышен

    def test_gain_holds_on_separate_data(self):
        proba, y = biased_problem(n=9000, seed=2)
        outcome = pp.holdout_gain(proba, y)
        self.assertGreater(outcome["gain"], 0.0)

    def test_choose_applies_weights_only_when_gain_is_confirmed(self):
        proba, y = biased_problem(n=9000, seed=3, shift=3.0)
        chosen = pp.choose_class_weights(proba, y)
        self.assertTrue(chosen["applied"])
        self.assertGreater(chosen["holdout_gain"], 0.05)
        self.assertGreater(chosen["weights"][2], 1.0)
        self.assertGreater(chosen["score_after"], chosen["score_before"])

    def test_choose_returns_ones_when_calibrated(self):
        proba, y = biased_problem(n=6000, seed=0, shift=1.2)  # веса здесь подгоняются под шум
        chosen = pp.choose_class_weights(proba, y)
        self.assertFalse(chosen["applied"])
        np.testing.assert_array_equal(chosen["weights"], np.ones(3))

    def test_weights_never_hurt_on_fit_data_and_stay_one_if_nothing_helps(self):
        y = np.repeat([0, 1, 2], 100)
        proba = np.eye(3)[y] * 0.9 + 0.05  # уже идеально
        result = pp.fit_class_weights(proba, y)
        self.assertEqual(result["score_after"], result["score_before"])
        np.testing.assert_array_equal(result["weights"], np.ones(3))

    def test_apply_class_weights(self):
        proba = np.array([[0.5, 0.4, 0.1], [0.3, 0.3, 0.4]])
        np.testing.assert_array_equal(pp.apply_class_weights(proba, np.ones(3)), [0, 2])
        np.testing.assert_array_equal(pp.apply_class_weights(proba, np.array([1.0, 2.0, 1.0])), [1, 1])

    def test_shape_errors(self):
        with self.assertRaises(ValueError):
            pp.fit_class_weights(np.ones((5, 3)), np.zeros(4, dtype=int))
        with self.assertRaises(ValueError):
            pp.fit_class_weights(np.ones(5), np.zeros(5, dtype=int))


if __name__ == "__main__":
    unittest.main()
