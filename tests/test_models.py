import os
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

import ml_metrics
from ml_models import BoostingModel, ModelError, default_n_jobs

N_JOBS = 2


def make_data(task, n=400, k=3, seed=0, with_nan=True):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "cat": pd.Categorical(rng.choice(["x", "y", "z"], size=n), categories=["x", "y", "z", "w"]),
        }
    )
    if with_nan:
        X.loc[rng.random(n) < 0.1, "a"] = np.nan
        X.loc[rng.random(n) < 0.1, "cat"] = np.nan
    signal = 2 * X["a"].fillna(0) - X["b"] + X["cat"].cat.codes.clip(lower=0)
    if task == "regression":
        return X, (signal + rng.normal(scale=0.2, size=n)).to_numpy()
    cuts = np.quantile(signal, np.linspace(0, 1, k + 1)[1:-1])
    return X, np.digitize(signal, cuts)


def model(name, task, **kwargs):
    kwargs.setdefault("max_rounds", 60)
    kwargs.setdefault("learning_rate", 0.2)
    return BoostingModel(name, task, n_jobs=N_JOBS, **kwargs)


class EveryModelTest(unittest.TestCase):
    def test_regression_learns_the_signal(self):
        X, y = make_data("regression")
        for name in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(name):
                m = model(name, "regression").fit(X.iloc[:300], y[:300])
                pred = m.predict(X.iloc[300:])
                self.assertEqual(pred.shape, (100,))
                self.assertLess(ml_metrics.compute("rmse", y[300:], pred), 0.6 * np.std(y[300:]))

    def test_multiclass_probabilities(self):
        X, y = make_data("classification", k=3)
        for name in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(name):
                m = model(name, "classification", n_classes=3).fit(X.iloc[:300], y[:300])
                proba = m.predict(X.iloc[300:])
                self.assertEqual(proba.shape, (100, 3))
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
                accuracy = ml_metrics.compute("accuracy", y[300:], proba.argmax(axis=1))
                self.assertGreater(accuracy, 0.7)

    def test_binary_gives_two_columns(self):
        X, y = make_data("classification", k=2)
        for name in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(name):
                proba = model(name, "classification", n_classes=2).fit(X, y).predict(X)
                self.assertEqual(proba.shape, (400, 2))

    def test_class_missing_from_fit_gets_zero_probability(self):
        X, y = make_data("classification", k=4)
        keep = y != 2
        for name in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(name):
                m = model(name, "classification", n_classes=4).fit(X[keep], y[keep])
                proba = m.predict(X)
                self.assertEqual(proba.shape, (400, 4))
                self.assertTrue((proba[:, 2] == 0).all())
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_unseen_and_missing_values_in_predict(self):
        X, y = make_data("regression")
        test = X.iloc[:5].copy()
        test["a"] = np.nan
        test["cat"] = pd.Categorical([None] * 5, categories=["x", "y", "z", "w"])
        for name in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(name):
                pred = model(name, "regression").fit(X, y).predict(test)
                self.assertTrue(np.isfinite(pred).all())


class BehaviourTest(unittest.TestCase):
    def test_early_stopping_records_best_iteration(self):
        X, y = make_data("regression", n=600)
        for name in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(name):
                m = model(name, "regression", max_rounds=500, early_stopping_rounds=10)
                m.fit(X.iloc[:400], y[:400], X.iloc[400:], y[400:])
                self.assertTrue(1 <= m.best_iteration_ < 500, m.best_iteration_)

    def test_n_rounds_overrides_without_validation(self):
        X, y = make_data("regression")
        m = model("lightgbm", "regression", max_rounds=500).fit(X, y, n_rounds=25)
        self.assertEqual(m.best_iteration_, 25)

    def test_time_budget_stops_boosting_early(self):
        X, y = make_data("regression", n=3000)
        for name in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(name):
                m = model(name, "regression", max_rounds=100000, time_budget=0.5, learning_rate=0.05)
                started = time.time()
                m.fit(X, y)
                self.assertLess(time.time() - started, 15)
                self.assertTrue(m.stopped_by_time_)
                self.assertTrue(0 < m.best_iteration_ < 100000)
                self.assertTrue(np.isfinite(m.predict(X)).all())

    def test_no_stop_flag_when_training_finished_on_its_own(self):
        # ревью: флаг ставился по часам даже без прерывания
        X, y = make_data("regression")
        for name in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(name):
                m = model(name, "regression", max_rounds=20, time_budget=60).fit(X, y)
                self.assertFalse(m.stopped_by_time_)
                self.assertEqual(m.best_iteration_, 20)

    def test_time_stop_keeps_best_validation_iteration(self):
        # ревью: остановка LightGBM по времени возвращала последнюю итерацию вместо лучшей
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(4000, 8)), columns=[f"f{i}" for i in range(8)])
        y = 3 * X["f0"] + rng.normal(scale=0.1, size=4000)
        Xv = pd.DataFrame(rng.normal(size=(1000, 8)), columns=X.columns)
        yv = -3 * Xv["f0"]  # валидация противоположна train: лучшая итерация — самая первая
        for name in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(name):
                m = BoostingModel(name, "regression", n_jobs=2, max_rounds=100000, learning_rate=0.1,
                                  early_stopping_rounds=100000, time_budget=0.3)
                m.fit(X, y, Xv, yv)
                self.assertTrue(m.stopped_by_time_)
                self.assertLessEqual(m.best_iteration_, 2)

    def test_same_seed_same_prediction(self):
        X, y = make_data("regression")
        a = model("lightgbm", "regression", seed=3).fit(X, y).predict(X)
        b = model("lightgbm", "regression", seed=3).fit(X, y).predict(X)
        np.testing.assert_array_equal(a, b)

    def test_catboost_does_not_write_files(self):
        X, y = make_data("regression")
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                model("catboost", "regression").fit(X, y)
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(previous)

    def test_round_synonyms_are_removed(self):
        # ревью Ивана (Y6): early_stopping_round ронял fit без валидации, num_trees урезал модель
        X, y = make_data("regression")
        for params in ({"early_stopping_round": 10}, {"num_trees": 5}, {"n_estimators": 5}, {"iterations": 5}):
            for name in ("lightgbm", "xgboost", "catboost"):
                with self.subTest(params=params, name=name):
                    m = model(name, "regression", params=params).fit(X, y, n_rounds=30)
                    self.assertEqual(m.best_iteration_, 30)
                    self.assertTrue(m.param_notes_)

    def test_catboost_accepts_parameters_of_other_libraries(self):
        # ревью Ивана (Y7): CatBoost падал на max_depth, colsample_bytree, min_child_weight, gamma, class_weight
        X, y = make_data("classification", k=3)
        params = {"max_depth": 4, "colsample_bytree": 0.8, "min_child_weight": 2, "gamma": 1,
                  "class_weight": "balanced", "num_leaves": 31, "reg_lambda": 2.0}
        m = model("catboost", "classification", n_classes=3, params=params).fit(X, y)
        self.assertEqual(m.predict(X).shape, (400, 3))
        self.assertEqual(m.params, {"depth": 4, "rsm": 0.8, "auto_class_weights": "Balanced", "l2_leaf_reg": 2.0})
        self.assertEqual(len(m.param_notes_), 7)

    def test_errors(self):
        X, y = make_data("regression")
        with self.assertRaises(ModelError):
            BoostingModel("forest", "regression")
        with self.assertRaises(ModelError):
            BoostingModel("lightgbm", "ranking")
        with self.assertRaises(ModelError):
            BoostingModel("lightgbm", "classification")
        with self.assertRaises(ModelError):
            model("lightgbm", "regression").predict(X)
        bad = y.copy()
        bad[0] = np.nan
        with self.assertRaises(ModelError):
            model("lightgbm", "regression").fit(X, bad)
        with self.assertRaises(ModelError):
            model("lightgbm", "classification", n_classes=3).fit(X, np.where(y > 0, 5, 0))
        with self.assertRaises(ModelError):
            model("lightgbm", "classification", n_classes=3).fit(X, np.zeros(len(X), dtype=int))

    def test_default_n_jobs(self):
        self.assertTrue(1 <= default_n_jobs() <= 8)
        os.environ["ML_N_JOBS"] = "3"
        try:
            self.assertEqual(default_n_jobs(), 3)
        finally:
            del os.environ["ML_N_JOBS"]


if __name__ == "__main__":
    unittest.main()
