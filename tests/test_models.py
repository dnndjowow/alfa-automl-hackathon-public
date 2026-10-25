import os
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

import ml_metrics
from ml_models import BoostingModel, ModelError, default_n_jobs, prepare_params

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
                self.assertTrue((proba[:, 2] > 0).all() and (proba[:, 2] <= 1e-4).all())
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
                self.assertFalse((proba.argmax(axis=1) == 2).any())

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

    def test_early_stopping_follows_the_task_metric(self):
        # наблюдение Ивана: остановка шла по метрике библиотеки (logloss, l2), а не по метрике задачи
        X, y = make_data("classification", k=3, n=600)
        expected = {"lightgbm": "f1_macro", "xgboost": "task_metric", "catboost": "TotalF1:average=Macro"}
        for name in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(name):
                m = model(name, "classification", n_classes=3, eval_metric="f1_macro", max_rounds=200, custom_eval=True)
                m.fit(X.iloc[:400], y[:400], X.iloc[400:], y[400:])
                self.assertIn("f1_macro" if name != "catboost" else "по TotalF1", m.eval_note_)
                if name == "lightgbm":
                    logged = m.model_.evals_result_["valid_0"]
                elif name == "xgboost":
                    logged = m.model_.evals_result()["validation_0"]
                else:
                    logged = m.model_.get_evals_result()["validation"]
                self.assertIn(expected[name], logged)

    def test_round_and_dart_aliases_are_removed(self):
        # код-ревью 25.10 (F15, F16): nrounds молча урезал модель до 5 деревьев, boost='dart' проходил
        X, y = make_data("regression")
        m = model("lightgbm", "regression", params={"nrounds": 5}).fit(X, y, n_rounds=30)
        self.assertEqual(m.best_iteration_, 30)
        self.assertTrue(m.param_notes_)
        clean, notes = prepare_params("lightgbm", "regression", {"boost": "dart", "num_leaves": 15})
        self.assertEqual(clean, {"num_leaves": 15})
        self.assertTrue(notes)

    def test_custom_stopping_is_off_by_default(self):
        X, y = make_data("classification", k=3, n=400)
        for name in ("lightgbm", "xgboost"):
            with self.subTest(name):
                m = model(name, "classification", n_classes=3, eval_metric="f1_macro", max_rounds=100)
                m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
                self.assertIn("custom_stopping=True", m.eval_note_)
                self.assertIsNone(m._stop_custom)
        # ревью Ивана (25.10, №2): ступенчатые метрики (F1, accuracy) без флага не останавливают обучение
        # и у CatBoost, и у встроенных binary_error/multi_error — остановка по метрике библиотеки
        for name, metric in (("catboost", "f1_macro"), ("lightgbm", "accuracy"), ("xgboost", "accuracy")):
            with self.subTest(name=name, metric=metric):
                m = model(name, "classification", n_classes=3, eval_metric=metric, max_rounds=50)
                m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
                self.assertIn("ступенями", m.eval_note_)
                self.assertEqual((m._stop_native, m._stop_custom), (None, None))
        m = model("catboost", "classification", n_classes=3, eval_metric="f1_macro", max_rounds=50, custom_eval=True)
        m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
        self.assertIn("TotalF1", m.eval_note_)  # с флагом — встроенная метрика CatBoost

    def test_native_metrics_and_approximations(self):
        X, y = make_data("regression", n=400)
        m = model("lightgbm", "regression", eval_metric="mae", max_rounds=100).fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
        self.assertIn("l1", m.model_.evals_result_["valid_0"])
        Xc, yc = make_data("classification", k=3, n=400)
        m = model("lightgbm", "classification", n_classes=3, eval_metric="roc_auc", max_rounds=50)
        m.fit(Xc.iloc[:300], yc[:300], Xc.iloc[300:], yc[300:])
        self.assertIn("приближение", m.eval_note_)
        m = model("catboost", "classification", n_classes=3, eval_metric="balanced_accuracy", max_rounds=30)
        m.fit(Xc.iloc[:300], yc[:300], Xc.iloc[300:], yc[300:])
        self.assertIn("по умолчанию", m.eval_note_)

    def test_failing_custom_metric_does_not_crash_training(self):
        X, y = make_data("classification", k=3, n=300)
        broken = ml_metrics.Metric("broken", lambda yt, yp, labels: (_ for _ in ()).throw(ml_metrics.MetricError("x")), True,
                           "classification", "labels")
        m = model("lightgbm", "classification", n_classes=3, eval_metric=broken, max_rounds=50, custom_eval=True)
        m.fit(X.iloc[:200], y[:200], X.iloc[200:], y[200:])
        self.assertEqual(m.predict(X).shape, (300, 3))

    def test_dart_is_removed(self):
        # наблюдение Ивана: dart отключает раннюю остановку, и часть валидации съедает весь бюджет
        X, y = make_data("regression")
        for name, key in (("lightgbm", "boosting_type"), ("xgboost", "booster")):
            with self.subTest(name):
                m = model(name, "regression", params={key: "dart"}, max_rounds=200, early_stopping_rounds=5)
                m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
                self.assertNotIn(key, m.params)
                self.assertTrue(any("dart" in n for n in m.param_notes_))
                self.assertLess(m.best_iteration_, 200)

    def test_logloss_stays_finite_when_a_class_is_missing_from_fit(self):
        # наблюдение Ивана: вероятность 0 у отсутствующего класса давала logloss около 34 на строку
        X, y = make_data("classification", k=4)
        keep = y != 3
        proba = model("lightgbm", "classification", n_classes=4).fit(X[keep], y[keep]).predict(X)
        value = ml_metrics.compute("logloss", y, proba, labels=[0, 1, 2, 3])
        self.assertLess(value, 5.0)

    def test_xgboost_handles_new_categories_one_versus_rest(self):
        # на спрятанном году регрессии новые категории давали прогнозы до 1576 при реальных 37–200
        rng = np.random.default_rng(0)
        n = 3000
        cats = [f"c{i}" for i in range(40)]
        effect = dict(zip(cats, rng.normal(scale=3, size=40)))
        X = pd.DataFrame({"a": rng.normal(size=n), "cat": rng.choice(cats[:30], size=n)})
        y = X["a"] + X["cat"].map(effect) + rng.normal(scale=0.1, size=n)
        X["cat"] = pd.Categorical(X["cat"], categories=cats)
        test = pd.DataFrame({"a": rng.normal(size=500), "cat": pd.Categorical([None] * 500, categories=cats)})
        m = model("xgboost", "regression", max_rounds=200).fit(X, y)
        self.assertEqual(m.model_.get_params()["max_cat_to_onehot"], 256)
        pred = m.predict(test)
        self.assertTrue(np.isfinite(pred).all())
        self.assertLess(pred.max(), y.max() + 1)
        self.assertGreater(pred.min(), y.min() - 1)
        override = model("xgboost", "regression", max_rounds=20, params={"max_cat_to_onehot": 4}).fit(X, y)
        self.assertEqual(override.model_.get_params()["max_cat_to_onehot"], 4)

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
