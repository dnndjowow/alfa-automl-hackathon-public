import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import ml_data
import ml_io
import ml_profile
import ml_schema
import ml_tools as mt
from tests.synth import make_task

DATA = Path(__file__).resolve().parent.parent / "data" / "public"
FAST = dict(max_rounds=80, learning_rate=0.2, n_jobs=2)


def prepare(task_md):
    data = ml_io.load_task(task_md)
    profile = ml_profile.build_profile(data)
    schema = ml_schema.resolve_schema(profile["schema"])
    return ml_data.split_task(data, schema), profile


class SyntheticToolsTest(unittest.TestCase):
    def prepare(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        task = make_task(tmp.name, **kwargs)
        split, profile = prepare(task.task_md)
        return split, profile, task

    def test_regression_eval_is_better_than_constant_and_json_safe(self):
        split, profile, _ = self.prepare(kind="regression", n_train=500)
        result = mt.train_eval(split, profile, metric="rmse", **FAST)
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual((result.metric, result.higher_is_better), ("rmse", False))
        baseline = float(np.std(split.y_train))
        self.assertLess(result.score, 0.6 * baseline)
        self.assertEqual(len(result.fold_scores), len(result.best_iterations))
        json.dumps(result.to_dict(), allow_nan=False)
        self.assertEqual(len(result.oof_pred), len(result.oof_index))

    def test_metric_defaults_come_from_profile_or_task(self):
        split, profile, _ = self.prepare(kind="multiclass", n_train=600)
        result = mt.train_eval(split, profile, **FAST)
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(result.metric, "f1_macro")  # в синтетическом task.md метрики нет
        self.assertEqual(result.oof_pred.shape[1], 3)
        self.assertGreater(result.score, 0.6)

    def test_classification_with_probability_metric(self):
        split, profile, _ = self.prepare(kind="binary", n_train=600)
        result = mt.train_eval(split, profile, metric="roc_auc", **FAST)
        self.assertEqual(result.status, "ok", result.error)
        self.assertGreater(result.score, 0.8)

    def test_log1p_target_transform_scores_on_original_scale(self):
        split, profile, _ = self.prepare(kind="regression", n_train=400)
        split.y_train = split.y_train - split.y_train.min() + 1.0
        plain = mt.train_eval(split, profile, **FAST)
        logged = mt.train_eval(split, profile, target_transform="log1p", **FAST)
        self.assertEqual((plain.status, logged.status), ("ok", "ok"))
        self.assertTrue(np.isfinite(logged.score))
        self.assertGreater(logged.oof_pred.min(), 0)

    def test_errors_are_results_not_exceptions(self):
        split, profile, _ = self.prepare(kind="regression")
        wrong_metric = mt.train_eval(split, profile, metric="f1_macro", **FAST)
        self.assertEqual(wrong_metric.status, "error")
        self.assertIn("f1_macro", wrong_metric.error)
        unknown = mt.train_eval(split, profile, metric="ndcg", **FAST)
        self.assertEqual(unknown.status, "error")
        no_model = mt.train_eval(split, profile, model="forest", **FAST)
        self.assertEqual(no_model.status, "error")
        multi = mt.train_eval(split, {**profile, "task": {"type": "multi_target"}}, **FAST)
        self.assertEqual(multi.status, "error")
        self.assertIn("не поддержан", multi.error)
        bad_transform = mt.train_eval(split, profile, target_transform="sqrt", **FAST)
        self.assertEqual(bad_transform.status, "error")

    def test_explicit_validation_overrides_suggestion(self):
        split, profile, _ = self.prepare(kind="regression")
        plan = {"strategy": "kfold", "n_splits": 3, "classification": False, "reason": "тест"}
        result = mt.train_eval(split, profile, validation=plan, **FAST)
        self.assertEqual(result.validation["reason"], "тест")
        self.assertEqual(len(result.fold_scores), 3)

    def test_subsample_and_time_budget(self):
        split, profile, _ = self.prepare(kind="regression", n_train=800)
        result = mt.train_eval(split, profile, subsample=0.5, time_budget=60, **FAST)
        self.assertEqual(result.status, "ok", result.error)
        self.assertTrue(any("подвыборка" in n for n in result.notes))
        self.assertEqual(len(result.oof_index), 400)

    def test_a_failing_fold_is_skipped_not_fatal(self):
        split, profile, _ = self.prepare(kind="regression", n_train=300)
        plan = {"strategy": "kfold", "n_splits": 3, "classification": False}
        real_score, calls = mt._score, []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("сбой во второй части")
            return real_score(*args, **kwargs)

        with mock.patch.object(mt, "_score", flaky):
            result = mt.train_eval(split, profile, validation=plan, **FAST)
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(len(result.fold_scores), 2)
        self.assertTrue(any("часть 2 пропущена" in n and "сбой" in n for n in result.notes))

    def test_all_folds_failing_gives_error_result(self):
        split, profile, _ = self.prepare(kind="regression", n_train=300)
        with mock.patch.object(mt, "_score", side_effect=RuntimeError("всегда")):
            result = mt.train_eval(split, profile, validation={"strategy": "kfold", "n_splits": 2}, **FAST)
        self.assertEqual(result.status, "error")
        self.assertIn("ни одна часть", result.error)

    def test_fit_predict_regression_and_classification(self):
        split, profile, task = self.prepare(kind="regression", n_train=500)
        evaluated = mt.train_eval(split, profile, **FAST)
        rounds = mt.rounds_from(evaluated)
        self.assertGreaterEqual(rounds, 1)
        final = mt.fit_predict(split, profile, n_rounds=rounds, **FAST)
        self.assertEqual(final["status"], "ok", final.get("error"))
        self.assertEqual(final["pred"].shape, (len(split.X_test),))
        self.assertTrue(np.isfinite(final["pred"]).all())

        split, profile, task = self.prepare(kind="multiclass", n_train=600)
        final = mt.fit_predict(split, profile, n_rounds=40, **FAST)
        self.assertEqual(final["proba"].shape, (len(split.X_test), 3))
        self.assertEqual(set(final["pred"]) <= set(final["classes"]), True)
        accuracy = float((final["pred"] == task.truth["y"].to_numpy()).mean())
        self.assertGreater(accuracy, 0.6)

    def test_fit_predict_reports_errors(self):
        split, profile, _ = self.prepare(kind="regression")
        self.assertEqual(mt.fit_predict(split, profile, model="forest")["status"], "error")

    def test_rounds_from_failed_result(self):
        self.assertIsNone(mt.rounds_from(mt.EvalResult(status="error")))

    def test_rounds_from_mean_by_default_and_largest_train_on_request(self):
        result = mt.EvalResult(status="ok", best_iterations=[62, 727, 29], train_sizes=[11000, 22000, 35000])
        self.assertEqual(mt.rounds_from(result, margin=1.0), 273)  # проверка на скрытых годах: среднее точнее
        self.assertEqual(mt.rounds_from(result, margin=1.0, how="largest_train"), 29)
        equal = mt.EvalResult(status="ok", best_iterations=[10, 20, 30, 40, 1000], train_sizes=[800] * 5)
        self.assertEqual(mt.rounds_from(equal, margin=1.0, how="largest_train"), 30)
        with self.assertRaises(ValueError):
            mt.rounds_from(result, how="max")

    def test_oof_index_points_into_the_original_train_after_subsample(self):
        # ревью: после подвыборки индексы указывали на строки уменьшенной таблицы
        split, profile, _ = self.prepare(kind="regression", n_train=1000)
        result = mt.train_eval(split, profile, subsample=0.5, **FAST)
        self.assertEqual(result.status, "ok", result.error)
        self.assertGreater(result.oof_index.max(), 600)
        truth = split.y_train.to_numpy()[result.oof_index]
        rmse = float(np.sqrt(np.mean((truth - result.oof_pred) ** 2)))
        self.assertAlmostEqual(rmse, result.oof_score, places=6)

    def test_exhausted_budget_skips_remaining_folds(self):
        # ревью: каждая следующая часть получала минимум секунду даже без остатка бюджета
        split, profile, _ = self.prepare(kind="regression", n_train=2000)
        plan = {"strategy": "kfold", "n_splits": 5, "classification": False}
        real_time, calls = mt.time.time, [0]

        def clock():  # первые вызовы — старт и первая часть, затем время «прыгает» за бюджет
            calls[0] += 1
            return real_time() + (0.0 if calls[0] <= 3 else 1000.0)

        with mock.patch.object(mt.time, "time", clock):
            result = mt.train_eval(split, profile, validation=plan, time_budget=10, **FAST)
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(len(result.fold_scores), 1)
        self.assertTrue(any("части 2–5 не посчитаны" in n for n in result.notes), result.notes)
        self.assertFalse(any("пропущена" in n for n in result.notes), result.notes)

    def test_effective_validation_is_reported(self):
        split, profile, _ = self.prepare(kind="multiclass", n_train=600)
        result = mt.train_eval(split, profile, **FAST)
        self.assertIn("effective", result.validation)
        self.assertEqual(result.validation["effective"]["n_splits"], len(result.fold_scores))

    def test_fit_predict_without_rounds_uses_early_stopping(self):
        # ревью: без n_rounds обучалось max_rounds раундов без остановки
        split, profile, _ = self.prepare(kind="regression", n_train=600)
        final = mt.fit_predict(split, profile, max_rounds=3000, learning_rate=0.3, n_jobs=2)
        self.assertEqual(final["status"], "ok", final.get("error"))
        self.assertLess(final["n_rounds"], 3000)
        self.assertTrue(any("подобрано" in n for n in final["notes"]))

    def test_fit_predict_rejects_unknown_transform(self):
        split, profile, _ = self.prepare(kind="regression")
        self.assertEqual(mt.fit_predict(split, profile, target_transform="sqrt", **FAST)["status"], "error")


@unittest.skipUnless(DATA.exists(), "публичные данные не скачаны в data/public")
class PublicToolsSmokeTest(unittest.TestCase):
    def test_regression_pipeline_on_a_subsample(self):
        split, profile = prepare(DATA / "regression_case" / "task.md")
        result = mt.train_eval(split, profile, subsample=0.25, max_rounds=120, learning_rate=0.15, n_jobs=4)
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(result.validation["strategy"], "temporal")
        self.assertEqual(len(result.fold_scores), 3)
        self.assertLess(result.score, float(np.std(split.y_train)))

    def test_multiclass_pipeline_on_a_subsample(self):
        split, profile = prepare(DATA / "multiclass_case" / "task.md")
        result = mt.train_eval(split, profile, subsample=0.2, max_rounds=60, learning_rate=0.2, n_jobs=4)
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(result.validation["strategy"], "group")
        self.assertEqual(result.metric, "f1_macro")
        self.assertEqual(len(result.classes), 10)


if __name__ == "__main__":
    unittest.main()
