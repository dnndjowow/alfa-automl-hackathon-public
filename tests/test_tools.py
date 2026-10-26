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

    def test_fit_predict_does_not_depend_on_the_metric_in_profile(self):
        # ревью Ивана (R2): rmse первой в task.md + task_type=classification -> fit_predict падал
        split, profile, _ = self.prepare(kind="multiclass", n_train=300)
        wrong = {**profile, "metric": {**profile["metric"], "primary": "rmse"}}
        final = mt.fit_predict(split, wrong, task_type="classification", n_rounds=20, **FAST)
        self.assertEqual(final["status"], "ok", final.get("error"))
        self.assertEqual(final["proba"].shape[1], 3)

    def test_params_from_llm_do_not_break_final_fit(self):
        # ревью Ивана (Y6): train_eval проходил, а fit_predict падал или обучал урезанную модель
        split, profile, _ = self.prepare(kind="regression", n_train=300)
        for params in ({"early_stopping_round": 10}, {"num_trees": 40}):
            ev = mt.train_eval(split, profile, params=params, **FAST)
            fp = mt.fit_predict(split, profile, params=params, n_rounds=70, **FAST)
            self.assertEqual((ev.status, fp["status"]), ("ok", "ok"), (ev.error, fp.get("error")))
            self.assertEqual(fp["n_rounds"], 70)
            self.assertTrue(any("убран" in n for n in ev.notes))
            self.assertTrue(any("убран" in n for n in fp["notes"]))

    def test_train_eval_reports_the_stopping_metric(self):
        split, profile, _ = self.prepare(kind="multiclass", n_train=400)
        default = mt.train_eval(split, profile, metric="f1_macro", **FAST)
        self.assertTrue(any("custom_stopping=True" in n for n in default.notes), default.notes)
        result = mt.train_eval(split, profile, metric="f1_macro", custom_stopping=True, **FAST)
        self.assertTrue(any("ранняя остановка по f1_macro" in n for n in result.notes), result.notes)
        step = mt.train_eval(split, profile, metric="accuracy", **FAST)
        self.assertTrue(any("ступенями" in n for n in step.notes), step.notes)
        native = mt.train_eval(split, profile, metric="logloss", **FAST)
        self.assertTrue(any("ранняя остановка по multi_logloss" in n for n in native.notes), native.notes)
        split, profile, _ = self.prepare(kind="regression", n_train=300)
        split.y_train = split.y_train - split.y_train.min() + 1
        logged = mt.train_eval(split, profile, metric="mae", target_transform="log1p", **FAST)
        self.assertFalse(any("ранняя остановка" in n for n in logged.notes))  # при log1p — метрика библиотеки

    def test_task_type_override_drives_the_validation_choice(self):
        # код-ревью 25.10 (F14): профиль binary + task_type='regression' без n_rounds — авто-раунды выбирали
        # stratified без y и падали «для стратификации нужен y»
        from ml_data import SplitData
        from ml_schema import Schema
        X = pd.DataFrame({"x": np.arange(100.0)})
        split = SplitData(X, pd.Series([1.0, 2.0] * 50), X.iloc[:5], pd.DataFrame(index=range(5)), Schema((), ("y",), ("x",)))
        profile = {"task": {"type": "binary"}, "metric": {"primary": None}}
        out = mt.fit_predict(split, profile, task_type="regression", max_rounds=10, n_jobs=1)
        self.assertEqual(out["status"], "ok", out.get("error"))
        ev = mt.train_eval(split, profile, task_type="regression", max_rounds=10, n_jobs=1)
        self.assertEqual(ev.status, "ok", ev.error)

    def test_classes_keep_original_label_types_in_eval_and_fit(self):
        # просьба Ивана (R6): классы одного типа в train_eval и fit_predict, числовые метки остаются числами
        split, profile, _ = self.prepare(kind="multiclass", n_train=300)
        codes = {name: i for i, name in enumerate(sorted(split.y_train.unique()))}
        split.y_train = split.y_train.map(codes)  # целые метки 0, 1, 2
        ev = mt.train_eval(split, profile, **FAST)
        fp = mt.fit_predict(split, profile, n_rounds=20, **FAST)
        self.assertEqual(ev.classes, fp["classes"])
        self.assertTrue(all(isinstance(c, (int, np.integer)) for c in ev.classes))
        self.assertTrue(all(isinstance(c, (int, np.integer)) for c in fp["classes"]))

    def test_fit_predict_reports_stop_by_time(self):
        # просьба Ивана (R4): понять, что финальное обучение урезано бюджетом, без сравнения числа раундов
        split, profile, _ = self.prepare(kind="regression", n_train=2000)
        normal = mt.fit_predict(split, profile, n_rounds=30, **FAST)
        self.assertIs(normal["stopped_by_time"], False)
        cut = mt.fit_predict(split, profile, n_rounds=100000, max_rounds=100000, learning_rate=0.01,
                             n_jobs=2, time_budget=0.3)
        self.assertEqual(cut["status"], "ok", cut.get("error"))
        self.assertIs(cut["stopped_by_time"], True)
        self.assertLess(cut["n_rounds"], 100000)

    def test_default_metric_is_reported_not_silent(self):
        # просьба Ивана (R1): не подставлять метрику молча
        split, profile, _ = self.prepare(kind="multiclass", n_train=300)  # в синтетическом task.md метрики нет
        default = mt.train_eval(split, profile, **FAST)
        self.assertEqual((default.metric, default.metric_source), ("f1_macro", "default"))
        self.assertTrue(any("по умолчанию f1_macro" in n for n in default.notes))
        self.assertEqual(default.to_dict()["metric_source"], "default")
        explicit = mt.train_eval(split, profile, metric="accuracy", **FAST)
        self.assertEqual(explicit.metric_source, "explicit")
        self.assertFalse(any("по умолчанию" in n for n in explicit.notes))
        from_task = mt.train_eval(split, {**profile, "metric": {**profile["metric"], "primary": "accuracy"}}, **FAST)
        self.assertEqual(from_task.metric_source, "profile")

    def test_fit_predict_auto_rounds_uses_given_validation_and_survives_failed_parts(self):
        # просьба Ивана (R5): подбор раундов учитывает переданную схему и не падает из-за одной части
        split, profile, _ = self.prepare(kind="regression", n_train=600)
        plan = {"strategy": "kfold", "n_splits": 3, "classification": False}
        given = mt.fit_predict(split, profile, validation=plan, **FAST)
        self.assertEqual(given["status"], "ok", given.get("error"))
        self.assertTrue(any("схема kfold" in n for n in given["notes"]))
        real_fit, calls = mt.BoostingModel.fit, []

        def flaky(model, *args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("сбой во второй части")
            return real_fit(model, *args, **kwargs)

        with mock.patch.object(mt.BoostingModel, "fit", flaky):
            survived = mt.fit_predict(split, profile, validation=plan, **FAST)
        self.assertEqual(survived["status"], "ok", survived.get("error"))
        self.assertTrue(any("часть 2 пропущена" in n for n in survived["notes"]))
        with mock.patch.object(mt.BoostingModel, "fit", side_effect=RuntimeError("всегда")):
            failed = mt.fit_predict(split, profile, validation=plan, **FAST)
        self.assertEqual(failed["status"], "error")
        self.assertIn("ни на одной части", failed["error"])

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

    def test_xgboost_on_a_hidden_future_year_stays_sane(self):
        # регрессия: учимся на годах 1–4, проверяем на спрятанном 5-м, где есть новые категории;
        # до max_cat_to_onehot XGBoost давал RMSE 124–154 при 58.7 у среднего
        split, profile = prepare(DATA / "regression_case" / "task.md")
        X, y = split.X_train, split.y_train
        past, future = (X.Year_Factor < 5).to_numpy(), (X.Year_Factor == 5).to_numpy()
        hidden = ml_data.SplitData(X[past].reset_index(drop=True), y[past].reset_index(drop=True),
                                   X[future].reset_index(drop=True), pd.DataFrame(index=range(future.sum())), split.schema)
        final = mt.fit_predict(hidden, profile, model="xgboost", n_rounds=230, target_transform="log1p", n_jobs=4)
        truth = y[future].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((truth - final["pred"]) ** 2)))
        self.assertLess(rmse, 55.0)
        self.assertLess(final["pred"].max(), 1000)

    def test_multiclass_pipeline_on_a_subsample(self):
        split, profile = prepare(DATA / "multiclass_case" / "task.md")
        result = mt.train_eval(split, profile, subsample=0.2, max_rounds=60, learning_rate=0.2, n_jobs=4)
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(result.validation["strategy"], "group")
        self.assertEqual(result.metric, "f1_macro")
        self.assertEqual(len(result.classes), 10)


if __name__ == "__main__":
    unittest.main()


class TwoStageTest(unittest.TestCase):
    """Двухступенчатая схема «фон против остальных» при сдвиге train/test (ТЗ Ивана, 25.10)."""

    @staticmethod
    def make(n_train=1500, n_test=800, shift=3.0, bg_train=0.4, bg_test=0.55, seed=0):
        from ml_data import SplitData
        from ml_schema import Schema
        rng = np.random.default_rng(seed)

        def part(n, bg, offset):
            y = np.where(rng.random(n) < bg, "N", rng.choice(["A", "B", "C"], n))
            centers = {"N": 0.0, "A": 2.0, "B": -2.0, "C": 4.0}
            X = pd.DataFrame({"x": [centers[v] for v in y] + rng.normal(0, 1.0, n),
                              "z": rng.normal(0, 1, n), "s": rng.normal(offset, 1, n)})
            return X, pd.Series(y)

        Xtr, ytr = part(n_train, bg_train, 0.0)
        Xte, yte = part(n_test, bg_test, shift)
        split = SplitData(Xtr, ytr, Xte, pd.DataFrame(index=range(n_test)), Schema((), ("y",), ("x", "z", "s")))
        return split, yte, {"task": {"type": "multiclass"}, "metric": {"primary": "f1_macro"}}

    def test_applies_under_shift_with_a_background_class(self):
        split, y_test, profile = self.make()
        out = mt.fit_predict_two_stage(split, profile, models=("lightgbm",), n_jobs=2)
        self.assertEqual(out["status"], "ok", out.get("error"))
        self.assertTrue(out["applied"])
        self.assertEqual(out["background"], "N")
        self.assertGreater(out["auc"], mt.SHIFT_AUC)
        self.assertEqual(len(out["pred"]), len(split.X_test))
        self.assertTrue(set(out["pred"]) <= {"N", "A", "B", "C"})
        share_pred = float(np.mean(out["pred"] == "N"))
        self.assertAlmostEqual(share_pred, out["share_test"], delta=0.01)  # доля фона в прогнозе = оценке EM
        self.assertGreater(out["share_test"], out["share_train"])            # EM видит, что фона в test больше

    def test_skipped_without_shift_or_background(self):
        split, _, profile = self.make(shift=0.0, bg_test=0.4)
        out = mt.fit_predict_two_stage(split, profile, models=("lightgbm",), n_jobs=2)
        self.assertEqual((out["status"], out["applied"]), ("skipped", False))
        self.assertIn("сдвига", out["reason"])
        split, _, profile = self.make(bg_train=0.25)  # все четыре класса по ~25% < 30%: фона нет
        out = mt.fit_predict_two_stage(split, profile, models=("lightgbm",), n_jobs=2)
        self.assertEqual(out["status"], "skipped")
        self.assertIn("фонового", out["reason"])
        out = mt.fit_predict_two_stage(split, {"task": {"type": "regression"}, "metric": {"primary": "rmse"}}, n_jobs=2)
        self.assertEqual(out["status"], "skipped")


class TwoStageSafetyTest(unittest.TestCase):
    """Границы применения и общий бюджет: часы/ML заменены, порядок вызовов и постобработка — реальные."""

    def setUp(self):
        self.split, _, self.profile = TwoStageTest.make(n_train=120, n_test=10, bg_train=0.6)
        self.check = {"auc": 0.95, "shifted": True}  # старый формат check тоже поддерживается
        self.now = 100.0

    def clock(self):
        return self.now

    def fake_eval(self, data, profile, **kwargs):
        classes = ml_data.LabelCodec().fit(data.y_train).classes_
        return mt.EvalResult(status="ok", score=0.5, best_iterations=[5], classes=list(classes),
                             oof_index=np.arange(len(data.y_train)),
                             oof_pred=np.full((len(data.y_train), len(classes)), 1 / len(classes)))

    def fake_fit(self, data, profile, **kwargs):
        classes = list(ml_data.LabelCodec().fit(data.y_train).classes_)
        n = len(data.X_test)
        proba = np.tile([0.4, 0.6], (n, 1)) if classes == ["bg", "rest"] else np.full((n, len(classes)), 1 / len(classes))
        return {"status": "ok", "classes": classes, "proba": proba}

    @staticmethod
    def unit_weights(proba, *args, **kwargs):
        return {"applied": False, "weights": np.ones(proba.shape[1])}

    def test_constant_probabilities_choose_exact_count_including_zero_and_all(self):
        # F20: раньше одинаковые p давали 100% фона даже при EM-доле 0.
        with mock.patch.object(mt, "train_eval", side_effect=self.fake_eval), \
             mock.patch.object(mt, "fit_predict", side_effect=self.fake_fit), \
             mock.patch.object(mt.ml_postprocess, "choose_class_weights", side_effect=self.unit_weights):
            for share, expected in ((0.0, 0), (0.2, 2), (1.0, 10)):
                with self.subTest(share=share), mock.patch.object(mt, "_em_share", return_value=share):
                    out = mt.fit_predict_two_stage(self.split, self.profile, models=("lightgbm",), check=self.check)
                    self.assertEqual(out["status"], "ok", out.get("error"))
                    selected = out["pred"] == out["background"]
                    self.assertEqual(int(selected.sum()), expected)
                    np.testing.assert_array_equal(np.flatnonzero(selected), np.arange(expected))
        np.testing.assert_array_equal(mt._background_mask(np.array([.1, .9, .8, .2]), .5), [False, True, True, False])

    def test_regression_with_a_background_is_skipped_before_shift_or_training(self):
        # F24: три числовых значения и частый ноль не должны становиться классификацией.
        self.split.y_train = self.split.y_train.map({"N": 0., "A": 1., "B": 2., "C": 3.})
        profile = {"task": {"type": "regression"}, "metric": {"primary": "rmse"}}
        with mock.patch.object(mt, "shift_check") as shift, mock.patch.object(mt, "train_eval") as train:
            out = mt.fit_predict_two_stage(self.split, profile)
        self.assertEqual((out["status"], out["applied"]), ("skipped", False))
        self.assertIn("не классификация", out["reason"])
        shift.assert_not_called()
        train.assert_not_called()

    def test_non_macro_or_missing_metric_skips_with_a_reason(self):
        # F22: принятие весов Macro F1 не должно менять accuracy/logloss/неизвестную метрику.
        for primary in ("accuracy", "balanced_accuracy", "f1_weighted", "logloss", "ndcg", None):
            profile = {**self.profile, "metric": {"primary": primary}}
            with self.subTest(primary=primary), mock.patch.object(mt, "shift_check") as shift, \
                 mock.patch.object(mt, "train_eval") as train:
                out = mt.fit_predict_two_stage(self.split, profile)
                self.assertEqual((out["status"], out["applied"]), ("skipped", False))
                self.assertIn("Macro F1", out["reason"])
                if primary is None:
                    self.assertTrue(out["notes"])
                shift.assert_not_called()
                train.assert_not_called()

    def test_macro_alias_and_legacy_check_are_accepted(self):
        profile = {**self.profile, "metric": {"primary": "Macro F1"}}
        with mock.patch.object(mt, "train_eval", side_effect=self.fake_eval), \
             mock.patch.object(mt, "fit_predict", side_effect=self.fake_fit), \
             mock.patch.object(mt.ml_postprocess, "choose_class_weights", side_effect=self.unit_weights), \
             mock.patch.object(mt, "shift_check") as shift:
            out = mt.fit_predict_two_stage(self.split, profile, models=("lightgbm",), check=self.check)
        self.assertEqual(out["status"], "ok", out.get("error"))
        shift.assert_not_called()

    def test_empty_budget_starts_neither_shift_nor_training(self):
        with mock.patch.object(mt, "shift_check") as shift, mock.patch.object(mt, "train_eval") as train:
            out = mt.fit_predict_two_stage(self.split, self.profile, time_budget=0)
        self.assertEqual((out["status"], out["applied"]), ("skipped", False))
        self.assertIn("бюджет", out["reason"])
        shift.assert_not_called()
        train.assert_not_called()

    def test_incomplete_shift_is_not_used_for_training(self):
        check = {"status": "skipped", "complete": False, "shifted": False, "auc": None, "reason": "бюджет detector"}
        with mock.patch.object(mt, "train_eval") as train:
            out = mt.fit_predict_two_stage(self.split, self.profile, check=check)
        self.assertEqual((out["status"], out["applied"]), ("skipped", False))
        self.assertIn("detector", out["reason"])
        train.assert_not_called()

    def test_shift_consuming_the_budget_prevents_all_stage_calls(self):
        def slow_shift(*args, **kwargs):
            self.now += 11
            return self.check

        with mock.patch.object(mt.time, "monotonic", side_effect=self.clock), \
             mock.patch.object(mt, "shift_check", side_effect=slow_shift) as shift, \
             mock.patch.object(mt, "train_eval") as train, mock.patch.object(mt, "fit_predict") as fit:
            out = mt.fit_predict_two_stage(self.split, self.profile, time_budget=10)
        self.assertEqual(shift.call_args.kwargs["time_budget"], 10)
        self.assertEqual((out["status"], out["applied"]), ("skipped", False))
        train.assert_not_called()
        fit.assert_not_called()

    def test_timeout_after_eval_or_predict_does_not_start_later_stages(self):
        for phase in ("eval", "predict"):
            self.now = 100

            def evaluate(*args, **kwargs):
                if phase == "eval":
                    self.now += 11
                return self.fake_eval(*args, **kwargs)

            def predict(*args, **kwargs):
                self.now += 11
                return self.fake_fit(*args, **kwargs)

            with self.subTest(phase=phase), mock.patch.object(mt.time, "monotonic", side_effect=self.clock), \
                 mock.patch.object(mt, "train_eval", side_effect=evaluate) as train, \
                 mock.patch.object(mt, "fit_predict", side_effect=predict) as fit:
                out = mt.fit_predict_two_stage(self.split, self.profile, check=self.check, time_budget=10)
                self.assertEqual((out["status"], out["applied"]), ("skipped", False))
                self.assertNotIn("pred", out)
                self.assertEqual(train.call_count, 1)
                self.assertEqual(fit.call_count, 0 if phase == "eval" else 1)

    def test_stage_quotas_use_the_remaining_time_after_every_operation(self):
        eval_costs, fit_costs, quotas = iter([30, 10]), iter([20, 5]), []

        def shift(*args, **kwargs):
            self.now += 20
            return self.check

        def evaluate(*args, **kwargs):
            quotas.append(kwargs["time_budget"])
            self.now += next(eval_costs)
            return self.fake_eval(*args, **kwargs)

        def predict(*args, **kwargs):
            quotas.append(kwargs["time_budget"])
            self.now += next(fit_costs)
            return self.fake_fit(*args, **kwargs)

        with mock.patch.object(mt.time, "monotonic", side_effect=self.clock), \
             mock.patch.object(mt, "shift_check", side_effect=shift), \
             mock.patch.object(mt, "train_eval", side_effect=evaluate), \
             mock.patch.object(mt, "fit_predict", side_effect=predict), \
             mock.patch.object(mt.ml_postprocess, "choose_class_weights", side_effect=self.unit_weights):
            out = mt.fit_predict_two_stage(self.split, self.profile, models=("lightgbm",), time_budget=100)
        self.assertEqual(out["status"], "ok", out.get("error"))
        np.testing.assert_allclose(quotas, [20, 50 / 3, 15, 20])
        self.assertEqual(out["seconds"], 85)

    def test_timeout_during_postprocess_never_returns_a_candidate(self):
        for phase in ("weights", "em"):
            self.now = 100

            def weights(*args, **kwargs):
                if phase == "weights":
                    self.now += 11
                return self.unit_weights(*args, **kwargs)

            def em(*args, **kwargs):
                self.now += 11
                return .2

            with self.subTest(phase=phase), mock.patch.object(mt.time, "monotonic", side_effect=self.clock), \
                 mock.patch.object(mt, "train_eval", side_effect=self.fake_eval), \
                 mock.patch.object(mt, "fit_predict", side_effect=self.fake_fit), \
                 mock.patch.object(mt.ml_postprocess, "choose_class_weights", side_effect=weights), \
                 mock.patch.object(mt, "_em_share", side_effect=em) as estimate:
                out = mt.fit_predict_two_stage(self.split, self.profile, models=("lightgbm",), check=self.check, time_budget=10)
                self.assertEqual((out["status"], out["applied"]), ("skipped", False))
                self.assertNotIn("pred", out)
                self.assertEqual(estimate.call_count, 0 if phase == "weights" else 1)

    def test_shift_check_timeout_after_preprocessing_skips_before_fit(self):
        pre = mock.Mock()

        def prepare(frame):
            self.now += 11
            return pre

        pre.fit.side_effect = prepare
        with mock.patch.object(mt.time, "monotonic", side_effect=self.clock), \
             mock.patch.object(mt, "TabularPreprocessor", return_value=pre), \
             mock.patch("lightgbm.LGBMClassifier") as model:
            out = mt.shift_check(self.split.X_train, self.split.X_test, time_budget=10)
        self.assertEqual((out["status"], out["complete"], out["shifted"], out["auc"]), ("skipped", False, False, None))
        pre.transform.assert_not_called()
        model.assert_not_called()

    def test_shift_check_stops_during_fit_or_after_predict_without_a_partial_auc(self):
        for phase in ("fit", "predict"):
            self.now = 100
            learner = mock.Mock()

            def fit(X, y, **kwargs):
                if phase == "fit":
                    self.now += 11
                    kwargs["callbacks"][0](None)
                return learner

            def predict(X):
                self.now += 11
                return np.tile([.5, .5], (len(X), 1))

            learner.fit.side_effect = fit
            learner.predict_proba.side_effect = predict
            with self.subTest(phase=phase), mock.patch.object(mt.time, "monotonic", side_effect=self.clock), \
                 mock.patch("lightgbm.LGBMClassifier", return_value=learner) as model:
                out = mt.shift_check(self.split.X_train, self.split.X_test, time_budget=10)
                self.assertEqual((out["status"], out["complete"], out["auc"]), ("skipped", False, None))
                self.assertEqual(model.call_count, 1)
                self.assertEqual(learner.predict_proba.call_count, 0 if phase == "fit" else 1)
