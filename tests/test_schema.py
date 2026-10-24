import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ml_io
import ml_schema as ms
from tests.synth import make_task

DATA = Path(__file__).resolve().parent.parent / "data" / "public"


def schema_of(data):
    return ms.infer_schema(data.train, data.test, data.sample_submission, data.data_dictionary)


class SyntheticSchemaTest(unittest.TestCase):
    def load(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return ml_io.load_task(make_task(tmp.name, **kwargs).task_md)

    def test_custom_names_are_found(self):
        suggestion = schema_of(self.load(kind="regression", target="price", id_col="row"))
        self.assertEqual(suggestion["id_cols"], ["row"])
        self.assertEqual(suggestion["target_cols"], ["price"])
        self.assertEqual(suggestion["confidence"], "high")
        self.assertNotIn("row", suggestion["feature_cols"])
        self.assertNotIn("price", suggestion["feature_cols"])
        self.assertEqual(suggestion["output_columns"], ["row", "price"])

    def test_without_dictionary_and_sample_confidence_drops(self):
        data = self.load(kind="binary")
        only_files = ms.infer_schema(data.train, data.test)
        self.assertEqual(only_files["target_cols"], ["y"])
        self.assertEqual(only_files["id_cols"], ["id"])  # по имени столбца
        self.assertIn(only_files["confidence"], ("low", "medium"))
        self.assertTrue(any("имени" in note for note in only_files["notes"]))

    def test_disagreeing_sources_give_low_confidence_and_resolve_refuses(self):
        data = self.load(kind="regression")
        dictionary = data.data_dictionary.copy()
        dictionary.loc[dictionary["feature"] == "a", "role"] = "target"
        suggestion = ms.infer_schema(data.train, data.test, data.sample_submission, dictionary)
        self.assertEqual(suggestion["confidence"], "low")
        with self.assertRaises(ms.SchemaError):
            ms.resolve_schema(suggestion)

    def test_override_wins_over_suggestion(self):
        suggestion = schema_of(self.load(kind="regression"))
        schema = ms.resolve_schema(suggestion, id_cols=["id"], target_cols=["y"])
        self.assertEqual(schema.target_cols, ("y",))
        self.assertNotIn("y", schema.feature_cols)
        with self.assertRaises(ms.SchemaError):
            ms.resolve_schema(suggestion, target_cols=["no_such_column"])

    def test_no_target_found(self):
        data = self.load(kind="regression")
        suggestion = ms.infer_schema(data.test, data.test)
        self.assertEqual(suggestion["confidence"], "none")
        with self.assertRaises(ms.SchemaError):
            ms.resolve_schema(suggestion)

    def test_sample_columns_missing_in_train_are_noted(self):
        data = self.load(kind="multiclass", n_classes=3)
        sample = data.sample_submission.copy()
        for name in ("alpha", "beta"):
            sample[f"p_{name}"] = 0.0
        suggestion = ms.infer_schema(data.train, data.test, sample, data.data_dictionary)
        self.assertTrue(any("нет в train" in note for note in suggestion["notes"]))


class TaskTypeTest(unittest.TestCase):
    def test_types(self):
        rng = np.random.default_rng(0)
        self.assertEqual(ms.suggest_task_type(pd.Series(rng.normal(size=100)))["type"], "regression")
        self.assertEqual(ms.suggest_task_type(pd.Series(["a", "b"] * 50))["type"], "binary")
        self.assertEqual(ms.suggest_task_type(pd.Series(list("abcde") * 20))["type"], "multiclass")
        self.assertEqual(ms.suggest_task_type(pd.Series([0, 1] * 50))["type"], "binary")
        self.assertEqual(ms.suggest_task_type(pd.Series(rng.integers(0, 1000, 500)))["type"], "regression")
        self.assertEqual(ms.suggest_task_type(None)["type"], "unknown")

    def test_small_integer_target_is_ambiguous_and_text_decides(self):
        y = pd.Series(np.random.default_rng(1).integers(0, 5, 200))
        guess = ms.suggest_task_type(y)
        self.assertEqual((guess["type"], guess["ambiguous"]), ("multiclass", True))
        self.assertEqual(ms.suggest_task_type(y, "Постройте регрессионную модель")["type"], "regression")
        confirmed = ms.suggest_task_type(y, "Задача многоклассовой классификации")
        self.assertEqual((confirmed["type"], confirmed["ambiguous"]), ("multiclass", False))

    def test_text_conflict_with_clear_data_keeps_data(self):
        y = pd.Series(np.random.default_rng(2).normal(size=100))
        guess = ms.suggest_task_type(y, "классификация классификация")
        self.assertEqual(guess["type"], "regression")
        self.assertTrue(guess["notes"])


class TaskTypeEvidenceTest(unittest.TestCase):
    def test_many_integer_classes_follow_the_metric(self):
        # ревью: 30 целых классов при «классификации» и Macro F1 давали regression
        y = pd.Series(np.tile(np.arange(30), 10))
        guess = ms.suggest_task_type(y, "Многоклассовая классификация. Основная метрика — Macro F1", "f1_macro")
        self.assertEqual((guess["type"], guess["ambiguous"]), ("multiclass", False))
        self.assertEqual(ms.suggest_task_type(y, "Многоклассовая классификация")["type"], "multiclass")
        self.assertEqual(ms.suggest_task_type(y)["type"], "regression")  # без доводов — число

    def test_two_numeric_values_with_regression_evidence(self):
        y = pd.Series([1.5, 7.5] * 50)
        self.assertEqual(ms.suggest_task_type(y, "Постройте регрессионную модель")["type"], "regression")
        self.assertEqual(ms.suggest_task_type(y, "", "rmse")["type"], "regression")
        self.assertEqual(ms.suggest_task_type(y)["type"], "binary")

    def test_metric_beats_text(self):
        y = pd.Series(np.random.default_rng(0).integers(0, 5, 200))
        self.assertEqual(ms.suggest_task_type(y, "регрессия регрессия", "f1_macro")["type"], "multiclass")

    def test_firm_data_wins_but_is_flagged(self):
        labels = pd.Series(["a", "b", "c"] * 30)
        guess = ms.suggest_task_type(labels, "", "rmse")
        self.assertEqual((guess["type"], guess["ambiguous"]), ("multiclass", True))
        self.assertTrue(guess["notes"])

    def test_unknown_metric_falls_back_to_text(self):
        y = pd.Series(np.tile(np.arange(30), 10))
        guess = ms.suggest_task_type(y, "задача классификации", "ndcg")
        self.assertEqual(guess["type"], "multiclass")
        self.assertTrue(any("неизвестна" in n for n in guess["notes"]))


class RoleOverlapTest(unittest.TestCase):
    def test_target_placeholder_in_test_is_not_an_id(self):
        # ревью: заглушка y=0 в test совпадала с sample и становилась id
        train = pd.DataFrame({"id": [1, 2, 3], "x": [1, 2, 3], "y": [5, 6, 7]})
        test = pd.DataFrame({"id": [4, 5], "x": [1, 2], "y": [0, 0]})
        sample = pd.DataFrame({"id": [4, 5], "y": [0, 0]})
        dictionary = pd.DataFrame({"feature": ["id", "x", "y"], "role": ["id", "feature", "target"]})
        for d in (dictionary, None):
            suggestion = ms.infer_schema(train, test, sample, d)
            self.assertEqual(suggestion["id_cols"], ["id"])
        suggestion = ms.infer_schema(train, test, sample, dictionary)
        self.assertEqual(suggestion["target_cols"], ["y"])
        schema = ms.resolve_schema(suggestion)
        self.assertEqual((schema.id_cols, schema.target_cols), (("id",), ("y",)))

    def test_id_found_when_sample_rows_are_in_another_order(self):
        # ревью Ивана (Y13): при другом порядке строк client_id попадал в таргеты
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"client_id": range(100, 160), "x": rng.normal(size=60), "y": rng.normal(size=60)})
        test = pd.DataFrame({"client_id": range(200, 220), "x": rng.normal(size=20)})
        sample = pd.DataFrame({"client_id": test["client_id"][::-1].to_numpy(), "y": 0.0})
        suggestion = ms.infer_schema(train, test, sample)
        self.assertEqual((suggestion["id_cols"], suggestion["target_cols"]), (["client_id"], ["y"]))
        self.assertIn(suggestion["confidence"], ("high", "medium"))
        self.assertTrue(any("порядок" in n for n in suggestion["notes"]))
        by_name = ms.infer_schema(train, test)
        self.assertEqual(by_name["id_cols"], ["client_id"])

    def test_resolve_refuses_overlapping_roles(self):
        train = pd.DataFrame({"id": [1, 2, 3], "x": [1, 2, 3], "y": [5, 6, 7]})
        test = pd.DataFrame({"id": [4, 5], "x": [1, 2]})
        suggestion = ms.infer_schema(train, test)
        with self.assertRaises(ms.SchemaError):
            ms.resolve_schema(suggestion, id_cols=["id", "y"], target_cols=["y"])


class MetricMentionTest(unittest.TestCase):
    def test_primary_metric_before_or_after_the_phrase(self):
        # ревью Ивана (Y2): метрика искалась только после фразы «основная метрика»
        cases = {
            "Macro F1 — основная метрика соревнования. Accuracy приводится для справки.": "f1_macro",
            "MAE является основной метрикой; RMSE выводится для справки.": "mae",
            "Решения ранжируются по ROC AUC (основная метрика). Также считаем logloss.": "roc_auc",
        }
        for text, expected in cases.items():
            self.assertEqual(ms.mentioned_metrics(text)["primary"], expected, text)

    def test_metric_spellings(self):
        # ревью Ивана (Y3): написания macro F1, PR-AUC, бинарный F1, R^2
        cases = {
            "Основная метрика: macro-averaged F1. Accuracy тоже считаем.": "f1_macro",
            "Метрика — F1-score (macro). Также смотрим accuracy.": "f1_macro",
            "Основная метрика — F1-мера (macro).": "f1_macro",
            "Основная метрика — оценка по F1-мере (macro).": "f1_macro",  # ревью Ивана (25.10, №4): падеж
            "Основная метрика — PR-AUC (average precision).": "average_precision",
            "Метрика — R^2.": "r2",
        }
        for text, expected in cases.items():
            self.assertEqual(ms.mentioned_metrics(text)["primary"], expected, text)
        self.assertEqual(ms.mentioned_metrics("Основная метрика — по F1-мере.")["primary"], "f1")
        plain = ms.mentioned_metrics("Основная метрика: F1-score.")
        self.assertEqual((plain["primary"], plain["supported"], plain["higher_is_better"]), ("f1", False, True))
        self.assertEqual(ms.metric_family("f1"), "classification")

    def test_metrics_outside_the_registry_are_named_not_replaced(self):
        # N1: при основной RMSLE/MAPE ядро возвращало «первую упомянутую» RMSE/MAE
        cases = {
            "RMSLE — основная метрика соревнования. RMSE приводится для справки.": "rmsle",
            "Основная метрика — RMSE между логарифмами прогноза и ответа.": "rmsle",
            "MAPE является основной метрикой; MAE выводится для справки.": "mape",
            "Основная метрика: symmetric mean absolute percentage error.": "smape",
            "Основная метрика: mean squared log error. MSE для справки.": "msle",
        }
        for text, expected in cases.items():
            found = ms.mentioned_metrics(text)
            self.assertEqual((found["primary"], found["supported"], found["higher_is_better"]),
                             (expected, False, False), text)
            self.assertEqual(ms.metric_family(expected), "regression")

    def test_sklearn_call_with_macro_average_is_one_mention(self):
        found = ms.mentioned_metrics('f1_score(y_true, y_pred, average="macro")')
        self.assertEqual(found["mentioned"], ["f1_macro"])

    def test_balanced_accuracy_before_primary_accuracy(self):
        # ревью: замена текста сдвигала позиции, и основной выбиралась balanced_accuracy
        found = ms.mentioned_metrics("Дополнительно balanced accuracy. Основная метрика — accuracy.")
        self.assertEqual(found["primary"], "accuracy")
        self.assertEqual(found["mentioned"], ["balanced_accuracy", "accuracy"])
        found = ms.mentioned_metrics("Основная метрика — balanced accuracy, ещё смотрим accuracy.")
        self.assertEqual(found["primary"], "balanced_accuracy")

    def test_primary_metric_phrases(self):
        text = "Метрика оценки — **RMSE** на исходной шкале.\nМеньшее значение лучше."
        found = ms.mentioned_metrics(text)
        self.assertEqual(found["primary"], "rmse")
        self.assertFalse(found["higher_is_better"])

    def test_primary_over_secondary_and_balanced_accuracy(self):
        text = "Основная метрика — Macro F1. Дополнительно Accuracy и Weighted F1, balanced accuracy."
        found = ms.mentioned_metrics(text)
        self.assertEqual(found["primary"], "f1_macro")
        self.assertEqual(set(found["mentioned"]), {"f1_macro", "accuracy", "f1_weighted", "balanced_accuracy"})
        self.assertTrue(found["higher_is_better"])

    def test_no_metric(self):
        found = ms.mentioned_metrics("Просто задача без метрики")
        self.assertIsNone(found["primary"])
        self.assertEqual(found["mentioned"], [])


@unittest.skipUnless(DATA.exists(), "публичные данные не скачаны в data/public")
class PublicSchemaTest(unittest.TestCase):
    def test_regression(self):
        data = ml_io.load_task(DATA / "regression_case" / "task.md")
        suggestion = schema_of(data)
        self.assertEqual((suggestion["id_cols"], suggestion["target_cols"]), (["id"], ["site_eui"]))
        self.assertEqual(suggestion["confidence"], "high")
        self.assertEqual(len(suggestion["feature_cols"]), 62)
        self.assertEqual(ms.suggest_task_type(data.train["site_eui"], data.text)["type"], "regression")
        self.assertEqual(ms.mentioned_metrics(data.text)["primary"], "rmse")

    def test_multiclass(self):
        data = ml_io.load_task(DATA / "multiclass_case" / "task.md")
        suggestion = schema_of(data)
        self.assertEqual((suggestion["id_cols"], suggestion["target_cols"]), (["id"], ["attack_cat"]))
        self.assertEqual(suggestion["confidence"], "high")
        self.assertEqual(len(suggestion["feature_cols"]), 42)
        self.assertEqual(ms.suggest_task_type(data.train["attack_cat"], data.text)["type"], "multiclass")
        found = ms.mentioned_metrics(data.text)
        self.assertEqual(found["primary"], "f1_macro")
        self.assertTrue({"accuracy", "f1_weighted"} <= set(found["mentioned"]))
        schema = ms.resolve_schema(suggestion)
        self.assertEqual(schema.target_cols, ("attack_cat",))


if __name__ == "__main__":
    unittest.main()
