import json
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

import ml_io
import ml_profile as mp
from tests.synth import make_task

DATA = Path(__file__).resolve().parent.parent / "data" / "public"


class ColumnProfileTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(mp.column_kind(pd.Series([1, 1, 1])), "constant")
        self.assertEqual(mp.column_kind(pd.Series([None, None], dtype=float)), "constant")
        self.assertEqual(mp.column_kind(pd.Series([0, 1, 0, 1])), "binary")
        self.assertEqual(mp.column_kind(pd.Series([True, False, True])), "binary")
        self.assertEqual(mp.column_kind(pd.Series([0.1, 0.5, 0.9])), "numeric")
        self.assertEqual(mp.column_kind(pd.Series(["a", "b", "c"])), "categorical")

    def test_numeric_and_categorical_details(self):
        num = mp.profile_column(pd.Series([1.0, 2.0, None, 4.0], name="x"))
        self.assertEqual((num["min"], num["max"], num["missing_frac"]), (1.0, 4.0, 0.25))
        cat = mp.profile_column(pd.Series(["a", "a", "b", "c"], name="c"))
        self.assertEqual(cat["top_values"][0], ["a", 0.5])

    def test_id_like(self):
        ids = mp.profile_column(pd.Series(range(200), name="row"))
        self.assertTrue(ids.get("id_like"))
        measure = mp.profile_column(pd.Series([i * 0.37 for i in range(200)], name="m"))
        self.assertFalse(measure.get("id_like", False))


class SyntheticProfileTest(unittest.TestCase):
    def profile(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return mp.profile_task(make_task(tmp.name, **kwargs).task_md)

    def test_regression_profile_is_valid_json_and_complete(self):
        profile = self.profile(kind="regression", target="price", id_col="row")
        json.loads(mp.to_json(profile, allow_nan=False))
        self.assertEqual(profile["task"]["type"], "regression")
        self.assertEqual(profile["schema"]["target_cols"], ["price"])
        self.assertIn("quantiles", profile["target"])
        self.assertEqual(profile["features"]["n_features"], 6)

    def test_classification_target_and_rare_classes(self):
        profile = self.profile(kind="multiclass", n_classes=4, n_train=600)
        self.assertEqual(profile["task"]["type"], "multiclass")
        self.assertEqual(len(profile["target"]["classes"]), 4)
        self.assertGreaterEqual(profile["target"]["imbalance_ratio"], 1.0)

    def test_unseen_category_and_missing_are_reported(self):
        profile = self.profile(kind="binary")
        self.assertIn("cat", profile["train_test"]["unseen_categories_share"])
        self.assertTrue(any(c["missing_frac"] > 0 for c in profile["features"]["columns"]))

    def test_multi_target_is_not_guessed(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        task = make_task(tmp.name, kind="regression")
        data = ml_io.load_task(task.task_md)
        data.train["y2"] = data.train["y"] * 2
        data.sample_submission["y2"] = 0.0
        profile = mp.build_profile(data)
        self.assertEqual(profile["task"]["type"], "multi_target")
        self.assertNotIn("target", profile)

    def test_missing_train_does_not_crash(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        task = make_task(tmp.name)
        (Path(tmp.name) / "data/participants/train.csv").unlink()
        profile = mp.profile_task(task.task_md)
        self.assertEqual(profile["task"]["type"], "unknown")

    def test_text_is_short_and_contains_key_facts(self):
        profile = self.profile(kind="multiclass")
        text = mp.format_profile(profile, max_columns=3)
        self.assertIn("тип задачи: multiclass", text)
        self.assertIn("таргет=['y']", text)
        self.assertIn("и ещё", text)
        self.assertLess(len(text), 4000)


@unittest.skipUnless(DATA.exists(), "публичные данные не скачаны в data/public")
class PublicProfileTest(unittest.TestCase):
    def test_regression(self):
        profile = mp.profile_task(DATA / "regression_case" / "task.md")
        json.loads(mp.to_json(profile, allow_nan=False))
        self.assertEqual(profile["task"]["type"], "regression")
        self.assertEqual(profile["metric"]["primary"], "rmse")
        self.assertEqual(profile["features"]["n_features"], 62)
        kinds = profile["features"]["kinds"]
        # State_Factor и facility_type — категории; building_class с двумя значениями считается бинарным
        self.assertEqual((kinds["categorical"], kinds["binary"]), (2, 1))
        time_risk = next(r for r in profile["risks"] if r["kind"] == "time_like_names")
        self.assertIn("Year_Factor", time_risk["columns"])
        for share in profile["train_test"]["unseen_categories_share"].values():
            self.assertTrue(0 < share <= 1)

    def test_multiclass(self):
        started = time.time()
        profile = mp.profile_task(DATA / "multiclass_case" / "task.md")
        json.loads(mp.to_json(profile, allow_nan=False))
        self.assertLess(time.time() - started, 60)
        self.assertEqual(profile["task"]["type"], "multiclass")
        self.assertEqual(profile["metric"]["primary"], "f1_macro")
        target = profile["target"]
        self.assertEqual(len(target["classes"]), 10)
        self.assertEqual(target["classes"]["Worms"]["count"], 65)
        self.assertIn("Worms", target["rare_classes"])
        self.assertEqual(profile["features"]["n_features"], 42)
        self.assertLess(len(mp.format_profile(profile)), 6000)


if __name__ == "__main__":
    unittest.main()
