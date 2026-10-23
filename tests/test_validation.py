import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ml_io
import ml_profile
import ml_schema
import ml_validation as mv

DATA = Path(__file__).resolve().parent.parent / "data" / "public"


def profile_with(task_type, out_of_range=None, columns=()):
    return {
        "task": {"type": task_type},
        "features": {"columns": list(columns)},
        "train_test": {"out_of_train_range_share": out_of_range or {}},
    }


class SuggestTest(unittest.TestCase):
    def test_temporal_when_test_is_beyond_train_in_a_small_integer_column(self):
        columns = [{"name": "period", "dtype": "int64", "n_unique": 5}]
        plan = mv.suggest_validation(profile_with("regression", {"period": 1.0}, columns))
        self.assertEqual((plan["strategy"], plan["column"], plan["n_splits"]), ("temporal", "period", 3))
        self.assertIn("period", plan["reason"])

    def test_float_or_high_cardinality_columns_do_not_trigger_temporal(self):
        columns = [{"name": "t", "dtype": "float64", "n_unique": 5},
                   {"name": "u", "dtype": "int64", "n_unique": 5000}]
        plan = mv.suggest_validation(profile_with("regression", {"t": 1.0, "u": 1.0}, columns))
        self.assertEqual(plan["strategy"], "kfold")

    def test_group_when_many_duplicates(self):
        X = pd.DataFrame({"a": [1, 1, 2, 2, 3, 4, 5, 6, 7, 8]})
        plan = mv.suggest_validation(profile_with("multiclass"), X)
        self.assertEqual(plan["strategy"], "group")
        self.assertTrue(plan["classification"])

    def test_defaults(self):
        X = pd.DataFrame({"a": range(10)})
        self.assertEqual(mv.suggest_validation(profile_with("binary"), X)["strategy"], "stratified")
        self.assertEqual(mv.suggest_validation(profile_with("regression"), X)["strategy"], "kfold")


class SplitsTest(unittest.TestCase):
    def check_partition(self, splits, n):
        all_valid = np.concatenate([v for _, v in splits])
        self.assertEqual(len(all_valid), len(set(all_valid)), "валидационные части пересекаются")
        for train, valid in splits:
            self.assertEqual(len(np.intersect1d(train, valid)), 0)
        return all_valid

    def test_temporal_validates_on_latest_values_only_from_the_past(self):
        X = pd.DataFrame({"year": np.repeat([1, 2, 3, 4, 5], 20)})
        plan = {"strategy": "temporal", "column": "year", "n_splits": 3}
        splits = mv.make_splits(plan, X)
        self.assertEqual([int(X["year"].iloc[v[0]]) for _, v in splits], [3, 4, 5])
        for train, valid in splits:
            self.assertLess(X["year"].iloc[train].max(), X["year"].iloc[valid].min())
            self.assertEqual(X["year"].iloc[valid].nunique(), 1)

    def test_temporal_errors(self):
        X = pd.DataFrame({"year": [1] * 10})
        with self.assertRaises(mv.ValidationError):
            mv.make_splits({"strategy": "temporal", "column": "year"}, X)
        with self.assertRaises(mv.ValidationError):
            mv.make_splits({"strategy": "temporal", "column": "nope"}, X)

    def test_group_keeps_identical_rows_together(self):
        rng = np.random.default_rng(0)
        base = pd.DataFrame({"a": rng.integers(0, 40, 300), "b": rng.integers(0, 3, 300)})
        y = pd.Series(rng.integers(0, 3, 300))
        for classification in (True, False):
            plan = {"strategy": "group", "n_splits": 5, "classification": classification}
            splits = mv.make_splits(plan, base, y)
            self.check_partition(splits, 300)
            keys = base.astype(str).agg("|".join, axis=1)
            for train, valid in splits:
                self.assertEqual(len(set(keys.iloc[train]) & set(keys.iloc[valid])), 0)

    def test_stratified_keeps_every_class_in_every_fold(self):
        y = pd.Series(["a"] * 80 + ["b"] * 15 + ["c"] * 5)
        X = pd.DataFrame({"x": range(100)})
        splits = mv.make_splits({"strategy": "stratified", "n_splits": 5}, X, y)
        self.assertEqual(len(splits), 5)
        for _, valid in splits:
            self.assertEqual(set(y.iloc[valid]), {"a", "b", "c"})
        self.assertEqual(len(self.check_partition(splits, 100)), 100)

    def test_stratified_with_a_very_rare_class_reduces_folds(self):
        y = pd.Series(["a"] * 95 + ["b"] * 3 + ["c"] * 2)
        splits = mv.make_splits({"strategy": "stratified", "n_splits": 5}, pd.DataFrame({"x": range(100)}), y)
        self.assertEqual(len(splits), 2)

    def test_singleton_class_falls_back_to_plain_kfold(self):
        y = pd.Series(["a"] * 99 + ["b"])
        splits = mv.make_splits({"strategy": "stratified", "n_splits": 5}, pd.DataFrame({"x": range(100)}), y)
        self.assertEqual(len(splits), 5)

    def test_kfold_is_deterministic_for_a_seed(self):
        X = pd.DataFrame({"x": range(50)})
        a = mv.make_splits({"strategy": "kfold", "n_splits": 4}, X, seed=1)
        b = mv.make_splits({"strategy": "kfold", "n_splits": 4}, X, seed=1)
        for (a1, a2), (b1, b2) in zip(a, b):
            np.testing.assert_array_equal(a2, b2)
        self.assertEqual(len(self.check_partition(a, 50)), 50)

    def test_unknown_strategy_and_tiny_data(self):
        with self.assertRaises(mv.ValidationError):
            mv.make_splits({"strategy": "magic"}, pd.DataFrame({"x": [1, 2, 3]}))
        with self.assertRaises(mv.ValidationError):
            mv.make_splits({"strategy": "kfold"}, pd.DataFrame({"x": [1]}))


@unittest.skipUnless(DATA.exists(), "публичные данные не скачаны в data/public")
class PublicValidationTest(unittest.TestCase):
    def setUp(self):
        pass

    def load(self, case):
        data = ml_io.load_task(DATA / case / "task.md")
        schema = ml_schema.resolve_schema(
            ml_schema.infer_schema(data.train, data.test, data.sample_submission, data.data_dictionary)
        )
        return data, schema, ml_profile.build_profile(data)

    def test_regression_is_validated_by_year(self):
        data, schema, profile = self.load("regression_case")
        X = data.train[list(schema.feature_cols)]
        plan = mv.suggest_validation(profile, X)
        self.assertEqual((plan["strategy"], plan["column"]), ("temporal", "Year_Factor"))
        splits = mv.make_splits(plan, X)
        years = [int(X["Year_Factor"].iloc[v[0]]) for _, v in splits]
        self.assertEqual(years, [3, 4, 5])
        self.assertEqual([len(v) for _, v in splits], [10879, 12946, 18308])

    def test_multiclass_is_validated_by_groups(self):
        data, schema, profile = self.load("multiclass_case")
        X, y = data.train[list(schema.feature_cols)], data.train["attack_cat"]
        self.assertEqual(mv.suggest_validation(profile)["strategy"], "stratified")  # без X дубликатов не видно
        plan = mv.suggest_validation(profile, X)
        self.assertEqual(plan["strategy"], "group")
        splits = mv.make_splits(plan, X, y)
        self.assertEqual(len(splits), 5)
        keys = pd.Series(mv.row_groups(X))
        for train, valid in splits:
            self.assertEqual(len(set(keys.iloc[train]) & set(keys.iloc[valid])), 0)
            self.assertGreater(y.iloc[valid].eq("Worms").sum(), 0)  # редкий класс есть в каждой части


if __name__ == "__main__":
    unittest.main()
