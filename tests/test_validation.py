import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ml_io
import ml_profile
import ml_schema
import ml_validation as mv

DATA = Path(__file__).resolve().parent.parent / "data" / "public"


def profile_with(task_type, out_of_range=None, columns=(), sides=None):
    return {
        "task": {"type": task_type},
        "features": {"columns": list(columns)},
        "train_test": {"out_of_train_range_share": out_of_range or {}, "out_of_train_range_side": sides or {}},
    }


class SuggestTest(unittest.TestCase):
    def test_temporal_when_test_is_beyond_train_in_a_small_integer_column(self):
        columns = [{"name": "period", "dtype": "int64", "n_unique": 5}]
        plan = mv.suggest_validation(profile_with("regression", {"period": 1.0}, columns))
        self.assertEqual((plan["strategy"], plan["column"], plan["n_splits"]), ("temporal", "period", 3))
        self.assertIn("period", plan["reason"])

    def test_test_before_train_validates_on_earliest_values(self):
        # ревью: при test раньше train проверка шла на самых поздних значениях
        columns = [{"name": "yr", "dtype": "int64", "n_unique": 5}]
        plan = mv.suggest_validation(profile_with("regression", {"yr": 1.0}, columns, {"yr": "below"}))
        self.assertEqual(plan["direction"], "backward")
        X = pd.DataFrame({"yr": np.repeat([5, 6, 7, 8, 9], 10)})
        splits = mv.make_splits(plan, X)
        self.assertEqual([int(X.yr.iloc[v[0]]) for _, v in splits], [7, 6, 5])
        for train, valid in splits:
            self.assertGreater(X.yr.iloc[train].min(), X.yr.iloc[valid].max())

    def test_dates_with_many_values_use_late_blocks(self):
        # ревью Ивана (Y4): по датам временная схема не включалась
        columns = [{"name": "date", "dtype": "object", "kind": "datetime", "n_unique": 400}]
        plan = mv.suggest_validation(profile_with("regression", {"date": 1.0}, columns, {"date": "above"}))
        self.assertEqual((plan["strategy"], plan["column"]), ("temporal", "date"))
        X = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=400).strftime("%Y-%m-%d")})
        splits = mv.make_splits(plan, X)
        self.assertEqual(len(splits), 3)
        days = pd.to_datetime(X["date"])
        covered = np.concatenate([v for _, v in splits])
        self.assertEqual(len(covered), len(set(covered)))
        for train, valid in splits:
            self.assertLess(days.iloc[train].max(), days.iloc[valid].min())
            self.assertGreater(len(valid), 50)
        backward = mv.make_splits({**plan, "direction": "backward"}, X)
        for train, valid in backward:
            self.assertGreater(days.iloc[train].min(), days.iloc[valid].max())

    def test_drift_on_both_sides_is_not_temporal(self):
        columns = [{"name": "yr", "dtype": "int64", "n_unique": 5}]
        plan = mv.suggest_validation(profile_with("regression", {"yr": 1.0}, columns, {"yr": "both"}))
        self.assertEqual(plan["strategy"], "kfold")

    def test_integer_year_with_missing_values_still_triggers_temporal(self):
        # ревью Ивана (Y5): из-за одного пропуска год читается как float64, и схема была kfold
        columns = [{"name": "year", "dtype": "float64", "n_unique": 4, "integer_valued": True}]
        plan = mv.suggest_validation(profile_with("regression", {"year": 1.0}, columns, {"year": "above"}))
        self.assertEqual((plan["strategy"], plan["column"]), ("temporal", "year"))
        X = pd.DataFrame({"year": [np.nan] * 3 + [1.0] * 10 + [2.0] * 10 + [3.0] * 10 + [4.0] * 10})
        splits = mv.make_splits(plan, X)
        self.assertEqual([float(X.year.iloc[v[0]]) for _, v in splits], [2.0, 3.0, 4.0])

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

    def test_rare_classes_keep_all_folds(self):
        # ревью Ивана (Y11): класс из 2–4 строк сокращал валидацию до 2–4 частей
        y = pd.Series(["a"] * 95 + ["b"] * 3 + ["c"] * 2)
        splits = mv.make_splits({"strategy": "stratified", "n_splits": 5}, pd.DataFrame({"x": range(100)}), y)
        self.assertEqual(len(splits), 5)
        self.assertEqual(splits.effective()["strategy"], "stratified")
        self.assertIn("объединены", splits.effective()["note"])
        self.assertEqual(len(self.check_partition(splits, 100)), 100)
        rare_in_valid = [int(y.iloc[v].isin(["b", "c"]).sum()) for _, v in splits]
        self.assertEqual(sum(rare_in_valid), 5)
        self.assertLessEqual(max(rare_in_valid), 2)  # редкие строки разнесены по частям

    def test_singleton_class_still_stratified(self):
        y = pd.Series(["a"] * 70 + ["b"] * 29 + ["c"])
        splits = mv.make_splits({"strategy": "stratified", "n_splits": 5}, pd.DataFrame({"x": range(100)}), y)
        self.assertEqual((len(splits), splits.effective()["strategy"]), (5, "stratified"))
        for _, valid in splits:
            self.assertEqual(int(y.iloc[valid].eq("b").sum()) in (5, 6), True)

    def test_group_stratified_is_fast_and_keeps_groups(self):
        # ревью Ивана (Y12): StratifiedGroupKFold на всех строках — секунды на каждый train_eval
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.integers(0, 30000, 50000), "b": rng.integers(0, 3, 50000)})
        y = rng.integers(0, 10, 50000)
        started = time.time()
        splits = mv.make_splits({"strategy": "group", "n_splits": 5, "classification": True}, X, y)
        elapsed = time.time() - started
        self.assertEqual(len(splits), 5)
        self.assertEqual(len(self.check_partition(splits, 50000)), 50000)
        keys = pd.Series(mv.row_groups(X))
        for train, valid in splits:
            self.assertEqual(len(set(keys.iloc[train]) & set(keys.iloc[valid])), 0)
        self.assertLess(elapsed, 5.0)  # на M2 ~1 с, на i5-11400F ~2.1 с; прежняя версия была в 8 раз медленнее

    def test_group_folds_do_not_change_when_features_are_added(self):
        # ревью Ивана (25.10, row_groups): номер группы был хэшем строки, и после добавления признаков
        # фолды перемешивались, хотя группы прежние; сравнение наборов признаков шло на разных фолдах
        rng = np.random.default_rng(0)
        base = pd.DataFrame({"a": rng.normal(size=1700).round(3), "b": rng.integers(0, 5, 1700)})
        X = pd.concat([base, base.iloc[:300]], ignore_index=True)  # 15% полных копий
        wider = X.assign(c=X["a"] * 2, d=X["b"] + 1, e=X["a"].abs())  # те же группы, новые столбцы
        y_reg, y_cls = rng.normal(size=len(X)), rng.integers(0, 2, len(X))
        for y, classification in ((y_reg, False), (y_cls, True)):
            plan = {"strategy": "group", "n_splits": 5, "classification": classification}
            before = mv.make_splits(plan, X, y)
            after = mv.make_splits(plan, wider, y)
            for (_, v1), (_, v2) in zip(before, after):
                self.assertEqual(sorted(v1), sorted(v2))

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
