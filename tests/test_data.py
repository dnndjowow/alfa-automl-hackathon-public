import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ml_io
import ml_schema
from ml_data import DataError, LabelCodec, TabularPreprocessor, split_task
from tests.synth import make_task

DATA = Path(__file__).resolve().parent.parent / "data" / "public"


def frame():
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0, 5.0],
            "cat": ["a", "b", "a", None, "c"],
            "flag": [True, False, True, True, False],
            "const": [7, 7, 7, 7, 7],
            "empty": [np.nan] * 5,
            "inf": [1.0, np.inf, 3.0, -np.inf, 5.0],
        }
    )


class PreprocessorFitTest(unittest.TestCase):
    def test_columns_are_classified(self):
        pre = TabularPreprocessor().fit(frame())
        self.assertEqual(pre.numeric_cols_, ["num", "flag", "inf"])
        self.assertEqual(pre.categorical_cols_, ["cat"])
        self.assertEqual(pre.dropped_, {"const": "constant", "empty": "all_missing"})
        self.assertEqual(pre.feature_names_, ["num", "cat", "flag", "inf"])

    def test_categories_ordered_by_frequency_then_name(self):
        pre = TabularPreprocessor().fit(frame())
        self.assertEqual(pre.categories_["cat"], ["a", "b", "c"])

    def test_nothing_left_is_an_error(self):
        with self.assertRaises(DataError):
            TabularPreprocessor().fit(pd.DataFrame({"c": [1, 1, 1]}))

    def test_not_fitted_and_bad_mode(self):
        with self.assertRaises(DataError):
            TabularPreprocessor().transform(frame())
        pre = TabularPreprocessor().fit(frame())
        with self.assertRaises(DataError):
            pre.transform(frame(), categorical="onehot")


class PreprocessorTransformTest(unittest.TestCase):
    def test_numeric_nan_kept_and_inf_replaced(self):
        out = TabularPreprocessor().fit_transform(frame())
        self.assertTrue(np.isnan(out.loc[2, "num"]))
        self.assertEqual(out["flag"].tolist(), [1.0, 0.0, 1.0, 1.0, 0.0])
        self.assertTrue(np.isnan(out.loc[1, "inf"]) and np.isnan(out.loc[3, "inf"]))
        self.assertEqual(out["num"].dtype, np.float64)

    def test_category_dtype_with_fixed_categories(self):
        pre = TabularPreprocessor().fit(frame())
        test = pd.DataFrame({"num": [1.0], "cat": ["zzz"], "flag": [True], "inf": [0.0]})
        out = pre.transform(test)
        self.assertEqual(str(out["cat"].dtype), "category")
        self.assertEqual(list(out["cat"].cat.categories), ["a", "b", "c"])
        self.assertTrue(pd.isna(out.loc[0, "cat"]))
        self.assertEqual(pre.unseen_share_, {"cat": 1.0})

    def test_codes_mode(self):
        pre = TabularPreprocessor().fit(frame())
        out = pre.transform(frame(), categorical="codes")
        self.assertEqual(out["cat"].tolist(), [0, 1, 0, -1, 2])
        self.assertEqual(out["cat"].dtype, np.int32)

    def test_missing_feature_is_filled_with_warning_and_extra_is_ignored(self):
        pre = TabularPreprocessor().fit(frame())
        test = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"], "extra": [9, 9]})
        out = pre.transform(test)
        self.assertEqual(list(out.columns), pre.feature_names_)
        self.assertTrue(out["flag"].isna().all())
        self.assertTrue(any("flag" in w for w in pre.warnings_))

    def test_no_leakage_from_test_into_fit(self):
        train = pd.DataFrame({"c": ["a", "a", "b"], "x": [1.0, 2.0, 3.0]})
        test = pd.DataFrame({"c": ["z", "z", "z"], "x": [100.0, 200.0, 300.0]})
        pre = TabularPreprocessor().fit(train)
        pre.transform(test)
        self.assertEqual(pre.categories_["c"], ["a", "b"])  # категория из test в словарь не попала

    def test_fit_is_deterministic_and_does_not_change_input(self):
        original = frame()
        copy = original.copy()
        a = TabularPreprocessor().fit_transform(original)
        b = TabularPreprocessor().fit_transform(original)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(original, copy)

    def test_value_or_missing_column_is_kept(self):
        # ревью: [1.0, NaN] удалялся как константа, хотя пропуск полностью определял класс
        df = pd.DataFrame({"measured": [1.0, np.nan] * 50})
        pre = TabularPreprocessor().fit(df)
        self.assertEqual(pre.feature_names_, ["measured"])
        self.assertEqual(TabularPreprocessor().fit(pd.DataFrame({"c": ["x", None] * 5})).feature_names_, ["c"])

    def test_integer_categories_match_across_text_and_float_files(self):
        # ревью: '1' в train (object) и 1.0 в test (float) не совпадали
        pre = TabularPreprocessor().fit(pd.DataFrame({"code": ["A", "1", "2", "1"]}))
        out = pre.transform(pd.DataFrame({"code": [1.0, 2.0, np.nan]}))
        self.assertEqual(out["code"].tolist()[:2], ["1", "2"])
        self.assertTrue(pd.isna(out.loc[2, "code"]))
        self.assertEqual(pre.unseen_share_, {})
        pre.transform(pd.DataFrame({"code": [1.5]}))
        self.assertEqual(pre.unseen_share_, {"code": 1.0})  # 1.5 действительно новая категория

    def test_dates_become_numeric_parts(self):
        # ревью Ивана (Y4): дата-строка становилась высококардинальной категорией
        train = pd.DataFrame({"when": ["2020-01-01", "2020-01-02", "2020-03-15", None], "x": [1, 2, 3, 4]})
        pre = TabularPreprocessor().fit(train)
        self.assertEqual(pre.datetime_cols_, ["when"])
        self.assertEqual(pre.feature_names_, ["when__days", "when__year", "when__month", "when__dayofweek", "x"])
        out = pre.transform(pd.DataFrame({"when": ["2021-06-01"], "x": [5]}))
        self.assertEqual(out.loc[0, "when__year"], 2021.0)
        self.assertAlmostEqual(out.loc[0, "when__days"], (pd.Timestamp("2021-06-01") - pd.Timestamp("1970-01-01")).days)
        self.assertTrue(pre.transform(train)["when__days"].isna().iloc[3])

    def test_codes_and_numbers_are_not_dates(self):
        from ml_data import parse_dates
        self.assertIsNone(parse_dates(pd.Series(["A77", "B12", "C3"])))
        self.assertIsNone(parse_dates(pd.Series([20200101, 20200102])))
        self.assertIsNotNone(parse_dates(pd.Series(["01.02.2020", "15.03.2021"])))

    def test_mixed_type_column_becomes_text_categories(self):
        df = pd.DataFrame({"m": [1, "a", 1, "a", 2.5]})
        pre = TabularPreprocessor().fit(df)
        self.assertEqual(pre.categorical_cols_, ["m"])
        self.assertEqual(set(pre.categories_["m"]), {"1", "a", "2.5"})


class LabelCodecTest(unittest.TestCase):
    def test_round_trip_with_strings(self):
        codec = LabelCodec().fit(["b", "a", "c", "a"])
        self.assertEqual(list(codec.classes_), ["a", "b", "c"])
        codes = codec.encode(["c", "a", "b"])
        self.assertEqual(codes.tolist(), [2, 0, 1])
        self.assertEqual(codec.decode(codes).tolist(), ["c", "a", "b"])

    def test_integer_labels_keep_their_values(self):
        codec = LabelCodec().fit([10, 3, 7])
        self.assertEqual(codec.decode([0, 2]).tolist(), [3, 10])

    def test_unknown_and_missing_labels_are_errors(self):
        codec = LabelCodec().fit(["a", "b"])
        with self.assertRaises(DataError):
            codec.encode(["a", "zzz"])
        with self.assertRaises(DataError):
            codec.decode([5])
        with self.assertRaises(DataError):
            LabelCodec().fit(["a", None])

    def test_decode_proba_columns_follow_class_order(self):
        codec = LabelCodec().fit(["x", "y", "z"])
        out = codec.decode_proba(np.full((2, 3), 1 / 3), index=[5, 6])
        self.assertEqual(list(out.columns), ["x", "y", "z"])
        self.assertEqual(list(out.index), [5, 6])
        with self.assertRaises(DataError):
            codec.decode_proba(np.zeros((2, 2)))


class SplitTaskTest(unittest.TestCase):
    def load(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return ml_io.load_task(make_task(tmp.name, **kwargs).task_md)

    def schema(self, data):
        suggestion = ml_schema.infer_schema(
            data.train, data.test, data.sample_submission, data.data_dictionary
        )
        return ml_schema.resolve_schema(suggestion)

    def test_split_shapes_and_roles(self):
        data = self.load(kind="regression", target="price", id_col="row")
        split = split_task(data, self.schema(data))
        self.assertEqual(split.X_train.shape, (300, 6))
        self.assertNotIn("price", split.X_train.columns)
        self.assertEqual(list(split.X_test.columns), list(split.X_train.columns))
        self.assertEqual(list(split.test_ids.columns), ["row"])
        self.assertEqual(split.test_ids["row"].tolist(), data.test["row"].tolist())
        self.assertEqual(split.warnings, [])

    def test_rows_without_target_are_dropped_with_warning(self):
        data = self.load(kind="regression")
        data.train.loc[:4, "y"] = np.nan
        split = split_task(data, self.schema(data))
        self.assertEqual(len(split.X_train), 295)
        self.assertEqual(len(split.y_train), 295)
        self.assertTrue(any("без таргета" in w for w in split.warnings))

    def test_misaligned_sample_submission_is_reported(self):
        data = self.load(kind="regression")
        data.sample_submission = data.sample_submission.iloc[::-1].reset_index(drop=True)
        split = split_task(data, self.schema(data))
        self.assertTrue(any("sample_submission" in w for w in split.warnings))

    def test_missing_pieces_raise(self):
        data = self.load(kind="regression")
        schema = self.schema(data)
        data.test = None
        with self.assertRaises(DataError):
            split_task(data, schema)

    def test_test_without_a_feature_warns_and_preprocessor_fills(self):
        data = self.load(kind="regression")
        schema = self.schema(data)
        data.test = data.test.drop(columns=["b"])
        split = split_task(data, schema)
        self.assertTrue(any("'b'" in w for w in split.warnings))
        out = TabularPreprocessor().fit(split.X_train).transform(split.X_test)
        self.assertTrue(out["b"].isna().all())


@unittest.skipUnless(DATA.exists(), "публичные данные не скачаны в data/public")
class PublicDataPipelineTest(unittest.TestCase):
    def run_case(self, case, rows, cols_after):
        data = ml_io.load_task(DATA / case / "task.md")
        schema = self.schema_for(data)
        split = split_task(data, schema)
        started = time.time()
        pre = TabularPreprocessor()
        train = pre.fit_transform(split.X_train)
        test = pre.transform(split.X_test)
        self.assertLess(time.time() - started, 30)
        self.assertEqual(train.shape, (rows, cols_after))
        self.assertEqual(list(train.columns), list(test.columns))
        self.assertEqual(split.warnings, [])
        return data, split, pre, train, test

    def schema_for(self, data):
        return ml_schema.resolve_schema(
            ml_schema.infer_schema(data.train, data.test, data.sample_submission, data.data_dictionary)
        )

    def test_regression(self):
        data, split, pre, train, test = self.run_case("regression_case", 53308, 62)
        self.assertEqual(sorted(pre.categorical_cols_), ["State_Factor", "building_class", "facility_type"])
        self.assertEqual(pre.dropped_, {})
        self.assertEqual(len(test), 22449)

    def test_multiclass(self):
        data, split, pre, train, test = self.run_case("multiclass_case", 87670, 42)
        self.assertEqual(sorted(pre.categorical_cols_), ["proto", "service", "state"])
        codec = LabelCodec().fit(split.y_train)
        self.assertEqual(len(codec.classes_), 10)
        self.assertEqual(codec.decode(codec.encode(split.y_train)).tolist(), split.y_train.tolist())
        self.assertIn("-", pre.categories_["service"])
        known = set(data.train["state"].dropna().astype(str))
        expected = (~data.test["state"].dropna().astype(str).isin(known)).sum() / len(data.test)
        self.assertGreater(expected, 0)
        self.assertAlmostEqual(pre.unseen_share_["state"], expected)


if __name__ == "__main__":
    unittest.main()
