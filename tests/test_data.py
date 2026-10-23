import unittest

import numpy as np
import pandas as pd

from ml_data import DataError, TabularPreprocessor


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

    def test_mixed_type_column_becomes_text_categories(self):
        df = pd.DataFrame({"m": [1, "a", 1, "a", 2.5]})
        pre = TabularPreprocessor().fit(df)
        self.assertEqual(pre.categorical_cols_, ["m"])
        self.assertEqual(set(pre.categories_["m"]), {"1", "a", "2.5"})


if __name__ == "__main__":
    unittest.main()
