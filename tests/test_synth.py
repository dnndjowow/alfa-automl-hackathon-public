import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests.synth import make_task


class SynthGeneratorTest(unittest.TestCase):
    def make(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return make_task(tmp.name, **kwargs), Path(tmp.name)

    def test_regression_files_and_shapes(self):
        task, root = self.make(kind="regression", n_train=50, n_test=20, target="price", id_col="row")
        part = root / "data" / "participants"
        train, test = pd.read_csv(part / "train.csv"), pd.read_csv(part / "test.csv")
        sample = pd.read_csv(part / "sample_submission.csv")
        self.assertEqual((len(train), len(test), len(sample)), (50, 20, 20))
        self.assertIn("price", train.columns)
        self.assertNotIn("price", test.columns)
        self.assertEqual(list(sample.columns), ["row", "price"])
        self.assertEqual(list(task.truth.columns), ["row", "price"])
        self.assertTrue(task.task_md.read_text(encoding="utf-8").count("`data/participants/") == 4)

    def test_classes_unseen_category_and_nans(self):
        task, root = self.make(kind="multiclass", n_classes=4, n_train=400, n_test=200, target="label")
        part = root / "data" / "participants"
        train, test = pd.read_csv(part / "train.csv"), pd.read_csv(part / "test.csv")
        self.assertEqual(train["label"].nunique(), 4)
        self.assertIn("violet", set(test["cat"].dropna()))
        self.assertNotIn("violet", set(train["cat"].dropna()))
        self.assertTrue(train["a"].isna().any() and train["cat"].isna().any())
        self.assertEqual(len(train["id"].unique()), len(train))

    def test_binary_has_two_classes_and_is_deterministic(self):
        first, _ = self.make(kind="binary", seed=3)
        second, _ = self.make(kind="binary", seed=3)
        self.assertEqual(first.truth["y"].nunique(), 2)
        pd.testing.assert_frame_equal(first.truth, second.truth)

    def test_flat_layout_crlf_and_bom(self):
        task, root = self.make(layout="flat", crlf=True, bom=True)
        raw = task.task_md.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"\r\n", raw)
        self.assertTrue((root / "train.csv").exists())


if __name__ == "__main__":
    unittest.main()
