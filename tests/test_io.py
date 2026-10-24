import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import ml_io
from tests.synth import make_task

DATA = Path(__file__).resolve().parent.parent / "data" / "public"


class SyntheticTaskLoadingTest(unittest.TestCase):
    def make(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return make_task(tmp.name, **kwargs), Path(tmp.name)

    def test_nested_layout_found_via_task_md(self):
        task, root = self.make(kind="regression", target="price", id_col="row")
        data = ml_io.load_task(task.task_md)
        self.assertEqual(data.files.warnings, [])
        self.assertEqual(data.files.paths["train"], (root / "data/participants/train.csv").resolve())
        self.assertEqual((len(data.train), len(data.test), len(data.sample_submission)), (300, 120, 120))
        self.assertIn("price", data.train.columns)
        self.assertIsNotNone(data.data_dictionary)

    def test_flat_layout_found_by_file_name(self):
        task, _ = self.make(layout="flat", kind="binary")
        # ссылки в task.md есть, но проверим и запасной поиск: уберём их из текста
        task.task_md.write_text("# Задача без ссылок на файлы\n", encoding="utf-8")
        data = ml_io.load_task(task.task_md)
        self.assertEqual(data.files.warnings, [])
        self.assertIsNotNone(data.train)
        self.assertIsNotNone(data.test)

    def test_crlf_and_bom_in_task_md(self):
        task, _ = self.make(crlf=True, bom=True)
        data = ml_io.load_task(task.task_md)
        self.assertNotIn("\r", data.text)
        self.assertFalse(data.text.startswith("﻿"))
        self.assertIsNotNone(data.train)

    def test_working_from_another_current_directory(self):
        task, root = self.make()
        elsewhere = tempfile.TemporaryDirectory()
        self.addCleanup(elsewhere.cleanup)
        previous = os.getcwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(root)
        from_task_dir = ml_io.load_task("task.md")
        os.chdir(elsewhere.name)
        from_elsewhere = ml_io.load_task(task.task_md)
        self.assertEqual(from_task_dir.files.paths, from_elsewhere.files.paths)
        self.assertTrue(from_elsewhere.files.paths["test"].is_absolute())
        self.assertEqual(len(from_elsewhere.test), 120)

    def test_missing_role_is_reported_not_raised(self):
        task, root = self.make(kind="regression")
        (root / "data/participants/sample_submission.csv").unlink()
        data = ml_io.load_task(task.task_md)
        self.assertIsNone(data.sample_submission)
        self.assertTrue(any("sample_submission" in w for w in data.files.warnings))
        self.assertIsNotNone(data.train)

    def test_unknown_csv_goes_to_extra_and_own_submission_is_ignored(self):
        task, root = self.make(extra_files=("weather_extra.csv", "submission.csv"))
        files = ml_io.find_task_files(task.task_md)
        names = {p.name for p in files.extra_csv}
        self.assertEqual(names, {"weather_extra.csv"})
        self.assertTrue(any("weather_extra.csv" in w for w in files.warnings))

    def test_recognised_names_with_suffixes(self):
        self.assertEqual(ml_io.classify_csv("train_data.csv"), "train")
        self.assertEqual(ml_io.classify_csv("Test.CSV"), "test")
        self.assertEqual(ml_io.classify_csv("x_test.csv"), "test")
        self.assertEqual(ml_io.classify_csv("sample_submission.csv"), "sample_submission")
        self.assertEqual(ml_io.classify_csv("SampleSubmission.csv"), "sample_submission")
        self.assertEqual(ml_io.classify_csv("data_dictionary.csv"), "data_dictionary")
        self.assertEqual(ml_io.classify_csv("column_descriptions.csv"), "data_dictionary")
        self.assertIsNone(ml_io.classify_csv("submission.csv"))
        self.assertIsNone(ml_io.classify_csv("weather.csv"))

    def test_substrings_do_not_assign_roles(self):
        # ревью: 'dict' внутри 'predictions', 'sample'+'sub' внутри 'subsample'
        for name in ("predictions.csv", "oof_predictions.csv", "train_predictions.csv", "test_preds.csv"):
            self.assertIsNone(ml_io.classify_csv(name), name)
        self.assertEqual(ml_io.classify_csv("train_subsample.csv"), "train")
        self.assertIsNone(ml_io.classify_csv("contest.csv"))

    def test_prediction_file_is_not_taken_as_dictionary(self):
        task, root = self.make(kind="regression")
        (root / "data/participants/data_dictionary.csv").unlink()
        (root / "data/participants/oof_predictions.csv").write_text("id,y\n1,0.5\n", encoding="utf-8")
        files = ml_io.find_task_files(task.task_md)
        self.assertIsNone(files.paths["data_dictionary"])

    def test_csv_outside_task_dir_relative_and_absolute(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "data").mkdir()
        (root / "task").mkdir()
        for name in ("train", "test"):
            pd.DataFrame({"id": [1, 2], "x": [1.0, 2.0]}).to_csv(root / "data" / f"{name}.csv", index=False)
        for text in ("Файлы: `../data/train.csv`, `../data/test.csv`",
                     f"Файлы: `{root}/data/train.csv`, `{root}/data/test.csv`"):
            (root / "task" / "task.md").write_text(text, encoding="utf-8")
            data = ml_io.load_task(root / "task" / "task.md")
            self.assertEqual(len(data.train), 2)
            self.assertEqual(data.files.paths["test"], (root / "data" / "test.csv").resolve())
            self.assertTrue(ml_io.display_path(data.files.paths["train"], data.files.task_dir).endswith("train.csv"))

    def test_only_empty_cell_is_missing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "t.csv"
        path.write_text("code,x\nNA,1\nNone,\nnull,3\n-,4\n", encoding="utf-8")
        df = ml_io.read_csv(path)
        self.assertEqual(df["code"].tolist(), ["NA", "None", "null", "-"])
        self.assertTrue(pd.isna(df.loc[1, "x"]))
        self.assertEqual(df["x"].isna().sum(), 1)

    def test_missing_task_md_raises(self):
        with self.assertRaises(ml_io.TaskLoadError):
            ml_io.read_task_text("/nonexistent/task.md")


@unittest.skipUnless(DATA.exists(), "публичные данные не скачаны в data/public")
class PublicTasksLoadingTest(unittest.TestCase):
    def test_regression(self):
        data = ml_io.load_task(DATA / "regression_case" / "task.md")
        self.assertEqual(data.files.warnings, [])
        self.assertEqual(data.train.shape, (53308, 64))
        self.assertEqual(data.test.shape, (22449, 63))
        self.assertEqual(data.sample_submission.shape, (22449, 2))
        self.assertEqual(list(data.sample_submission.columns), ["id", "site_eui"])
        self.assertEqual(len(data.data_dictionary), 64)
        self.assertEqual(data.test["id"].tolist(), data.sample_submission["id"].tolist())

    def test_multiclass(self):
        data = ml_io.load_task(DATA / "multiclass_case" / "task.md")
        self.assertEqual(data.files.warnings, [])
        self.assertEqual(data.train.shape, (87670, 44))
        self.assertEqual(data.test.shape, (41166, 43))
        self.assertEqual(data.sample_submission.shape, (41166, 2))
        self.assertEqual(data.train["attack_cat"].nunique(), 10)
        self.assertNotIn("label", data.train.columns)
        self.assertEqual(data.test["id"].tolist(), data.sample_submission["id"].tolist())
        # '-' в service — значение, а не пропуск
        self.assertIn("-", set(data.train["service"]))


if __name__ == "__main__":
    unittest.main()
