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

    def test_more_file_name_variants(self):
        # ревью Ивана (Y8): training/testing/trainset, submission_example/template не распознавались
        cases = {"training.csv": "train", "testing.csv": "test", "trainset.csv": "train", "testset.csv": "test",
                 "submission_example.csv": "sample_submission", "example_submission.csv": "sample_submission",
                 "submission_template.csv": "sample_submission", "train_labels.csv": "train_labels",
                 "y_train.csv": "train_labels", "X_train.csv": "train", "test_labels.csv": None}
        for name, role in cases.items():
            self.assertEqual(ml_io.classify_csv(name), role, name)

    def test_features_and_labels_in_separate_files(self):
        # ревью Ивана (Y8): train_labels.csv отбрасывался, таргет терялся
        by_id = self.write_task({"train_features.csv": "id,x\n1,0.5\n2,1.5\n3,2.5\n",
                                 "train_labels.csv": "id,y\n3,30\n1,10\n2,20\n",
                                 "test_features.csv": "id,x\n4,3.5\n"}, "`train_features.csv` `test_features.csv`")
        data = ml_io.load_task(by_id / "task.md")
        self.assertEqual(data.train["y"].tolist(), [10, 20, 30])
        self.assertEqual(data.files.paths["test"].name, "test_features.csv")
        by_order = self.write_task({"X_train.csv": "a,b\n1,2\n3,4\n", "y_train.csv": "target\n0\n1\n",
                                    "X_test.csv": "a,b\n5,6\n"}, "")
        data = ml_io.load_task(by_order / "task.md")
        self.assertEqual(list(data.train.columns), ["a", "b", "target"])
        self.assertTrue(any("по порядку" in w for w in data.files.warnings))

    def test_only_labels_file_is_used_as_train(self):
        # ревью Ивана (25.10, №5): только train_labels.csv / train_target.csv с полным train — train был None
        for name in ("train_labels.csv", "train_target.csv"):
            with self.subTest(name):
                root = self.write_task({name: "id,x,y\n1,0.5,0\n2,1.5,1\n3,2.5,0\n", "test.csv": "id,x\n4,3.5\n"},
                                       f"`{name}` `test.csv`")
                data = ml_io.load_task(root / "task.md")
                self.assertEqual(data.train["y"].tolist(), [0, 1, 0])
                self.assertIsNone(data.files.train_labels)
                self.assertFalse(any("роли 'train' не найден" in w for w in data.files.warnings))
                self.assertTrue(any("как train используется" in w for w in data.files.warnings))

    def test_unselected_candidates_are_kept_in_extra(self):
        root = self.write_task({"train.csv": "id,x,y\n1,2,3\n", "training.csv": "id,x,y\n1,2,3\n",
                                "test.csv": "id,x\n2,3\n"}, "")
        files = ml_io.find_task_files(root / "task.md")
        self.assertEqual(files.paths["train"].name, "train.csv")
        self.assertEqual([p.name for p in files.extra_csv], ["training.csv"])

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

    def test_codes_with_leading_zeros_keep_train_type_in_test(self):
        # ревью Ивана (Y14): '00123'/'A77' в train — текст, а в test читались числом 123
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task.md").write_text("`train.csv` `test.csv`", encoding="utf-8")
        pd.DataFrame({"id": range(4), "code": ["00123", "A77", "00456", "A77"], "y": range(4)}).to_csv(root / "train.csv", index=False)
        (root / "test.csv").write_text("id,code\n4,00123\n5,00456\n", encoding="utf-8")
        data = ml_io.load_task(root / "task.md")
        self.assertEqual(data.test["code"].tolist(), ["00123", "00456"])
        self.assertEqual(data.test["id"].tolist(), [4, 5])  # остальные столбцы как были
        self.assertTrue(any("как текст" in w for w in data.files.warnings))

    def test_numeric_train_column_is_not_forced_to_text(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task.md").write_text("`train.csv` `test.csv`", encoding="utf-8")
        (root / "train.csv").write_text("id,x,y\n1,1.5,0\n2,2.5,1\n", encoding="utf-8")
        (root / "test.csv").write_text("id,x\n3,oops\n", encoding="utf-8")
        data = ml_io.load_task(root / "task.md")
        self.assertEqual(data.train["x"].dtype, float)
        self.assertFalse(any("как текст" in w for w in data.files.warnings))

    def test_existing_submission_referenced_in_task_md_is_ignored(self):
        # ревью Ивана (Y15): после первого запуска рядом лежит submission.csv, и было ложное предупреждение
        task, root = self.make(kind="regression")
        text = task.task_md.read_text(encoding="utf-8")
        self.assertIn("`submission.csv`", text)
        (root / "submission.csv").write_text("id,y\n1,2\n", encoding="utf-8")
        files = ml_io.find_task_files(task.task_md)
        self.assertEqual(files.warnings, [])
        self.assertEqual(files.extra_csv, [])

    def write_task(self, files, text="`train.csv` `test.csv`"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task.md").write_text(text, encoding="utf-8")
        for name, content in files.items():
            (root / name).write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        return root

    def test_numbers_with_missing_markers_become_numeric(self):
        # ревью Ивана (Y1): «NA», «?», «null» превращали числовой признак в категорию, а таргет ломали
        rows = ["id,x,y,country"] + [f"{i},{i * 0.5},{i * 2},{'NA' if i % 3 else 'RU'}" for i in range(40)]
        rows[5] = "4,NA,8,RU"
        rows[6] = "5,?,null,RU"
        rows[7] = "6, ,12,RU"
        root = self.write_task({"train.csv": "\n".join(rows) + "\n", "test.csv": "id,x,country\n100,1.5,NA\n101,NA,RU\n"})
        data = ml_io.load_task(root / "task.md")
        self.assertTrue(pd.api.types.is_float_dtype(data.train["x"]))
        self.assertTrue(pd.api.types.is_float_dtype(data.train["y"]))
        self.assertEqual(int(data.train["x"].isna().sum()), 3)
        self.assertTrue(pd.isna(data.train.loc[5, "y"]))
        self.assertTrue(pd.isna(data.test.loc[1, "x"]))
        self.assertEqual(set(data.train["country"]), {"NA", "RU"})  # настоящая категория «NA» не тронута
        self.assertTrue(any("переведены в числа" in w for w in data.files.warnings))

    def test_rare_legit_class_is_not_erased(self):
        # ревью Ивана (25.10, №3): до 5% нечисловых значений считались мусором, и редкий класс «A»
        # в таргете из «1»/«2»/«3» превращался в пропуск, а split_task выбрасывал эти строки
        labels = ["1", "2", "3"] * 33 + ["A"] * 4
        rows = ["id,x,target"] + [f"{i},{i * 0.5},{label}" for i, label in enumerate(labels)]
        root = self.write_task({"train.csv": "\n".join(rows) + "\n", "test.csv": "id,x\n500,1.5\n"})
        data = ml_io.load_task(root / "task.md")
        self.assertEqual(int((data.train["target"].astype(str) == "A").sum()), 4)
        self.assertFalse(data.train["target"].isna().any())

    def test_target_class_is_kept_whatever_the_junk_rule_says(self):
        # код-ревью 25.10 (F02): одиночный класс «A» считался мусором, а класс «NA» в классификации — пропуском
        for labels, text in ((["1", "2"] * 50 + ["A"], "`train.csv` `test.csv`"),
                             (["1", "2"] * 50 + ["NA"] * 30, "Классификация, основная метрика — Macro F1. `train.csv` `test.csv`")):
            rows = ["id,x,target"] + [f"{i},{i * 0.5},{label}" for i, label in enumerate(labels)]
            root = self.write_task({"train.csv": "\n".join(rows) + "\n", "test.csv": "id,x\n500,1.5\n"}, text)
            data = ml_io.load_task(root / "task.md")
            self.assertFalse(data.train["target"].isna().any(), labels[-1])
            self.assertEqual(int((data.train["target"].astype(str) == labels[-1]).sum()), labels.count(labels[-1]))
        # регрессия: «NA» в числовом таргете — по-прежнему пропуск (Y1)
        rows = ["id,x,y"] + [f"{i},{i},{'NA' if i % 10 == 0 else i * 2}" for i in range(100)]
        root = self.write_task({"train.csv": "\n".join(rows) + "\n", "test.csv": "id,x\n500,1\n"},
                               "Регрессия, основная метрика — RMSE. `train.csv` `test.csv`")
        data = ml_io.load_task(root / "task.md")
        self.assertTrue(pd.api.types.is_float_dtype(data.train["y"]))

    def test_single_junk_value_still_becomes_numeric(self):
        # единичный мусор (не больше max(2, 0.1%) строк, одно-два значения) по-прежнему не мешает
        rows = ["id,x,y"] + [f"{i},{'oops' if i == 3 else i * 0.5},{i}" for i in range(200)]
        root = self.write_task({"train.csv": "\n".join(rows) + "\n", "test.csv": "id,x\n500,1.5\n"})
        data = ml_io.load_task(root / "task.md")
        self.assertTrue(pd.api.types.is_float_dtype(data.train["x"]))
        self.assertTrue(pd.isna(data.train.loc[3, "x"]))

    def test_mostly_text_column_is_not_converted(self):
        rows = ["id,code,y"] + [f"{i},{'A' + str(i) if i % 2 else str(i)},{i}" for i in range(20)]
        data = ml_io.load_task(self.write_task({"train.csv": "\n".join(rows) + "\n", "test.csv": "id,code\n50,A3\n"}) / "task.md")
        self.assertFalse(pd.api.types.is_numeric_dtype(data.train["code"]))

    def test_semicolon_separator_cp1251_and_tsv(self):
        # ревью Ивана (Y9): «;» читался одним столбцом, cp1251 падал, .tsv не искался
        root = self.write_task({"train.csv": "id;x;y\n1;2;3\n2;3;4\n", "test.csv": "id;x\n3;4\n"})
        self.assertEqual(list(ml_io.load_task(root / "task.md").train.columns), ["id", "x", "y"])
        root = self.write_task({"train.csv": "id,город,y\n1,Москва,3\n2,Тула,4\n".encode("cp1251"),
                                "test.csv": "id,город\n3,Москва\n".encode("cp1251")})
        data = ml_io.load_task(root / "task.md")
        self.assertEqual(list(data.train.columns), ["id", "город", "y"])
        self.assertEqual(data.train.loc[0, "город"], "Москва")
        root = self.write_task({"train.tsv": "id\tx\ty\n1\t2\t3\n", "test.tsv": "id\tx\n3\t4\n"}, "`train.tsv` `test.tsv`")
        data = ml_io.load_task(root / "task.md")
        self.assertEqual(list(data.train.columns), ["id", "x", "y"])
        self.assertEqual(data.files.paths["test"].name, "test.tsv")

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
