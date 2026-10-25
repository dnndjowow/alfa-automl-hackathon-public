import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import agent_llm
import agent_tools
from tests.synth import make_task

FAST = {"learning_rate": 0.2, "max_rounds": 80}


class ScriptedChat:
    """Вместо LLM: отдаёт заранее заданные вызовы инструментов и запоминает, что видела."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.usage = None

    def __call__(self, messages, tools):
        self.seen.append(json.loads(json.dumps(messages, ensure_ascii=False, default=str)))
        if not self.script:
            return {"content": "готово", "tool_calls": []}
        step = self.script.pop(0)
        if callable(step):
            step = step(messages)
        if isinstance(step, str):
            return {"content": step, "tool_calls": []}
        name, args = step
        return {"content": "", "tool_calls": [{"id": f"c{len(self.seen)}", "name": name,
                                                "arguments": json.dumps(args, ensure_ascii=False)}]}


def session_for(tmp, deadline_s=600, **kwargs):
    task = make_task(tmp, **kwargs)
    session = agent_tools.Session(task.task_md, deadline=time.monotonic() + deadline_s, log=lambda m: None, n_jobs=2)
    return task, session


class AgentLoopTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_regression_dialog_writes_valid_submission(self):
        task, session = session_for(self.tmp, kind="regression", n_train=400, target="price", id_col="row")
        chat = ScriptedChat([
            ("set_task", {"target_cols": ["price"], "id_cols": ["row"], "task_type": "regression", "metric": "RMSE"}),
            ("evaluate", {"model": "lightgbm", "hypothesis": "база", **FAST}),
            ("evaluate", {"model": "catboost", "hypothesis": "сравнить", **FAST}),
            ("finalize", {}),
            ("finish", {"reason": "достаточно"}),
        ])
        summary = agent_llm.run_agent(session, chat, log=lambda m: None)
        self.assertEqual(summary["finished"], "достаточно")
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertEqual(list(sub.columns), ["row", "price"])
        self.assertTrue((sub["row"].to_numpy() == task.truth["row"].to_numpy()).all())
        self.assertEqual(session.written["run_id"], session.best_run())
        # LLM видела результат своего эксперимента (метрику), а не только «ок»
        tool_msgs = [m for m in chat.seen[-1] if m["role"] == "tool"]
        self.assertIn("score", json.loads(tool_msgs[1]["content"]))

    def test_multiclass_labels_and_early_submission(self):
        task, session = session_for(self.tmp, kind="multiclass", n_train=500, target="label")
        chat = ScriptedChat([
            ("set_task", {"target_cols": ["label"], "id_cols": ["id"], "task_type": "multiclass", "metric": "Macro F1"}),
            ("evaluate", {"model": "lightgbm", "hypothesis": "база", **FAST}),
            ("finish", {"reason": "стоп"}),
        ])
        agent_llm.run_agent(session, chat, log=lambda m: None)
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertEqual(list(sub.columns), ["id", "label"])
        self.assertTrue(set(sub["label"]) <= set(task.truth["label"]) | {"alpha", "beta", "gamma"})
        self.assertEqual(session.written["run_id"], "r1")  # ранний ответ записан без finalize от LLM

    def test_bad_calls_return_errors_instead_of_crashing(self):
        _, session = session_for(self.tmp, kind="regression")
        self.assertFalse(session.call("evaluate", {"model": "lightgbm"})["ok"])          # до set_task
        self.assertFalse(session.call("set_task", {"target_cols": ["y"], "task_type": "regression",
                                                   "metric": "f1_macro"})["ok"])          # не та метрика
        self.assertFalse(session.call("set_task", {"target_cols": ["nope"], "task_type": "regression",
                                                   "metric": "rmse"})["ok"])              # нет столбца
        self.assertFalse(session.call("unknown", {})["ok"])
        ok = session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "mae"})
        self.assertTrue(ok["ok"], ok)
        res = session.call("evaluate", {"model": "lightgbm", "params": {"n_jobs": 64, "num_leaves": 15}, **FAST})
        self.assertIn("n_jobs", res["dropped_params"])
        json.dumps(res)  # результат всегда сериализуется

    def test_worse_run_does_not_overwrite_written_answer(self):
        _, session = session_for(self.tmp, kind="regression", n_train=400)
        session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
        session.call("evaluate", {"model": "lightgbm", **FAST})
        bad = session.call("evaluate", {"model": "lightgbm", "learning_rate": 0.001, "max_rounds": 5})
        self.assertEqual(session.written["run_id"], "r1")
        out = session.call("finalize", {"run_id": bad["run_id"]})
        self.assertFalse(out["ok"])
        self.assertEqual(session.written["run_id"], "r1")

    def test_text_tool_calls_are_parsed(self):
        calls = agent_llm.parse_text_calls('Думаю так: {"tool": "finish", "arguments": {"reason": "x"}}')
        self.assertEqual(calls[0]["name"], "finish")
        self.assertEqual(agent_llm.parse_text_calls("просто текст {x}"), [])
        nested = agent_llm.parse_text_calls('{"type":"function","function":{"name":"finish","arguments":"{\\"reason\\":\\"y\\"}"}}')
        self.assertEqual(json.loads(nested[0]["arguments"]), {"reason": "y"})
        two = agent_llm.parse_text_calls('<think>план {a}</think><tool_call>{"name":"evaluate","arguments":{"model":"lightgbm"}}'
                                         '</tool_call><tool_call>{"name":"finish","arguments":{}}</tool_call>')
        self.assertEqual([c["name"] for c in two], ["evaluate", "finish"])

    def test_model_without_tool_calls_stops(self):
        _, session = session_for(self.tmp, kind="regression")
        summary = agent_llm.run_agent(session, ScriptedChat(["a", "b", "c", "d"]), log=lambda m: None)
        self.assertLessEqual(summary["steps"], 3)

    def test_constant_safety_answer(self):
        task, session = session_for(self.tmp, kind="binary", target="t")
        self.assertIsNotNone(session.write_constant())
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertEqual(len(sub), len(task.truth))


class GeneralizationTest(unittest.TestCase):
    """Постановки из проверки на внешних датасетах (tests/external_tasks.py)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_positive_class_from_task_text(self):
        y = pd.Series(["good"] * 70 + ["bad"] * 30)
        text = "Оцените вероятность того, что заёмщик плохой (`risk` = `bad`; хорошие — `good`)."
        self.assertEqual(agent_tools.guess_positive_class(["bad", "good"], y, text), "bad")
        self.assertEqual(agent_tools.guess_positive_class([0, 1], pd.Series([0, 1, 1]), ""), 1)
        self.assertEqual(agent_tools.guess_positive_class(["a", "b"], pd.Series(["a"] * 9 + ["b"]), ""), "b")
        text = "вероятность того, что доход больше (`income` = `>50K`; иначе `<=50K`)"
        self.assertEqual(agent_tools.guess_positive_class(["<=50K", ">50K"], pd.Series(["<=50K", ">50K"]), text), ">50K")

    def test_binary_probability_of_named_class(self):
        task = make_task(self.tmp, kind="binary", n_train=600, target="t")
        text = task.task_md.read_text(encoding="utf-8") + "\nНужна вероятность класса `alpha`; метрика ROC AUC.\n"
        task.task_md.write_text(text, encoding="utf-8")
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
        out = session.call("set_task", {"target_cols": ["t"], "id_cols": ["id"], "task_type": "binary", "metric": "roc_auc"})
        self.assertEqual(out["positive_class"], {"t": "alpha"})
        session.call("evaluate", {"model": "lightgbm", **FAST})
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        from sklearn.metrics import roc_auc_score
        self.assertGreater(roc_auc_score(task.truth["t"] == "alpha", sub["t"]), 0.7)
        bad = session.call("set_task", {"target_cols": ["t"], "id_cols": ["id"], "task_type": "binary",
                                        "metric": "roc_auc", "positive_class": "omega"})
        self.assertFalse(bad["ok"])

    def test_two_targets_are_both_modelled(self):
        task = make_task(self.tmp, kind="regression", n_train=500)
        folder = task.task_md.parent / "data" / "participants"
        train = pd.read_csv(folder / "train.csv")
        train["y2"] = -3 * train["y"] + 1
        train.to_csv(folder / "train.csv", index=False)
        sample = pd.read_csv(folder / "sample_submission.csv")
        sample["y2"] = 0.0
        sample.to_csv(folder / "sample_submission.csv", index=False)
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
        out = session.call("set_task", {"target_cols": ["y", "y2"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
        self.assertTrue(out["ok"], out)
        res = session.call("evaluate", {"model": "lightgbm", **FAST})
        self.assertEqual(set(res["per_target"]), {"y", "y2"})
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertLess(abs(np.corrcoef(sub["y"], sub["y2"])[0, 1] + 1), 0.05)  # y2 ≈ -3y: обе модели настоящие

    def test_rmsle_metric_and_nonnegative_predictions(self):
        task = make_task(self.tmp, kind="regression", n_train=500)
        folder = task.task_md.parent / "data" / "participants"
        train = pd.read_csv(folder / "train.csv")
        train["y"] = np.exp(train["y"] / 2).round(1)
        train.to_csv(folder / "train.csv", index=False)
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
        out = session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "RMSLE"})
        self.assertEqual(out["metric"], "rmsle")
        res = session.call("evaluate", {"model": "lightgbm", "target_transform": "log1p", **FAST})
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["metric"], "rmsle")
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertTrue((sub["y"] >= 0).all())


class BlendTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_blend_uses_oof_weights_and_reuses_fitted_members(self):
        for kind, metric in (("regression", "rmse"), ("multiclass", "f1_macro")):
            task = make_task(self.tmp / kind, kind=kind, n_train=600, target="t")
            session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
            session.call("set_task", {"target_cols": ["t"], "id_cols": ["id"], "task_type": kind, "metric": metric})
            session.call("evaluate", {"model": "lightgbm", **FAST})
            session.call("evaluate", {"model": "catboost", "learning_rate": 0.2, "max_rounds": 120})
            out = session.call("blend", {})
            if not out["ok"]:
                self.assertIn("не лучше", out["error"])   # смесь не создаётся, если не помогает
                continue
            self.assertTrue(session._is_better(out["oof_score"], out["best_single_oof"]))
            fin = session.call("finalize", {"run_id": out["run_id"]})
            self.assertTrue(fin["ok"], fin)
            self.assertEqual(session.written["run_id"], out["run_id"])
            sub = pd.read_csv(task.task_md.parent / "submission.csv")
            self.assertEqual(len(sub), len(task.truth))

    def test_blend_needs_two_runs_and_simplex_grid(self):
        self.assertEqual(len(agent_tools._simplex_grid(2, 10)), 11)
        self.assertTrue(all(abs(w.sum() - 1) < 1e-9 for w in agent_tools._simplex_grid(3, 5)))
        task = make_task(self.tmp, kind="regression", n_train=300)
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
        session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
        session.call("evaluate", {"model": "lightgbm", **FAST})
        self.assertFalse(session.call("blend", {})["ok"])

    def test_core_names_agent_metrics_as_primary(self):
        import ml_schema
        self.assertEqual(ml_schema.mentioned_metrics("Качество — **RMSE между логарифмами** цены")["primary"], "rmsle")
        self.assertEqual(ml_schema.mentioned_metrics("RMSLE — основная метрика. RMSE для справки")["primary"], "rmsle")
        self.assertEqual(ml_schema.mentioned_metrics("Метрика — RMSE, меньше — лучше")["primary"], "rmse")


class CoreReviewWorkaroundTest(unittest.TestCase):
    """Обходы дефектов ML-ядра из docs/REVIEW_FINDINGS.md на стороне агента."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_round_aliases_and_catboost_params_do_not_break_final_fit(self):
        _, session = session_for(self.tmp, kind="regression", n_train=400)
        session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
        res = session.call("evaluate", {"model": "lightgbm", "params": {"early_stopping": 20, "num_trees": 40}, **FAST})
        self.assertTrue(res["ok"] and {"early_stopping", "num_trees"} <= set(res["dropped_params"]), res)
        self.assertEqual(session.written["run_id"], res["run_id"])     # финальное обучение прошло
        cat = session.call("evaluate", {"model": "catboost", "learning_rate": 0.2, "max_rounds": 80,
                                        "params": {"max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8,
                                                   "min_child_weight": 3}})
        self.assertTrue(cat["ok"], cat)
        self.assertEqual(cat["params"], {"depth": 4, "subsample": 0.8, "bootstrap_type": "Bernoulli", "rsm": 0.8})

    def test_binary_f1_of_positive_class(self):
        task, session = session_for(self.tmp, kind="binary", n_train=500, target="t")
        out = session.call("set_task", {"target_cols": ["t"], "id_cols": ["id"], "task_type": "binary",
                                        "metric": "F1-score", "positive_class": "beta"})
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["metric"], "f1")
        res = session.call("evaluate", {"model": "lightgbm", **FAST})
        self.assertTrue(res["ok"], res)
        self.assertFalse(session.call("set_task", {"target_cols": ["t"], "task_type": "multiclass", "metric": "f1"})["ok"])


class SecondReviewTest(unittest.TestCase):
    """Находки второго ревью агента (docs/REVIEW_FINDINGS.md, раздел «Агент Ивана»)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_numeric_positive_class_ignores_probability_range_text(self):
        y = pd.Series([0] * 80 + [1] * 20)
        for text in ("Нужна вероятность от 0 до 1 того, что клиент уйдёт (t = 1)",
                     "Predict the probability (between 0 and 1) that t is 1", "вероятность в диапазоне [0, 1]"):
            self.assertEqual(agent_tools.guess_positive_class([0, 1], y, text), 1)

    def test_label_metric_writes_labels_even_if_text_mentions_probability(self):
        task = make_task(self.tmp, kind="binary", n_train=400, target="t")
        text = task.task_md.read_text(encoding="utf-8") + "\nМетрика — Macro F1. В ответе метка класса (не вероятность!).\n"
        task.task_md.write_text(text, encoding="utf-8")
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
        out = session.call("set_task", {"target_cols": ["t"], "id_cols": ["id"], "task_type": "binary", "metric": "f1_macro"})
        self.assertFalse(out["answer_as_probability"])
        session.call("evaluate", {"model": "lightgbm", **FAST})
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertTrue(set(sub["t"]) <= {"alpha", "beta"})

    def test_prediction_goes_to_renamed_output_column(self):
        task = make_task(self.tmp, kind="regression", n_train=400)
        folder = task.task_md.parent / "data" / "participants"
        pd.DataFrame({"id": task.truth["id"], "prediction": 0.0}).to_csv(folder / "sample_submission.csv", index=False)
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
        out = session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
        self.assertTrue(out["ok"], out)
        session.call("evaluate", {"model": "lightgbm", **FAST})
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertEqual(list(sub.columns), ["id", "prediction"])
        self.assertGreater(sub["prediction"].nunique(), 10)

    def test_fallback_does_not_regress_a_binary_second_target(self):
        import solution
        task = make_task(self.tmp, kind="regression", n_train=400)
        folder = task.task_md.parent / "data" / "participants"
        train = pd.read_csv(folder / "train.csv")
        train["flag2"] = (train["y"] > 0).astype(int)
        train.to_csv(folder / "train.csv", index=False)
        sample = pd.read_csv(folder / "sample_submission.csv")
        sample["flag2"] = 0
        sample.to_csv(folder / "sample_submission.csv", index=False)
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 150, log=lambda m: None, n_jobs=2)
        solution.fallback_policy(session)
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertTrue(set(sub["flag2"]) <= {0, 1})
        self.assertGreater(sub["y"].nunique(), 10)

    def test_garbled_names_do_not_capture_other_parameters(self):
        fitted, ignored = agent_tools._fit_args(agent_tools.Session.tool_evaluate,
                                                {"learning_rate_decay": 0.9, "hypothesisary": "x", "modl": "catboost"})
        self.assertNotIn("learning_rate", fitted)
        self.assertEqual(fitted.get("hypothesis"), "x")
        self.assertIn("learning_rate_decay", ignored)

    def test_compaction_keeps_a_single_summary(self):
        _, session = session_for(self.tmp, kind="regression")
        messages = agent_llm.initial_messages(session)
        base = messages[1]
        for step in range(30):
            messages = agent_llm.compact(messages, session, keep=6, base_first=base)
            messages += [{"role": "assistant", "content": f"шаг {step}"}, {"role": "user", "content": "ok"}]
        self.assertEqual(messages[1]["content"].count("Что уже сделано"), 1)

    def test_client_keeps_tools_after_transient_error_and_bounds_timeout(self):
        class Flaky:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(("tools" in kwargs, kwargs["timeout"]))
                if len(self.calls) == 1:
                    error = RuntimeError("Service Unavailable")
                    error.status_code = 503
                    raise error
                message = type("M", (), {"content": "ok", "tool_calls": None})
                return type("R", (), {"usage": None, "choices": [type("C", (), {"message": message})]})
        chat = agent_llm.OpenAIChat.__new__(agent_llm.OpenAIChat)
        flaky = Flaky()
        chat.client = type("Client", (), {"chat": type("Chat", (), {"completions": flaky})})
        chat.model, chat.retries, chat.timeout, chat.tools_supported = "m", 3, 240.0, True
        chat.usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
        chat.time_left = lambda: 200.0
        original = agent_llm.time.sleep
        agent_llm.time.sleep = lambda s: None
        try:
            chat([{"role": "user", "content": "x"}], [])
        finally:
            agent_llm.time.sleep = original
        self.assertTrue(chat.tools_supported)
        self.assertEqual([c[0] for c in flaky.calls], [True, True])
        self.assertLessEqual(flaky.calls[0][1], 140.0)


class MaxQualityTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_tune_registers_trials_and_seeds_average_final_fits(self):
        task, session = session_for(self.tmp, kind="regression", n_train=400)
        session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
        out = session.call("tune", {"model": "lightgbm", "n_trials": 4, "time_budget_s": 120})
        self.assertTrue(out["ok"], out)
        self.assertGreaterEqual(out["trials"], 2)
        self.assertEqual(session.runs[out["top"][-1]["run_id"]]["config"]["model"], "lightgbm")
        best = session.best_run()
        first = session.call("finalize", {"run_id": best})
        single = pd.read_csv(task.task_md.parent / "submission.csv")["y"].to_numpy()
        bagged = session.call("finalize", {"run_id": best, "seeds": 3})
        self.assertTrue(bagged["ok"], bagged)
        self.assertEqual(session.written["seeds"], 3)
        averaged = pd.read_csv(task.task_md.parent / "submission.csv")["y"].to_numpy()
        self.assertFalse(np.allclose(single, averaged))      # действительно другие сиды
        self.assertEqual(session.estimate_final_seconds(best, 3), 0.0)  # все три сида в кэше
        self.assertTrue(first["ok"] or "уже записан" in first.get("note", ""))


class LLMArgumentTest(unittest.TestCase):
    def test_garbled_argument_names_are_mapped_or_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, session = session_for(tmp, kind="regression", n_train=300)
            session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            out = session.call("tune", {"model": "lightgbm", "hypothesisary": "x", "hydrogen-": "x",
                                        "n_trials-": "2", "time_budget_s-": "60"})
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["trials"], 2)
            self.assertEqual(set(out["ignored_args"]), {"hydrogen-"})

    def test_dialog_stops_when_model_loops_on_failing_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, session = session_for(tmp, kind="regression")
            chat = ScriptedChat([("evaluate", {"model": "nope"})] * 20)
            summary = agent_llm.run_agent(session, chat, log=lambda m: None)
            self.assertLessEqual(summary["steps"], 8)


class PlanItem2Test(unittest.TestCase):
    def test_date_column_allowed_for_llm_temporal_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(tmp, kind="regression", n_train=400)
            folder = task.task_md.parent / "data" / "participants"
            for name, start in (("train", "2020-01-01"), ("test", "2021-06-01")):
                frame = pd.read_csv(folder / f"{name}.csv")
                frame["signup"] = pd.date_range(start, periods=len(frame), freq="D").strftime("%Y-%m-%d")
                frame.to_csv(folder / f"{name}.csv", index=False)
            session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
            session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            out = session.call("evaluate", {"model": "lightgbm", "validation": {"strategy": "temporal", "column": "signup"},
                                            **FAST})
            self.assertTrue(out["ok"], out)

    def test_threshold_for_binary_f1_on_imbalanced_classes(self):
        rng = np.random.default_rng(0)
        codes = (rng.random(3000) < 0.1).astype(int)
        score = codes * 0.35 + rng.random(3000) * 0.6        # позитивы редкие, порог 0.5 для них слишком высок
        proba = np.column_stack([1 - score, score])
        metric = agent_tools._binary_metric("f1", 1)
        choice = agent_tools.choose_weights_for_metric(proba, codes, metric)
        self.assertTrue(choice["applied"], choice)
        before = metric(codes, proba.argmax(axis=1))
        after = metric(codes, (proba * choice["weights"]).argmax(axis=1))
        self.assertGreater(after, before + 0.05)


class AddFeaturesTest(unittest.TestCase):
    def make(self, tmp):
        task = make_task(tmp, kind="regression", n_train=1500, n_test=300)
        folder = task.task_md.parent / "data" / "participants"
        rng = np.random.default_rng(1)
        for name in ("train", "test"):
            frame = pd.read_csv(folder / f"{name}.csv")
            frame["num"] = rng.uniform(1, 10, len(frame))
            frame["den"] = rng.uniform(1, 10, len(frame))
            if name == "train":
                frame["y"] = 5 * frame["num"] / frame["den"] + rng.normal(0, 0.05, len(frame))
            frame.to_csv(folder / f"{name}.csv", index=False)
        return task

    def test_useful_ratio_is_accepted_and_used_in_final_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = self.make(tmp)
            session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
            session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            out = session.call("add_features", {"features": ["num / den", {"op": "count", "col": "cat"}, "nope / den"],
                                                "hypothesis": "y — отношение"})
            self.assertTrue(out["ok"] and out["accepted"], out)
            self.assertIn("num__ratio__den", out["features"])
            self.assertEqual(len(out["errors"]), 1)                       # 'nope' — понятная ошибка, не падение
            self.assertIn("num__ratio__den", session.setup["split"].X_test.columns)
            fin = session.call("finalize", {"run_id": out["new_run"]})
            self.assertTrue(fin["ok"], fin)
            base_fin = session._fit_run(out["base_run"])                   # старый запуск — на старых признаках
            self.assertEqual(len(base_fin["y"]["pred"]), 300)

    def test_as_category_turns_codes_into_text(self):
        X = pd.DataFrame({"mgr": [10, 20, 10, None]})
        train, test, applied, errors = agent_tools.build_features(X.iloc[:3], X.iloc[3:], [{"op": "as_category", "col": "mgr"}])
        self.assertEqual(applied[0]["name"], "mgr__cat")
        self.assertEqual(list(train["mgr__cat"]), ["10.0", "20.0", "10.0"])
        self.assertTrue(pd.isna(test["mgr__cat"].iloc[0]))

    def test_target_mean_is_out_of_fold_on_train_and_full_on_test(self):
        X = pd.DataFrame({"area": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]})
        y = pd.Series(np.arange(20, dtype=float))
        test = pd.DataFrame({"area": [1, 2, 3]})
        train, test_out, applied, errors = agent_tools.build_features(X, test, [{"op": "target_mean", "by": "area"}],
                                                                      y_train=y, task_type="regression")
        self.assertEqual(applied[0]["name"], "te__area")
        self.assertEqual(list(test_out["te__area"].iloc[:2]), [4.5, 14.5])     # среднее по всему train
        self.assertTrue(pd.isna(test_out["te__area"].iloc[2]))                # новой сущности нет в train
        self.assertEqual(list(test_out["te__area__n"]), [8, 8, 0])     # шкала train: 4 из 5 фолдов
        # временная схема: строка видит только более ранние периоды, test — весь train
        times = np.tile([1.0, 2.0], 10)
        train, test_out, _, _ = agent_tools.build_features(X, test, [{"op": "target_mean", "by": "area"}], y_train=y,
                                                           task_type="regression", te_scheme={"times": times})
        self.assertTrue(train["te__area"][times == 1].isna().all())
        self.assertEqual(train["te__area"].iloc[1], y[(X.area == 1) & (times == 1)].mean())
        self.assertEqual(list(test_out["te__area"].iloc[:2]), [4.5, 14.5])
        self.assertFalse(np.allclose(train["te__area"], np.where(X.area == 1, 4.5, 14.5)))  # своя строка не входит
        unique = pd.DataFrame({"area": range(20)})
        _, _, applied, errors = agent_tools.build_features(unique, test, [{"op": "target_mean", "by": ["area"]}],
                                                           y_train=y)
        self.assertEqual(applied, [])
        self.assertIn("почти уникален", errors[0])

    def test_agent_loop_checks_entity_target_mean_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(tmp, kind="regression", n_train=1500, n_test=300)
            folder = task.task_md.parent / "data" / "participants"
            rng = np.random.default_rng(3)
            effect = rng.normal(0, 10, 150)
            for name in ("train", "test"):
                frame = pd.read_csv(folder / f"{name}.csv")
                frame["area"] = rng.integers(0, 150, len(frame)) * 37.5 + 100   # «площадь здания»
                if name == "train":
                    frame["y"] = effect[((frame["area"] - 100) / 37.5).astype(int)] + rng.normal(0, 1, len(frame))
                frame.to_csv(folder / f"{name}.csv", index=False)
            session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
            chat = ScriptedChat([
                ("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"}),
                ("evaluate", {"model": "lightgbm", "hypothesis": "база"}),
                ("finish", {"reason": "тест"}),
            ])
            agent_llm.run_agent(session, chat, log=lambda m: None)
            reply = json.loads(chat.seen[2][-1]["content"])
            self.assertIn("area", json.dumps(json.loads(chat.seen[1][-1]["content"]).get("entity_key_hint")))
            self.assertTrue(reply["auto_entity_features"]["accepted"], reply["auto_entity_features"])
            self.assertIn("te__area", session.setup["split"].X_test.columns)

    def test_entity_keys_add_columns_that_stay_constant_per_entity(self):
        rng = np.random.default_rng(0)
        buildings = pd.DataFrame({"area": rng.uniform(1000, 9000, 300).round(1),
                                  "kind": rng.choice(list("ABCDEFG"), 300), "built": rng.integers(1900, 2020, 300)})
        def years(n):
            rows = buildings.sample(n, replace=True, random_state=n)
            return rows.assign(temp=rng.normal(10, 5, n).round(1), rare=np.where(rng.random(n) < 0.99, 0, 1))
        keys = agent_tools.entity_keys(years(2000).reset_index(drop=True), years(500).reset_index(drop=True))
        self.assertEqual(keys[0], ["area"])
        self.assertEqual(keys[-1][0], "area")
        self.assertEqual(set(keys[-1][1:]), {"kind", "built"})   # погода и почти константный столбец — нет

    def test_target_mean_on_date_blocks_ignores_targets_of_the_same_block(self):
        import ml_validation
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"t": rng.uniform(0, 1000, 2000), "ent": rng.integers(0, 40, 2000)})
        y = pd.Series(X.ent + X.t / 100 + rng.normal(0, 1, 2000))
        plan = {"strategy": "temporal", "column": "t", "direction": "forward", "n_splits": 3}
        splits = list(ml_validation.make_splits(plan, X, None))
        scheme = {"times": ml_validation._time_values(X.t), "splits": splits}
        keys, before = X.ent.astype(str), agent_tools._target_mean(X.ent.astype(str), pd.Series(["1"]), y, True, scheme)
        for _, valid in splits:   # ревью Юры, п. 1: таргеты своего проверочного блока не видны
            changed = y.copy()
            changed.iloc[valid] = rng.normal(0, 100, len(valid))
            after = agent_tools._target_mean(keys, pd.Series(["1"]), changed, True, scheme)
            for suffix in ("", "__last", "__gap"):
                np.testing.assert_allclose(before[suffix][0][valid], after[suffix][0][valid])

    def test_target_mean_is_not_built_under_group_validation(self):
        X = pd.DataFrame({"area": [1.0, 1, 2, 2]})
        _, _, applied, errors = agent_tools.build_features(X, X.iloc[:1], [{"op": "target_mean", "by": "area"}],
                                                           y_train=pd.Series([1.0, 2, 3, 4]), task_type="regression",
                                                           te_scheme={"skip": "групповая схема"})
        self.assertEqual(applied, [])                          # ревью Юры, F01
        self.assertIn("групповая схема", errors[0])

    def test_failed_row_order_request_leaves_data_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = self.make(tmp)
            session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
            session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            out = session.call("evaluate", {"validation": {"strategy": "temporal", "column": "row_order",
                                                           "direction": "bad"}})
            self.assertFalse(out["ok"])                            # ревью Юры, F12
            self.assertEqual(session.featureset, 0)
            self.assertNotIn(agent_tools.ROW_ORDER, session.setup["split"].X_train.columns)

    def test_useless_feature_leaves_data_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = self.make(tmp)
            session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
            session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            columns = list(session.setup["split"].X_train.columns)
            out = session.call("add_features", {"features": [{"op": "nan_count"}]})
            if not out["accepted"]:
                self.assertEqual(list(session.setup["split"].X_train.columns), columns)
                self.assertEqual(session.featureset, 0)


class StabilizeTest(unittest.TestCase):
    def test_revalidation_scores_runs_and_blends_on_new_folds(self):
        import solution
        with tempfile.TemporaryDirectory() as tmp:
            task, session = session_for(tmp, kind="regression", n_train=500)
            session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            session.call("evaluate", {"model": "lightgbm", **FAST})
            session.call("evaluate", {"model": "catboost", "learning_rate": 0.2, "max_rounds": 120})
            blend = session.call("blend", {})
            ids = ["r1", "r2"] + ([blend["run_id"]] if blend.get("ok") else [])
            scores = session.revalidate(ids)
            self.assertEqual(set(scores), set(ids))
            for value in scores.values():
                self.assertNotEqual(value["first"], value["second"])     # действительно другое разбиение
            solution.log = lambda m: None
            import os
            os.environ["AGENT_STABILIZE"] = "on"
            try:
                solution.stabilize(session)                               # выбирает и записывает без ошибок
            finally:
                del os.environ["AGENT_STABILIZE"]
            self.assertIn(session.written["run_id"], ids)


class LoggingTest(unittest.TestCase):
    def test_log_survives_unencodable_characters(self):
        import io, sys, solution
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1251", errors="strict")
        original = sys.stdout
        sys.stdout = stream
        try:
            solution.log("ответ LLM: площадь ft² → 18⅓ °C ✓")
            stream.flush()
        finally:
            sys.stdout = original
        self.assertIn(b"LLM", raw.getvalue())


class EarlyFinishTest(unittest.TestCase):
    def test_first_early_finish_is_refused_on_large_data_second_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, session = session_for(tmp, kind="regression", n_train=6000, deadline_s=3600)
            chat = ScriptedChat([
                ("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"}),
                ("evaluate", {"model": "lightgbm", "hypothesis": "база", **FAST}),
                ("finish", {"reason": "хватит"}),
                ("finish", {"reason": "точно хватит"}),
            ])
            summary = agent_llm.run_agent(session, chat, log=lambda m: None)
            self.assertEqual(summary["finished"], "точно хватит")
            refused = [m for m in chat.seen[-1] if m["role"] == "tool" and "Рано останавливаться" in m["content"]]
            self.assertEqual(len(refused), 1)

    def test_small_data_finish_is_accepted_at_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, session = session_for(tmp, kind="regression", n_train=500, deadline_s=3600)
            chat = ScriptedChat([
                ("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"}),
                ("evaluate", {"model": "lightgbm", "hypothesis": "база", **FAST}),
                ("finish", {"reason": "мало данных"}),
            ])
            self.assertEqual(agent_llm.run_agent(session, chat, log=lambda m: None)["finished"], "мало данных")


class RowOrderTest(unittest.TestCase):
    def test_row_order_temporal_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, session = session_for(tmp, kind="regression", n_train=600)
            text = task.task_md.read_text(encoding="utf-8") + "\nСтроки упорядочены по времени; test — более поздние.\n"
            task.task_md.write_text(text, encoding="utf-8")
            session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
            self.assertTrue(session.rows_look_time_ordered())
            self.assertIsNotNone(session.overview()["suggested"]["validation_hint"])
            session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            out = session.call("evaluate", {"model": "lightgbm", "validation": {"strategy": "temporal", "column": "row_order"},
                                            **FAST})
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["validation"]["strategy"], "temporal")
            X_train, X_test = session.setup["split"].X_train, session.setup["split"].X_test
            self.assertEqual(X_train["__row_order__"].max() + 1, X_test["__row_order__"].min())
            fin = session.call("finalize", {"run_id": out["run_id"]})
            self.assertTrue(fin["ok"] or "уже записан" in fin.get("note", ""), fin)


class SecondaryTargetTest(unittest.TestCase):
    def test_second_target_of_other_type_gets_its_own_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(tmp, kind="regression", n_train=600)
            folder = task.task_md.parent / "data" / "participants"
            train = pd.read_csv(folder / "train.csv")
            train["flag2"] = (train["a"] > 0).astype(int)
            train.to_csv(folder / "train.csv", index=False)
            sample = pd.read_csv(folder / "sample_submission.csv")
            sample["flag2"] = 0
            sample.to_csv(folder / "sample_submission.csv", index=False)
            session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
            session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            session.call("evaluate", {"model": "lightgbm", **FAST})
            sub = pd.read_csv(task.task_md.parent / "submission.csv")
            self.assertTrue(set(sub["flag2"]) == {0, 1})                  # не константа и метки 0/1
            test = pd.read_csv(folder / "test.csv")
            self.assertGreater(((test["a"] > 0).astype(int).to_numpy() == sub["flag2"].to_numpy()).mean(), 0.8)


class ThirdReviewTest(unittest.TestCase):
    def test_few_folds_compare_pooled_oof_with_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, session = session_for(tmp, kind="regression")
            session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            scores = {"a": 48.98, "b": 49.28, "c": 49.27}
            session._score = lambda r: scores[r]
            session._paired = lambda n, o: {"gain": 0.008, "se": 0.013, "wins": 1, "folds": 3}
            self.assertTrue(session._significantly_better("a", "b"))     # явно лучше по общему OOF
            self.assertFalse(session._significantly_better("c", "b"))    # почти равны — не меняем
            session._paired = lambda n, o: {"gain": 0.008, "se": 0.013, "wins": 2, "folds": 5}
            self.assertFalse(session._significantly_better("a", "b"))    # 5 фолдов — прежнее парное правило

    def test_blend_of_two_targets_can_be_finalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(tmp, kind="regression", n_train=600)
            folder = task.task_md.parent / "data" / "participants"
            train = pd.read_csv(folder / "train.csv")
            train["y2"] = train["y"] * 2 + train["b"]
            train.to_csv(folder / "train.csv", index=False)
            sample = pd.read_csv(folder / "sample_submission.csv")
            sample["y2"] = 0.0
            sample.to_csv(folder / "sample_submission.csv", index=False)
            session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
            session.call("set_task", {"target_cols": ["y", "y2"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            session.call("evaluate", {"model": "lightgbm", **FAST})
            session.call("evaluate", {"model": "catboost", "learning_rate": 0.2, "max_rounds": 120})
            blend = session.call("blend", {})
            if blend.get("ok"):
                fin = session.call("finalize", {"run_id": blend["run_id"]})
                self.assertTrue(fin["ok"], fin)
            else:   # смесь могла не дать выигрыша — тогда проверяем напрямую сборку смеси из двух участников
                session.runs["bx"] = {"config": {"model": "blend", "members": ["r1", "r2"],
                                                 "weights": {"y": [0.5, 0.5], "y2": [0.5, 0.5]}, "subsample": None},
                                      "result": {"status": "ok"}}
                fits = session._fit_run("bx")
                self.assertEqual(set(fits), {"y", "y2"})

    def test_row_order_probe_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, session = session_for(tmp, kind="regression", n_train=600)
            session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
            out = session.call("evaluate", {"model": "lightgbm", "subsample": 0.5, **FAST,
                                            "validation": {"strategy": "temporal", "column": "__row_order__"}})
            self.assertNotIn("__row_order__", session.setup["split"].X_train.columns)
            self.assertEqual(session.featureset, 0)


class RobustnessTest(unittest.TestCase):
    """Сценарии, найденные ревью: каждый раньше портил ответ или ронял агента."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def set_regression(self, session):
        out = session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
        self.assertTrue(out["ok"], out)

    def test_subsample_probe_never_replaces_full_answer(self):
        _, session = session_for(self.tmp, kind="regression", n_train=800)
        self.set_regression(session)
        session.call("evaluate", {"model": "lightgbm", **FAST})
        session.call("evaluate", {"model": "lightgbm", "subsample": 0.2, "learning_rate": 0.01, "max_rounds": 5})
        self.assertEqual(session.best_run(), "r1")
        self.assertFalse(session.call("finalize", {"run_id": "r2"})["ok"])
        self.assertEqual(session.written["run_id"], "r1")

    def test_failed_validation_override_keeps_working_plan(self):
        _, session = session_for(self.tmp, kind="regression", n_train=600)
        self.set_regression(session)
        session.call("evaluate", {"model": "lightgbm", **FAST})
        bad = session.call("evaluate", {"model": "lightgbm", "validation": {"strategy": "temporal", "column": "cat"}, **FAST})
        self.assertFalse(bad["ok"])
        again = session.call("evaluate", {"model": "lightgbm", "learning_rate": 0.1, "max_rounds": 80})
        self.assertTrue(again["ok"], again)
        group_col = session.call("evaluate", {"model": "lightgbm", "validation": {"strategy": "group", "column": "grp"}})
        self.assertFalse(group_col["ok"])

    def test_extra_output_column_is_not_filled_with_predictions(self):
        task = make_task(self.tmp, kind="regression", n_train=400)
        folder = task.task_md.parent / "data" / "participants"
        sample = pd.read_csv(folder / "sample_submission.csv")
        sample.insert(0, "row_id", range(len(sample)))
        sample.to_csv(folder / "sample_submission.csv", index=False)
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
        out = session.call("set_task", {"target_cols": ["y"], "id_cols": [], "task_type": "regression", "metric": "rmse"})
        self.assertIn("id", out["id_cols"])       # id из шаблона ответа не стал признаком
        session.call("evaluate", {"model": "lightgbm", **FAST})
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertEqual(list(sub.columns), ["row_id", "id", "y"])
        self.assertEqual(sub["row_id"].tolist(), list(range(len(sub))))
        self.assertEqual(sub["id"].tolist(), task.truth["id"].tolist())

    def test_per_class_probability_columns(self):
        task = make_task(self.tmp, kind="multiclass", n_train=500, target="label")
        folder = task.task_md.parent / "data" / "participants"
        sample = pd.DataFrame({"id": task.truth["id"], "proba_alpha": 0.3, "proba_beta": 0.3, "proba_gamma": 0.4})
        sample.to_csv(folder / "sample_submission.csv", index=False)
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
        out = session.call("set_task", {"target_cols": ["label"], "id_cols": ["id"], "task_type": "multiclass", "metric": "logloss"})
        self.assertTrue(out["ok"], out)
        session.call("evaluate", {"model": "lightgbm", **FAST})
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertEqual(list(sub.columns), ["id", "proba_alpha", "proba_beta", "proba_gamma"])
        self.assertTrue(((sub.iloc[:, 1:].sum(axis=1) - 1).abs() < 1e-6).all())

    def test_binary_float_target_written_as_int_and_proba_when_asked(self):
        task = make_task(self.tmp, kind="binary", n_train=500, target="t")
        folder = task.task_md.parent / "data" / "participants"
        train = pd.read_csv(folder / "train.csv")
        train["t"] = (train["t"] == "beta").astype(float)
        train.loc[:5, "t"] = float("nan")
        train.to_csv(folder / "train.csv", index=False)
        pd.DataFrame({"id": task.truth["id"], "t": 0}).to_csv(folder / "sample_submission.csv", index=False)
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
        session.call("set_task", {"target_cols": ["t"], "id_cols": ["id"], "task_type": "binary", "metric": "f1_macro"})
        session.call("evaluate", {"model": "lightgbm", **FAST})
        text = (task.task_md.parent / "submission.csv").read_text().splitlines()[1]
        self.assertRegex(text, r",[01]$")
        session.call("set_task", {"target_cols": ["t"], "id_cols": ["id"], "task_type": "binary", "metric": "roc_auc"})
        session.call("evaluate", {"model": "lightgbm", **FAST})
        session.call("finalize", {})
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertTrue(sub["t"].between(0, 1).all() and sub["t"].nunique() > 2)

    def test_second_target_gets_own_model_and_low_confidence_still_gets_safety(self):
        task = make_task(self.tmp, kind="regression", n_train=400)
        folder = task.task_md.parent / "data" / "participants"
        train = pd.read_csv(folder / "train.csv")
        train["y2"] = train["y"] * 2
        train.to_csv(folder / "train.csv", index=False)
        sample = pd.read_csv(folder / "sample_submission.csv")
        sample["y2"] = 0.0
        sample.to_csv(folder / "sample_submission.csv", index=False)
        session = agent_tools.Session(task.task_md, deadline=time.monotonic() + 600, log=lambda m: None, n_jobs=2)
        self.assertIsNotNone(session.write_constant())
        session.call("set_task", {"target_cols": ["y"], "id_cols": ["id"], "task_type": "regression", "metric": "rmse"})
        session.call("evaluate", {"model": "lightgbm", **FAST})
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertEqual(list(sub.columns), ["id", "y", "y2"])
        # второй таргет не в постановке: получает свою модель (раньше — константу), и она осмысленна: y2 = 2y
        self.assertGreater(sub["y2"].nunique(), 1)
        self.assertGreater(np.corrcoef(sub["y"], sub["y2"])[0, 1], 0.9)

    def test_solution_runs_fallback_when_model_never_calls_tools(self):
        import solution
        task = make_task(self.tmp, kind="regression", n_train=300)
        chat = ScriptedChat(["думаю", "ещё думаю", "всё ещё думаю"])
        original = agent_llm.OpenAIChat
        agent_llm.OpenAIChat = lambda *a, **k: chat
        chat.model = "scripted"
        import os
        os.environ["AGENT_TIME_LIMIT"], os.environ["AGENT_SAFETY_S"] = "150", "30"
        solution.START = time.monotonic()   # модуль мог быть импортирован раньше другим тестом
        try:
            code = solution.main(["solution.py", str(task.task_md)])
        finally:
            agent_llm.OpenAIChat = original
            del os.environ["AGENT_TIME_LIMIT"], os.environ["AGENT_SAFETY_S"]
        self.assertEqual(code, 0)
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertGreater(sub["y"].nunique(), 1)   # не константа: сработала запасная политика


if __name__ == "__main__":
    unittest.main()
