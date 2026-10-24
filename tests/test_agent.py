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
        self.assertEqual(agent_tools.detect_extra_metric("Метрика — **RMSLE**"), "rmsle")


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

    def test_log_rmse_phrase_is_rmsle(self):
        self.assertEqual(agent_tools.detect_extra_metric("Качество — **RMSE между логарифмами** цены"), "rmsle")
        self.assertIsNone(agent_tools.detect_extra_metric("Метрика — RMSE, меньше — лучше"))


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

    def test_extra_metric_needs_to_be_the_named_one(self):
        detect = agent_tools.detect_extra_metric
        self.assertIsNone(detect("Метрика — RMSE (log-преобразование таргета разрешено).", "rmse"))
        self.assertIsNone(detect("Evaluation metric: RMSE, log files are in logs/.", "rmse"))
        self.assertIsNone(detect("Основная метрика — RMSE. Для информации также показывается MAPE.", "rmse"))
        self.assertEqual(detect("Основная метрика — **MAPE**; RMSE показывается для справки.", "rmse"), "mape")
        self.assertEqual(detect("Качество — **RMSE между логарифмами** цены", "rmse"), "rmsle")

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

    def test_second_target_gets_constant_and_low_confidence_still_gets_safety(self):
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
        self.assertEqual(sub["y2"].nunique(), 1)
        self.assertGreater(sub["y"].nunique(), 1)

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
