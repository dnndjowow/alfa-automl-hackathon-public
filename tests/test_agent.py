import json
import tempfile
import time
import unittest
from pathlib import Path

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
        try:
            code = solution.main(["solution.py", str(task.task_md)])
        finally:
            agent_llm.OpenAIChat = original
        self.assertEqual(code, 0)
        sub = pd.read_csv(task.task_md.parent / "submission.csv")
        self.assertGreater(sub["y"].nunique(), 1)   # не константа: сработала запасная политика


if __name__ == "__main__":
    unittest.main()
