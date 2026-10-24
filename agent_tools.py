"""Инструменты, которые вызывает LLM: JSON на входе и выходе, внутри — ML-ядро (ml_*).

Session хранит состояние одного запуска: данные, выбранную постановку, эксперименты,
лучший результат и записанный ответ. Методы tool_* не бросают исключений наружу:
ошибка возвращается словарём {"ok": false, "error": ...}, чтобы LLM могла её учесть.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import ml_data
import ml_io
import ml_metrics
import ml_postprocess
import ml_profile
import ml_schema
import ml_tools
import ml_validation
from ml_models import MODELS

TASK_TYPES = ("regression", "binary", "multiclass")
PROBA_METRICS = ("roc_auc", "logloss")
# параметры, которые задаёт сам код: потоки, сид, число раундов, запись файлов, GPU
_RESERVED_PARAMS = {
    "n_jobs", "thread_count", "nthread", "num_threads", "verbose", "verbosity", "silent",
    "allow_writing_files", "train_dir", "callbacks", "random_state", "random_seed", "seed",
    "n_estimators", "iterations", "num_boost_round", "num_iterations", "early_stopping_rounds",
    "eval_set", "task_type", "device", "device_type", "gpu_id", "gpu_platform_id", "objective_type",
}
_VALIDATION_KEYS = ("strategy", "n_splits", "column", "direction")
_PROBA_WORDS = re.compile(r"вероятност|probabilit|\bgini\b|джини|\bauc\b|log[\s_-]*loss", re.IGNORECASE)
_TEXT_LIMIT = 600


def _short(text, limit=_TEXT_LIMIT):
    text = str(text)
    return text if len(text) <= limit else text[: limit // 2] + " … " + text[-limit // 2:]


def _jsonable(value):
    """numpy и прочее -> обычные Python-значения; NaN/inf -> None."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _validation_key(plan: Optional[dict]) -> str:
    """Запуски сравнимы только при одинаковой схеме валидации."""
    plan = plan or {}
    return json.dumps({k: plan.get(k) for k in _VALIDATION_KEYS}, sort_keys=True)


def _label_text(value) -> str:
    """Метка класса как текст без «.0»: 1.0 из float-таргета совпадает со столбцом «1»."""
    text = str(value)
    return text[:-2] if re.fullmatch(r"-?\d+\.0", text) else text


def _clean_labels(labels):
    """Целые метки, прочитанные как float (из-за пропусков в train), вернуть к int."""
    labels = np.asarray(labels)
    if labels.dtype.kind == "f" and np.isfinite(labels).all() and (labels % 1 == 0).all():
        return labels.astype("int64")
    return labels


def match_class_columns(columns, classes) -> Optional[dict]:
    """Столбец ответа -> класс: точное имя, затем имя с префиксом/суффиксом (proba_beta, class_1)."""
    names = {c: _label_text(c) for c in classes}
    mapping = {}
    for col in columns:
        low = str(col).lower()
        exact = [c for c, name in names.items() if str(col) == name]
        loose = [c for c, name in names.items() if low.endswith(name.lower()) or low.startswith(name.lower())]
        hits = exact or sorted(loose, key=lambda c: -len(names[c]))[:1]
        if hits:
            mapping[col] = hits[0]
    if len(mapping) == len(columns) and len(set(mapping.values())) == len(columns):
        return mapping
    return None


class SubmissionError(ValueError):
    """Ответ не соответствует ожидаемому формату."""


def write_submission(frame: pd.DataFrame, task_dir: Path, n_test: int, id_cols=()) -> Path:
    """Проверить ответ и атомарно записать task_dir/submission.csv (временный файл -> os.replace)."""
    if len(frame) != n_test:
        raise SubmissionError(f"строк в ответе {len(frame)}, а в test {n_test}")
    if frame.columns.duplicated().any():
        raise SubmissionError(f"повторяющиеся столбцы: {list(frame.columns)}")
    for col in id_cols:
        if col in frame.columns and frame[col].duplicated().any():
            raise SubmissionError(f"повторяющиеся значения id в {col!r}")
    for col in frame.columns:
        values = frame[col]
        if values.isna().any():
            raise SubmissionError(f"в столбце {col!r} есть пропуски")
        if pd.api.types.is_numeric_dtype(values) and not np.isfinite(values.to_numpy(dtype=float)).all():
            raise SubmissionError(f"в столбце {col!r} есть inf")
    path = Path(task_dir) / "submission.csv"
    tmp = path.with_name(f".submission.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def copy_sample_submission(task_md) -> bool:
    """Последний рубеж, если данные даже не загрузились: шаблон ответа нужной формы вместо пустоты."""
    try:
        files = ml_io.find_task_files(task_md)
        sample, test = files.paths.get("sample_submission"), files.paths.get("test")
        if sample is None:
            return False
        if test is not None and len(ml_io.read_csv(sample)) != len(ml_io.read_csv(test)):
            return False
        target = files.task_dir / "submission.csv"
        tmp = target.with_name(f".submission.{os.getpid()}.tmp")
        shutil.copyfile(sample, tmp)
        os.replace(tmp, target)
        return True
    except Exception:
        return False


class Session:
    def __init__(self, task_md, deadline: float, log=print, n_jobs: Optional[int] = None, seed: int = 0,
                 auto_early: bool = True, final_slack: float = 300.0):
        self.task_md = Path(task_md).resolve()
        self.task_dir = self.task_md.parent
        self.deadline = deadline          # time.monotonic(), после которого новые эксперименты не начинаем
        self.final_slack = final_slack    # сколько финальное обучение может зайти за deadline (часть запаса)
        self.log = log
        self.n_jobs, self.seed, self.auto_early = n_jobs, seed, auto_early
        self.data = ml_io.load_task(self.task_md)
        self.profile = ml_profile.build_profile(self.data)
        self.setup = None                 # выбранная постановка (set_task)
        self.runs = {}                    # run_id -> запуск текущей постановки
        self.plan = None                  # текущая схема валидации
        self.plan_key = None
        self.written = None               # что сейчас лежит в submission.csv
        self.finished = None              # причина остановки от LLM
        self.history = []                 # короткий журнал вызовов
        self._counter = 0                 # номера запусков не повторяются даже после нового set_task

    # ---- время ---------------------------------------------------------------
    def time_left(self) -> float:
        return self.deadline - time.monotonic()

    def estimate_final_seconds(self, run_id) -> float:
        """Сколько займёт финальное обучение: время одной части валидации с запасом на весь train."""
        run = self.runs[run_id]
        result = run["result"]
        folds = max(1, len(result.get("fold_scores") or []))
        per_fold = result.get("seconds", 60.0) / folds / (run["config"]["subsample"] or 1.0)
        return 1.6 * per_fold + 30.0

    def can_finalize(self, run_id) -> bool:
        return self.estimate_final_seconds(run_id) <= self.time_left() + self.final_slack

    # ---- обзор задачи -------------------------------------------------------
    def overview(self, max_columns: int = 60) -> dict:
        schema = self.profile["schema"]
        return {
            "profile_text": ml_profile.format_profile(self.profile, max_columns=max_columns),
            "suggested": {
                "target_cols": schema["target_cols"], "id_cols": schema["id_cols"],
                "output_columns": schema["output_columns"], "confidence": schema["confidence"],
                "task_type": self.profile.get("task", {}).get("type"),
                "metric": self.profile["metric"].get("primary"),
                "metric_source": self.profile["metric"].get("source"),
                "answer_looks_like_probability": self.sample_suggests_proba(),
            },
            "all_train_columns": [str(c) for c in self.data.train.columns] if self.data.train is not None else [],
            "extra_csv": [ml_io.display_path(p, self.task_dir) for p in self.data.files.extra_csv],
            "available": {"models": list(MODELS), "metrics": ml_metrics.available_metrics(),
                          "task_types": list(TASK_TYPES), "validation": list(ml_validation.STRATEGIES),
                          "target_transforms": [t for t in ml_tools.TARGET_TRANSFORMS if t]},
            "time_left_s": round(self.time_left()),
        }

    def sample_suggests_proba(self) -> bool:
        """Похоже ли, что в ответе нужна вероятность: дробная заглушка в шаблоне или слова в условии."""
        sample, train = self.data.sample_submission, self.data.train
        targets = self.profile["schema"]["target_cols"]
        if sample is not None and train is not None and len(targets) == 1 and targets[0] in sample.columns:
            placeholder, truth = sample[targets[0]], train[targets[0]]
            if pd.api.types.is_float_dtype(placeholder) and truth.dropna().nunique() == 2:
                if not pd.api.types.is_float_dtype(truth) or (placeholder.dropna() % 1 != 0).any():
                    return True
        return bool(_PROBA_WORDS.search(self.data.text or ""))

    # ---- инструменты ---------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        """Единая точка вызова инструмента: ошибки -> {"ok": false}, журнал, время."""
        method = getattr(self, f"tool_{name}", None)
        started = time.monotonic()
        if method is None:
            out = {"ok": False, "error": f"неизвестный инструмент {name!r}"}
        elif not isinstance(args, dict):
            out = {"ok": False, "error": "аргументы должны быть JSON-объектом"}
        else:
            try:
                out = method(**args)
            except TypeError as error:
                out = {"ok": False, "error": f"неверные аргументы: {error}"}
            except Exception as error:  # инструмент не должен ронять агента
                out = {"ok": False, "error": f"{type(error).__name__}: {_short(error)}",
                       "trace": _short(traceback.format_exc(limit=2))}
        out["time_left_s"] = round(self.time_left())
        self.history.append({"tool": name, "args": args, "ok": out.get("ok"),
                             "seconds": round(time.monotonic() - started, 1)})
        return _jsonable(out)

    def tool_set_task(self, target_cols, task_type, metric, id_cols=None, drop_features=None,
                      prediction_format="auto", reason=""):
        """Зафиксировать постановку: таргет, id, тип задачи, метрику и формат ответа."""
        if isinstance(target_cols, str):
            target_cols = [target_cols]
        if isinstance(id_cols, str):
            id_cols = [id_cols]
        if task_type not in TASK_TYPES:
            return {"ok": False, "error": f"task_type должен быть одним из {TASK_TYPES}"}
        if not target_cols or len(target_cols) != 1:
            return {"ok": False, "error": "обучается ровно один таргет: выберите главный; остальные столбцы "
                                          "ответа код заполнит константой по train"}
        try:
            resolved = ml_metrics.get_metric(metric)
        except ml_metrics.MetricError as error:
            return {"ok": False, "error": str(error)}
        family = "regression" if task_type == "regression" else "classification"
        if resolved.task != family:
            return {"ok": False, "error": f"метрика {resolved.name} для {resolved.task}, а задача {task_type}"}
        if prediction_format not in ("auto", "label", "proba"):
            return {"ok": False, "error": "prediction_format: auto | label | proba"}

        notes = []
        suggestion = self.profile["schema"]
        ids = list(id_cols) if id_cols is not None else list(suggestion["id_cols"])
        test_cols = set(self.data.test.columns) if self.data.test is not None else set()
        for col in suggestion.get("output_columns") or []:
            # столбец шаблона ответа, который есть в test и не таргет, — идентификатор, а не признак
            if col not in ids and col not in target_cols and col in test_cols:
                ids.append(col)
                notes.append(f"{col!r} есть в шаблоне ответа и в test: считаю его id и убираю из признаков")
        schema = ml_schema.resolve_schema(suggestion, id_cols=ids, target_cols=target_cols)
        drop = [c for c in (drop_features or []) if c in schema.feature_cols]
        if drop:
            schema = ml_schema.Schema(schema.id_cols, schema.target_cols,
                                      tuple(c for c in schema.feature_cols if c not in drop), schema.output_columns)
        split = ml_data.split_task(self.data, schema)
        y = split.y_train
        if family == "regression" and not pd.api.types.is_numeric_dtype(y):
            return {"ok": False, "error": f"таргет {target_cols[0]!r} не числовой: регрессия невозможна"}
        if family == "classification" and y.nunique() < 2:
            return {"ok": False, "error": "в таргете меньше двух классов"}
        if task_type == "binary" and y.nunique() != 2:
            return {"ok": False, "error": f"в таргете {y.nunique()} классов, а не 2: выберите multiclass"}

        # профиль для ml_tools с выбранными типом задачи и метрикой (иначе там останется эвристика)
        profile = copy.deepcopy(self.profile)
        profile["task"] = {"type": task_type, "ambiguous": False, "notes": ["выбрано агентом"]}
        profile["metric"] = dict(profile["metric"], primary=resolved.name, higher_is_better=resolved.higher_is_better)
        plan = ml_validation.suggest_validation(profile, split.X_train)
        if plan.get("column") and plan["column"] not in split.X_train.columns:
            # временной столбец убран из признаков: схема по нему невозможна, берём следующую по смыслу
            profile_no_time = copy.deepcopy(profile)
            profile_no_time.get("train_test", {})["out_of_train_range_share"] = {}
            notes.append(f"столбца {plan['column']!r} нет в признаках: временная схема заменена")
            plan = ml_validation.suggest_validation(profile_no_time, split.X_train)

        if self.written and self.written.get("run_id"):
            self.written["stale"] = True  # ответ от прошлой постановки остаётся на диске до лучшего
        self.setup = {"schema": schema, "split": split, "profile": profile, "task_type": task_type,
                      "metric": resolved.name, "higher_is_better": resolved.higher_is_better,
                      "prediction_format": prediction_format, "reason": reason}
        self.plan, self.plan_key = plan, _validation_key(plan)
        self.runs = {}
        self.log(f"[set_task] target={list(schema.target_cols)} id={list(schema.id_cols)} {task_type} "
                 f"metric={resolved.name} features={len(schema.feature_cols)} validation={plan['strategy']}")
        return {
            "ok": True, "n_train": len(split.X_train), "n_test": len(split.X_test),
            "n_features": len(schema.feature_cols), "id_cols": list(schema.id_cols), "dropped_features": drop,
            "target_summary": ml_profile.profile_target(y, task_type),
            "metric": resolved.name, "higher_is_better": resolved.higher_is_better,
            "validation": plan, "output_columns": self._output_columns(),
            "answer_as_probability": self._wants_proba() if task_type == "binary" else False,
            "notes": notes, "warnings": split.warnings,
        }

    def tool_evaluate(self, model="lightgbm", params=None, learning_rate=0.05, target_transform=None,
                      validation=None, subsample=None, time_budget_s=None, max_rounds=3000, hypothesis=""):
        """Оценить конфигурацию на валидации. Сравнение — только с запусками на той же схеме."""
        if self.setup is None:
            return {"ok": False, "error": "сначала вызовите set_task"}
        if model not in MODELS:
            return {"ok": False, "error": f"model должна быть одной из {MODELS}"}
        params, dropped = self._clean_params(params)
        if target_transform in ("none", "", "null"):
            target_transform = None
        if target_transform and self.setup["task_type"] != "regression":
            target_transform = None
        plan, error = self._plan_from(validation)
        if error:
            return {"ok": False, "error": error}
        try:
            subsample = float(subsample) if subsample is not None else None
        except (TypeError, ValueError):
            subsample = None
        if subsample is not None and not (0.05 <= subsample < 1.0):
            subsample = None
        budget = self._eval_budget(time_budget_s)
        if budget is None:
            return {"ok": False, "error": "времени на новый эксперимент не осталось: вызовите finalize или finish"}

        self._counter += 1
        run_id = f"r{self._counter}"
        self.log(f"[evaluate {run_id}] {model} lr={learning_rate} transform={target_transform} params={params} "
                 f"validation={plan['strategy']} subsample={subsample} budget={budget:.0f}s "
                 f"hypothesis={_short(hypothesis, 200)}")
        result = ml_tools.train_eval(
            self.setup["split"], self.setup["profile"], model=model, metric=self.setup["metric"], validation=plan,
            params=params, task_type=self.setup["task_type"], target_transform=target_transform, n_jobs=self.n_jobs,
            seed=self.seed, time_budget=budget, max_rounds=int(max_rounds), learning_rate=float(learning_rate),
            subsample=subsample,
        )
        summary = result.to_dict()
        summary["notes"] = [_short(n, 300) for n in summary.get("notes", [])][-5:]
        summary["validation"] = {k: summary.get("validation", {}).get(k) for k in _VALIDATION_KEYS + ("effective",)}
        key = _validation_key(plan)
        changed = key != self.plan_key
        if changed and result.status == "ok" and subsample is None:
            # новая схема становится текущей только после успешного полного запуска
            self.plan, self.plan_key = plan, key
        config = {"model": model, "params": params, "learning_rate": float(learning_rate),
                  "target_transform": target_transform, "max_rounds": int(max_rounds), "subsample": subsample}
        self.runs[run_id] = {"config": config, "result": summary, "eval": result, "key": key,
                             "hypothesis": hypothesis}
        best = self.best_run()
        self.log(f"[evaluate {run_id}] status={result.status} score={summary.get('score')} "
                 f"({result.seconds:.0f}s) best={best}")
        out = {"ok": result.status == "ok", "run_id": run_id, **summary, "best_run_id": best,
               "best_score": self.runs[best]["result"]["score"] if best else None, "dropped_params": dropped,
               "comparable_runs": [r for r, v in self.runs.items() if v["key"] == self.plan_key
                                   and v["config"]["subsample"] is None]}
        if subsample is not None:
            out["note"] = "запуск на подвыборке: только для прикидки, с полными запусками не сравнивается"
        elif changed:
            out["note"] = ("схема валидации изменилась: прежние запуски с этим несравнимы" if result.status == "ok"
                           else "новая схема не сработала: текущей осталась прежняя")
        if self.auto_early and out["ok"] and subsample is None and not (self.written or {}).get("run_id"):
            # ранний настоящий ответ: страхует от сбоев дальнейшего диалога
            try:
                early = self._finalize(run_id)
            except Exception as error:
                early = {"ok": False, "error": f"{type(error).__name__}: {_short(error, 200)}"}
            out["early_submission"] = {k: early.get(k) for k in ("ok", "run_id", "error", "rows")}
        return out

    def tool_finalize(self, run_id=None, class_weights="auto", reason=""):
        """Обучить выбранный запуск на всём train и записать submission.csv (если он лучше записанного)."""
        if self.setup is None:
            return {"ok": False, "error": "сначала вызовите set_task"}
        run_id = run_id or self.best_run()
        if run_id not in self.runs or self.runs[run_id]["result"]["status"] != "ok":
            return {"ok": False, "error": f"нет успешного запуска {run_id!r}; успешные: {self._ok_runs()}"}
        if (self.written or {}).get("run_id") == run_id:
            return {"ok": True, "note": "этот запуск уже записан", "written": self.written}
        if not self._better_than_written(run_id, automatic=False):
            return {"ok": False, "error": f"{run_id} не лучше уже записанного {self.written['run_id']} "
                                          "на той же валидации (или это подвыборка); ответ не перезаписан"}
        if not self.can_finalize(run_id):
            return {"ok": False, "error": f"не хватит времени: нужно ~{self.estimate_final_seconds(run_id):.0f}s"}
        return self._finalize(run_id, class_weights)

    def tool_finish(self, reason=""):
        """Закончить исследование. Если ответ не записан, код сам обучит лучший запуск."""
        self.finished = reason or "без причины"
        self.log(f"[finish] {self.finished}")
        return {"ok": True, "best_run_id": self.best_run(), "written": self.written}

    # ---- выбор запусков -------------------------------------------------------
    def _ok_runs(self, full_only=False):
        return [r for r, v in self.runs.items() if v["result"]["status"] == "ok"
                and not (full_only and v["config"]["subsample"] is not None)]

    def best_run(self) -> Optional[str]:
        """Лучший полный запуск на текущей схеме; если таких нет — на последней схеме с успешными."""
        full = self._ok_runs(full_only=True)
        pool = [r for r in full if self.runs[r]["key"] == self.plan_key]
        if not pool and full:
            last_key = self.runs[full[-1]]["key"]
            pool = [r for r in full if self.runs[r]["key"] == last_key]
        if not pool:
            pool = self._ok_runs()  # одни подвыборки — лучше, чем ничего
        best = None
        for run_id in pool:
            if best is None or ml_metrics.is_better(self.setup["metric"], self.runs[run_id]["result"]["score"],
                                                    self.runs[best]["result"]["score"]):
                best = run_id
        return best

    def _better_than_written(self, run_id, automatic: bool) -> bool:
        """Можно ли заменить записанный ответ этим запуском.

        automatic=True — решение кода (ensure_final): только при честном сравнении.
        automatic=False — явная просьба LLM: несравнимые схемы допускаются, подвыборка — нет."""
        written = self.written or {}
        if not written.get("run_id"):
            return True                      # на диске константа или ничего
        new = self.runs[run_id]
        old = self.runs.get(written["run_id"])
        if old is None:
            return True                      # ответ от прошлой постановки: LLM сама её сменила
        if new["config"]["subsample"] is not None and old["config"]["subsample"] is None:
            return False
        if old["key"] != new["key"]:
            return not automatic
        return ml_metrics.is_better(self.setup["metric"], new["result"]["score"], old["result"]["score"])

    # ---- внутреннее ----------------------------------------------------------
    def _plan_from(self, validation):
        """Схема из запроса LLM поверх текущей; (plan, error)."""
        if not validation:
            return dict(self.plan), None
        if not isinstance(validation, dict):
            return None, "validation должна быть объектом {strategy, n_splits, column, direction}"
        strategy = validation.get("strategy", self.plan["strategy"])
        if strategy not in ml_validation.STRATEGIES:
            return None, f"validation.strategy: {ml_validation.STRATEGIES}"
        plan = {"strategy": strategy, "n_splits": int(validation.get("n_splits") or self.plan.get("n_splits") or 5),
                "column": None, "direction": None,
                "classification": self.setup["task_type"] != "regression", "reason": "выбрано агентом"}
        if strategy == "temporal":
            column = validation.get("column") or (self.plan.get("column") if self.plan["strategy"] == "temporal" else None)
            X = self.setup["split"].X_train
            if column not in X.columns or not pd.api.types.is_numeric_dtype(X[column]):
                return None, f"для temporal нужен числовой признак column; {column!r} не подходит"
            direction = validation.get("direction") or "forward"
            if direction not in ("forward", "backward"):
                return None, "direction: forward | backward"
            plan.update(column=column, direction=direction, n_splits=min(plan["n_splits"], 5))
        elif strategy == "group" and validation.get("column"):
            return None, ("группировка по столбцу не поддержана: group объединяет одинаковые строки train; "
                          "уберите column")
        return plan, None

    def _eval_budget(self, requested) -> Optional[float]:
        """Бюджет эксперимента: не больше остатка за вычетом финального обучения; None — времени нет."""
        reserve = 120.0
        best = self.best_run()
        if best:
            reserve = max(reserve, self.estimate_final_seconds(best))
        left = self.time_left() - reserve
        if left < 20:
            return None
        default = max(60.0, min(1800.0, left / 3))
        try:
            requested = float(requested) if requested is not None else default
        except (TypeError, ValueError):
            requested = default
        return max(min(30.0, left), min(requested, left))

    def _clean_params(self, params):
        if not isinstance(params, dict):
            return {}, []
        clean, dropped = {}, []
        for key, value in params.items():
            if key in _RESERVED_PARAMS or not isinstance(value, (int, float, str, bool)):
                dropped.append(key)
            else:
                clean[key] = value
        return clean, dropped

    def _output_columns(self, schema=None) -> list:
        schema = schema or self.setup["schema"]
        if schema.output_columns:
            return list(schema.output_columns)
        return list(schema.id_cols) + list(schema.target_cols)

    def _wants_proba(self, setup=None) -> bool:
        """Для бинарной задачи: писать вероятность, а не метку?"""
        setup = setup or self.setup
        fmt = setup["prediction_format"]
        if fmt != "auto":
            return fmt == "proba"
        return setup["metric"] in PROBA_METRICS or self.sample_suggests_proba()

    def _finalize(self, run_id, class_weights="auto") -> dict:
        run, setup = self.runs[run_id], self.setup
        config, ev = run["config"], run["eval"]
        n_rounds = ml_tools.rounds_from(ev)
        budget = max(10.0, self.time_left() + self.final_slack)
        self.log(f"[finalize {run_id}] {config['model']} n_rounds={n_rounds} budget={budget:.0f}s")
        out = ml_tools.fit_predict(
            setup["split"], setup["profile"], model=config["model"], n_rounds=n_rounds, params=config["params"],
            task_type=setup["task_type"], target_transform=config["target_transform"], n_jobs=self.n_jobs,
            seed=self.seed, time_budget=budget, max_rounds=config["max_rounds"], learning_rate=config["learning_rate"],
        )
        if out.get("status") != "ok":
            return {"ok": False, "error": f"финальное обучение не удалось: {_short(out.get('error'))}"}
        notes = list(out.get("notes", []))
        if n_rounds and out.get("n_rounds") and out["n_rounds"] < 0.8 * n_rounds:
            notes.append(f"обучение урезано по времени: {out['n_rounds']} из {n_rounds} раундов")
            if (self.written or {}).get("run_id"):
                return {"ok": False, "error": "финальное обучение урезано по времени; прежний ответ сохранён",
                        "notes": notes}
        pred, proba, classes, weights_info = out["pred"], out.get("proba"), out.get("classes"), None
        if setup["task_type"] != "regression":
            pred, weights_info = self._class_weights(run, out, class_weights)
        frame, frame_notes = self._assemble(setup, pred, proba, classes)
        path = write_submission(frame, self.task_dir, len(setup["split"].X_test), setup["schema"].id_cols)
        self.written = {"run_id": run_id, "score": run["result"]["score"], "model": config["model"]}
        self.log(f"[finalize {run_id}] записан {path} ({len(frame)} строк), cv={run['result']['score']}")
        return {"ok": True, "path": str(path), "rows": len(frame), "columns": list(frame.columns),
                "run_id": run_id, "cv_score": run["result"]["score"], "n_rounds": out.get("n_rounds"),
                "class_weights": weights_info, "notes": notes + frame_notes, "warnings": out.get("warnings", [])[:5]}

    def _class_weights(self, run, out, mode):
        """Веса классов под Macro F1 по OOF-вероятностям; применяются, только если честно помогают."""
        setup, ev = self.setup, run["eval"]
        labels = out["pred"]
        if mode == "off" or setup["metric"] != "f1_macro" or run["config"]["subsample"]:
            return labels, None
        if ev.oof_pred is None or ev.oof_index is None or len(ev.oof_index) < 200:
            return labels, None
        try:
            codec = ml_data.LabelCodec().fit(setup["split"].y_train)
            truth = codec.encode(setup["split"].y_train.iloc[ev.oof_index])
            groups = ml_validation.row_groups(setup["split"].X_train.iloc[ev.oof_index])
            choice = ml_postprocess.choose_class_weights(ev.oof_pred, truth, groups=groups)
            info = {"applied": bool(choice["applied"]), "holdout_gain": choice["holdout_gain"]}
            if choice["applied"]:
                labels = codec.decode(ml_postprocess.apply_class_weights(out["proba"], choice["weights"]))
                info["weights"] = dict(zip([str(c) for c in codec.classes_], choice["weights"]))
            self.log(f"[class_weights] {info}")
            return labels, info
        except Exception as error:
            return labels, {"applied": False, "error": _short(error, 200)}

    def _assemble(self, setup, pred, proba=None, classes=None):
        """Собрать ответ в порядке test по столбцам шаблона. Каждый столбец заполняется по своему правилу;
        предсказание пишется только в столбец таргета или в столбцы вероятностей классов."""
        schema, split = setup["schema"], setup["split"]
        n = len(split.X_test)
        ids = split.test_ids.reset_index(drop=True)
        test = self.data.test.reset_index(drop=True)
        train, sample = self.data.train, self.data.sample_submission
        columns = self._output_columns(schema)
        target = schema.target_cols[0]
        classification = setup["task_type"] != "regression"
        notes, frame = [], pd.DataFrame(index=range(n))

        class_map = None
        if classification and proba is not None and setup["prediction_format"] != "label":
            candidates = [c for c in columns if c not in ids.columns and c != target]
            if len(candidates) >= 2:
                class_map = match_class_columns(candidates, classes)
                if class_map is None and len(candidates) == len(classes):
                    class_map = dict(zip(candidates, classes))
                    notes.append(f"столбцы {candidates} сопоставлены классам {list(map(str, classes))} по порядку")
        proba_target = (classification and proba is not None and len(classes) == 2
                        and target in columns and self._wants_proba(setup))
        labels = _clean_labels(pred) if classification else np.asarray(pred, dtype=float)
        if classification and not proba_target and sample is not None and target in sample.columns \
                and pd.api.types.is_integer_dtype(sample[target]) and labels.dtype.kind == "f":
            labels = labels.astype("int64")

        for col in columns:
            if col in ids.columns:
                frame[col] = ids[col].to_numpy()
            elif col == target:
                if proba_target:
                    frame[col] = np.asarray(proba)[:, 1]
                    notes.append(f"{col}: вероятность класса {_label_text(classes[1])!r}")
                else:
                    frame[col] = labels
            elif class_map and col in class_map:
                frame[col] = np.asarray(proba)[:, list(classes).index(class_map[col])]
            elif col in test.columns:
                frame[col] = test[col].to_numpy()
                notes.append(f"{col}: скопирован из test")
            elif train is not None and col in train.columns:
                values = train[col].dropna()
                fill = float(values.median()) if pd.api.types.is_numeric_dtype(values) and values.nunique() > 20 \
                    else values.mode().iloc[0]
                frame[col] = fill
                notes.append(f"{col}: не моделируется, константа {fill!r} по train")
            elif sample is not None and len(sample) == n and col in sample.columns:
                frame[col] = sample[col].to_numpy()
                notes.append(f"{col}: скопирован из sample_submission")
            else:
                raise SubmissionError(f"не знаю, чем заполнить столбец ответа {col!r}")
        return frame[columns], notes

    def write_constant(self) -> Optional[dict]:
        """Страховочный ответ до любых экспериментов: медиана / частый класс / доли классов."""
        try:
            suggestion = self.profile["schema"]
            if not suggestion["target_cols"]:
                return None
            target = suggestion["target_cols"][0]
            schema = ml_schema.resolve_schema(suggestion, id_cols=suggestion["id_cols"], target_cols=[target])
            split = ml_data.split_task(self.data, schema)
            y = split.y_train
            kind = self.profile.get("task", {}).get("type")
            regression = kind == "regression" or (kind not in ("binary", "multiclass")
                                                  and pd.api.types.is_numeric_dtype(y) and y.nunique() > 20)
            n = len(split.X_test)
            setup = {"schema": schema, "split": split, "metric": self.profile["metric"].get("primary") or "",
                     "task_type": "regression" if regression else ("binary" if y.nunique() == 2 else "multiclass"),
                     "prediction_format": "auto"}
            if regression:
                frame, notes = self._assemble(setup, np.full(n, float(y.median())))
            else:
                counts = y.value_counts(normalize=True)
                classes = np.array(sorted(counts.index, key=str))
                priors = np.tile([counts[c] for c in classes], (n, 1))
                frame, notes = self._assemble(setup, np.repeat(counts.index[0], n), priors, classes)
            write_submission(frame, self.task_dir, n, schema.id_cols)
            self.written = {"run_id": None, "score": None, "model": "constant"}
            self.log(f"[safety] записан страховочный ответ {list(frame.columns)}; {notes[:3]}")
            return self.written
        except Exception as error:
            self.log(f"[safety] страховочный ответ не записан: {type(error).__name__}: {_short(error, 200)}")
            return None


TOOL_SPECS = [
    {
        "name": "set_task",
        "description": "Зафиксировать постановку задачи по task.md и профилю данных: таргет, id, тип задачи, "
                       "метрику качества и формат ответа. Вызвать первым; повторный вызов начинает эксперименты "
                       "заново (записанный ответ остаётся на диске, пока его не заменит лучший).",
        "parameters": {
            "type": "object",
            "properties": {
                "target_cols": {"type": "array", "items": {"type": "string"},
                                "description": "Столбец-таргет, ровно один. Если таргетов несколько, выберите главный: "
                                               "остальные столбцы ответа заполнятся константой"},
                "id_cols": {"type": "array", "items": {"type": "string"},
                            "description": "Столбцы-идентификаторы строк для ответа; [] если их нет"},
                "task_type": {"type": "string", "enum": list(TASK_TYPES)},
                "metric": {"type": "string", "description": "Метрика из task.md, одно из available.metrics"},
                "drop_features": {"type": "array", "items": {"type": "string"},
                                  "description": "Признаки, которые нельзя использовать (утечка, служебные)"},
                "prediction_format": {"type": "string", "enum": ["auto", "label", "proba"],
                                      "description": "Классификация: метки или вероятности; auto — по метрике и шаблону ответа"},
                "reason": {"type": "string", "description": "Коротко: почему так"},
            },
            "required": ["target_cols", "task_type", "metric"],
        },
    },
    {
        "name": "evaluate",
        "description": "Обучить и оценить конфигурацию на валидации (ответ не записывает). Возвращает метрику по частям, "
                       "лучшие итерации, время и лучший запуск среди сравнимых.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "enum": list(MODELS)},
                "params": {"type": "object",
                           "description": "Гиперпараметры библиотеки: lightgbm num_leaves, min_child_samples, "
                                          "colsample_bytree, reg_lambda; xgboost max_depth, min_child_weight; "
                                          "catboost depth, l2_leaf_reg"},
                "learning_rate": {"type": "number"},
                "target_transform": {"type": "string", "enum": ["none", "log1p"],
                                     "description": "Только регрессия: log1p для неотрицательного скошенного таргета"},
                "validation": {"type": "object",
                               "description": "Сменить схему: {strategy: temporal|group|stratified|kfold, n_splits, "
                                              "column (только temporal: числовой признак времени), "
                                              "direction: forward|backward}. group = группы одинаковых строк"},
                "subsample": {"type": "number",
                              "description": "Доля train 0.05..1 для быстрой прикидки; с полными запусками не сравнивается "
                                             "и в финал не идёт"},
                "time_budget_s": {"type": "number", "description": "Лимит времени на этот эксперимент"},
                "hypothesis": {"type": "string", "description": "Что проверяем и почему, по результатам прошлых запусков"},
            },
            "required": ["model", "hypothesis"],
        },
    },
    {
        "name": "finalize",
        "description": "Обучить запуск на всём train и записать submission.csv. Без run_id берётся лучший. "
                       "Ответ заменяется, только если запуск лучше уже записанного.",
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "class_weights": {"type": "string", "enum": ["auto", "off"],
                                  "description": "Подбор весов классов под Macro F1 по OOF"},
                "reason": {"type": "string"},
            },
        },
    },
    {
        "name": "finish",
        "description": "Остановить исследование, когда дальнейшие эксперименты не окупаются по времени или качеству.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
    },
]


def openai_tools() -> list:
    return [{"type": "function", "function": spec} for spec in TOOL_SPECS]
