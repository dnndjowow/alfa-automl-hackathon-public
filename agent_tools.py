"""Инструменты, которые вызывает LLM: JSON на входе и выходе, внутри — ML-ядро (ml_*).

Session хранит состояние одного запуска: данные, выбранную постановку, эксперименты,
лучший результат и записанный ответ. Методы tool_* не бросают исключений наружу:
ошибка возвращается словарём {"ok": false, "error": ...}, чтобы LLM могла её учесть.
Несколько таргетов обучаются по очереди одной конфигурацией; score — среднее по таргетам.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
import re
import shutil
import time
import traceback
from dataclasses import replace
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
# параметры, которые задаёт сам код: потоки, сид, число раундов, запись файлов, GPU
_RESERVED_PARAMS = {
    "n_jobs", "thread_count", "nthread", "num_threads", "verbose", "verbosity", "silent",
    "allow_writing_files", "train_dir", "callbacks", "random_state", "random_seed", "seed",
    "n_estimators", "iterations", "num_boost_round", "num_iterations", "early_stopping_rounds",
    "eval_set", "task_type", "device", "device_type", "gpu_id", "gpu_platform_id", "objective_type",
    # синонимы числа раундов и ранней остановки: роняют fit_predict или молча урезают модель (ml_models)
    "early_stopping", "early_stopping_round", "n_iter_no_change", "num_trees", "num_round", "num_rounds",
    "n_iter", "max_iter", "num_boost_round", "eta", "learning_rate", "shrinkage_rate",
}
# CatBoost не понимает типичные параметры LightGBM/XGBoost: переводим или отбрасываем
_CATBOOST_RENAME = {"max_depth": "depth", "colsample_bytree": "rsm", "feature_fraction": "rsm",
                    "colsample_bylevel": "rsm", "reg_lambda": "l2_leaf_reg", "lambda_l2": "l2_leaf_reg"}
# XGBoost молча игнорирует эти параметры («Parameters: { class_weight } are not used»): LLM должна это видеть
_XGBOOST_DROP = {"class_weight", "is_unbalance", "auto_class_weights", "num_leaves", "min_child_samples"}
# class_weight не выбрасываем: ядро переводит его в auto_class_weights='Balanced'
_CATBOOST_DROP = {"num_leaves", "min_child_samples", "min_data_in_leaf", "min_child_weight", "gamma",
                  "reg_alpha", "lambda_l1", "subsample_freq", "bagging_freq", "max_bin_by_feature"}
_VALIDATION_KEYS = ("strategy", "n_splits", "column", "direction")
_PROBA_WORDS = re.compile(r"вероятност|probabilit|\bgini\b|джини|\bauc\b|log[\s_-]*loss", re.IGNORECASE)
_POSITIVE_WORDS = {"1", "true", "yes", "да", "positive", "pos"}
_TEXT_LIMIT = 600


def _rmsle(yt, yp, labels):
    return float(np.sqrt(np.mean((np.log1p(np.clip(yp, 0, None)) - np.log1p(np.clip(yt, 0, None))) ** 2)))


def _msle(yt, yp, labels):
    return float(np.mean((np.log1p(np.clip(yp, 0, None)) - np.log1p(np.clip(yt, 0, None))) ** 2))


def _mape(yt, yp, labels):
    return float(np.mean(np.abs(yt - yp) / np.maximum(np.abs(yt), 1e-9)))


def _smape(yt, yp, labels):
    return float(np.mean(2 * np.abs(yt - yp) / np.maximum(np.abs(yt) + np.abs(yp), 1e-9)))


# метрики, которых нет в ml_metrics: тот же класс Metric, ml_tools принимает его вместо имени
EXTRA_METRICS = {m.name: m for m in (
    ml_metrics.Metric("rmsle", _rmsle, False, "regression", "values"),
    ml_metrics.Metric("msle", _msle, False, "regression", "values"),
    ml_metrics.Metric("mape", _mape, False, "regression", "values"),
    ml_metrics.Metric("smape", _smape, False, "regression", "values"),
)}
_METRIC_ALIASES = {"gini": "roc_auc", "giniindex": "roc_auc", "rocaucscore": "roc_auc",
                   "multiclasslogloss": "logloss", "multilogloss": "logloss",
                   "rootmeansquaredlogarithmicerror": "rmsle", "meanabsolutepercentageerror": "mape"}


def resolve_metric(name) -> ml_metrics.Metric:
    """Метрика по имени: ml_metrics, затем свои (RMSLE, MAPE, ...) и синонимы (Gini ~ ROC AUC)."""
    try:
        return ml_metrics.get_metric(name)
    except ml_metrics.MetricError:
        key = re.sub(r"[^a-z0-9]", "", str(name).lower())
        key = _METRIC_ALIASES.get(key, key)
        if key in EXTRA_METRICS:
            return EXTRA_METRICS[key]
        try:
            return ml_metrics.get_metric(key)
        except ml_metrics.MetricError:
            known = ml_metrics.available_metrics() + list(EXTRA_METRICS)
            raise ml_metrics.MetricError(f"неизвестная метрика {name!r}; доступны: {', '.join(known)}")


_BINARY_METRICS = {"f1": "f1", "f1score": "f1", "f1binary": "f1", "precision": "precision", "recall": "recall",
                   "sensitivity": "recall"}


def _binary_metric(kind: str, positive_code: int) -> ml_metrics.Metric:
    """F1 / precision / recall положительного класса; метки приходят кодами 0..K-1."""
    from sklearn import metrics as skm
    scorer = {"f1": skm.f1_score, "precision": skm.precision_score, "recall": skm.recall_score}[kind]

    def fn(yt, yp, labels):
        return float(scorer(np.asarray(yt) == positive_code, np.asarray(yp) == positive_code, zero_division=0))
    return ml_metrics.Metric(kind, fn, True, "classification", "labels")


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


def guess_positive_class(classes, y: pd.Series, text: str):
    """Класс, вероятность которого пишем в бинарной задаче.

    Числовые метки: 1 (или большая) — так принято, а «вероятность от 0 до 1» в тексте не должна сбивать.
    Нечисловые: 1) класс, названный в условии рядом со словом «вероятность»; 2) yes / true / да;
    3) более редкий класс (обычно это событие: дефолт, отток, мошенничество)."""
    names = {c: _label_text(c) for c in classes}
    numeric = all(re.fullmatch(r"-?\d+(?:\.\d+)?", n) for n in names.values())
    if numeric:
        return max(classes, key=lambda c: float(names[c]))
    clean = (text or "").replace("*", "")
    for phrase in re.finditer(r"вероятност|probabilit", clean, flags=re.IGNORECASE):
        window = clean[phrase.end(): phrase.end() + 160]
        found = []
        for cls, name in names.items():
            match = re.search(r"(?<![\w<>=.-])" + re.escape(name) + r"(?![\w<>=.-])", window)
            if match:
                found.append((match.start(), cls))
        if found:
            return min(found, key=lambda item: item[0])[1]
    for cls, name in names.items():
        if name.lower() in _POSITIVE_WORDS:
            return cls
    counts = y.value_counts()
    return min(classes, key=lambda c: counts.get(c, 0))


def _suggest_params(trial, model: str):
    """Пространство поиска гиперпараметров для Optuna: (params, learning_rate)."""
    lr = trial.suggest_float("learning_rate", 0.015, 0.15, log=True)
    if model == "lightgbm":
        params = {"num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                  "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                  "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                  "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
                  "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, 1.0, log=True)}
    elif model == "xgboost":
        params = {"max_depth": trial.suggest_int("max_depth", 3, 12),
                  "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 30.0, log=True),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                  "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                  "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True)}
    else:
        params = {"depth": trial.suggest_int("depth", 4, 10),
                  "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
                  "rsm": trial.suggest_float("rsm", 0.4, 1.0),
                  "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True)}
    if trial.number == 0:
        params = {}   # первая попытка — параметры по умолчанию
    return params, lr


def choose_weights_for_metric(proba, codes, metric, groups=None, repeats=3, min_gain=0.002) -> dict:
    """Веса классов (для двух классов — это порог) под произвольную метрику по меткам: подбор на одной части
    OOF и проверка на другой; применяются, только если средний прирост на проверке не меньше min_gain."""
    n_classes = proba.shape[1]
    grid = np.geomspace(0.2, 5.0, 25)

    def score(rows, weights):
        return metric(codes[rows], (proba[rows] * weights).argmax(axis=1), labels=list(range(n_classes)))

    def fit(rows):
        weights = np.ones(n_classes)
        best = score(rows, weights)
        for _ in range(2):
            for k in range(n_classes):
                for value in grid:
                    trial = weights.copy()
                    trial[k] = value
                    current = score(rows, trial)
                    if current > best + 1e-6:
                        best, weights = current, trial
        return weights

    gains = []
    for seed in range(repeats):
        fit_rows, check_rows = ml_postprocess._holdout_split(len(codes), 0.6, seed, groups)
        if len(fit_rows) < 50 or len(check_rows) < 50:
            return {"applied": False, "weights": np.ones(n_classes), "holdout_gain": 0.0}
        weights = fit(fit_rows)
        gains.append(score(check_rows, weights) - score(check_rows, np.ones(n_classes)))
    gain = float(np.mean(gains))
    if gain < min_gain:
        return {"applied": False, "weights": np.ones(n_classes), "holdout_gain": gain}
    return {"applied": True, "weights": fit(np.arange(len(codes))), "holdout_gain": gain}


FEATURE_GRAMMAR = {
    "ratio/diff/prod/sum": {"op": "ratio", "a": "числовой столбец", "b": "числовой столбец"},
    "group_mean/group_std/group_dev/group_count": {"op": "group_dev", "col": "числовой столбец",
                                                    "by": "категориальный столбец"},
    "count": {"op": "count", "col": "любой столбец (частота значения)"},
    "cross": {"op": "cross", "a": "категория", "b": "категория"},
    "nan_count": {"op": "nan_count"},
    "as_category": {"op": "as_category", "col": "числовой столбец-код (id менеджера, MCC, регион)"},
}
_FORMULA = re.compile(r"^\s*([\w.]+)\s*([-+*/])\s*([\w.]+)\s*$")


def _parse_feature(spec):
    """Спецификация признака: словарь грамматики или формула вида «a / b»."""
    if isinstance(spec, str):
        match = _FORMULA.match(spec)
        if not match:
            raise ValueError(f"не понимаю {spec!r}; формулы вида a/b, a-b, a*b, a+b или словарь {FEATURE_GRAMMAR}")
        a, sign, b = match.groups()
        return {"op": {"/": "ratio", "-": "diff", "*": "prod", "+": "sum"}[sign], "a": a, "b": b}
    if not isinstance(spec, dict) or "op" not in spec:
        raise ValueError(f"нужен словарь с op: {list(FEATURE_GRAMMAR)}")
    return spec


def build_features(X_train: pd.DataFrame, X_test: pd.DataFrame, specs: list):
    """Посчитать признаки на train и test вместе (только по признакам, без таргета).
    Возвращает (X_train, X_test, применённые [{name, spec}], ошибки)."""
    both = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    n_train = len(X_train)
    applied, errors, new = [], [], {}

    def numeric(col):
        if col not in both.columns:
            raise ValueError(f"нет столбца {col!r}")
        values = pd.to_numeric(both[col], errors="coerce")
        if values.notna().mean() < 0.5:
            raise ValueError(f"{col!r} не числовой")
        return values.astype(float)

    for raw in specs:
        try:
            spec = _parse_feature(raw)
            op = str(spec["op"]).lower()
            if op in ("ratio", "diff", "prod", "sum"):
                a, b = numeric(spec["a"]), numeric(spec["b"])
                name = f"{spec['a']}__{op}__{spec['b']}"
                value = {"ratio": a / b.where(b != 0), "diff": a - b, "prod": a * b, "sum": a + b}[op]
            elif op in ("group_mean", "group_std", "group_dev", "group_count"):
                by = spec["by"]
                if by not in both.columns:
                    raise ValueError(f"нет столбца {by!r}")
                keys = both[by].astype(str)
                if op == "group_count":
                    name, value = f"count_by__{by}", keys.map(keys.value_counts()).astype(float)
                else:
                    col = numeric(spec["col"])
                    stat = col.groupby(keys).transform("mean" if op != "group_std" else "std")
                    name = f"{spec['col']}__{op}__{by}"
                    value = col - stat if op == "group_dev" else stat
            elif op == "count":
                col = spec["col"]
                if col not in both.columns:
                    raise ValueError(f"нет столбца {col!r}")
                keys = both[col].astype(str)
                name, value = f"count__{col}", keys.map(keys.value_counts()).astype(float)
            elif op == "cross":
                a, b = spec["a"], spec["b"]
                if a not in both.columns or b not in both.columns:
                    raise ValueError(f"нет столбца {a!r} или {b!r}")
                value = both[a].astype(str) + "|" + both[b].astype(str)
                if value.nunique() > 0.5 * len(value):
                    raise ValueError(f"cross {a}×{b}: почти все значения уникальны")
                name = f"{a}__x__{b}"
            elif op == "as_category":
                # числовой код (менеджер, MCC, регион) как категория: CatBoost считает по нему статистики таргета
                col = spec["col"]
                if col not in both.columns:
                    raise ValueError(f"нет столбца {col!r}")
                name = f"{col}__cat"
                value = both[col].astype("string").where(both[col].notna()).astype(object)
            elif op == "nan_count":
                name, value = "nan_count", both.isna().sum(axis=1).astype(float)
            else:
                raise ValueError(f"неизвестная операция {op!r}; доступны {list(FEATURE_GRAMMAR)}")
            if name in both.columns or name in new:
                continue
            if isinstance(value, pd.Series) and value.dtype.kind == "f":
                value = value.replace([np.inf, -np.inf], np.nan)
            new[name] = value
            applied.append({"name": name, "spec": spec})
        except Exception as error:
            errors.append(f"{raw}: {_short(error, 150)}")
    if not new:
        return X_train, X_test, [], errors
    extra = pd.DataFrame(new)
    train_out = pd.concat([X_train.reset_index(drop=True), extra.iloc[:n_train].reset_index(drop=True)], axis=1)
    test_out = pd.concat([X_test.reset_index(drop=True), extra.iloc[n_train:].reset_index(drop=True)], axis=1)
    train_out.index, test_out.index = X_train.index, X_test.index
    return train_out, test_out, applied, errors


ROW_ORDER = "__row_order__"
ROW_ORDER_ALIASES = {ROW_ORDER, "row_order", "row_number"}
_TIME_ORDER_WORDS = re.compile(
    r"упорядочен\w*\s+по\s+(времени|дате)|в\s+порядке\s+времени|хронологич|ordered\s+by\s+(time|date)|chronolog"
    r"|test\w*[^.\n]{0,60}(более\s+поздн|позже|позднее|later|future|будущ)"
    r"|(более\s+поздн|позже|later)[^.\n]{0,40}(test|тест)"
    r"|(test|тестов)\w*[^.\n]{0,40}(последн|latest|most\s+recent)", re.IGNORECASE)


def _fit_args(method, args: dict):
    """Аргументы от LLM к сигнатуре инструмента: «n_trials-» -> n_trials, «hypothesisary» -> hypothesis
    (по префиксу), незнакомые ключи отбрасываются и возвращаются списком, а не роняют вызов."""
    names = list(inspect.signature(method).parameters)
    fitted, ignored = {}, []
    for key, value in args.items():
        norm = re.sub(r"[^a-z0-9_]", "", str(key).lower()).strip("_")
        if norm in names:
            target = norm
        else:
            # «hypothesisary» -> hypothesis, «hypoth» -> hypothesis; но «learning_rate_decay» — другой параметр
            close = [n for n in names if len(norm) >= 4 and (
                (norm.startswith(n) and "_" not in norm[len(n):] and len(norm) - len(n) <= 4) or n.startswith(norm))]
            target = max(close, key=len) if close else None
        if target is None or target in fitted:
            ignored.append(str(key))
        else:
            fitted[target] = value
    return fitted, ignored


def _simplex_grid(k: int, steps: int) -> list:
    """Все веса k моделей с шагом 1/steps, сумма которых равна 1."""
    def rec(left, parts):
        if parts == 1:
            return [[left]]
        return [[i] + rest for i in range(left + 1) for rest in rec(left - i, parts - 1)]
    return [np.array(w, dtype=float) / steps for w in rec(steps, k)]


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
        self._fits = {}                   # (run_id, seed) -> обученные на всём train предсказания
        self._fold_cache = {}             # (схема, таргет) -> номер фолда каждой строки train
        self._reval_cache = {}            # (run_id, сдвиг сида) -> OOF на другом разбиении
        self._fold_metric_cache = {}      # (run_id, таргет, схема) -> метрики по фолдам

    # ---- время ---------------------------------------------------------------
    def time_left(self) -> float:
        return self.deadline - time.monotonic()

    def estimate_final_seconds(self, run_id, seeds: int = 1) -> float:
        """Сколько займёт финальное обучение: время одной части валидации с запасом на весь train,
        умноженное на число ещё не обученных сидов."""
        run = self.runs[run_id]
        if run["config"]["model"] == "blend":
            return sum(self.estimate_final_seconds(m, seeds) for m in run["config"]["members"])
        missing = sum(1 for k in range(seeds) if (run_id, k) not in self._fits)
        if not missing:
            return 0.0
        result = run["result"]
        folds = max(1, len(result.get("fold_scores") or []))
        per_fold = result.get("seconds", 60.0) / folds / (run["config"]["subsample"] or 1.0)
        return missing * (1.6 * per_fold + 10.0)

    def can_finalize(self, run_id, seeds: int = 1) -> bool:
        return self.estimate_final_seconds(run_id, seeds) <= self.time_left() + self.final_slack

    # ---- обзор задачи -------------------------------------------------------
    def overview(self, max_columns: int = 60) -> dict:
        schema = self.profile["schema"]
        return {
            "profile_text": ml_profile.format_profile(self.profile, max_columns=max_columns),
            "suggested": {
                "target_cols": schema["target_cols"], "id_cols": schema["id_cols"],
                "output_columns": schema["output_columns"], "confidence": schema["confidence"],
                "task_type": self.profile.get("task", {}).get("type"),
                "metric": self.suggested_metric(),
                "metric_source": self.profile["metric"].get("source"),
                "metric_computed_by": "агент" if self.profile["metric"].get("supported") is False else "ядро",
                "answer_looks_like_probability": self.sample_suggests_proba(),
                "validation_hint": (f"по условию test позже train. Если есть столбцы времени (год, месяц, дата) — "
                                    f"лучше временная схема по ним. Иначе можно validation={{strategy: temporal, "
                                    f"column: {ROW_ORDER}}}: номер строки станет и признаком; на одних данных это "
                                    f"помогает, на других мешает — сравнивай с обычной схемой осторожно"
                                    if self.rows_look_time_ordered() else None),
            },
            "all_train_columns": [str(c) for c in self.data.train.columns] if self.data.train is not None else [],
            "extra_csv": [ml_io.display_path(p, self.task_dir) for p in self.data.files.extra_csv],
            "available": {"models": list(MODELS), "metrics": ml_metrics.available_metrics() + list(EXTRA_METRICS)
                          + ["f1 (binary)", "precision (binary)", "recall (binary)"],
                          "task_types": list(TASK_TYPES), "validation": list(ml_validation.STRATEGIES),
                          "target_transforms": [t for t in ml_tools.TARGET_TRANSFORMS if t]},
            "time_left_s": round(self.time_left()),
        }

    def suggested_metric(self) -> Optional[str]:
        """Основная метрика из task.md по ядру (mentioned_metrics): ядро называет и те, что считает агент
        (rmsle/msle/mape/smape, бинарная f1), с supported=False."""
        return self.profile["metric"].get("primary")

    def sample_suggests_proba(self) -> bool:
        """Похоже ли, что ответ — вероятность, когда метрика неизвестна: дробная заглушка в шаблоне при
        двух классах (или строковых метках), иначе слова в условии."""
        sample, train = self.data.sample_submission, self.data.train
        targets = self.profile["schema"]["target_cols"]
        if sample is not None and train is not None and len(targets) == 1 and targets[0] in sample.columns:
            placeholder, truth = sample[targets[0]], train[targets[0]]
            if truth.dropna().nunique() == 2:
                if not pd.api.types.is_float_dtype(placeholder):
                    return False                     # целая или строковая заглушка — ждут метку
                fractional = bool((placeholder.dropna() % 1 != 0).any())
                return fractional or not pd.api.types.is_numeric_dtype(truth)
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
            args, ignored = _fit_args(method, args)
            try:
                out = method(**args)
            except TypeError as error:
                out = {"ok": False, "error": f"неверные аргументы: {error}"}
            except Exception as error:  # инструмент не должен ронять агента
                out = {"ok": False, "error": f"{type(error).__name__}: {_short(error)}",
                       "trace": _short(traceback.format_exc(limit=2))}
            if ignored:
                out["ignored_args"] = ignored
                out["accepted_args"] = [p for p in inspect.signature(method).parameters]
        out["time_left_s"] = round(self.time_left())
        self.history.append({"tool": name, "args": args, "ok": out.get("ok"),
                             "seconds": round(time.monotonic() - started, 1)})
        return _jsonable(out)

    def tool_set_task(self, target_cols, task_type, metric, id_cols=None, drop_features=None,
                      prediction_format="auto", positive_class=None, reason=""):
        """Зафиксировать постановку: таргеты, id, тип задачи, метрику и формат ответа."""
        if isinstance(target_cols, str):
            target_cols = [target_cols]
        if isinstance(id_cols, str):
            id_cols = [id_cols]
        if task_type not in TASK_TYPES:
            return {"ok": False, "error": f"task_type должен быть одним из {TASK_TYPES}"}
        if not target_cols:
            return {"ok": False, "error": "укажите target_cols"}
        binary_kind = _BINARY_METRICS.get(re.sub(r"[^a-z0-9]", "", str(metric).lower()))
        if binary_kind and task_type != "binary":
            return {"ok": False, "error": f"{metric} считается по положительному классу: нужна binary-задача "
                                          "(для многоклассовой — f1_macro / f1_weighted)"}
        try:
            # бинарная F1/precision/recall: метрика создаётся ниже, когда известен положительный класс
            resolved = _binary_metric(binary_kind, 1) if binary_kind else resolve_metric(metric)
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
        schema = ml_schema.resolve_schema(suggestion, id_cols=ids, target_cols=list(target_cols))
        drop = [c for c in (drop_features or []) if c in schema.feature_cols]
        if drop:
            schema = replace(schema, feature_cols=tuple(c for c in schema.feature_cols if c not in drop))
        splits = {t: ml_data.split_task(self.data, replace(schema, target_cols=(t,))) for t in schema.target_cols}
        for target, split in splits.items():
            y = split.y_train
            if family == "regression" and not pd.api.types.is_numeric_dtype(y):
                return {"ok": False, "error": f"таргет {target!r} не числовой: регрессия невозможна"}
            if family == "classification" and y.nunique() < 2:
                return {"ok": False, "error": f"в таргете {target!r} меньше двух классов"}
            if task_type == "binary" and y.nunique() != 2:
                return {"ok": False, "error": f"в {target!r} {y.nunique()} классов, а не 2: выберите multiclass"}
        main = splits[schema.target_cols[0]]
        positive = {}
        if task_type == "binary":
            for target, split in splits.items():
                classes = ml_data.LabelCodec().fit(split.y_train).classes_
                chosen = None
                if positive_class is not None:
                    chosen = next((c for c in classes if _label_text(c) == _label_text(positive_class)), None)
                    if chosen is None:
                        return {"ok": False, "error": f"positive_class {positive_class!r} нет среди классов "
                                                      f"{[_label_text(c) for c in classes]}"}
                positive[target] = chosen if chosen is not None else guess_positive_class(classes, split.y_train,
                                                                                          self.data.text)
            if binary_kind:
                first = splits[schema.target_cols[0]]
                classes = list(ml_data.LabelCodec().fit(first.y_train).classes_)
                resolved = _binary_metric(binary_kind, classes.index(positive[schema.target_cols[0]]))

        # профиль для ml_tools с выбранными типом задачи и метрикой (иначе там останется эвристика);
        # своей метрики ml_tools по имени не знает — для его проверки семейства подставляем стандартную
        profile = copy.deepcopy(self.profile)
        profile["task"] = {"type": task_type, "ambiguous": False, "notes": ["выбрано агентом"]}
        known = resolved.name if resolved.name in ml_metrics.available_metrics() else \
            ("rmse" if family == "regression" else "f1_macro")
        profile["metric"] = dict(profile["metric"], primary=known, higher_is_better=resolved.higher_is_better)
        plan = ml_validation.suggest_validation(profile, main.X_train)
        if plan.get("column") and plan["column"] not in main.X_train.columns:
            # временной столбец убран из признаков: схема по нему невозможна, берём следующую по смыслу
            profile_no_time = copy.deepcopy(profile)
            profile_no_time.get("train_test", {})["out_of_train_range_share"] = {}
            notes.append(f"столбца {plan['column']!r} нет в признаках: временная схема заменена")
            plan = ml_validation.suggest_validation(profile_no_time, main.X_train)

        if self.written and self.written.get("run_id"):
            self.written["stale"] = True  # ответ от прошлой постановки остаётся на диске до лучшего
        self.featuresets = {0: {"splits": splits, "specs": []}}   # набор признаков -> разрезанные данные
        self.featureset = 0
        self.setup = {"schema": schema, "splits": splits, "split": main, "profile": profile, "task_type": task_type,
                      "metric": resolved.name, "metric_obj": resolved, "higher_is_better": resolved.higher_is_better,
                      "prediction_format": prediction_format, "positive": positive, "reason": reason}
        self.plan, self.plan_key = plan, _validation_key(plan)
        self.runs, self._fits, self._fold_cache, self._reval_cache, self._fold_metric_cache = {}, {}, {}, {}, {}
        self.log(f"[set_task] target={list(schema.target_cols)} id={list(schema.id_cols)} {task_type} "
                 f"metric={resolved.name} features={len(schema.feature_cols)} validation={plan['strategy']}"
                 + (f" positive={ {t: _label_text(c) for t, c in positive.items()} }" if positive else ""))
        out = {
            "ok": True, "n_train": len(main.X_train), "n_test": len(main.X_test),
            "n_features": len(schema.feature_cols), "id_cols": list(schema.id_cols), "dropped_features": drop,
            "target_summary": {t: ml_profile.profile_target(s.y_train, task_type) for t, s in splits.items()},
            "metric": resolved.name, "higher_is_better": resolved.higher_is_better,
            "validation": plan, "output_columns": self._output_columns(),
            "notes": notes, "warnings": main.warnings,
        }
        if task_type == "binary":
            out["answer_as_probability"] = self._wants_proba()
            out["positive_class"] = {t: _label_text(c) for t, c in positive.items()}
        if len(splits) > 1:
            out["note"] = f"{len(splits)} таргета обучаются по очереди одной конфигурацией; score — среднее"
        return out

    def tool_evaluate(self, model="lightgbm", params=None, learning_rate=0.05, target_transform=None,
                      validation=None, subsample=None, time_budget_s=None, max_rounds=3000, hypothesis=""):
        """Оценить конфигурацию на валидации. Сравнение — только с запусками на той же схеме."""
        if self.setup is None:
            return {"ok": False, "error": "сначала вызовите set_task"}
        if model not in MODELS:
            return {"ok": False, "error": f"model должна быть одной из {MODELS}"}
        params, dropped = self._clean_params(params, model)
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
        evals, splits, started = {}, self.setup["splits"], time.monotonic()
        for number, (target, split) in enumerate(splits.items()):
            # бюджет делится поровну; недоиспользованное время переходит следующим таргетам
            share = max(10.0, (budget - (time.monotonic() - started)) / (len(splits) - number))
            evals[target] = ml_tools.train_eval(
                split, self.setup["profile"], model=model, metric=self.setup["metric_obj"], validation=plan,
                params=params, task_type=self.setup["task_type"], target_transform=target_transform, n_jobs=self.n_jobs,
                seed=self.seed, time_budget=share, max_rounds=int(max_rounds),
                learning_rate=float(learning_rate), subsample=subsample,
            )
        summary = self._summary(evals)
        key = _validation_key(plan)
        changed = key != self.plan_key
        status = summary["status"]
        if changed and status == "ok" and subsample is None:
            # новая схема становится текущей только после успешного полного запуска
            self.plan, self.plan_key = plan, key
        rollback, self._row_order_rollback = getattr(self, "_row_order_rollback", None), None
        if rollback and (status != "ok" or subsample is not None):
            self.featureset, self.setup["splits"], self.setup["split"] = rollback
            self.log(f"[validation] {ROW_ORDER} убран: запуск не удался или был пробой")
        config = {"model": model, "params": params, "learning_rate": float(learning_rate),
                  "target_transform": target_transform, "max_rounds": int(max_rounds), "subsample": subsample,
                  "featureset": self.featureset}
        self.runs[run_id] = {"config": config, "result": summary, "eval": evals, "key": key, "hypothesis": hypothesis,
                             "plan": dict(plan)}
        best = self.best_run()
        self.log(f"[evaluate {run_id}] status={status} score={summary.get('score')} "
                 f"({summary['seconds']:.0f}s) best={best}")
        out = {"ok": status == "ok", "run_id": run_id, **summary, "best_run_id": best,
               "best_score": self._score(best) if best else None, "dropped_params": dropped,
               "comparable_runs": [r for r, v in self.runs.items() if v["key"] == self.plan_key
                                   and v["config"]["subsample"] is None]}
        if subsample is not None:
            out["note"] = "запуск на подвыборке: только для прикидки, с полными запусками не сравнивается"
        elif changed:
            out["note"] = ("схема валидации изменилась: прежние запуски с этим несравнимы" if status == "ok"
                           else "новая схема не сработала: текущей осталась прежняя")
        if self.auto_early and out["ok"] and subsample is None and not (self.written or {}).get("run_id"):
            # ранний настоящий ответ: страхует от сбоев дальнейшего диалога
            try:
                early = self._finalize(run_id)
            except Exception as error:
                early = {"ok": False, "error": f"{type(error).__name__}: {_short(error, 200)}"}
            out["early_submission"] = {k: early.get(k) for k in ("ok", "run_id", "error", "rows")}
        return out

    def tool_finalize(self, run_id=None, class_weights="auto", seeds=1, reason=""):
        """Обучить выбранный запуск на всём train и записать submission.csv (если он лучше записанного).
        seeds > 1 — среднее нескольких обучений с разными сидами (меньше разброс предсказаний)."""
        if self.setup is None:
            return {"ok": False, "error": "сначала вызовите set_task"}
        try:
            seeds = max(1, min(10, int(seeds)))
        except (TypeError, ValueError):
            seeds = 1
        run_id = run_id or self.best_run()
        if run_id not in self.runs or self.runs[run_id]["result"]["status"] != "ok":
            return {"ok": False, "error": f"нет успешного запуска {run_id!r}; успешные: {self._ok_runs()}"}
        written = self.written or {}
        if written.get("run_id") == run_id:
            if seeds <= written.get("seeds", 1):
                return {"ok": True, "note": "этот запуск уже записан", "written": self.written}
            if not self.can_finalize(run_id, seeds):
                return {"ok": False, "error": f"не хватит времени на {seeds} сидов"}
            return self._finalize(run_id, class_weights, seeds)
        if not self._better_than_written(run_id, automatic=False):
            return {"ok": False, "error": f"{run_id} не лучше уже записанного {self.written['run_id']} "
                                          "на той же валидации (или это подвыборка); ответ не перезаписан"}
        if not self.can_finalize(run_id, seeds):
            return {"ok": False, "error": f"не хватит времени: нужно ~{self.estimate_final_seconds(run_id, seeds):.0f}s"}
        return self._finalize(run_id, class_weights, seeds)

    def tool_tune(self, model="lightgbm", n_trials=15, time_budget_s=None, target_transform=None, hypothesis=""):
        """Подбор гиперпараметров Optuna (TPE) на текущей схеме валидации. Каждая попытка — обычный запуск
        evaluate: её можно финализировать и смешивать. Останавливается по n_trials или по бюджету."""
        if self.setup is None:
            return {"ok": False, "error": "сначала вызовите set_task"}
        if model not in MODELS:
            return {"ok": False, "error": f"model должна быть одной из {MODELS}"}
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        budget = self._eval_budget(time_budget_s if time_budget_s is not None else self.time_left() / 3)
        if budget is None:
            return {"ok": False, "error": "времени на подбор не осталось"}
        try:
            n_trials = max(1, min(100, int(n_trials)))
        except (TypeError, ValueError):
            n_trials = 15
        started, runs, failures = time.monotonic(), [], 0
        sampler = optuna.samplers.TPESampler(seed=self.seed + self._counter)
        study = optuna.create_study(direction="maximize" if self.setup["higher_is_better"] else "minimize",
                                    sampler=sampler)
        # первая попытка — параметры по умолчанию: подбор не может оказаться хуже базы
        study.enqueue_trial({"learning_rate": 0.05})
        for number in range(n_trials):
            left = budget - (time.monotonic() - started)
            if left < 20:
                break
            trial = study.ask()
            params, lr = _suggest_params(trial, model)
            per_trial = max(20.0, min(left, 2.5 * budget / n_trials))
            out = self.tool_evaluate(model=model, params=params, learning_rate=lr, target_transform=target_transform,
                                     time_budget_s=per_trial, hypothesis=f"tune {model} #{number + 1}: {hypothesis}")
            score = out.get("oof_score") if out.get("oof_score") is not None else out.get("score")
            if not out.get("ok") or score is None:
                failures += 1
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                if failures >= 3 and not runs:
                    break
                continue
            study.tell(trial, score)
            runs.append({"run_id": out["run_id"], "score": score, "params": params, "learning_rate": lr})
        if not runs:
            return {"ok": False, "error": "ни одна попытка подбора не удалась", "failures": failures}
        ranked = sorted(runs, key=lambda r: r["score"], reverse=self.setup["higher_is_better"])
        self.log(f"[tune] {model}: {len(runs)} попыток за {time.monotonic() - started:.0f}s, лучшая "
                 f"{ranked[0]['run_id']} = {ranked[0]['score']}")
        return {"ok": True, "model": model, "trials": len(runs), "failed": failures,
                "seconds": round(time.monotonic() - started), "best": ranked[0], "top": ranked[:5],
                "best_run_id": self.best_run(), "best_score": self._score(self.best_run())}

    def tool_add_features(self, features, hypothesis="", model="lightgbm"):
        """Добавить признаки из фиксированной грамматики (без таргета, поэтому без утечки). Партия принимается,
        только если быстрая модель на тех же фолдах устойчиво лучше без неё; иначе данные не меняются."""
        if self.setup is None:
            return {"ok": False, "error": "сначала вызовите set_task"}
        if isinstance(features, (str, dict)):
            features = [features]
        if not isinstance(features, list) or not features:
            return {"ok": False, "error": "features: список спецификаций, например {\"op\": \"ratio\", \"a\": \"x\", \"b\": \"y\"}"}
        current = self.featuresets[self.featureset]
        new_splits, applied, errors = {}, [], []
        for target, split in current["splits"].items():
            X_train, X_test, applied, errors = build_features(split.X_train, split.X_test, features[:40])
            new_splits[target] = replace(split, X_train=X_train, X_test=X_test)
        if not applied:
            return {"ok": False, "error": "ни один признак не построен", "errors": errors[:10],
                    "grammar": FEATURE_GRAMMAR}
        # база: лучший полный запуск выбранной модели с параметрами по умолчанию на текущем наборе, иначе новый
        base = next((r for r in reversed(self._ok_runs(full_only=True))
                     if self.runs[r]["config"]["model"] == model and not self.runs[r]["config"]["params"]
                     and self.runs[r]["config"].get("featureset", 0) == self.featureset
                     and self.runs[r]["key"] == self.plan_key), None)
        if base is None:
            out = self.tool_evaluate(model=model, hypothesis=f"база для проверки признаков: {hypothesis}")
            if not out.get("ok"):
                return {"ok": False, "error": f"база не посчиталась: {out.get('error')}"}
            base = out["run_id"]
        config = self.runs[base]["config"]
        previous = (self.featureset, self.setup["splits"], self.setup["split"])
        number = max(self.featuresets) + 1
        self.featuresets[number] = {"splits": new_splits, "specs": current["specs"] + applied}
        self.featureset = number
        self.setup["splits"], self.setup["split"] = new_splits, new_splits[next(iter(new_splits))]
        out = self.tool_evaluate(model=model, params=config["params"], learning_rate=config["learning_rate"],
                                 target_transform=config["target_transform"],
                                 hypothesis=f"признаки {[a['name'] for a in applied][:8]}: {hypothesis}")
        candidate = out.get("run_id")
        accepted = bool(out.get("ok")) and self._significantly_better(candidate, base)
        paired = self._paired(candidate, base) if out.get("ok") else None
        if not accepted:
            self.featureset, self.setup["splits"], self.setup["split"] = previous
            del self.featuresets[number]
            if candidate in self.runs:
                del self.runs[candidate]
        self.log(f"[features] {'приняты' if accepted else 'отклонены'} {[a['name'] for a in applied]} "
                 f"base={base} {self._score(base)} new={out.get('oof_score')} {paired}")
        return {"ok": True, "accepted": accepted, "features": [a["name"] for a in applied], "errors": errors[:10],
                "base_run": base, "base_oof": self._score(base), "new_run": candidate if accepted else None,
                "new_oof": out.get("oof_score"), "paired": paired, "featureset": self.featureset,
                "n_features": len(self.setup["split"].X_train.columns),
                "note": "признаки приняты: дальнейшие evaluate/tune используют их" if accepted else
                        "признаки не дали устойчивого выигрыша: данные не изменены"}

    def tool_finish(self, reason=""):
        """Закончить исследование. Если ответ не записан, код сам обучит лучший запуск."""
        self.finished = reason or "без причины"
        self.log(f"[finish] {self.finished}")
        return {"ok": True, "best_run_id": self.best_run(), "written": self.written}

    # ---- выбор запусков -------------------------------------------------------
    def _summary(self, evals: dict) -> dict:
        """Сводка запуска для LLM: один таргет — как EvalResult.to_dict(), несколько — среднее и разбивка."""
        parts = {}
        for target, result in evals.items():
            part = result.to_dict()
            part["notes"] = [_short(n, 300) for n in part.get("notes", [])][-5:]
            part["validation"] = {k: part.get("validation", {}).get(k) for k in _VALIDATION_KEYS + ("effective",)}
            parts[target] = part
        first = next(iter(parts.values()))
        if len(parts) == 1:
            return first
        ok = all(p["status"] == "ok" for p in parts.values())
        scores = [p["score"] for p in parts.values()]
        oof_scores = [p.get("oof_score") for p in parts.values()]
        return {"status": "ok" if ok else "error", "model": first["model"], "metric": first["metric"],
                "higher_is_better": first["higher_is_better"],
                "score": float(np.mean(scores)) if ok and None not in scores else None,
                "oof_score": float(np.mean(oof_scores)) if ok and None not in oof_scores else None,
                "fold_scores": first["fold_scores"], "seconds": float(sum(p["seconds"] for p in parts.values())),
                "validation": first["validation"],
                "error": "; ".join(f"{t}: {p['error']}" for t, p in parts.items() if p["error"]) or None,
                "per_target": {t: {k: p[k] for k in ("score", "fold_scores", "best_iterations", "seconds", "notes")}
                               for t, p in parts.items()}}

    def _ok_runs(self, full_only=False):
        return [r for r, v in self.runs.items() if v["result"]["status"] == "ok"
                and not (full_only and v["config"]["subsample"] is not None)]

    def _is_better(self, new, old) -> bool:
        return ml_metrics.is_better(self.setup["metric_obj"], new, old)

    def _fold_ids(self, target) -> np.ndarray:
        """Номер фолда текущей схемы для каждой строки train (-1 — строка не проверялась). Разбиение
        детерминировано (тот же план и сид, что в train_eval), поэтому OOF любого запуска раскладывается по фолдам."""
        key = (self.plan_key, target)
        if key not in self._fold_cache:
            split = self.setup["splits"][target]
            y = None
            if self.setup["task_type"] != "regression":
                y = ml_data.LabelCodec().fit(split.y_train).encode(split.y_train)
            ids = np.full(len(split.X_train), -1)
            try:
                for number, (_, valid) in enumerate(ml_validation.make_splits(self.plan, split.X_train, y, seed=self.seed)):
                    ids[valid] = number
            except Exception:
                pass
            self._fold_cache[key] = ids
        return self._fold_cache[key]

    def _fold_metrics(self, target, index, pred) -> dict:
        """Метрика задачи по каждому фолду: {номер фолда: значение}."""
        folds = self._fold_ids(target)[np.asarray(index)]
        out = {}
        for number in np.unique(folds[folds >= 0]):
            mask = folds == number
            if mask.sum() >= 5:
                try:
                    out[int(number)] = self._oof_metric(target, np.asarray(index)[mask], np.asarray(pred)[mask])
                except Exception:
                    pass
        return out

    def _run_fold_metrics(self, run_id, target) -> dict:
        """Метрики запуска по фолдам текущей схемы (с кэшем: запуски неизменяемы)."""
        key = (run_id, target, self.plan_key)
        if key not in self._fold_metric_cache:
            self._fold_metric_cache[key] = self._fold_metrics(target, *self._oof(run_id, target))
        return self._fold_metric_cache[key]

    def _paired(self, new_id, old_id) -> Optional[dict]:
        """Парное сравнение двух запусков по общим фолдам: относительные разницы (плюс — new лучше)."""
        sign = 1.0 if self.setup["higher_is_better"] else -1.0
        diffs = []
        for target in self.setup["splits"]:
            a = self._run_fold_metrics(new_id, target)
            b = self._run_fold_metrics(old_id, target)
            for fold in set(a) & set(b):
                diffs.append(sign * (a[fold] - b[fold]) / max(abs(b[fold]), 1e-12))
        if len(diffs) < 2:
            return None
        diffs = np.asarray(diffs)
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        return {"gain": float(diffs.mean()), "se": se, "wins": int((diffs > 0).sum()), "folds": len(diffs)}

    def _significantly_better(self, new_id, old_id) -> bool:
        """Новый запуск лучше, только если выигрыш устойчив по фолдам: среднее > 0.5·SE и победа в большинстве
        фолдов. Без парных данных — обычное сравнение средних."""
        if not self._is_better(self._score(new_id), self._score(old_id)):
            return False
        try:
            paired = self._paired(new_id, old_id)
        except Exception:
            paired = None
        if paired is None or paired["folds"] < 5:
            # мало фолдов (временная схема — обычно 3): SE по 3 точкам шумит и «застревает» на первом
            # запуске; сравниваем общий OOF с небольшим относительным порогом
            old = self._score(old_id)
            return abs(self._score(new_id) - old) > 0.0025 * max(abs(old), 1e-12)
        return paired["gain"] > 0.5 * paired["se"] and paired["wins"] * 2 >= paired["folds"]

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
        for run_id in pool:   # в порядке появления: новый запуск сменяет лучший только при устойчивом выигрыше
            if best is None or self._significantly_better(run_id, best):
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
        if automatic:
            return self._significantly_better(run_id, written["run_id"])
        return self._is_better(self._score(run_id), self._score(written["run_id"]))

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
            if column in ROW_ORDER_ALIASES:
                if ROW_ORDER not in self.setup["split"].X_train.columns:
                    # откат, если запуск с этой схемой не удастся или окажется пробой на подвыборке
                    self._row_order_rollback = (self.featureset, self.setup["splits"], self.setup["split"])
                column = self._ensure_row_order()
            X = self.setup["split"].X_train
            kinds = {c["name"]: c.get("kind") for c in self.profile.get("features", {}).get("columns", [])}
            if column not in X.columns or not (pd.api.types.is_numeric_dtype(X[column]) or kinds.get(column) == "datetime"):
                return None, f"для temporal нужен числовой признак времени или дата в column; {column!r} не подходит"
            direction = validation.get("direction") or "forward"
            if direction not in ("forward", "backward"):
                return None, "direction: forward | backward"
            plan.update(column=column, direction=direction, n_splits=min(plan["n_splits"], 5))
        elif strategy == "group" and validation.get("column"):
            return None, ("группировка по столбцу не поддержана: group объединяет одинаковые строки train; "
                          "уберите column")
        return plan, None

    def _ensure_row_order(self) -> str:
        """Столбец с номером строки (train 0..n-1, test продолжает) для временной схемы, когда строки
        упорядочены по времени, а явного столбца времени нет. Добавляется новым набором признаков."""
        if ROW_ORDER in self.setup["split"].X_train.columns:
            return ROW_ORDER
        current = self.featuresets[self.featureset]
        new_splits = {}
        for target, split in current["splits"].items():
            n_train, n_test = len(self.data.train), len(split.X_test)
            positions = pd.Series(np.arange(n_train), index=self.data.train.index)
            X_train = split.X_train.copy()
            X_train[ROW_ORDER] = positions.reindex(split.X_train.index).to_numpy(dtype=float)
            X_test = split.X_test.copy()
            X_test[ROW_ORDER] = np.arange(n_train, n_train + n_test, dtype=float)
            new_splits[target] = replace(split, X_train=X_train, X_test=X_test)
        number = max(self.featuresets) + 1
        self.featuresets[number] = {"splits": new_splits, "specs": current["specs"] + [{"name": ROW_ORDER,
                                                                                       "spec": {"op": "row_order"}}]}
        self.featureset = number
        self.setup["splits"], self.setup["split"] = new_splits, new_splits[next(iter(new_splits))]
        self.log(f"[validation] добавлен {ROW_ORDER}: временная схема по порядку строк")
        return ROW_ORDER

    def rows_look_time_ordered(self) -> bool:
        """Условие говорит, что строки упорядочены по времени / test позже train, а столбца времени нет."""
        return bool(_TIME_ORDER_WORDS.search(self.data.text or ""))

    def _eval_budget(self, requested) -> Optional[float]:
        """Бюджет эксперимента: не больше остатка за вычетом финального обучения; None — времени нет."""
        # пока моделей нет, резерв маленький: первая модель важнее всего; дальше — время финала лучшей
        reserve = 30.0
        best = self.best_run() if self.setup else None
        if best:
            reserve = max(60.0, self.estimate_final_seconds(best))
        left = self.time_left() - reserve
        if left < 20:
            return None
        default = max(60.0, min(1800.0, left / 3))
        try:
            requested = float(requested) if requested is not None else default
        except (TypeError, ValueError):
            requested = default
        return max(min(30.0, left), min(requested, left))

    def _clean_params(self, params, model="lightgbm"):
        """Параметры LLM без тех, что задаёт код, и в словаре нужной библиотеки; (params, отброшенные)."""
        if not isinstance(params, dict):
            return {}, []
        clean, dropped = {}, []
        for key, value in params.items():
            if key in _RESERVED_PARAMS or not isinstance(value, (int, float, str, bool)):
                dropped.append(key)
            elif str(value).lower() == "dart":
                dropped.append(key)  # dart отключает раннюю остановку и съедает весь бюджет
            elif model == "xgboost" and key in _XGBOOST_DROP:
                dropped.append(key)
            elif model == "catboost" and key in _CATBOOST_DROP:
                dropped.append(key)
            elif model == "catboost" and key in _CATBOOST_RENAME:
                clean[_CATBOOST_RENAME[key]] = value
            elif model == "catboost" and key in ("subsample", "bagging_fraction"):
                clean.update(subsample=value, bootstrap_type="Bernoulli")
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
        metric = setup.get("metric_obj")
        if metric is None and setup.get("metric"):
            try:
                metric = resolve_metric(setup["metric"])
            except ml_metrics.MetricError:
                metric = None
        if metric is not None:
            return metric.needs == "proba"           # метрика по меткам (F1, accuracy) — пишем метки
        return self.sample_suggests_proba()

    # ---- финальное обучение и ансамбль --------------------------------------
    def _oof(self, run_id, target):
        """OOF-предсказания запуска: (позиции строк train, значения (n,) или вероятности (n, K))."""
        run = self.runs[run_id]
        if run["config"]["model"] == "blend":
            return run["oof"][target]
        ev = run["eval"][target]
        return ev.oof_index, ev.oof_pred

    def _fit_one(self, run_id, k: int) -> dict:
        """Одно финальное обучение запуска с сидом seed+k (с кэшем): таргет -> {pred, proba, classes, ...}."""
        if (run_id, k) in self._fits:
            return self._fits[(run_id, k)]
        run, setup = self.runs[run_id], self.setup
        config = run["config"]
        fits = {}
        splits = self.featuresets[config.get("featureset", 0)]["splits"]
        for target, split in splits.items():
            ev = run["eval"][target]
            n_rounds = ml_tools.rounds_from(ev)
            budget = max(10.0, self.time_left() + self.final_slack)
            self.log(f"[fit {run_id}/seed{k}] {target}: {config['model']} n_rounds={n_rounds} budget={budget:.0f}s")
            out = ml_tools.fit_predict(
                split, setup["profile"], model=config["model"], n_rounds=n_rounds, params=config["params"],
                task_type=setup["task_type"], target_transform=config["target_transform"], n_jobs=self.n_jobs,
                seed=self.seed + k, time_budget=budget, max_rounds=config["max_rounds"],
                learning_rate=config["learning_rate"],
            )
            if out.get("status") != "ok":
                raise RuntimeError(f"финальное обучение {run_id}/{target} не удалось: {_short(out.get('error'))}")
            truncated = bool(n_rounds and out.get("n_rounds") and out["n_rounds"] < 0.8 * n_rounds)
            notes = list(out.get("notes", []))
            if truncated:
                notes.append(f"{target}: обучение урезано по времени: {out['n_rounds']} из {n_rounds} раундов")
            fits[target] = {"pred": out["pred"], "proba": out.get("proba"), "classes": out.get("classes"),
                            "notes": notes, "truncated": truncated}
        self._fits[(run_id, k)] = fits
        return fits

    def _combine(self, parts: list, weights: list, targets=None) -> dict:
        """Взвешенное среднее предсказаний: значения регрессии или вероятности классов."""
        fits = {}
        for target in (targets or self.setup["splits"]):
            items = [p[target] for p in parts]
            if self.setup["task_type"] == "regression":
                pred = sum(w * np.asarray(i["pred"], dtype=float) for w, i in zip(weights, items))
                fits[target] = {"pred": pred, "proba": None, "classes": None}
            else:
                proba = sum(w * np.asarray(i["proba"], dtype=float) for w, i in zip(weights, items))
                classes = items[0]["classes"]
                fits[target] = {"pred": np.asarray(classes)[proba.argmax(axis=1)], "proba": proba, "classes": classes}
            fits[target]["notes"] = [n for i in items for n in i.get("notes", [])]
            fits[target]["truncated"] = any(i["truncated"] for i in items)
        return fits

    def _fit_run(self, run_id, seeds: int = 1) -> dict:
        """Финальное обучение запуска: среднее по seeds сидам; смесь — взвешенное среднее участников."""
        config = self.runs[run_id]["config"]
        if config["model"] == "blend":
            parts = [self._fit_run(m, seeds) for m in config["members"]]
            per_target = {}
            for target in self.setup["splits"]:
                per_target[target] = self._combine([{target: p[target]} for p in parts], config["weights"][target],
                                                   targets=[target])[target]
            return per_target
        parts = [self._fit_one(run_id, k) for k in range(max(1, int(seeds)))]
        return self._combine(parts, [1.0 / len(parts)] * len(parts))

    def _finalize(self, run_id, class_weights="auto", seeds: int = 1) -> dict:
        run, setup = self.runs[run_id], self.setup
        try:
            fits = self._fit_run(run_id, seeds)
        except RuntimeError as error:
            return {"ok": False, "error": str(error)}
        notes = [n for f in fits.values() for n in f["notes"]]
        if any(f["truncated"] for f in fits.values()) and (self.written or {}).get("run_id"):
            return {"ok": False, "error": "финальное обучение урезано по времени; прежний ответ сохранён", "notes": notes}
        predictions, weights_info = {}, {}
        for target, split in setup["splits"].items():
            fit = fits[target]
            pred = fit["pred"]
            if setup["task_type"] != "regression":
                pred, weights_info[target] = self._class_weights(run_id, target, fit, class_weights)
            elif split.y_train.min() >= 0:
                pred = np.clip(pred, 0, None)  # неотрицательный таргет: отрицательный прогноз только вредит
            predictions[target] = (pred, fit["proba"], fit["classes"])
        setup["secondary"] = self._secondary_targets()
        frame, frame_notes = self._assemble(setup, predictions)
        path = write_submission(frame, self.task_dir, len(setup["split"].X_test), setup["schema"].id_cols)
        self.written = {"run_id": run_id, "score": self._score(run_id), "model": run["config"]["model"],
                        "seeds": int(seeds)}
        self.log(f"[finalize {run_id}] записан {path} ({len(frame)} строк), cv={self._score(run_id)}")
        return {"ok": True, "path": str(path), "rows": len(frame), "columns": list(frame.columns),
                "run_id": run_id, "cv_score": self._score(run_id),
                "class_weights": {t: w for t, w in weights_info.items() if w} or None,
                "notes": notes + frame_notes}

    def _secondary_targets(self) -> dict:
        """Таргеты из шаблона ответа вне основной постановки (например, другого типа): вместо константы —
        простая модель LightGBM на тех же признаках; тип по данным. Ошибка -> столбец получит константу."""
        if "_secondary" in self.setup:
            return self.setup["_secondary"]
        out = {}
        schema = self.setup["schema"]
        test_cols = set(self.data.test.columns) if self.data.test is not None else set()
        candidates = [c for c in self._output_columns() if c not in schema.target_cols and c not in schema.id_cols
                      and c not in test_cols and self.data.train is not None and c in self.data.train.columns]
        for col in candidates:
            if self.time_left() < 120:
                break
            try:
                y = self.data.train[col]
                kind = ml_schema.suggest_task_type(y, "")["type"]
                if kind not in TASK_TYPES:
                    continue
                target_schema = replace(schema, target_cols=(col,))
                split = ml_data.split_task(self.data, target_schema)
                profile = copy.deepcopy(self.setup["profile"])
                profile["task"] = {"type": kind, "ambiguous": False, "notes": ["вторичный таргет"]}
                result = ml_tools.fit_predict(split, profile, model="lightgbm", task_type=kind, n_jobs=self.n_jobs,
                                              seed=self.seed, time_budget=min(600.0, self.time_left() / 4))
                if result.get("status") != "ok":
                    continue
                pred = result["pred"]
                if kind == "regression":
                    pred = np.asarray(pred, dtype=float)
                    if float(y.dropna().min()) >= 0:
                        pred = np.clip(pred, 0, None)
                else:
                    pred = _clean_labels(pred)
                out[col] = pred
                self.log(f"[secondary] {col}: {kind}, модель LightGBM")
            except Exception as error:
                self.log(f"[secondary] {col}: {type(error).__name__}: {_short(error, 150)}")
        self.setup["_secondary"] = out
        return out

    def _class_weights(self, run_id, target, fit, mode):
        """Веса классов под Macro F1 по OOF-вероятностям; применяются, только если честно помогают."""
        setup, run = self.setup, self.runs[run_id]
        split = setup["splits"][target]
        labels = fit["pred"]
        metric = setup["metric_obj"]
        if mode == "off" or metric.needs != "labels" or setup["metric"] == "accuracy" \
                or run["config"]["subsample"] or fit["proba"] is None:
            return labels, None
        index, oof = self._oof(run_id, target)
        if oof is None or index is None or len(index) < 200:
            return labels, None
        try:
            codec = ml_data.LabelCodec().fit(split.y_train)
            truth = codec.encode(split.y_train.iloc[index])
            groups = ml_validation.row_groups(split.X_train.iloc[index])
            if setup["metric"] == "f1_macro":
                choice = ml_postprocess.choose_class_weights(oof, truth, groups=groups)
            else:   # бинарная F1, precision/recall, f1_weighted, balanced accuracy: веса = порог
                choice = choose_weights_for_metric(np.asarray(oof), truth, metric, groups=groups)
            info = {"applied": bool(choice["applied"]), "holdout_gain": choice["holdout_gain"]}
            if choice["applied"]:
                labels = codec.decode(ml_postprocess.apply_class_weights(fit["proba"], choice["weights"]))
                info["weights"] = dict(zip([str(c) for c in codec.classes_], choice["weights"]))
            self.log(f"[class_weights] {target}: {info}")
            return labels, info
        except Exception as error:
            return labels, {"applied": False, "error": _short(error, 200)}

    def _oof_metric(self, target, index, pred) -> float:
        """Метрика задачи на OOF-предсказаниях (как oof_score в ml_tools)."""
        split, metric = self.setup["splits"][target], self.setup["metric_obj"]
        y = split.y_train.iloc[index]
        if self.setup["task_type"] == "regression":
            return metric(y.to_numpy(dtype=float), pred)
        codec = ml_data.LabelCodec().fit(split.y_train)
        codes = codec.encode(y)
        if metric.needs == "labels":
            return metric(codes, np.asarray(pred).argmax(axis=1))
        return metric(codes, pred, labels=list(range(len(codec.classes_))))

    def tool_blend(self, run_ids=None, reason=""):
        """Смесь запусков текущей схемы по OOF: жадный отбор Caruana с возвратом и равновесное среднее топ-k;
        берётся лучший вариант и создаётся, только если он устойчиво лучше лучшего одиночного запуска."""
        if self.setup is None:
            return {"ok": False, "error": "сначала вызовите set_task"}
        pool = [r for r in self._ok_runs(full_only=True) if self.runs[r]["key"] == self.plan_key
                and self.runs[r]["config"]["model"] != "blend"]
        if run_ids:
            missing = [r for r in run_ids if r not in pool]
            if missing:
                return {"ok": False, "error": f"{missing} не годятся: нужны успешные полные запуски текущей схемы {pool}"}
            candidates = list(dict.fromkeys(run_ids))
        else:
            candidates = sorted(pool, key=lambda r: self._score(r), reverse=self.setup["higher_is_better"])[:10]
        if len(candidates) < 2:
            return {"ok": False, "error": f"для смеси нужно хотя бы два успешных полных запуска; есть {pool}"}
        best_single = candidates[0]
        for run_id in candidates[1:]:
            if self._is_better(self._score(run_id), self._score(best_single)):
                best_single = run_id
        targets = list(self.setup["splits"])
        aligned, commons = {}, {}
        for target in targets:
            parts = {m: self._oof(m, target) for m in candidates}
            common = None
            for index, _ in parts.values():
                common = np.asarray(index) if common is None else np.intersect1d(common, index)
            if common is None or len(common) < 20:
                return {"ok": False, "error": "у запусков слишком мало общих OOF-строк"}
            commons[target] = common
            aligned[target] = {}
            for m, (index, pred) in parts.items():
                order = pd.Series(np.arange(len(index)), index=index).loc[common].to_numpy()
                aligned[target][m] = np.asarray(pred, dtype=float)[order]

        def score_of(weights: dict) -> float:
            total = sum(weights.values())
            values = []
            for target in targets:
                mix = sum(w * aligned[target][m] for m, w in weights.items()) / total
                values.append(self._oof_metric(target, commons[target], mix))
            return float(np.mean(values))

        # Caruana: стартуем с лучшего, добавляем (с возвратом) то, что сильнее всего улучшает смесь
        counts = {best_single: 1}
        trail = [(dict(counts), score_of(counts))]
        for _ in range(30):
            step = None
            for m in candidates:
                trial = dict(counts)
                trial[m] = trial.get(m, 0) + 1
                value = score_of(trial)
                if step is None or self._is_better(value, step[1]):
                    step = (trial, value)
            counts = step[0]
            trail.append((dict(counts), step[1]))
        caruana = trail[0]
        for item in trail[1:]:
            if self._is_better(item[1], caruana[1]):
                caruana = item
        options = [("caruana", caruana[0], caruana[1])]
        for k in (2, 3, 4):
            if len(candidates) >= k:
                equal = {m: 1 for m in candidates[:k]}
                options.append((f"equal_top{k}", equal, score_of(equal)))
        kind, chosen, blended = options[0]
        for option in options[1:]:
            if self._is_better(option[2], blended):
                kind, chosen, blended = option
        members = [m for m in candidates if chosen.get(m)]
        if len(members) < 2:
            return {"ok": False, "error": f"смесь сводится к одному запуску {best_single}: она не лучше лучшего одиночного"}
        total = sum(chosen.values())
        weights = {t: [chosen[m] / total for m in members] for t in targets}
        oof = {t: (commons[t], sum(w * aligned[t][m] for m, w in zip(members, weights[t]))) for t in targets}
        self._counter += 1
        run_id = f"b{self._counter}"
        seconds = float(sum(self.runs[m]["result"].get("seconds", 0) for m in members))
        self.runs[run_id] = {
            "config": {"model": "blend", "members": members, "weights": weights, "subsample": None, "params": {},
                       "learning_rate": None, "target_transform": None, "max_rounds": 0, "kind": kind},
            "result": {"status": "ok", "score": blended, "oof_score": blended, "fold_scores": [], "seconds": seconds,
                       "metric": self.setup["metric"], "model": "blend"},
            "eval": None, "oof": oof, "key": self.plan_key, "hypothesis": reason}
        paired = self._paired(run_id, best_single)
        if not self._significantly_better(run_id, best_single):
            del self.runs[run_id]
            return {"ok": False, "error": "смесь не даёт устойчивого выигрыша над лучшим одиночным запуском",
                    "best_single": best_single, "blend_oof": blended, "single_oof": self._score(best_single),
                    "paired": paired}
        self.log(f"[blend {run_id}] {kind} {dict(zip(members, [round(w, 2) for w in weights[targets[0]]]))} -> "
                 f"{blended} (лучший одиночный {best_single} {self._score(best_single)}; {paired})")
        return {"ok": True, "run_id": run_id, "kind": kind, "members": members,
                "weights": {m: round(w, 3) for m, w in zip(members, weights[targets[0]])},
                "oof_score": blended, "best_single": best_single, "best_single_oof": self._score(best_single),
                "paired_vs_best_single": paired, "best_run_id": self.best_run(),
                "note": "обучить и записать смесь: finalize(run_id)"}

    def _reval_oof(self, run_id, offset: int) -> dict:
        """OOF запуска на другом разбиении (сид фолдов + offset): таргет -> (позиции, предсказания)."""
        key = (run_id, offset)
        if key in self._reval_cache:
            return self._reval_cache[key]
        run = self.runs[run_id]
        config = run["config"]
        if config["model"] == "blend":
            parts = [self._reval_oof(m, offset) for m in config["members"]]
            out = {}
            for target in self.setup["splits"]:
                common = None
                for part in parts:
                    common = np.asarray(part[target][0]) if common is None else np.intersect1d(common, part[target][0])
                mix = 0
                for part, weight in zip(parts, config["weights"][target]):
                    index, pred = part[target]
                    order = pd.Series(np.arange(len(index)), index=index).loc[common].to_numpy()
                    mix = mix + weight * np.asarray(pred, dtype=float)[order]
                out[target] = (common, mix)
        else:
            splits = self.featuresets[config.get("featureset", 0)]["splits"]
            out = {}
            for target, split in splits.items():
                result = ml_tools.train_eval(
                    split, self.setup["profile"], model=config["model"], metric=self.setup["metric_obj"],
                    validation=run.get("plan") or self.plan, params=config["params"], task_type=self.setup["task_type"],
                    target_transform=config["target_transform"], n_jobs=self.n_jobs, seed=self.seed + offset,
                    time_budget=max(20.0, self.time_left() / 4), max_rounds=config["max_rounds"],
                    learning_rate=config["learning_rate"])
                if result.status != "ok":
                    raise RuntimeError(f"перепроверка {run_id}: {result.error}")
                out[target] = (result.oof_index, result.oof_pred)
        self._reval_cache[key] = out
        return out

    def revalidate(self, run_ids, offset: int = 1) -> dict:
        """Оценка кандидатов на новом разбиении: run_id -> {"first", "second", "mean"} (метрика по всем OOF)."""
        scores = {}
        for run_id in run_ids:
            try:
                oof = self._reval_oof(run_id, offset)
                second = float(np.mean([self._oof_metric(t, *oof[t]) for t in self.setup["splits"]]))
            except Exception as error:
                self.log(f"[revalidate] {run_id}: {type(error).__name__}: {_short(error, 150)}")
                continue
            first = self._score(run_id)
            scores[run_id] = {"first": first, "second": second, "mean": (first + second) / 2}
        return scores

    def _score(self, run_id):
        """Оценка для сравнения запусков: метрика по всем OOF-строкам сразу, иначе среднее по частям."""
        result = self.runs[run_id]["result"]
        return result.get("oof_score") if result.get("oof_score") is not None else result.get("score")

    def _assemble(self, setup, predictions: dict):
        """Собрать ответ в порядке test по столбцам шаблона. predictions: таргет -> (метки/значения, proba, classes).
        Каждый столбец заполняется по своему правилу; прогноз пишется только в столбец своего таргета
        или в столбцы вероятностей классов."""
        schema, split = setup["schema"], setup["split"]
        n = len(split.X_test)
        ids = split.test_ids.reset_index(drop=True)
        test = self.data.test.reset_index(drop=True)
        train, sample = self.data.train, self.data.sample_submission
        columns = self._output_columns(schema)
        classification = setup["task_type"] != "regression"
        notes, frame = [], pd.DataFrame(index=range(n))
        main = schema.target_cols[0]

        class_map = None
        pred, proba, classes = predictions[main]
        if classification and proba is not None and setup["prediction_format"] != "label" and len(predictions) == 1:
            candidates = [c for c in columns if c not in ids.columns and c != main]
            if len(candidates) >= 2:
                class_map = match_class_columns(candidates, classes)
                if class_map is None and len(candidates) == len(classes):
                    class_map = dict(zip(candidates, classes))
                    notes.append(f"столбцы {candidates} сопоставлены классам {list(map(str, classes))} по порядку")
        wants_proba = classification and setup["task_type"] == "binary" and self._wants_proba(setup)

        alias = {}
        if len(predictions) == 1 and main not in columns:
            # столбец ответа назван иначе, чем таргет (prediction, target, ...): прогноз пишем в него
            unknown = [c for c in columns if c not in ids.columns and not (class_map and c in class_map)
                       and c not in test.columns and (train is None or c not in train.columns)]
            if len(unknown) == 1:
                alias[unknown[0]] = main
                notes.append(f"{unknown[0]}: прогноз таргета {main!r} (столбец ответа назван иначе)")
        for col in columns:
            source = alias.get(col, col)
            if col in ids.columns:
                frame[col] = ids[col].to_numpy()
            elif source in predictions:
                pred, proba, classes = predictions[source]
                if wants_proba and proba is not None and len(classes) == 2:
                    positive = setup["positive"].get(source, classes[1])
                    frame[col] = np.asarray(proba)[:, list(classes).index(positive)]
                    notes.append(f"{col}: вероятность класса {_label_text(positive)!r}")
                elif classification:
                    labels = _clean_labels(pred)
                    if sample is not None and col in sample.columns and labels.dtype.kind == "f" \
                            and pd.api.types.is_integer_dtype(sample[col]):
                        labels = labels.astype("int64")
                    frame[col] = labels
                else:
                    frame[col] = np.asarray(pred, dtype=float)
            elif class_map and col in class_map:
                frame[col] = np.asarray(predictions[main][1])[:, list(predictions[main][2]).index(class_map[col])]
            elif col in test.columns:
                frame[col] = test[col].to_numpy()
                notes.append(f"{col}: скопирован из test")
            elif col in setup.get("secondary", {}):
                frame[col] = setup["secondary"][col]
                notes.append(f"{col}: отдельная модель LightGBM (таргет вне основной постановки)")
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
            schema = ml_schema.resolve_schema(suggestion, id_cols=suggestion["id_cols"],
                                              target_cols=suggestion["target_cols"])
            kind = self.profile.get("task", {}).get("type")
            predictions, task_type, positive = {}, None, {}
            split = None
            for target in schema.target_cols:
                split = split or ml_data.split_task(self.data, replace(schema, target_cols=(target,)))
                y = self.data.train[target].dropna()
                n = len(split.X_test)
                regression = kind == "regression" or (kind not in ("binary", "multiclass")
                                                      and pd.api.types.is_numeric_dtype(y) and y.nunique() > 20)
                if regression:
                    task_type = task_type or "regression"
                    predictions[target] = (np.full(n, float(y.median())), None, None)
                else:
                    task_type = task_type or ("binary" if y.nunique() == 2 else "multiclass")
                    counts = y.value_counts(normalize=True)
                    classes = np.array(sorted(counts.index, key=str))
                    priors = np.tile([counts[c] for c in classes], (n, 1))
                    predictions[target] = (np.repeat(counts.index[0], n), priors, classes)
                    if len(classes) == 2:
                        positive[target] = guess_positive_class(classes, y, self.data.text)
            setup = {"schema": schema, "split": split, "task_type": task_type, "positive": positive,
                     "metric": self.suggested_metric() or "",
                     "prediction_format": "auto"}
            frame, notes = self._assemble(setup, predictions)
            write_submission(frame, self.task_dir, len(split.X_test), schema.id_cols)
            self.written = {"run_id": None, "score": None, "model": "constant"}
            self.log(f"[safety] записан страховочный ответ {list(frame.columns)}; {notes[:3]}")
            return self.written
        except Exception as error:
            self.log(f"[safety] страховочный ответ не записан: {type(error).__name__}: {_short(error, 200)}")
            return None


TOOL_SPECS = [
    {
        "name": "set_task",
        "description": "Зафиксировать постановку задачи по task.md и профилю данных: таргеты, id, тип задачи, "
                       "метрику качества и формат ответа. Вызвать первым; повторный вызов начинает эксперименты "
                       "заново (записанный ответ остаётся на диске, пока его не заменит лучший).",
        "parameters": {
            "type": "object",
            "properties": {
                "target_cols": {"type": "array", "items": {"type": "string"},
                                "description": "Столбцы-таргеты. Несколько таргетов одного типа обучаются по очереди"},
                "id_cols": {"type": "array", "items": {"type": "string"},
                            "description": "Столбцы-идентификаторы строк для ответа; [] если их нет"},
                "task_type": {"type": "string", "enum": list(TASK_TYPES)},
                "metric": {"type": "string", "description": "Метрика из task.md, одно из available.metrics"},
                "drop_features": {"type": "array", "items": {"type": "string"},
                                  "description": "Признаки, которые нельзя использовать (утечка, служебные)"},
                "prediction_format": {"type": "string", "enum": ["auto", "label", "proba"],
                                      "description": "Классификация: метки или вероятности; auto — по метрике и шаблону ответа"},
                "positive_class": {"type": "string",
                                   "description": "Бинарная задача с вероятностями: класс, вероятность которого просят в условии"},
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
                                          "colsample_bytree, reg_lambda, class_weight=balanced; xgboost max_depth, "
                                          "min_child_weight (class_weight не поддерживает); catboost depth, l2_leaf_reg, "
                                          "auto_class_weights=Balanced. Неподдержанные ключи вернутся в dropped_params"},
                "learning_rate": {"type": "number"},
                "target_transform": {"type": "string", "enum": ["none", "log1p"],
                                     "description": "Только регрессия: log1p для неотрицательного скошенного таргета "
                                                    "и для метрик RMSLE/MSLE"},
                "validation": {"type": "object",
                               "description": "Сменить схему: {strategy: temporal|group|stratified|kfold, n_splits, "
                                              "column (только temporal: числовой признак времени, дата или "
                                              "__row_order__ — порядок строк, если test позже train), "
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
                "seeds": {"type": "integer", "description": "1..10: среднее обучений с разными сидами; 3–5 обычно "
                                                            "немного лучше на test, стоит seeds× времени финала"},
                "reason": {"type": "string"},
            },
        },
    },
    {
        "name": "tune",
        "description": "Подбор гиперпараметров модели (Optuna TPE) на текущей схеме валидации в пределах бюджета. "
                       "Каждая попытка — обычный запуск (можно finalize/blend). Первая попытка — параметры по умолчанию.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "enum": list(MODELS)},
                "n_trials": {"type": "integer", "description": "Сколько попыток (каждая — полная валидация)"},
                "time_budget_s": {"type": "number", "description": "Лимит на весь подбор"},
                "target_transform": {"type": "string", "enum": ["none", "log1p"]},
                "hypothesis": {"type": "string"},
            },
            "required": ["model"],
        },
    },
    {
        "name": "add_features",
        "description": "Добавить признаки из грамматики (считаются без таргета). Принимаются партией, только если "
                       "быстрая модель на тех же фолдах устойчиво лучше; иначе данные не меняются. Формулы 'a/b', "
                       "'a-b', 'a*b', 'a+b' или словари: {op: ratio|diff|prod|sum, a, b}, {op: group_mean|group_std|"
                       "group_dev|group_count, col, by}, {op: count, col}, {op: cross, a, b}, {op: nan_count}, "
                       "{op: as_category, col} — числовой код как категория (id, MCC, регион).",
        "parameters": {
            "type": "object",
            "properties": {
                "features": {"type": "array", "items": {}, "description": "До 40 спецификаций признаков"},
                "hypothesis": {"type": "string", "description": "Почему эти признаки должны помочь (смысл из task.md)"},
                "model": {"type": "string", "enum": list(MODELS)},
            },
            "required": ["features"],
        },
    },
    {
        "name": "blend",
        "description": "Смешать предсказания нескольких успешных полных запусков текущей схемы (разные модели "
                       "ошибаются по-разному). Веса подбираются по OOF; смесь создаётся, только если она лучше лучшего "
                       "одиночного запуска. Затем finalize(run_id смеси).",
        "parameters": {
            "type": "object",
            "properties": {
                "run_ids": {"type": "array", "items": {"type": "string"},
                            "description": "Запуски для смеси; по умолчанию — до 4 лучших"},
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
