"""Инструменты, которые вызывает LLM: JSON на входе и выходе, внутри — ML-ядро (ml_*).

Session хранит состояние одного запуска: данные, выбранную постановку, эксперименты,
лучший результат и записанный ответ. Методы tool_* не бросают исключений наружу:
ошибка возвращается словарём {"ok": false, "error": ...}, чтобы LLM могла её учесть.
Несколько таргетов обучаются по очереди одной конфигурацией; score — среднее по таргетам.
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
PROBA_METRICS = ("roc_auc", "logloss")
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
_CATBOOST_DROP = {"num_leaves", "min_child_samples", "min_data_in_leaf", "min_child_weight", "gamma",
                  "reg_alpha", "lambda_l1", "class_weight", "subsample_freq", "bagging_freq", "max_bin_by_feature"}
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
_EXTRA_PATTERNS = {"rmsle": r"\brmsle\b|root mean squared log|rmse\s+(?:между\s+|of\s+(?:the\s+)?)(?:log|логарифм)",
                   "msle": r"\bmsle\b|mean squared log", "mape": r"\bmape\b|mean absolute percentage",
                   "smape": r"\bsmape\b|symmetric mean absolute"}
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


def detect_extra_metric(text: str, core_primary: Optional[str] = None) -> Optional[str]:
    """Метрика из EXTRA_METRICS, упомянутая в условии (её не находит ml_schema).

    Если ядро нашло основную метрику, своя побеждает, только если она ближе к фразе
    «основная метрика / метрика — / качество —», чем метрика ядра."""
    text = (text or "").replace("*", "").replace("`", "")
    hits = []
    for name, pattern in _EXTRA_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            hits.append((match.start(), name))
    if not hits:
        return None
    if not core_primary:
        return min(hits)[1]
    core_pattern = getattr(ml_schema, "_METRIC_PATTERNS", {}).get(core_primary)
    phrase = re.compile(getattr(ml_schema, "_PRIMARY_PHRASE").pattern + r"|качеств\w*\s*[—:-]", re.IGNORECASE)
    for found in phrase.finditer(text):
        after = text[found.end(): found.end() + 150]
        core = re.search(core_pattern, after, flags=re.IGNORECASE) if core_pattern else None
        extra = [(re.search(_EXTRA_PATTERNS[n], after, flags=re.IGNORECASE), n) for _, n in hits]
        extra = [(m.start(), n) for m, n in extra if m]
        if extra and (core is None or min(extra)[0] <= core.start()):
            return min(extra)[1]
        if core is not None:
            return None
    return None


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
        self._fits = {}                   # run_id -> обученные на всём train предсказания (для смесей)

    # ---- время ---------------------------------------------------------------
    def time_left(self) -> float:
        return self.deadline - time.monotonic()

    def estimate_final_seconds(self, run_id) -> float:
        """Сколько займёт финальное обучение: время одной части валидации с запасом на весь train."""
        run = self.runs[run_id]
        if run_id in self._fits:
            return 0.0
        if run["config"]["model"] == "blend":
            return sum(self.estimate_final_seconds(m) for m in run["config"]["members"])
        result = run["result"]
        folds = max(1, len(result.get("fold_scores") or []))
        per_fold = result.get("seconds", 60.0) / folds / (run["config"]["subsample"] or 1.0)
        return 1.6 * per_fold + 30.0

    def can_finalize(self, run_id) -> bool:
        return self.estimate_final_seconds(run_id) <= self.time_left() + self.final_slack

    # ---- обзор задачи -------------------------------------------------------
    def overview(self, max_columns: int = 60) -> dict:
        schema = self.profile["schema"]
        extra = self.suggested_extra_metric()
        return {
            "profile_text": ml_profile.format_profile(self.profile, max_columns=max_columns),
            "suggested": {
                "target_cols": schema["target_cols"], "id_cols": schema["id_cols"],
                "output_columns": schema["output_columns"], "confidence": schema["confidence"],
                "task_type": self.profile.get("task", {}).get("type"),
                "metric": extra or self.profile["metric"].get("primary"),
                "metric_source": "своя метрика агента" if extra else self.profile["metric"].get("source"),
                "answer_looks_like_probability": self.sample_suggests_proba(),
            },
            "all_train_columns": [str(c) for c in self.data.train.columns] if self.data.train is not None else [],
            "extra_csv": [ml_io.display_path(p, self.task_dir) for p in self.data.files.extra_csv],
            "available": {"models": list(MODELS), "metrics": ml_metrics.available_metrics() + list(EXTRA_METRICS)
                          + ["f1 (binary)", "precision (binary)", "recall (binary)"],
                          "task_types": list(TASK_TYPES), "validation": list(ml_validation.STRATEGIES),
                          "target_transforms": [t for t in ml_tools.TARGET_TRANSFORMS if t]},
            "time_left_s": round(self.time_left()),
        }

    def suggested_extra_metric(self) -> Optional[str]:
        """Своя метрика агента, если условие называет её основной (RMSLE, MAPE, бинарная F1)."""
        core = self.profile["metric"].get("primary")
        extra = detect_extra_metric(self.data.text, core)
        if not extra and not core and re.search(r"\bf1\b", self.data.text or "", re.I):
            extra = "f1"
        return extra

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
        self.setup = {"schema": schema, "splits": splits, "split": main, "profile": profile, "task_type": task_type,
                      "metric": resolved.name, "metric_obj": resolved, "higher_is_better": resolved.higher_is_better,
                      "prediction_format": prediction_format, "positive": positive, "reason": reason}
        self.plan, self.plan_key = plan, _validation_key(plan)
        self.runs, self._fits = {}, {}
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
        config = {"model": model, "params": params, "learning_rate": float(learning_rate),
                  "target_transform": target_transform, "max_rounds": int(max_rounds), "subsample": subsample}
        self.runs[run_id] = {"config": config, "result": summary, "eval": evals, "key": key, "hypothesis": hypothesis}
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
            if best is None or self._is_better(self._score(run_id), self._score(best)):
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
        best = self.best_run() if self.setup else None
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

    def _fit_run(self, run_id) -> dict:
        """Обучить запуск на всём train (с кэшем): таргет -> {pred, proba, classes, notes, truncated}."""
        if run_id in self._fits:
            return self._fits[run_id]
        run, setup = self.runs[run_id], self.setup
        config = run["config"]
        if config["model"] == "blend":
            members = {m: self._fit_run(m) for m in config["members"]}
            fits = {}
            for target in setup["splits"]:
                weights = config["weights"][target]
                parts = [members[m][target] for m in config["members"]]
                if setup["task_type"] == "regression":
                    pred = sum(w * np.asarray(p["pred"], dtype=float) for w, p in zip(weights, parts))
                    fits[target] = {"pred": pred, "proba": None, "classes": None}
                else:
                    proba = sum(w * np.asarray(p["proba"], dtype=float) for w, p in zip(weights, parts))
                    classes = parts[0]["classes"]
                    fits[target] = {"pred": np.asarray(classes)[proba.argmax(axis=1)], "proba": proba, "classes": classes}
                fits[target]["notes"] = []
                fits[target]["truncated"] = any(p["truncated"] for p in parts)
            self._fits[run_id] = fits
            return fits
        fits = {}
        for target, split in setup["splits"].items():
            ev = run["eval"][target]
            n_rounds = ml_tools.rounds_from(ev)
            budget = max(10.0, self.time_left() + self.final_slack)
            self.log(f"[fit {run_id}] {target}: {config['model']} n_rounds={n_rounds} budget={budget:.0f}s")
            out = ml_tools.fit_predict(
                split, setup["profile"], model=config["model"], n_rounds=n_rounds, params=config["params"],
                task_type=setup["task_type"], target_transform=config["target_transform"], n_jobs=self.n_jobs,
                seed=self.seed, time_budget=budget, max_rounds=config["max_rounds"],
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
        self._fits[run_id] = fits
        return fits

    def _finalize(self, run_id, class_weights="auto") -> dict:
        run, setup = self.runs[run_id], self.setup
        try:
            fits = self._fit_run(run_id)
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
        frame, frame_notes = self._assemble(setup, predictions)
        path = write_submission(frame, self.task_dir, len(setup["split"].X_test), setup["schema"].id_cols)
        self.written = {"run_id": run_id, "score": self._score(run_id), "model": run["config"]["model"]}
        self.log(f"[finalize {run_id}] записан {path} ({len(frame)} строк), cv={self._score(run_id)}")
        return {"ok": True, "path": str(path), "rows": len(frame), "columns": list(frame.columns),
                "run_id": run_id, "cv_score": self._score(run_id),
                "class_weights": {t: w for t, w in weights_info.items() if w} or None,
                "notes": notes + frame_notes}

    def _class_weights(self, run_id, target, fit, mode):
        """Веса классов под Macro F1 по OOF-вероятностям; применяются, только если честно помогают."""
        setup, run = self.setup, self.runs[run_id]
        split = setup["splits"][target]
        labels = fit["pred"]
        if mode == "off" or setup["metric"] != "f1_macro" or run["config"]["subsample"] or fit["proba"] is None:
            return labels, None
        index, oof = self._oof(run_id, target)
        if oof is None or index is None or len(index) < 200:
            return labels, None
        try:
            codec = ml_data.LabelCodec().fit(split.y_train)
            truth = codec.encode(split.y_train.iloc[index])
            groups = ml_validation.row_groups(split.X_train.iloc[index])
            choice = ml_postprocess.choose_class_weights(oof, truth, groups=groups)
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
        """Смесь нескольких запусков: веса по OOF-предсказаниям на общей схеме валидации."""
        if self.setup is None:
            return {"ok": False, "error": "сначала вызовите set_task"}
        pool = [r for r in self._ok_runs(full_only=True) if self.runs[r]["key"] == self.plan_key
                and self.runs[r]["config"]["model"] != "blend"]
        if run_ids:
            missing = [r for r in run_ids if r not in pool]
            if missing:
                return {"ok": False, "error": f"{missing} не годятся: нужны успешные полные запуски текущей схемы {pool}"}
            members = list(dict.fromkeys(run_ids))
        else:
            members = sorted(pool, key=lambda r: self._score(r), reverse=self.setup["higher_is_better"])[:4]
        if len(members) < 2:
            return {"ok": False, "error": f"для смеси нужно хотя бы два успешных полных запуска; есть {pool}"}
        grid = _simplex_grid(len(members), 10 if len(members) <= 4 else 5)
        weights, oof, per_target, gain = {}, {}, {}, []
        for target in self.setup["splits"]:
            parts = [self._oof(m, target) for m in members]
            common = parts[0][0]
            for index, _ in parts[1:]:
                common = np.intersect1d(common, index)
            if len(common) < 20:
                return {"ok": False, "error": "у запусков слишком мало общих OOF-строк"}
            aligned = []
            for index, pred in parts:
                order = pd.Series(np.arange(len(index)), index=index).loc[common].to_numpy()
                aligned.append(np.asarray(pred, dtype=float)[order])
            singles = [self._oof_metric(target, common, p) for p in aligned]
            best_w, best_s = None, None
            for w in grid:
                score = self._oof_metric(target, common, sum(wi * p for wi, p in zip(w, aligned)))
                if best_s is None or self._is_better(score, best_s):
                    best_w, best_s = w, score
            best_single = singles[0]
            for value in singles[1:]:
                if self._is_better(value, best_single):
                    best_single = value
            weights[target] = [float(w) for w in best_w]
            oof[target] = (common, sum(wi * p for wi, p in zip(best_w, aligned)))
            per_target[target] = {"blend": best_s, "best_single": best_single,
                                  "weights": dict(zip(members, [round(float(w), 2) for w in best_w]))}
            gain.append(best_s - best_single)
        blended = float(np.mean([v["blend"] for v in per_target.values()]))
        single = float(np.mean([v["best_single"] for v in per_target.values()]))
        if not self._is_better(blended, single):
            return {"ok": False, "error": "смесь не лучше лучшего одиночного запуска на OOF", "per_target": per_target}
        self._counter += 1
        run_id = f"b{self._counter}"
        seconds = float(sum(self.runs[m]["result"].get("seconds", 0) for m in members))
        self.runs[run_id] = {
            "config": {"model": "blend", "members": members, "weights": weights, "subsample": None, "params": {},
                       "learning_rate": None, "target_transform": None, "max_rounds": 0},
            "result": {"status": "ok", "score": blended, "oof_score": blended, "fold_scores": [], "seconds": seconds,
                       "metric": self.setup["metric"], "model": "blend"},
            "eval": None, "oof": oof, "key": self.plan_key, "hypothesis": reason}
        self.log(f"[blend {run_id}] {members} -> {blended} (лучший одиночный {single})")
        return {"ok": True, "run_id": run_id, "members": members, "oof_score": blended, "best_single_oof": single,
                "per_target": per_target, "best_run_id": self.best_run(),
                "note": "обучить и записать смесь: finalize(run_id)"}

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
                     "metric": self.suggested_extra_metric() or self.profile["metric"].get("primary") or "",
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
                                          "colsample_bytree, reg_lambda; xgboost max_depth, min_child_weight; "
                                          "catboost depth, l2_leaf_reg"},
                "learning_rate": {"type": "number"},
                "target_transform": {"type": "string", "enum": ["none", "log1p"],
                                     "description": "Только регрессия: log1p для неотрицательного скошенного таргета "
                                                    "и для метрик RMSLE/MSLE"},
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
