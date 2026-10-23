"""Инструменты агента: оценка модели на валидации (train_eval) и финальное обучение (fit_predict).

Оба вызова возвращают результат, а не поднимают исключение для ожидаемых проблем:
статус "error" и понятное сообщение. Ничего не пишется на диск.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import ml_metrics
import ml_validation
from ml_data import LabelCodec, SplitData, TabularPreprocessor
from ml_models import BoostingModel

TARGET_TRANSFORMS = (None, "log1p")


@dataclass
class EvalResult:
    status: str                       # "ok" | "error"
    model: str = ""
    task_type: str = ""
    metric: str = ""
    higher_is_better: Optional[bool] = None
    score: Optional[float] = None     # среднее по частям валидации
    oof_score: Optional[float] = None  # метрика по всем проверенным строкам сразу
    fold_scores: list = field(default_factory=list)
    best_iterations: list = field(default_factory=list)
    seconds: float = 0.0
    validation: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    error: Optional[str] = None
    classes: Optional[list] = None
    oof_index: Optional[np.ndarray] = None   # позиции строк train, попавшие в проверку
    oof_pred: Optional[np.ndarray] = None    # значения (регрессия) или вероятности (n, K) в порядке oof_index

    def to_dict(self) -> dict:
        """JSON-безопасная сводка без массивов."""
        def num(v):
            return None if v is None or not np.isfinite(v) else float(v)
        return {
            "status": self.status, "model": self.model, "task_type": self.task_type, "metric": self.metric,
            "higher_is_better": self.higher_is_better, "score": num(self.score), "oof_score": num(self.oof_score),
            "fold_scores": [num(s) for s in self.fold_scores], "best_iterations": list(self.best_iterations),
            "seconds": round(self.seconds, 2), "validation": self.validation, "params": self.params,
            "notes": self.notes, "error": self.error,
        }


def _error(message: str, **kwargs) -> EvalResult:
    return EvalResult(status="error", error=message, **kwargs)


def _prepare_target(y: pd.Series, task_type: str, transform):
    """Возвращает (кодек или None, числовой таргет для модели)."""
    if task_type == "classification":
        codec = LabelCodec().fit(y)
        return codec, codec.encode(y)
    values = y.to_numpy(dtype=float)
    if transform == "log1p":
        if (values <= -1).any():
            raise ValueError("log1p невозможен: в таргете есть значения <= -1")
        values = np.log1p(values)
    return None, values


def _inverse(values: np.ndarray, task_type: str, transform):
    return np.expm1(values) if task_type == "regression" and transform == "log1p" else values


def _score(metric: ml_metrics.Metric, y_true, pred, task_type: str, n_classes: int) -> float:
    if task_type == "regression":
        return metric(y_true, pred)
    if metric.needs == "labels":
        return metric(y_true, pred.argmax(axis=1))
    return metric(y_true, pred, labels=list(range(n_classes)))


def _resolve_task_and_metric(profile: dict, task_type, metric):
    kind = task_type or profile.get("task", {}).get("type")
    if kind in ("binary", "multiclass"):
        kind = "classification"
    if kind not in ("regression", "classification"):
        raise ValueError(f"тип задачи {kind!r} пока не поддержан инструментами (нужна регрессия или классификация)")
    name = metric or profile.get("metric", {}).get("primary") or ("rmse" if kind == "regression" else "f1_macro")
    resolved = ml_metrics.get_metric(name)
    if resolved.task != kind:
        raise ValueError(f"метрика {resolved.name!r} относится к задаче {resolved.task}, а задача — {kind}")
    return kind, resolved


def train_eval(split: SplitData, profile: dict, model: str = "lightgbm", metric: Optional[str] = None,
               validation: Optional[dict] = None, params: Optional[dict] = None,
               task_type: Optional[str] = None, target_transform: Optional[str] = None,
               n_jobs: Optional[int] = None, seed: int = 0, time_budget: Optional[float] = None,
               max_rounds: int = 3000, learning_rate: float = 0.05, subsample: Optional[float] = None,
               ) -> EvalResult:
    """Оценить модель на валидации. Данные в каждой части готовятся только по её train-строкам."""
    started = time.time()
    try:
        kind, resolved = _resolve_task_and_metric(profile, task_type, metric)
        if target_transform not in TARGET_TRANSFORMS:
            raise ValueError(f"преобразование таргета {target_transform!r} не поддержано: {TARGET_TRANSFORMS}")
        if not isinstance(split.y_train, pd.Series):
            raise ValueError("несколько таргетов пока не поддержаны")
        base = dict(model=model, task_type=kind, metric=resolved.name,
                    higher_is_better=resolved.higher_is_better, params=dict(params or {}))
        X, y = split.X_train, split.y_train
        notes = []
        if subsample is not None and subsample < 1.0:
            keep = np.sort(np.random.default_rng(seed).choice(len(X), max(2, int(len(X) * subsample)), replace=False))
            X, y = X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)
            notes.append(f"использована случайная подвыборка train: {len(X)} строк")
        codec, target = _prepare_target(y, kind, target_transform)
        n_classes = len(codec.classes_) if codec else 0
        plan = validation or ml_validation.suggest_validation(profile, X)
        splits = ml_validation.make_splits(plan, X, target if kind == "classification" else None, seed=seed)

        fold_scores, iterations = [], []
        oof_index, oof_parts = [], []
        for number, (tr, va) in enumerate(splits, 1):
            left = len(splits) - number + 1
            budget = None if time_budget is None else max(1.0, (time_budget - (time.time() - started)) / left)
            try:
                pre = TabularPreprocessor()
                X_tr, X_va = pre.fit_transform(X.iloc[tr]), pre.transform(X.iloc[va])
                boosting = BoostingModel(
                    model, kind, n_classes=n_classes or None, params=params, n_jobs=n_jobs, seed=seed,
                    max_rounds=max_rounds, learning_rate=learning_rate, time_budget=budget,
                )
                boosting.fit(X_tr, target[tr], X_va, target[va])
                raw = boosting.predict(X_va)
                truth = target[va] if kind == "classification" else y.to_numpy(dtype=float)[va]
                pred = _inverse(raw, kind, target_transform)
                fold_scores.append(_score(resolved, truth, pred, kind, n_classes))
                iterations.append(boosting.best_iteration_)
                oof_index.append(va)
                oof_parts.append(pred)
                if boosting.stopped_by_time_:
                    notes.append(f"часть {number}: обучение остановлено по бюджету времени")
            except Exception as error:  # одна неудачная часть не должна ронять всю оценку
                notes.append(f"часть {number} пропущена: {type(error).__name__}: {error}")
        if not fold_scores:
            return _error("ни одна часть валидации не посчитана; " + "; ".join(notes[-3:]),
                          seconds=time.time() - started, validation=plan, notes=notes, **base)
        index = np.concatenate(oof_index)
        oof = np.concatenate(oof_parts)
        truth_all = target[index] if kind == "classification" else y.to_numpy(dtype=float)[index]
        return EvalResult(
            status="ok", score=float(np.mean(fold_scores)),
            oof_score=_score(resolved, truth_all, oof, kind, n_classes),
            fold_scores=[float(s) for s in fold_scores], best_iterations=iterations,
            seconds=time.time() - started, validation=plan, notes=notes,
            classes=[str(c) for c in codec.classes_] if codec else None,
            oof_index=index, oof_pred=oof, **base,
        )
    except Exception as error:
        return _error(f"{type(error).__name__}: {error}", model=model, seconds=time.time() - started,
                      notes=[traceback.format_exc(limit=3)])


def rounds_from(result: EvalResult, margin: float = 1.15) -> Optional[int]:
    """Число раундов для финального обучения на всём train: среднее лучших итераций с запасом."""
    if result.status != "ok" or not result.best_iterations:
        return None
    return max(1, int(round(float(np.mean(result.best_iterations)) * margin)))


def fit_predict(split: SplitData, profile: dict, model: str = "lightgbm", n_rounds: Optional[int] = None,
                params: Optional[dict] = None, task_type: Optional[str] = None,
                target_transform: Optional[str] = None, n_jobs: Optional[int] = None, seed: int = 0,
                time_budget: Optional[float] = None, max_rounds: int = 3000, learning_rate: float = 0.05) -> dict:
    """Обучить на всём train и предсказать test.

    Регрессия: pred (n,) в исходной шкале таргета. Классификация: proba (n, K) и classes —
    порядок столбцов; pred содержит метки классов (по наибольшей вероятности)."""
    started = time.time()
    try:
        kind, _ = _resolve_task_and_metric(profile, task_type, None)
        if not isinstance(split.y_train, pd.Series):
            raise ValueError("несколько таргетов пока не поддержаны")
        codec, target = _prepare_target(split.y_train, kind, target_transform)
        pre = TabularPreprocessor()
        X_tr, X_te = pre.fit_transform(split.X_train), pre.transform(split.X_test)
        boosting = BoostingModel(
            model, kind, n_classes=len(codec.classes_) if codec else None, params=params, n_jobs=n_jobs,
            seed=seed, max_rounds=max_rounds, learning_rate=learning_rate, time_budget=time_budget,
        )
        boosting.fit(X_tr, target, n_rounds=n_rounds)
        raw = boosting.predict(X_te)
        out = {"status": "ok", "model": model, "task_type": kind, "n_rounds": boosting.best_iteration_,
               "seconds": time.time() - started, "warnings": list(pre.warnings_) + list(split.warnings)}
        if kind == "regression":
            out["pred"] = _inverse(raw, kind, target_transform)
        else:
            out.update(proba=raw, classes=list(codec.classes_), pred=codec.decode(raw.argmax(axis=1)))
        return out
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}", "seconds": time.time() - started}
