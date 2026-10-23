"""Реестр метрик: имя, направление оптимизации и тип входа.

Метрика берётся по имени или алиасу из task.md («RMSE», «Macro F1», ...).
Некорректные входы (NaN, разная длина, неизвестное имя) дают MetricError,
а не молчаливую подмену результата.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn import metrics as skm


class MetricError(ValueError):
    """Неизвестная метрика или некорректные входы метрики."""


def _rmse(yt, yp, labels):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def _mse(yt, yp, labels):
    return float(np.mean((yt - yp) ** 2))


def _mae(yt, yp, labels):
    return float(np.mean(np.abs(yt - yp)))


def _r2(yt, yp, labels):
    return float(skm.r2_score(yt, yp))


def _accuracy(yt, yp, labels):
    return float(skm.accuracy_score(yt, yp))


def _balanced_accuracy(yt, yp, labels):
    return float(skm.balanced_accuracy_score(yt, yp))


def _f1(average):
    def fn(yt, yp, labels):
        return float(skm.f1_score(yt, yp, average=average, labels=labels, zero_division=0))
    return fn


def _proba_2d(proba):
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.ndim != 2:
        raise MetricError("вероятности должны быть массивом (n,) или (n, k)")
    return proba


def _logloss(yt, proba, labels):
    return float(skm.log_loss(yt, _proba_2d(proba), labels=labels))


def _roc_auc(yt, proba, labels):
    proba = _proba_2d(proba)
    classes = np.asarray(labels) if labels is not None else np.unique(yt)
    if proba.shape[1] != len(classes):
        raise MetricError(
            f"столбцов вероятностей {proba.shape[1]}, а классов {len(classes)}: передайте labels"
        )
    if len(classes) == 2:
        return float(skm.roc_auc_score(yt == classes[1], proba[:, 1]))
    return float(
        skm.roc_auc_score(yt, proba, multi_class="ovr", average="macro", labels=classes)
    )


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable
    higher_is_better: bool
    task: str   # "regression" | "classification"
    needs: str  # "values" | "labels" | "proba"

    def __call__(self, y_true, y_pred, labels=None) -> float:
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        if len(yt) != len(yp):
            raise MetricError(f"{self.name}: длины не совпадают ({len(yt)} и {len(yp)})")
        if len(yt) == 0:
            raise MetricError(f"{self.name}: пустые данные")
        try:
            if self.needs == "values":
                yt, yp = self._numeric(yt, yp)
            elif self.needs == "labels":
                self._check_labels(yt, yp)
            else:
                if pd.isna(yt).any():
                    raise MetricError(f"{self.name}: в y_true есть пропуски")
                if not np.isfinite(np.asarray(yp, dtype=float)).all():
                    raise MetricError(f"{self.name}: в вероятностях есть NaN/inf")
            return self.fn(yt, yp, labels)
        except MetricError:
            raise
        except ValueError as error:
            raise MetricError(f"{self.name}: {error}") from error

    def _numeric(self, yt, yp):
        if yp.ndim == 2 and yp.shape[1] == 1:
            yp = yp.ravel()
        if yt.ndim != 1 or yp.ndim != 1:
            raise MetricError(f"{self.name}: ожидаются одномерные массивы")
        yt, yp = yt.astype(float), yp.astype(float)
        if not np.isfinite(yt).all():
            raise MetricError(f"{self.name}: в y_true есть NaN/inf")
        if not np.isfinite(yp).all():
            raise MetricError(f"{self.name}: в предсказаниях есть NaN/inf")
        return yt, yp

    def _check_labels(self, yt, yp):
        if yt.ndim != 1 or yp.ndim != 1:
            raise MetricError(f"{self.name}: ожидаются одномерные метки")
        if pd.isna(yt).any() or pd.isna(yp).any():
            raise MetricError(f"{self.name}: в метках есть пропуски")


_METRICS = {
    m.name: m
    for m in (
        Metric("rmse", _rmse, False, "regression", "values"),
        Metric("mse", _mse, False, "regression", "values"),
        Metric("mae", _mae, False, "regression", "values"),
        Metric("r2", _r2, True, "regression", "values"),
        Metric("accuracy", _accuracy, True, "classification", "labels"),
        Metric("balanced_accuracy", _balanced_accuracy, True, "classification", "labels"),
        Metric("f1_macro", _f1("macro"), True, "classification", "labels"),
        Metric("f1_weighted", _f1("weighted"), True, "classification", "labels"),
        Metric("f1_micro", _f1("micro"), True, "classification", "labels"),
        Metric("logloss", _logloss, False, "classification", "proba"),
        Metric("roc_auc", _roc_auc, True, "classification", "proba"),
    )
}

# Ключи — нормализованные имена (нижний регистр, только буквы и цифры).
_ALIASES = {
    "rootmeansquarederror": "rmse",
    "meansquarederror": "mse",
    "meanabsoluteerror": "mae",
    "r2score": "r2",
    "rsquared": "r2",
    "acc": "accuracy",
    "macrof1": "f1_macro",
    "f1macro": "f1_macro",
    "weightedf1": "f1_weighted",
    "f1weighted": "f1_weighted",
    "microf1": "f1_micro",
    "f1micro": "f1_micro",
    "bacc": "balanced_accuracy",
    "crossentropy": "logloss",
    "logarithmicloss": "logloss",
    "auc": "roc_auc",
    "aucroc": "roc_auc",
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def get_metric(name) -> Metric:
    """Вернуть метрику по имени или алиасу; Metric возвращается как есть."""
    if isinstance(name, Metric):
        return name
    key = _normalize(name)
    for canonical in _METRICS:
        if _normalize(canonical) == key:
            return _METRICS[canonical]
    if key in _ALIASES:
        return _METRICS[_ALIASES[key]]
    raise MetricError(f"неизвестная метрика {name!r}; доступны: {', '.join(_METRICS)}")


def available_metrics() -> list[str]:
    return list(_METRICS)


def compute(name, y_true, y_pred, labels=None) -> float:
    return get_metric(name)(y_true, y_pred, labels)


def _missing(value) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def is_better(name, new, best) -> bool:
    """Строго ли `new` лучше `best` с учётом направления метрики.

    None и NaN (например, score неудачного эксперимента) не лучше ничего; любое число лучше них."""
    if _missing(new):
        return False
    if _missing(best):
        return True
    return new > best if get_metric(name).higher_is_better else new < best


def worst_value(name) -> float:
    return -np.inf if get_metric(name).higher_is_better else np.inf
