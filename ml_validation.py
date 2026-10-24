"""Схемы валидации и выбор схемы по паспорту задачи.

suggest_validation() только предлагает схему и объясняет причину; вызывающий код
может заменить её своей. make_splits() превращает схему в список (train_idx, valid_idx).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

STRATEGIES = ("temporal", "group", "stratified", "kfold")


class ValidationError(ValueError):
    """Схему валидации нельзя построить на этих данных."""


class Splits(list):
    """Список пар (train_idx, valid_idx) и схема, которая применилась на самом деле.

    Она может отличаться от запрошенной: например, стратификация невозможна при классе
    из одной строки, и тогда используется обычный KFold; note объясняет почему."""

    def __init__(self, pairs, strategy, n_splits, note=None):
        super().__init__(pairs)
        self.strategy, self.n_splits, self.note = strategy, n_splits, note

    def effective(self) -> dict:
        return {"strategy": self.strategy, "n_splits": len(self), "note": self.note}


def row_groups(X: pd.DataFrame) -> np.ndarray:
    """Одинаковые по признакам строки получают один и тот же номер группы."""
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def suggest_validation(profile: dict, X_train: pd.DataFrame = None, n_splits: int = 5,
                       duplicate_threshold: float = 0.05) -> dict:
    """Предложить схему: temporal, group, stratified или kfold, с причиной."""
    classification = profile["task"]["type"] in ("binary", "multiclass")
    columns = {c["name"]: c for c in profile.get("features", {}).get("columns", [])}
    train_test = profile.get("train_test", {})
    out_of_range = train_test.get("out_of_train_range_share", {})
    sides = train_test.get("out_of_train_range_side", {})
    for name, share in out_of_range.items():
        info = columns.get(name)
        side = sides.get(name, "above")
        if (
            share >= 0.9 and info is not None and side in ("above", "below")
            and (info["dtype"].startswith(("int", "uint")) or info.get("integer_valued"))
            and 3 <= info["n_unique"] <= 50
        ):
            direction = "forward" if side == "above" else "backward"
            where = "более поздних" if direction == "forward" else "более ранних"
            return {
                "strategy": "temporal", "column": name, "direction": direction,
                "n_splits": min(3, info["n_unique"] - 1), "classification": classification,
                "reason": f"{share:.0%} строк test лежат {'выше' if side == 'above' else 'ниже'} границ train "
                          f"по целочисленному признаку {name!r} ({info['n_unique']} значений): "
                          f"проверяем на {where} значениях",
            }
    if X_train is not None and len(X_train):
        shared = float(X_train.duplicated(keep=False).mean())
        if shared >= duplicate_threshold:
            return {
                "strategy": "group", "column": None, "n_splits": n_splits, "classification": classification,
                "reason": f"{shared:.0%} строк train имеют полные копии по признакам: "
                          "делим по группам одинаковых строк, чтобы копии не попали в обе части",
            }
    if classification:
        return {"strategy": "stratified", "column": None, "n_splits": n_splits, "classification": True,
                "reason": "классификация: сохраняем доли классов в каждой части"}
    return {"strategy": "kfold", "column": None, "n_splits": n_splits, "classification": False,
            "reason": "особенностей, требующих другой схемы, не найдено"}


def _temporal(plan, X):
    column = plan.get("column")
    if column not in X.columns:
        raise ValidationError(f"столбца {column!r} для временной схемы нет в признаках")
    values = X[column].to_numpy(dtype=float)
    uniq = np.unique(values[~np.isnan(values)])
    k = min(int(plan.get("n_splits", 3)), len(uniq) - 1)
    if k < 1:
        raise ValidationError(f"в {column!r} меньше двух значений: временная схема невозможна")
    backward = plan.get("direction", "forward") == "backward"
    splits = []
    for v in (uniq[:k][::-1] if backward else uniq[-k:]):
        train = np.flatnonzero(values > v) if backward else np.flatnonzero(values < v)
        valid = np.flatnonzero(values == v)
        if len(train) and len(valid):
            splits.append((train, valid))
    return Splits(splits, "temporal", len(splits))


def make_splits(plan: dict, X: pd.DataFrame, y=None, seed: int = 0) -> list:
    """plan — словарь из suggest_validation() или собранный вручную (strategy, n_splits, column,
    classification). Возвращает список пар индексов-позиций (train_idx, valid_idx)."""
    strategy = plan.get("strategy")
    if strategy not in STRATEGIES:
        raise ValidationError(f"неизвестная схема {strategy!r}; доступны: {STRATEGIES}")
    n_rows, n_splits = len(X), int(plan.get("n_splits", 5))
    if n_rows < 2:
        raise ValidationError("для валидации нужно хотя бы две строки")
    if strategy == "temporal":
        return _temporal(plan, X)
    n_splits = max(2, min(n_splits, n_rows))
    if strategy == "group":
        groups = row_groups(X)
        n_splits = min(n_splits, len(np.unique(groups)))
        if n_splits < 2:
            raise ValidationError("меньше двух групп: групповая схема невозможна")
        if y is not None and plan.get("classification"):
            folds = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups)
            return Splits(folds, "group_stratified", n_splits)
        return Splits(GroupKFold(n_splits).split(X, y, groups), "group", n_splits)
    note = None
    if strategy == "stratified":
        if y is None:
            raise ValidationError("для стратификации нужен y")
        smallest = int(pd.Series(np.asarray(y)).value_counts().min())
        if smallest >= 2:
            if smallest < n_splits:
                note = f"частей {smallest} вместо {n_splits}: в самом редком классе {smallest} строк"
            n_splits = min(n_splits, smallest)
            folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
            return Splits(folds, "stratified", n_splits, note)
        note = "стратификация невозможна: есть класс из одной строки; использован обычный KFold"
    return Splits(KFold(n_splits, shuffle=True, random_state=seed).split(X), "kfold", n_splits, note)

