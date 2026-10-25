"""Схемы валидации и выбор схемы по паспорту задачи.

suggest_validation() только предлагает схему и объясняет причину; вызывающий код
может заменить её своей. make_splits() превращает схему в список (train_idx, valid_idx).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from ml_data import date_days, to_dates

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
    """Одинаковые по признакам строки получают один и тот же номер группы.

    Номера идут по первому появлению строки, а не по значению хэша: от номеров зависит, в какой фолд
    попадёт группа, и при добавлении признаков (группы те же) фолды не должны меняться."""
    return pd.factorize(pd.util.hash_pandas_object(X, index=False))[0]


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
            and (
                info.get("kind") == "datetime" and info["n_unique"] >= 3
                or (info["dtype"].startswith(("int", "uint")) or info.get("integer_valued"))
                and 3 <= info["n_unique"] <= 50
            )
        ):
            direction = "forward" if side == "above" else "backward"
            where = "более поздних" if direction == "forward" else "более ранних"
            return {
                "strategy": "temporal", "column": name, "direction": direction,
                "n_splits": min(3, info["n_unique"] - 1), "classification": classification,
                "reason": f"{share:.0%} строк test лежат {'выше' if side == 'above' else 'ниже'} границ train "
                          f"по {'дате' if info.get('kind') == 'datetime' else 'целочисленному признаку'} {name!r} "
                          f"({info['n_unique']} значений): проверяем на {where} значениях",
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


_PER_VALUE_LIMIT = 50  # до стольких значений временного столбца проверяем по одному значению (году)


def _time_values(series: pd.Series) -> np.ndarray:
    """Числа как есть, текстовые даты — в днях от 1970-01-01."""
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float)
    return date_days(to_dates(series)).to_numpy(dtype=float)


def _temporal(plan, X):
    column = plan.get("column")
    if column not in X.columns:
        raise ValidationError(f"столбца {column!r} для временной схемы нет в признаках")
    values = _time_values(X[column])
    known = values[~np.isnan(values)]
    uniq = np.unique(known)
    k = min(int(plan.get("n_splits", 3)), len(uniq) - 1)
    if k < 1:
        raise ValidationError(f"в {column!r} меньше двух значений: временная схема невозможна")
    backward = plan.get("direction", "forward") == "backward"
    if len(uniq) <= _PER_VALUE_LIMIT:
        # каждая часть — одно значение (например, год); учимся на более ранних (или поздних)
        blocks = [(v, v) for v in (uniq[:k][::-1] if backward else uniq[-k:])]
    else:
        # много значений (даты): k последних (или первых) блоков по 1/(k+2) строк
        edges = np.quantile(known, np.linspace(0, 1, k + 3))
        pairs = list(zip(edges[:-1], edges[1:]))
        blocks = pairs[:k][::-1] if backward else pairs[-k:]
    splits = []
    for low, high in blocks:
        last = high == known.max() or low == high
        valid = np.flatnonzero((values >= low) & ((values <= high) if last else (values < high)))
        if backward:
            train = np.flatnonzero(values > high) if low == high else np.flatnonzero(values >= high)
        else:
            train = np.flatnonzero(values < low)
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
        n_groups = len(np.unique(groups))
        n_splits = min(n_splits, n_groups)
        if n_splits < 2:
            raise ValidationError("меньше двух групп: групповая схема невозможна")
        if y is not None and plan.get("classification"):
            return _group_stratified(np.asarray(y), groups, n_splits, seed)
        return Splits(GroupKFold(n_splits).split(X, y, groups), "group", n_splits)
    if strategy == "stratified":
        if y is None:
            raise ValidationError("для стратификации нужен y")
        strata, note = _strata(np.asarray(y), n_splits)
        folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, strata)
        return Splits(folds, "stratified", n_splits, note)
    return Splits(KFold(n_splits, shuffle=True, random_state=seed).split(X), "kfold", n_splits)


def _strata(y: np.ndarray, n_splits: int):
    """Метки для стратификации: классы реже n_splits строк объединяются в общую страту,
    чтобы не уменьшать число частей (раньше класс из 2 строк оставлял 2 части)."""
    labels = pd.Series(y).astype(str)
    counts = labels.value_counts()
    rare = counts[counts < n_splits].index
    if len(rare) == 0:
        return labels.to_numpy(), None
    strata = labels.where(~labels.isin(rare), "__rare__")
    if (strata == "__rare__").sum() < n_splits:
        strata = strata.replace("__rare__", counts.index[0])  # даже вместе мало: к самому частому классу
    note = f"классы {sorted(map(str, rare))[:10]} реже {n_splits} строк объединены в общую страту"
    return strata.to_numpy(), note


def _group_stratified(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> Splits:
    """Группы одинаковых строк не разрываются, доли классов сохраняются.

    StratifiedGroupKFold медленный на больших данных, поэтому он применяется только к строкам,
    у которых есть копии; строки без копий делятся обычным StratifiedKFold, и части объединяются."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    copied = counts[inverse] > 1
    single, multi = np.flatnonzero(~copied), np.flatnonzero(copied)
    valid = [[] for _ in range(n_splits)]
    if len(single) >= n_splits:
        strata, _ = _strata(y[single], n_splits)
        for i, (_, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(single, strata)):
            valid[i].append(single[va])
    else:
        for i, idx in enumerate(single):
            valid[i % n_splits].append(np.array([idx]))
    if len(multi):
        multi_groups = groups[multi]
        if len(np.unique(multi_groups)) >= n_splits:
            folds = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(multi, y[multi], multi_groups)
            for i, (_, va) in enumerate(folds):
                valid[i].append(multi[va])
        else:
            for i, g in enumerate(np.unique(multi_groups)):
                valid[i % n_splits].append(multi[multi_groups == g])
    everything = np.arange(len(y))
    pairs = []
    for parts in valid:
        va = np.sort(np.concatenate(parts)) if parts else np.array([], dtype=int)
        if len(va):
            pairs.append((np.setdiff1d(everything, va), va))
    return Splits(pairs, "group_stratified", len(pairs))
