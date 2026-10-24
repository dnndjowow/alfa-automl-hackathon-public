"""Подстройка вероятностей под Macro F1: веса классов, подобранные по предсказаниям вне обучения.

Метка выбирается как argmax(proba * weights). Веса ищутся покоординатным спуском по
геометрической сетке; fit_class_weights() сообщает прирост на данных подбора, а
holdout_gain() — честный прирост на непересекающейся части.
"""

from __future__ import annotations

import numpy as np


def macro_f1_codes(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro F1 для целых меток; классы, которых нет ни в y_true, ни в y_pred, не учитываются (как в sklearn)."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    present = (support > 0) | (predicted > 0)
    denom = (support + predicted).astype(float)
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    return float(f1[present].mean())


def apply_class_weights(proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (np.asarray(proba) * np.asarray(weights)).argmax(axis=1)


def fit_class_weights(proba: np.ndarray, y_true: np.ndarray, rounds: int = 3, low: float = 0.2,
                      high: float = 5.0, steps: int = 25, min_gain: float = 1e-4) -> dict:
    """Вернуть {'weights', 'score_before', 'score_after'}; веса остаются 1, если выигрыша нет."""
    proba, y_true = np.asarray(proba, dtype=float), np.asarray(y_true, dtype=np.int64)
    if proba.ndim != 2 or len(proba) != len(y_true):
        raise ValueError("ожидаются proba (n, K) и y_true (n,)")
    n_classes = proba.shape[1]
    weights = np.ones(n_classes)
    best = before = macro_f1_codes(y_true, proba.argmax(axis=1), n_classes)
    grid = np.geomspace(low, high, steps)
    for _ in range(rounds):
        improved = False
        for k in range(n_classes):
            current = weights[k]
            top_value, top_score = current, best
            for candidate in grid:
                weights[k] = candidate
                score = macro_f1_codes(y_true, apply_class_weights(proba, weights), n_classes)
                if score > top_score + min_gain:
                    top_value, top_score = candidate, score
            weights[k] = top_value
            if top_score > best:
                best, improved = top_score, True
        if not improved:
            break
    return {"weights": weights, "score_before": before, "score_after": best}


def _holdout_split(n: int, fit_fraction: float, seed: int, groups=None):
    """Разделить строки на подбор и проверку; при groups одинаковые группы не попадают в обе части."""
    rng = np.random.default_rng(seed)
    if groups is None:
        order = rng.permutation(n)
        cut = int(n * fit_fraction)
        return order[:cut], order[cut:]
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError("groups должен иметь по значению на каждую строку")
    unique = rng.permutation(np.unique(groups))
    fit_groups = unique[: int(len(unique) * fit_fraction)]
    in_fit = np.isin(groups, fit_groups)
    return np.flatnonzero(in_fit), np.flatnonzero(~in_fit)


def holdout_gain(proba: np.ndarray, y_true: np.ndarray, fit_fraction: float = 0.6, seed: int = 0,
                 groups=None, **kwargs) -> dict:
    """Подобрать веса на одной части и измерить Macro F1 на другой: показывает, не переобучились ли веса.

    groups — номера групп одинаковых строк (ml_validation.row_groups): без них копии строк попадают
    в обе части, и прирост на проверке завышен."""
    proba, y_true = np.asarray(proba), np.asarray(y_true)
    fit_idx, test_idx = _holdout_split(len(y_true), fit_fraction, seed, groups)
    fitted = fit_class_weights(proba[fit_idx], y_true[fit_idx], **kwargs)
    n_classes = proba.shape[1]
    base = macro_f1_codes(y_true[test_idx], proba[test_idx].argmax(axis=1), n_classes)
    tuned = macro_f1_codes(y_true[test_idx], apply_class_weights(proba[test_idx], fitted["weights"]), n_classes)
    return {"holdout_before": base, "holdout_after": tuned, "gain": tuned - base, "weights": fitted["weights"]}


def choose_class_weights(proba: np.ndarray, y_true: np.ndarray, min_holdout_gain: float = 0.002,
                         repeats: int = 3, fit_fraction: float = 0.6, groups=None, **kwargs) -> dict:
    """Применять веса, только если средний прирост на отложенных частях не меньше порога.

    Иначе возвращает единичные веса: подбор на шуме не должен портить результат.
    kwargs уходят в fit_class_weights (rounds, low, high, steps, min_gain)."""
    proba = np.asarray(proba)
    gains = [holdout_gain(proba, y_true, fit_fraction=fit_fraction, seed=seed, groups=groups, **kwargs)["gain"]
             for seed in range(repeats)]
    mean_gain = float(np.mean(gains))
    if mean_gain < min_holdout_gain:
        return {"applied": False, "weights": np.ones(proba.shape[1]), "holdout_gain": mean_gain}
    fitted = fit_class_weights(proba, y_true, **kwargs)
    return {"applied": True, "weights": fitted["weights"], "holdout_gain": mean_gain,
            "score_before": fitted["score_before"], "score_after": fitted["score_after"]}
