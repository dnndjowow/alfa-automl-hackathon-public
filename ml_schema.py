"""Подсказки по схеме задачи: id, таргет, признаки, тип задачи и метрика.

Всё здесь — предложения, а не окончательные решения: вызывающий код может
переопределить их в resolve_schema(). Если подсказка неуверенная, resolve_schema
поднимает SchemaError вместо угадывания.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import ml_metrics

_MAX_INT_CLASSES = 20
_CLS_WORDS = re.compile(r"классификац|classif", re.IGNORECASE)
_REG_WORDS = re.compile(r"регресс|regress", re.IGNORECASE)
_ID_NAME = re.compile(r"id|index|row_?id|uid", re.IGNORECASE)

_METRIC_PATTERNS = {
    "rmse": r"\brmse\b|root mean squared error",
    "mse": r"\bmse\b",
    "mae": r"\bmae\b",
    "r2": r"\br2\b|r²|r-squared",
    "accuracy": r"\baccuracy\b",
    "balanced_accuracy": r"balanced[\s_-]*accuracy",
    "f1_macro": r"macro[\s_-]*f1|f1[\s_-]*macro|average\s*=\s*[\"']macro[\"']",
    "f1_weighted": r"weighted[\s_-]*f1|f1[\s_-]*weighted",
    "f1_micro": r"micro[\s_-]*f1|f1[\s_-]*micro",
    "logloss": r"log[\s_-]*loss|cross[\s_-]*entropy",
    "roc_auc": r"roc[\s_-]*auc|\bauc\b",
}
_PRIMARY_PHRASE = re.compile(
    r"основн\w*\s+метрик|метрик\w*\s+оценк|метрика\s*[—:-]|primary\s+metric|main\s+metric"
    r"|evaluation\s+metric|scoring\s+metric",
    re.IGNORECASE,
)


class SchemaError(ValueError):
    """Схему нельзя определить уверенно: нужно решение вызывающего кода."""


@dataclass(frozen=True)
class Schema:
    id_cols: tuple
    target_cols: tuple
    feature_cols: tuple
    output_columns: Optional[tuple] = None  # столбцы sample_submission, если он есть


def _dictionary_roles(dictionary) -> dict:
    if dictionary is None or not {"feature", "role"} <= set(dictionary.columns):
        return {}
    return dict(zip(dictionary["feature"].astype(str), dictionary["role"].astype(str).str.lower()))


def infer_schema(train, test=None, sample_submission=None, dictionary=None) -> dict:
    """Предложить id, таргеты и признаки по sample_submission, разнице train/test и словарю."""
    train_cols = list(train.columns) if train is not None else []
    test_cols = list(test.columns) if test is not None else []
    roles = _dictionary_roles(dictionary)
    notes, sources = [], {}

    id_cols = []
    if sample_submission is not None and test is not None and len(sample_submission) == len(test):
        id_cols = [
            c for c in sample_submission.columns
            if c in test.columns and np.array_equal(sample_submission[c].to_numpy(), test[c].to_numpy())
        ]
        if id_cols:
            sources["id"] = "значения sample_submission совпадают с test"
    if not id_cols:
        id_cols = [c for c, r in roles.items() if r == "id" and c in train_cols]
        if id_cols:
            sources["id"] = "data_dictionary"
    if not id_cols:
        id_cols = [c for c in test_cols if _ID_NAME.fullmatch(c)]
        if id_cols:
            sources["id"] = "по имени столбца"
            notes.append("id определён только по имени столбца")

    sample_targets = (
        [c for c in sample_submission.columns if c not in id_cols] if sample_submission is not None else []
    )
    options = [
        ("sample_submission", [c for c in sample_targets if c in train_cols]),
        ("train_minus_test", [c for c in train_cols if test is not None and c not in test_cols and c not in id_cols]),
        ("data_dictionary", [c for c, r in roles.items() if r == "target" and c in train_cols]),
    ]
    options = [(name, cols) for name, cols in options if cols]
    targets = options[0][1] if options else []
    agree = [name for name, cols in options if set(cols) == set(targets)]
    disagree = [(name, cols) for name, cols in options if set(cols) != set(targets)]
    sources["target"] = agree
    for name, cols in disagree:
        notes.append(f"источник {name} называет таргетом {cols}, а выбран {targets}")
    missing_in_train = [c for c in sample_targets if c not in train_cols]
    if missing_in_train:
        notes.append(f"в sample_submission есть столбцы, которых нет в train: {missing_in_train}")

    if not targets:
        confidence = "none"
    elif disagree:
        confidence = "low"
    elif len(agree) >= 2:
        confidence = "high"
    elif len(agree) == 1 and id_cols:
        confidence = "medium"
    else:
        confidence = "low"

    shared = [c for c in test_cols if c in train_cols] if test is not None else list(train_cols)
    return {
        "id_cols": id_cols,
        "target_cols": targets,
        "shared_columns": shared,
        "feature_cols": [c for c in shared if c not in id_cols and c not in targets],
        "output_columns": list(sample_submission.columns) if sample_submission is not None else None,
        "confidence": confidence,
        "sources": sources,
        "notes": notes,
    }


def resolve_schema(suggestion: dict, id_cols=None, target_cols=None) -> Schema:
    """Принять подсказку или переопределить её; при неуверенной подсказке — SchemaError."""
    overridden = id_cols is not None or target_cols is not None
    ids = list(id_cols) if id_cols is not None else list(suggestion["id_cols"])
    targets = list(target_cols) if target_cols is not None else list(suggestion["target_cols"])
    if not targets:
        raise SchemaError("таргет не определён: передайте target_cols")
    if not overridden and suggestion["confidence"] in ("none", "low"):
        raise SchemaError(
            f"схема неуверенная ({suggestion['confidence']}): {suggestion['notes'] or 'один источник'}; "
            "передайте id_cols и target_cols"
        )
    pool = suggestion["shared_columns"]
    known = set(pool) | set(suggestion["target_cols"]) | set(suggestion["id_cols"])
    unknown = [c for c in targets + ids if c not in known]
    if unknown:
        raise SchemaError(f"столбцов {unknown} нет в данных задачи")
    features = tuple(c for c in pool if c not in ids and c not in targets)
    output = suggestion.get("output_columns")
    return Schema(tuple(ids), tuple(targets), features, tuple(output) if output else None)


def _text_hint(text: str):
    classes, regressions = len(_CLS_WORDS.findall(text)), len(_REG_WORDS.findall(text))
    if classes > regressions:
        return "classification"
    if regressions > classes:
        return "regression"
    return None


def suggest_task_type(y: Optional[pd.Series], text: str = "") -> dict:
    """regression | binary | multiclass | unknown; ambiguous=True, если данные допускают оба прочтения."""
    notes = []
    if y is None or len(y.dropna()) == 0:
        return {"type": "unknown", "ambiguous": True, "notes": ["нет таргета для анализа"]}
    values = y.dropna()
    n_unique = int(values.nunique())
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    ambiguous = False
    if not numeric:
        kind = "binary" if n_unique == 2 else "multiclass"
    elif n_unique == 2:
        kind = "binary"
    elif bool((values % 1 == 0).all()) and n_unique <= _MAX_INT_CLASSES:
        kind, ambiguous = "multiclass", True
    else:
        kind = "regression"

    hint = _text_hint(text)
    if hint:
        family = "regression" if kind == "regression" else "classification"
        if hint != family:
            if ambiguous:
                kind = "regression" if hint == "regression" else kind
                ambiguous = False
                notes.append(f"тип выбран по тексту задачи ({hint})")
            else:
                notes.append(f"текст задачи говорит {hint}, а данные — {family}; выбраны данные")
        elif ambiguous:
            ambiguous = False
            notes.append(f"тип подтверждён текстом задачи ({hint})")
    return {"type": kind, "ambiguous": ambiguous, "notes": notes}


def mentioned_metrics(text: str) -> dict:
    """Метрики из task.md. primary — по фразам вроде «основная метрика», иначе первое упоминание."""
    clean = text.replace("*", "").replace("`", "")
    hits = []
    for name, pattern in _METRIC_PATTERNS.items():
        scan = re.sub(_METRIC_PATTERNS["balanced_accuracy"], " ", clean, flags=re.IGNORECASE) if name == "accuracy" else clean
        hits += [(m.start(), name) for m in re.finditer(pattern, scan, flags=re.IGNORECASE)]
    hits.sort()
    mentioned = list(dict.fromkeys(name for _, name in hits))
    primary, source = None, None
    for phrase in _PRIMARY_PHRASE.finditer(clean):
        near = [name for pos, name in hits if 0 <= pos - phrase.end() <= 150]
        if near:
            primary, source = near[0], "фраза об основной метрике"
            break
    if primary is None and mentioned:
        primary, source = mentioned[0], "первое упоминание"
    metric = ml_metrics.get_metric(primary) if primary else None
    return {
        "primary": primary,
        "mentioned": mentioned,
        "higher_is_better": metric.higher_is_better if metric else None,
        "source": source,
    }
