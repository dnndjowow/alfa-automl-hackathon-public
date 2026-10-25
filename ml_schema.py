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

_DEFAULT_INT_CLASSES = 20   # целый таргет с не более чем столькими значениями по умолчанию считаем классами
_MAX_INT_CLASSES = 200      # до этого порога целый таргет допускает оба прочтения
_CLS_WORDS = re.compile(r"классификац|classif", re.IGNORECASE)
_REG_WORDS = re.compile(r"регресс|regress", re.IGNORECASE)
_ID_NAME = re.compile(r"id|index|row_?id|uid|\w+_id", re.IGNORECASE)

def _f1_variant(kind: str) -> str:
    """macro F1 во всех встреченных написаниях: «macro F1», «macro-averaged F1», «F1-score (macro)», «F1-мера macro», «по F1-мере (macro)»."""
    return (rf"f1_score\s*\([^)]*average\s*=\s*[\"']{kind}[\"']\s*\)"
            rf"|{kind}[\s_-]*(?:averaged[\s_-]+)?f1"
            rf"|f1(?:[\s_-]*(?:score|мер\w*))?[\s_-]*\(?\s*{kind}"
            rf"|average\s*=\s*[\"']{kind}[\"']")


# Если совпадения пересекаются, остаётся самое длинное: «accuracy» внутри «balanced accuracy»,
# «auc» внутри «PR-AUC» и «F1» внутри «macro F1» отдельными метриками не считаются.
_METRIC_PATTERNS = {
    "rmse": r"\brmse\b|root mean squared error",
    "mse": r"\bmse\b",
    "mae": r"\bmae\b",
    "r2": r"\br\^?2\b|r²|r-squared|coefficient of determination",
    "accuracy": r"\baccuracy\b",
    "balanced_accuracy": r"balanced[\s_-]*accuracy",
    "f1_macro": _f1_variant("macro"),
    "f1_weighted": _f1_variant("weighted"),
    "f1_micro": _f1_variant("micro"),
    "f1": r"\bf1(?:[\s_-]*(?:score|мер\w*))?\b",
    "logloss": r"log[\s_-]*loss|cross[\s_-]*entropy",
    "rmsle": r"\brmsle\b|root mean squared log\w*(?: error)?|rmse\s+(?:между\s+|of\s+(?:the\s+)?)(?:log|логарифм)\w*",
    "msle": r"\bmsle\b|mean squared log\w* error",
    "mape": r"\bmape\b|mean absolute percentage(?: error)?",
    "smape": r"\bsmape\b|symmetric mean absolute(?: percentage)?(?: error)?",
    "average_precision": r"pr[\s_-]*auc|auc[\s_-]*pr\b|average[\s_-]+precision",
    "roc_auc": r"roc[\s_-]*auc|\bauc\b",
}
# Метрики, которые распознаются, но не считаются ядром: бинарному F1 нужен выбор положительного
# класса, RMSLE/MSLE/MAPE/SMAPE считает агент. Их всё равно надо называть основными, а не подменять.
_UNSUPPORTED = {
    "f1": {"task": "classification", "higher_is_better": True},
    "rmsle": {"task": "regression", "higher_is_better": False},
    "msle": {"task": "regression", "higher_is_better": False},
    "mape": {"task": "regression", "higher_is_better": False},
    "smape": {"task": "regression", "higher_is_better": False},
}
_SENTENCE_END = re.compile(r"[.;!?]|\n\s*\n")
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
        # id уникален, не помечен словарём как таргет (заглушка таргета в test тоже может совпасть)
        # и содержит те же значения, что в test; порядок строк в шаблоне может отличаться
        id_cols = [
            c for c in sample_submission.columns
            if c in test.columns and roles.get(c) != "target" and test[c].is_unique
            and set(sample_submission[c].astype(str)) == set(test[c].astype(str))
        ]
        if id_cols:
            sources["id"] = "значения sample_submission совпадают с test"
            if not all(np.array_equal(sample_submission[c].to_numpy(), test[c].to_numpy()) for c in id_cols):
                notes.append("порядок строк sample_submission отличается от test")
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
    overlap = sorted(set(ids) & set(targets))
    if overlap:
        raise SchemaError(f"столбцы {overlap} не могут быть одновременно id и таргетом")
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


def suggest_task_type(y: Optional[pd.Series], text: str = "", metric: Optional[str] = None) -> dict:
    """regression | binary | multiclass | unknown.

    Данные дают первое прочтение. Если они допускают оба (два числа, целые числа с умеренным
    числом значений), решает семейство основной метрики, а без неё — текст задачи. Если довод
    спорит с однозначными данными, данные сохраняются, но ambiguous=True и есть заметка.
    """
    notes = []
    if y is None or len(y.dropna()) == 0:
        return {"type": "unknown", "ambiguous": True, "notes": ["нет таргета для анализа"]}
    values = y.dropna()
    n_unique = int(values.nunique())
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    if not numeric:
        kind, firm = ("binary" if n_unique == 2 else "multiclass"), True
    elif n_unique == 2:
        kind, firm = "binary", False
    elif bool((values % 1 == 0).all()) and n_unique <= _MAX_INT_CLASSES:
        kind, firm = ("multiclass" if n_unique <= _DEFAULT_INT_CLASSES else "regression"), False
    else:
        kind, firm = "regression", True

    evidence, source = None, None
    if metric:
        evidence, source = metric_family(metric), f"метрика {metric}"
        if evidence is None:
            notes.append(f"метрика {metric!r} неизвестна, тип по ней не определить")
    if evidence is None:
        evidence, source = _text_hint(text), "текст задачи"
    family = "regression" if kind == "regression" else "classification"
    if evidence is None:
        return {"type": kind, "ambiguous": not firm, "notes": notes}
    if evidence == family:
        if not firm:
            notes.append(f"тип подтверждён: {source}")
        return {"type": kind, "ambiguous": False, "notes": notes}
    if not firm:
        kind = "regression" if evidence == "regression" else ("binary" if n_unique == 2 else "multiclass")
        notes.append(f"тип выбран по доводу «{source}» ({evidence}); данные допускали оба прочтения")
        return {"type": kind, "ambiguous": False, "notes": notes}
    notes.append(f"{source} указывает на {evidence}, а данные однозначно — {family}; выбраны данные")
    return {"type": kind, "ambiguous": True, "notes": notes}


def metric_family(name: Optional[str]) -> Optional[str]:
    """regression / classification для метрики из реестра или из распознаваемых без поддержки."""
    if not name:
        return None
    if name in _UNSUPPORTED:
        return _UNSUPPORTED[name]["task"]
    try:
        return ml_metrics.get_metric(name).task
    except ml_metrics.MetricError:
        return None


def _metric_hits(clean: str) -> list:
    """(start, end, name) для всех упоминаний метрик; из пересекающихся остаётся самое длинное."""
    found = [(m.start(), m.end(), name)
             for name, pattern in _METRIC_PATTERNS.items()
             for m in re.finditer(pattern, clean, flags=re.IGNORECASE)]
    kept = []
    for hit in sorted(found, key=lambda h: (-(h[1] - h[0]), h[0])):
        if all(hit[1] <= k[0] or hit[0] >= k[1] for k in kept):
            kept.append(hit)
    return sorted(kept)


def _sentence(clean: str, start: int, end: int):
    """Границы предложения вокруг [start, end): по . ; ! ? и пустой строке."""
    left = 0
    for m in _SENTENCE_END.finditer(clean, 0, start):
        left = m.end()
    right_match = _SENTENCE_END.search(clean, end)
    return left, right_match.start() if right_match else len(clean)


def mentioned_metrics(text: str) -> dict:
    """Метрики из task.md.

    primary — метрика, ближайшая к фразе вроде «основная метрика» в том же предложении,
    до или после неё; без такой фразы — первое упоминание. supported=False значит, что метрика
    распознана, но ядро её не считает (например, бинарный F1 без выбранного положительного класса).
    """
    clean = text.replace("*", "").replace("`", "")
    hits = _metric_hits(clean)
    mentioned = list(dict.fromkeys(name for _, _, name in hits))
    primary, source = None, None
    for phrase in _PRIMARY_PHRASE.finditer(clean):
        left, right = _sentence(clean, phrase.start(), phrase.end())
        candidates = []
        for start, end, name in hits:
            if start < left or end > right:
                continue
            if start >= phrase.end():
                distance = start - phrase.end()
            elif end <= phrase.start():
                distance = phrase.start() - end
            else:
                distance = 0
            candidates.append((distance, start < phrase.start(), name))  # при равенстве — после фразы
        if candidates:
            primary, source = min(candidates)[2], "фраза об основной метрике"
            break
    if primary is None and mentioned:
        primary, source = mentioned[0], "первое упоминание"
    supported, higher = True, None
    if primary in _UNSUPPORTED:
        supported, higher = False, _UNSUPPORTED[primary]["higher_is_better"]
    elif primary:
        higher = ml_metrics.get_metric(primary).higher_is_better
    return {"primary": primary, "mentioned": mentioned, "higher_is_better": higher,
            "supported": supported, "source": source}
