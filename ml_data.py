"""Подготовка данных для моделей без утечки: всё, что учится, учится только на train."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

import ml_io
import ml_schema


class DataError(ValueError):
    """Данные нельзя подготовить: нет нужных столбцов, неизвестные метки и т. п."""


@dataclass
class SplitData:
    """Задача, разрезанная по схеме: признаки и таргет train, признаки test и его id."""

    X_train: pd.DataFrame
    y_train: Union[pd.Series, pd.DataFrame]
    X_test: pd.DataFrame
    test_ids: pd.DataFrame  # id-столбцы test в исходном порядке (может не иметь столбцов)
    schema: ml_schema.Schema
    warnings: list = field(default_factory=list)


def split_task(data: ml_io.TaskData, schema: ml_schema.Schema) -> SplitData:
    if data.train is None or data.test is None:
        raise DataError("для обучения нужны и train, и test")
    train, test, warnings = data.train, data.test, []
    for col in schema.target_cols:
        if col not in train.columns:
            raise DataError(f"таргета {col!r} нет в train")
    absent_train = [c for c in schema.feature_cols if c not in train.columns]
    if absent_train:
        raise DataError(f"признаков {absent_train} нет в train")
    absent_test = [c for c in schema.feature_cols if c not in test.columns]
    if absent_test:
        warnings.append(f"в test нет признаков {absent_test}: они будут заполнены пропусками")
    absent_ids = [c for c in schema.id_cols if c not in test.columns]
    if absent_ids:
        raise DataError(f"id-столбцов {absent_ids} нет в test")

    targets = list(schema.target_cols)
    has_target = train[targets].notna().all(axis=1)
    if not has_target.all():
        warnings.append(f"{int((~has_target).sum())} строк train без таргета исключены")
        train = train.loc[has_target]
    y = train[targets[0]] if len(targets) == 1 else train[targets]

    sample = data.sample_submission
    if sample is not None and schema.id_cols and all(c in sample.columns for c in schema.id_cols):
        same = len(sample) == len(test) and all(
            np.array_equal(sample[c].to_numpy(), test[c].to_numpy()) for c in schema.id_cols
        )
        if not same:
            warnings.append("id в sample_submission не совпадают с test по порядку или числу строк")

    return SplitData(
        X_train=train[list(schema.feature_cols)],
        y_train=y,
        X_test=test[[c for c in schema.feature_cols if c in test.columns]],
        test_ids=test[list(schema.id_cols)].copy(),
        schema=schema,
        warnings=warnings,
    )


class LabelCodec:
    """Метки классов <-> целые коды 0..K-1; порядок классов фиксируется в fit()."""

    def fit(self, y) -> "LabelCodec":
        y = pd.Series(np.asarray(y))
        if y.isna().any():
            raise DataError("в метках классов есть пропуски")
        try:
            self.classes_ = np.array(sorted(y.unique()))
        except TypeError:
            self.classes_ = np.array(sorted(y.unique(), key=str), dtype=object)
        return self

    def encode(self, y) -> np.ndarray:
        codes = pd.Index(self.classes_).get_indexer(pd.Series(np.asarray(y)))
        if (codes < 0).any():
            unknown = sorted({str(v) for v, c in zip(np.asarray(y), codes) if c < 0})[:5]
            raise DataError(f"метки {unknown} не встречались при fit()")
        return codes.astype("int64")

    def decode(self, codes) -> np.ndarray:
        codes = np.asarray(codes, dtype="int64")
        if codes.size and (codes.min() < 0 or codes.max() >= len(self.classes_)):
            raise DataError("код класса вне диапазона")
        return self.classes_[codes]

    def decode_proba(self, proba, index=None) -> pd.DataFrame:
        proba = np.asarray(proba)
        if proba.ndim != 2 or proba.shape[1] != len(self.classes_):
            raise DataError(f"ожидается матрица (n, {len(self.classes_)})")
        return pd.DataFrame(proba, columns=[str(c) for c in self.classes_], index=index)


class TabularPreprocessor:
    """Приводит признаки к виду для градиентного бустинга.

    fit() запоминает только то, что видно в переданной таблице (список признаков,
    типы, словари категорий). Числа остаются с пропусками (LightGBM, XGBoost и
    CatBoost умеют с ними работать), ±inf заменяется на NaN. Категории кодируются по
    словарю из fit(); значения, которых там не было, превращаются в пропуск.
    """

    def __init__(self, drop_constant: bool = True):
        self.drop_constant = drop_constant

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        if not isinstance(X, pd.DataFrame) or X.shape[1] == 0:
            raise DataError("ожидается таблица хотя бы с одним признаком")
        self.numeric_cols_, self.categorical_cols_ = [], []
        self.categories_, self.dropped_, self.warnings_ = {}, {}, []
        for col in X.columns:
            s = X[col]
            if s.isna().all():
                self.dropped_[col] = "all_missing"
            elif self.drop_constant and s.nunique(dropna=True) <= 1:
                self.dropped_[col] = "constant"
            elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                self.numeric_cols_.append(col)
            else:
                counts = s.dropna().astype(str).value_counts()
                order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                self.categories_[col] = [k for k, _ in order]
                self.categorical_cols_.append(col)
        self.feature_names_ = [c for c in X.columns if c not in self.dropped_]
        if not self.feature_names_:
            raise DataError("после удаления пустых и константных столбцов признаков не осталось")
        return self

    def transform(self, X: pd.DataFrame, categorical: str = "category") -> pd.DataFrame:
        """categorical: 'category' — dtype category с фиксированными категориями;
        'codes' — целые коды (−1 для пропуска и неизвестных значений)."""
        if not hasattr(self, "feature_names_"):
            raise DataError("сначала вызовите fit()")
        if categorical not in ("category", "codes"):
            raise DataError("categorical должен быть 'category' или 'codes'")
        self.warnings_ = []
        self.unseen_share_ = {}
        out = {}
        for col in self.feature_names_:
            if col not in X.columns:
                self.warnings_.append(f"в данных нет признака {col!r}: заполнен пропусками")
                values = pd.Series(np.nan, index=X.index)
            else:
                values = X[col]
            if col in self.categorical_cols_:
                present = values.notna()
                text = values.astype(str).where(present)
                cat = pd.Categorical(text, categories=self.categories_[col])
                unseen = float((present & pd.isna(cat)).mean()) if len(X) else 0.0
                if unseen > 0:
                    self.unseen_share_[col] = unseen
                out[col] = (
                    pd.Series(cat, index=X.index)
                    if categorical == "category"
                    else pd.Series(np.asarray(cat.codes, dtype="int32"), index=X.index)
                )
            else:
                numbers = pd.to_numeric(values, errors="coerce").astype("float64")
                out[col] = numbers.replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame(out, index=X.index, columns=self.feature_names_)

    def fit_transform(self, X: pd.DataFrame, categorical: str = "category") -> pd.DataFrame:
        return self.fit(X).transform(X, categorical=categorical)
