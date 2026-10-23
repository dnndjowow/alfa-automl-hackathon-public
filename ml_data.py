"""Подготовка данных для моделей без утечки: всё, что учится, учится только на train."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataError(ValueError):
    """Данные нельзя подготовить: нет нужных столбцов, неизвестные метки и т. п."""


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
