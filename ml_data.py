"""Подготовка данных для моделей без утечки: всё, что учится, учится только на train."""

from __future__ import annotations

import re
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


def category_text(series: pd.Series) -> pd.Series:
    """Значения категории как текст; целые числа без «.0», чтобы 1 из одного файла и 1.0 из другого совпали."""
    text = series.astype(str).where(series.notna())
    return text.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)


_ISO_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?:Z|[+-]\d{2}:?\d{2})?$")
_ANY_DATE = re.compile(r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?$")
_DAY_MONTH = re.compile(r"^(\d{1,2})([-/.])(\d{1,2})[-/.]\d{2,4}(?:$|[ T])")
_DATE_SHARE = 0.95
_EPOCH = pd.Timestamp("1970-01-01")
DATE_PARTS = ("days", "year", "month", "dayofweek")


def parse_dates(series: pd.Series) -> Optional[pd.Series]:
    """Дата-время из текстового столбца, если не меньше 95% непустых значений похожи на даты и разбираются;
    иначе None. Числа датами не считаются."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    text = series.dropna().astype(str).str.strip()
    if text.empty:
        return None
    iso = text.str.match(_ISO_DATE).mean() >= _DATE_SHARE
    if not iso and text.str.match(_ANY_DATE).mean() < _DATE_SHARE:
        return None
    if iso:
        # utc=True: смешанные смещения (+01:00 зимой, +02:00 летом) иначе дают object, и .dt падал
        parsed = pd.to_datetime(series.astype(str).str.strip().where(series.notna()), errors="coerce",
                                format="ISO8601", utc=True).dt.tz_convert(None)
    else:
        parsed = to_dates(series)
    return parsed if parsed[series.notna()].notna().mean() >= _DATE_SHARE else None


def date_dayfirst(series: pd.Series) -> bool:
    """Идёт ли день перед месяцем в датах вида 05.01.2020 или 05/01/2020.

    Поле больше 12 решает однозначно (первое — день, второе — месяц); если все поля не больше 12 —
    день первым при разделителе «.» (русский и банковский формат), иначе месяц первым (как в pandas).
    Даты с годом в начале (2020-01-05) так не разбираются: для них False.
    """
    text = series.dropna().astype(str).str.strip()
    parts = text.str.extract(_DAY_MONTH).dropna()
    if parts.empty or len(parts) < 0.5 * len(text):
        return False
    first, second = parts[0].astype(int), parts[2].astype(int)
    if (first > 12).any():
        return True
    if (second > 12).any():
        return False
    return bool((parts[1] == ".").mean() >= 0.5)


def to_dates(series: pd.Series, dayfirst: Optional[bool] = None) -> pd.Series:
    """Текст в дату-время (неразобранное — NaT). dayfirst=None — порядок дня и месяца по самим значениям;
    чтобы train и test разбирались одинаково, передавайте флаг, найденный на train."""
    if dayfirst is None:
        dayfirst = date_dayfirst(series)
    values = series.astype(str).str.strip().where(series.notna())
    # utc=True — общая шкала для дат со смещениями (в том числе разными); даты без смещения не сдвигаются
    return pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=dayfirst, utc=True).dt.tz_convert(None)


def date_days(parsed: pd.Series) -> pd.Series:
    """Дни от 1970-01-01 (дробные) — числовое представление даты для порядка и границ."""
    return (parsed - _EPOCH) / pd.Timedelta(days=1)


class TabularPreprocessor:
    """Приводит признаки к виду для градиентного бустинга.

    fit() запоминает только то, что видно в переданной таблице (список признаков,
    типы, словари категорий). Числа остаются с пропусками (LightGBM, XGBoost и
    CatBoost умеют с ними работать), ±inf заменяется на NaN. Категории кодируются по
    словарю из fit(); значения, которых там не было, превращаются в пропуск.
    Текстовые даты становятся числами: <столбец>__days (дни от 1970-01-01), __year, __month, __dayofweek.
    """

    def __init__(self, drop_constant: bool = True):
        self.drop_constant = drop_constant

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        if not isinstance(X, pd.DataFrame) or X.shape[1] == 0:
            raise DataError("ожидается таблица хотя бы с одним признаком")
        self.numeric_cols_, self.categorical_cols_, self.datetime_cols_ = [], [], []
        self.categories_, self.dropped_, self.warnings_, self.dayfirst_ = {}, {}, [], {}
        for col in X.columns:
            s = X[col]
            if s.isna().all():
                self.dropped_[col] = "all_missing"
            elif self.drop_constant and s.nunique(dropna=False) <= 1:
                # «одно значение или пропуск» не константа: сам факт пропуска может быть сигналом
                self.dropped_[col] = "constant"
            elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                self.numeric_cols_.append(col)
            elif parse_dates(s) is not None:
                self.datetime_cols_.append(col)
                self.dayfirst_[col] = date_dayfirst(s)  # test разбирается так же, как train
            else:
                counts = category_text(s).dropna().value_counts()
                order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                self.categories_[col] = [k for k, _ in order]
                self.categorical_cols_.append(col)
        self.input_cols_ = [c for c in X.columns if c not in self.dropped_]
        # имена частей даты не должны совпадать с уже существующими столбцами (when__year в данных):
        # иначе вычисленный год перезаписывался исходным столбцом, а LightGBM падал на повторе имени
        taken = {str(c) for c in self.input_cols_ if c not in self.datetime_cols_}
        self.date_names_ = {}
        for c in self.datetime_cols_:
            names = {}
            for part in DATE_PARTS:
                name = f"{c}__{part}"
                while name in taken:
                    name += "_"
                taken.add(name)
                names[part] = name
            self.date_names_[c] = names
        self.feature_names_ = [
            name for c in self.input_cols_
            for name in (list(self.date_names_[c].values()) if c in self.datetime_cols_ else [c])
        ]
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
        for col in self.input_cols_:
            if col not in X.columns:
                self.warnings_.append(f"в данных нет признака {col!r}: заполнен пропусками")
                values = pd.Series(np.nan, index=X.index)
            else:
                values = X[col]
            if col in self.datetime_cols_:
                parsed = to_dates(values, self.dayfirst_.get(col, False))
                names = self.date_names_[col]
                out[names["days"]] = date_days(parsed).astype("float64")
                out[names["year"]] = parsed.dt.year.astype("float64")
                out[names["month"]] = parsed.dt.month.astype("float64")
                out[names["dayofweek"]] = parsed.dt.dayofweek.astype("float64")
            elif col in self.categorical_cols_:
                present = values.notna()
                text = category_text(values)
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
