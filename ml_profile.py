"""Паспорт задачи: компактное JSON-описание данных для LLM и кода агента.

build_profile() ничего не решает: схема, тип задачи и метрика даются как подсказки
(см. ml_schema). format_profile() превращает паспорт в короткий текст для промпта.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import ml_io
import ml_schema

_TIME_HINT = re.compile(r"year|date|time|month|week|day|period|quarter|дата|год|месяц|период", re.IGNORECASE)
_HIGH_CARDINALITY = 50
_RARE_SHARE = 0.01
_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def _py(value):
    """numpy-значения -> обычные Python; NaN и inf -> None (чтобы паспорт был валидным JSON)."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def _round(value, digits=4):
    value = _py(value)
    return round(value, digits) if isinstance(value, float) else value


def column_kind(series: pd.Series) -> str:
    n_unique = series.nunique(dropna=True)
    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series) or n_unique == 2:
        return "binary"
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


def profile_column(series: pd.Series) -> dict:
    n_rows = len(series)
    kind = column_kind(series)
    n_unique = int(series.nunique(dropna=True))
    info = {
        "name": str(series.name),
        "kind": kind,
        "dtype": str(series.dtype),
        "missing_frac": _round(series.isna().mean()),
        "n_unique": n_unique,
    }
    non_null = max(n_rows - int(series.isna().sum()), 1)
    if kind == "numeric":
        info.update(
            min=_round(series.min()), max=_round(series.max()), mean=_round(series.mean()),
            std=_round(series.std()), median=_round(series.median()),
        )
    elif kind in ("categorical", "binary"):
        top = series.value_counts(dropna=True).head(5)
        info["top_values"] = [[str(k), _round(v / n_rows)] for k, v in top.items()]
    integer_like = pd.api.types.is_integer_dtype(series) or not pd.api.types.is_numeric_dtype(series)
    if n_rows >= 50 and integer_like and n_unique / non_null > 0.99:
        info["id_like"] = True
    return info


def profile_target(y: pd.Series, task_type: str) -> dict:
    out = {"name": str(y.name), "missing_frac": _round(y.isna().mean()), "n_unique": int(y.nunique())}
    if task_type == "regression":
        q = y.quantile(list(_QUANTILES))
        out.update(
            quantiles={f"{p:g}": _round(v) for p, v in zip(_QUANTILES, q)},
            mean=_round(y.mean()), std=_round(y.std()), skew=_round(y.skew()),
            share_zero=_round((y == 0).mean()), share_negative=_round((y < 0).mean()),
        )
    else:
        counts = y.value_counts()
        shares = counts / counts.sum()
        out["classes"] = {str(k): {"count": int(counts[k]), "share": _round(shares[k])} for k in counts.index}
        out["imbalance_ratio"] = _round(counts.max() / counts.min())
        out["rare_classes"] = [str(k) for k in counts.index if shares[k] < _RARE_SHARE]
    return out


def compare_train_test(train: pd.DataFrame, test: pd.DataFrame, columns, top=10) -> dict:
    unseen, out_of_range, missing_shift = {}, {}, {}
    for col in columns:
        if col not in train.columns or col not in test.columns:
            continue
        a, b = train[col], test[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            bn = b.dropna()
            if len(bn) and a.notna().any():
                share = float(((bn < a.min()) | (bn > a.max())).mean())
                if share > 0:
                    out_of_range[col] = _round(share)
        else:
            known = set(a.dropna().astype(str))
            bs = b.dropna().astype(str)
            if len(bs):
                share = float((~bs.isin(known)).mean())
                if share > 0:
                    unseen[col] = _round(share)
        shift = abs(float(b.isna().mean()) - float(a.isna().mean()))
        if shift > 0.1:
            missing_shift[col] = _round(shift)

    def best(d):
        return dict(sorted(d.items(), key=lambda kv: -kv[1])[:top])

    return {
        "unseen_categories_share": best(unseen),
        "out_of_train_range_share": best(out_of_range),
        "missing_frac_shift": best(missing_shift),
    }


def find_risks(train: pd.DataFrame, columns, column_infos) -> list:
    risks = []

    def add(kind, cols=None, detail=""):
        risks.append({"kind": kind, "columns": cols or [], "detail": detail})

    by_kind = {}
    for info in column_infos:
        by_kind.setdefault(info["kind"], []).append(info["name"])
    if by_kind.get("constant"):
        add("constant_features", by_kind["constant"], "не несут информации")
    high_card = [i["name"] for i in column_infos if i["kind"] == "categorical" and i["n_unique"] > _HIGH_CARDINALITY]
    if high_card:
        add("high_cardinality", high_card, f"больше {_HIGH_CARDINALITY} категорий: осторожно с кодированием")
    ids = [i["name"] for i in column_infos if i.get("id_like")]
    if ids:
        add("possible_id_in_features", ids, "почти все значения уникальны: возможен идентификатор или утечка")
    heavy = [i["name"] for i in column_infos if i["missing_frac"] and i["missing_frac"] > 0.5]
    if heavy:
        add("mostly_missing", heavy, "больше половины пропусков")
    time_like = [c for c in columns if _TIME_HINT.search(str(c))]
    if time_like:
        add("time_like_names", time_like, "возможен временной или групповой сплит: проверить по описанию")
    if len(train) and len(columns):
        dup = float(train[list(columns)].duplicated().mean())
        if dup > 0.01:
            add("duplicate_rows", [], f"{dup:.1%} строк train дублируют друг друга по признакам")
    if len(train) < 1000:
        add("small_train", [], f"всего {len(train)} строк: валидация шумная")
    return risks


def build_profile(data: ml_io.TaskData) -> dict:
    train, test = data.train, data.test
    suggestion = ml_schema.infer_schema(train, test, data.sample_submission, data.data_dictionary)
    files = {}
    for role, path in data.files.paths.items():
        frame = getattr(data, role)
        if path is not None and frame is not None:
            files[role] = {
                "path": ml_io.display_path(path, data.files.task_dir),
                "rows": int(len(frame)), "cols": int(frame.shape[1]),
            }
    profile = {
        "files": files,
        "warnings": list(data.files.warnings),
        "schema": suggestion,
        "metric": ml_schema.mentioned_metrics(data.text),
    }
    if train is None:
        profile.update(task={"type": "unknown", "ambiguous": True, "notes": ["нет train"]})
        return profile

    targets = suggestion["target_cols"]
    y = train[targets[0]] if len(targets) == 1 else None
    if len(targets) > 1:
        task = {"type": "multi_target", "ambiguous": True, "notes": [f"несколько таргетов: {targets}"]}
    else:
        task = ml_schema.suggest_task_type(y, data.text)
    profile["task"] = task
    if y is not None and task["type"] != "unknown":
        profile["target"] = profile_target(y, task["type"])

    features = suggestion["feature_cols"]
    infos = [profile_column(train[c]) for c in features]
    counts = {}
    for info in infos:
        counts[info["kind"]] = counts.get(info["kind"], 0) + 1
    profile["features"] = {"n_features": len(features), "kinds": counts, "columns": infos}
    if test is not None:
        profile["train_test"] = compare_train_test(train, test, features)
    profile["risks"] = find_risks(train, features, infos)
    return profile


def profile_task(task_md) -> dict:
    return build_profile(ml_io.load_task(task_md))


def to_json(profile: dict, **kwargs) -> str:
    return json.dumps(profile, ensure_ascii=False, **kwargs)


def format_profile(profile: dict, max_columns: int = 40) -> str:
    """Короткий текст для промпта LLM: сначала главное, столбцы — до max_columns строк."""
    lines = []
    schema, task, metric = profile["schema"], profile["task"], profile["metric"]
    for role, f in profile["files"].items():
        lines.append(f"файл {role}: {f['path']} ({f['rows']} строк, {f['cols']} столбцов)")
    lines.append(
        f"схема (уверенность {schema['confidence']}): id={schema['id_cols']}, "
        f"таргет={schema['target_cols']}, признаков={len(schema['feature_cols'])}"
    )
    lines.append(f"тип задачи: {task['type']}" + (" (неоднозначно)" if task.get("ambiguous") else ""))
    if metric["primary"]:
        direction = "больше лучше" if metric["higher_is_better"] else "меньше лучше"
        lines.append(f"метрика (по task.md): {metric['primary']} — {direction}; упомянуты: {metric['mentioned']}")
    target = profile.get("target")
    if target and "classes" in target:
        classes = ", ".join(f"{k}={v['count']}" for k, v in target["classes"].items())
        lines.append(f"классы: {classes}; дисбаланс {target['imbalance_ratio']}; редкие: {target['rare_classes']}")
    elif target:
        lines.append(
            f"таргет: среднее {target['mean']}, std {target['std']}, перекос {target['skew']}, "
            f"квантили {target['quantiles']}"
        )
    for note in schema["notes"] + task.get("notes", []) + profile["warnings"]:
        lines.append(f"заметка: {note}")
    for risk in profile.get("risks", []):
        lines.append(f"риск {risk['kind']}: {risk['columns'][:8]} {risk['detail']}")
    tt = profile.get("train_test", {})
    for key, values in tt.items():
        if values:
            lines.append(f"train/test {key}: {values}")
    columns = profile.get("features", {}).get("columns", [])
    if columns:
        lines.append(f"признаки по типам: {profile['features']['kinds']}")
    for info in columns[:max_columns]:
        extra = (
            f"[{info['min']}..{info['max']}] среднее {info['mean']}"
            if info["kind"] == "numeric" else f"топ {info.get('top_values', [])[:3]}"
        )
        lines.append(f"- {info['name']}: {info['kind']}, пропусков {info['missing_frac']}, уникальных {info['n_unique']}, {extra}")
    if len(columns) > max_columns:
        lines.append(f"... и ещё {len(columns) - max_columns} признаков")
    return "\n".join(lines)
