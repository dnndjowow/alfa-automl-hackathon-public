"""Инструменты агента: оценка модели на валидации (train_eval) и финальное обучение (fit_predict).

Оба вызова возвращают результат, а не поднимают исключение для ожидаемых проблем:
статус "error" и понятное сообщение. Ничего не пишется на диск.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

import ml_metrics
import ml_postprocess
import ml_validation
from ml_data import LabelCodec, SplitData, TabularPreprocessor
from ml_models import BoostingModel, default_n_jobs, prepare_params

TARGET_TRANSFORMS = (None, "log1p")


@dataclass
class EvalResult:
    status: str                       # "ok" | "error"
    model: str = ""
    task_type: str = ""
    metric: str = ""
    metric_source: str = ""          # "explicit" | "profile" | "default" (подставлена ядром)
    higher_is_better: Optional[bool] = None
    score: Optional[float] = None     # среднее по частям валидации
    oof_score: Optional[float] = None  # метрика по всем проверенным строкам сразу
    fold_scores: list = field(default_factory=list)
    best_iterations: list = field(default_factory=list)
    train_sizes: list = field(default_factory=list)  # строк обучения в каждой посчитанной части
    seconds: float = 0.0
    validation: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    error: Optional[str] = None
    classes: Optional[list] = None   # исходные метки классов в порядке столбцов oof_pred (как fit_predict)
    oof_index: Optional[np.ndarray] = None   # позиции строк train, попавшие в проверку
    oof_pred: Optional[np.ndarray] = None    # значения (регрессия) или вероятности (n, K) в порядке oof_index

    def to_dict(self) -> dict:
        """JSON-безопасная сводка без массивов."""
        def num(v):
            return None if v is None or not np.isfinite(v) else float(v)
        return {
            "status": self.status, "model": self.model, "task_type": self.task_type, "metric": self.metric,
            "metric_source": self.metric_source,
            "higher_is_better": self.higher_is_better, "score": num(self.score), "oof_score": num(self.oof_score),
            "fold_scores": [num(s) for s in self.fold_scores], "best_iterations": list(self.best_iterations),
            "train_sizes": list(self.train_sizes),
            "seconds": round(self.seconds, 2), "validation": self.validation, "params": self.params,
            "notes": self.notes, "error": self.error,
        }


def _error(message: str, **kwargs) -> EvalResult:
    return EvalResult(status="error", error=message, **kwargs)


def _prepare_target(y: pd.Series, task_type: str, transform):
    """Возвращает (кодек или None, числовой таргет для модели)."""
    if task_type == "classification":
        codec = LabelCodec().fit(y)
        return codec, codec.encode(y)
    values = y.to_numpy(dtype=float)
    if transform == "log1p":
        if (values <= -1).any():
            raise ValueError("log1p невозможен: в таргете есть значения <= -1")
        values = np.log1p(values)
    return None, values


def _inverse(values: np.ndarray, task_type: str, transform):
    return np.expm1(values) if task_type == "regression" and transform == "log1p" else values


def _score(metric: ml_metrics.Metric, y_true, pred, task_type: str, n_classes: int) -> float:
    if task_type == "regression":
        return metric(y_true, pred)
    if metric.needs == "labels":
        return metric(y_true, pred.argmax(axis=1))
    return metric(y_true, pred, labels=list(range(n_classes)))


def _resolve_task(profile: dict, task_type) -> str:
    kind = task_type or profile.get("task", {}).get("type")
    if kind in ("binary", "multiclass"):
        kind = "classification"
    if kind not in ("regression", "classification"):
        raise ValueError(f"тип задачи {kind!r} пока не поддержан инструментами (нужна регрессия или классификация)")
    return kind


def _resolve_task_and_metric(profile: dict, task_type, metric):
    """(тип задачи, метрика, откуда метрика: "explicit" | "profile" | "default")."""
    kind = _resolve_task(profile, task_type)
    if metric is not None:
        name, source = metric, "explicit"
    elif profile.get("metric", {}).get("primary"):
        name, source = profile["metric"]["primary"], "profile"
    else:
        name, source = ("rmse" if kind == "regression" else "f1_macro"), "default"
    resolved = ml_metrics.get_metric(name)
    if resolved.task != kind:
        raise ValueError(f"метрика {resolved.name!r} относится к задаче {resolved.task}, а задача — {kind}")
    return kind, resolved, source


def _suggested_plan(profile: dict, X: pd.DataFrame, kind: str) -> dict:
    """suggest_validation по уже разрешённому семейству задачи: явный task_type важнее типа в профиле
    (иначе профиль binary при task_type='regression' давал stratified без y и ValidationError)."""
    task = dict(profile.get("task") or {})
    if (task.get("type") in ("binary", "multiclass")) != (kind == "classification"):
        task["type"] = "multiclass" if kind == "classification" else "regression"
        profile = {**profile, "task": task}
    return ml_validation.suggest_validation(profile, X)


def train_eval(split: SplitData, profile: dict, model: str = "lightgbm", metric: Optional[str] = None,
               validation: Optional[dict] = None, params: Optional[dict] = None,
               task_type: Optional[str] = None, target_transform: Optional[str] = None,
               n_jobs: Optional[int] = None, seed: int = 0, time_budget: Optional[float] = None,
               max_rounds: int = 3000, learning_rate: float = 0.05, subsample: Optional[float] = None,
               custom_stopping: bool = False) -> EvalResult:
    """Оценить модель на валидации. Данные в каждой части готовятся только по её train-строкам.

    Ранняя остановка идёт по метрике задачи, если у библиотеки есть такая встроенная (MAE, accuracy,
    AUC, у CatBoost и Macro F1); иначе — по метрике библиотеки. custom_stopping=True для таких метрик
    считает метрику задачи своей функцией на каждой итерации (дольше, выигрыш не подтверждён)."""
    started = time.time()
    try:
        kind, resolved, metric_source = _resolve_task_and_metric(profile, task_type, metric)
        if target_transform not in TARGET_TRANSFORMS:
            raise ValueError(f"преобразование таргета {target_transform!r} не поддержано: {TARGET_TRANSFORMS}")
        if not isinstance(split.y_train, pd.Series):
            raise ValueError("несколько таргетов пока не поддержаны")
        base = dict(model=model, task_type=kind, metric=resolved.name, metric_source=metric_source,
                    higher_is_better=resolved.higher_is_better, params=dict(params or {}))
        X, y = split.X_train, split.y_train
        notes = list(prepare_params(model, kind, params)[1]) if model in ("lightgbm", "xgboost", "catboost") else []
        if metric_source == "default":
            notes.append(f"метрика не передана и не найдена в task.md: взята по умолчанию {resolved.name}")
        positions = np.arange(len(X))  # позиции строк в исходном split.X_train
        if subsample is not None and subsample < 1.0:
            positions = np.sort(np.random.default_rng(seed).choice(len(X), max(2, int(len(X) * subsample)), replace=False))
            X, y = X.iloc[positions].reset_index(drop=True), y.iloc[positions].reset_index(drop=True)
            notes.append(f"использована случайная подвыборка train: {len(X)} строк")
        codec, target = _prepare_target(y, kind, target_transform)
        n_classes = len(codec.classes_) if codec else 0
        plan = dict(validation or _suggested_plan(profile, X, kind))
        splits = ml_validation.make_splits(plan, X, target if kind == "classification" else None, seed=seed)
        plan["effective"] = splits.effective()

        fold_scores, iterations, sizes = [], [], []
        oof_index, oof_parts = [], []
        for number, (tr, va) in enumerate(splits, 1):
            budget = None
            if time_budget is not None:
                remaining = time_budget - (time.time() - started)
                if remaining <= 0:
                    notes.append(f"бюджет времени исчерпан: части {number}–{len(splits)} не посчитаны")
                    break
                budget = remaining / (len(splits) - number + 1)
            try:
                pre = TabularPreprocessor()
                X_tr, X_va = pre.fit_transform(X.iloc[tr]), pre.transform(X.iloc[va])
                boosting = BoostingModel(
                    model, kind, n_classes=n_classes or None, params=params, n_jobs=n_jobs, seed=seed,
                    max_rounds=max_rounds, learning_rate=learning_rate, time_budget=budget,
                    # при log1p модель видит таргет в другой шкале: метрика задачи там не годится
                    eval_metric=None if target_transform else resolved, custom_eval=custom_stopping,
                )
                boosting.fit(X_tr, target[tr], X_va, target[va])
                if number == 1 and boosting.eval_note_:
                    notes.append(boosting.eval_note_)
                raw = boosting.predict(X_va)
                truth = target[va] if kind == "classification" else y.to_numpy(dtype=float)[va]
                pred = _inverse(raw, kind, target_transform)
                fold_scores.append(_score(resolved, truth, pred, kind, n_classes))
                iterations.append(boosting.best_iteration_)
                sizes.append(len(tr))
                oof_index.append(va)
                oof_parts.append(pred)
                if boosting.stopped_by_time_:
                    notes.append(f"часть {number}: обучение остановлено по бюджету времени")
            except Exception as error:  # одна неудачная часть не должна ронять всю оценку
                notes.append(f"часть {number} пропущена: {type(error).__name__}: {error}")
        if not fold_scores:
            return _error("ни одна часть валидации не посчитана; " + "; ".join(notes[-3:]),
                          seconds=time.time() - started, validation=plan, notes=notes, **base)
        index = np.concatenate(oof_index)
        oof = np.concatenate(oof_parts)
        truth_all = target[index] if kind == "classification" else y.to_numpy(dtype=float)[index]
        return EvalResult(
            status="ok", score=float(np.mean(fold_scores)),
            oof_score=_score(resolved, truth_all, oof, kind, n_classes),
            fold_scores=[float(s) for s in fold_scores], best_iterations=iterations, train_sizes=sizes,
            seconds=time.time() - started, validation=plan, notes=notes,
            classes=list(codec.classes_) if codec else None,  # исходные метки, как в fit_predict
            oof_index=positions[index], oof_pred=oof, **base,
        )
    except Exception as error:
        return _error(f"{type(error).__name__}: {error}", model=model, seconds=time.time() - started,
                      notes=[traceback.format_exc(limit=3)])


def rounds_from(result: EvalResult, margin: float = 1.15, how: str = "mean") -> Optional[int]:
    """Число раундов для финального обучения на всём train.

    how="mean" (по умолчанию): среднее лучших итераций по частям. how="largest_train": медиана
    по частям, где обучающих строк не меньше 90% от максимума. На скрытых годах регрессии
    (24.10.2026) среднее дало RMSE 47.0 против 47.8 у largest_train в среднем по четырём проверкам,
    поэтому оно остаётся по умолчанию. margin — запас на то, что данных в финале больше.
    """
    if result.status != "ok" or not result.best_iterations:
        return None
    best = np.asarray(result.best_iterations, dtype=float)
    if how == "largest_train" and len(result.train_sizes) == len(best):
        sizes = np.asarray(result.train_sizes, dtype=float)
        value = float(np.median(best[sizes >= 0.9 * sizes.max()]))
    elif how in ("largest_train", "mean"):
        value = float(np.mean(best))
    else:
        raise ValueError(f"неизвестный способ {how!r}: mean или largest_train")
    return max(1, int(round(value * margin)))


def _auto_rounds(X, target, kind, profile, n_classes, model, params, n_jobs, seed, max_rounds,
                 learning_rate, time_budget, validation=None, margin=1.15):
    """Подобрать число раундов ранней остановкой на каждой части валидации и усреднить (как rounds_from).

    validation — схема от вызывающего кода (иначе suggest_validation). Сбой одной части не роняет
    подбор; если не удалась ни одна, поднимается ValueError. Возвращает (раунды, схема, заметки)."""
    started = time.time()
    plan = dict(validation or _suggested_plan(profile, X, kind))
    splits = ml_validation.make_splits(plan, X, target if kind == "classification" else None, seed=seed)
    iterations, notes = [], []
    for number, (tr, va) in enumerate(splits):
        budget = None
        if time_budget is not None:
            remaining = time_budget - (time.time() - started)
            if remaining <= 0 and iterations:
                notes.append(f"подбор раундов: бюджет исчерпан, посчитано частей {len(iterations)} из {len(splits)}")
                break
            budget = max(remaining, 0.0) / (len(splits) - number)
        try:
            pre = TabularPreprocessor()
            boosting = BoostingModel(model, kind, n_classes=n_classes, params=params, n_jobs=n_jobs, seed=seed,
                                     max_rounds=max_rounds, learning_rate=learning_rate, time_budget=budget)
            boosting.fit(pre.fit_transform(X.iloc[tr]), target[tr], pre.transform(X.iloc[va]), target[va])
            iterations.append(boosting.best_iteration_)
        except Exception as error:  # одна неудачная часть не должна ронять финальное обучение
            notes.append(f"подбор раундов: часть {number + 1} пропущена: {type(error).__name__}: {error}")
    if not iterations:
        raise ValueError("число раундов не подобрано ни на одной части; " + "; ".join(notes[-3:]))
    return max(1, int(round(float(np.mean(iterations)) * margin))), plan["strategy"], notes


def fit_predict(split: SplitData, profile: dict, model: str = "lightgbm", n_rounds: Optional[int] = None,
                params: Optional[dict] = None, task_type: Optional[str] = None,
                target_transform: Optional[str] = None, n_jobs: Optional[int] = None, seed: int = 0,
                time_budget: Optional[float] = None, max_rounds: int = 3000, learning_rate: float = 0.05,
                validation: Optional[dict] = None) -> dict:
    """Обучить на всём train и предсказать test.

    n_rounds — число раундов (обычно rounds_from(результат train_eval)). Если не задано, оно
    подбирается ранней остановкой по частям валидации (схема validation или suggest_validation);
    обучать max_rounds без остановки не будем. Регрессия: pred (n,) в исходной шкале таргета. Классификация: proba (n, K)
    и classes — порядок столбцов; pred содержит метки классов (по наибольшей вероятности)."""
    started = time.time()
    try:
        kind = _resolve_task(profile, task_type)  # метрика финальному обучению не нужна
        if target_transform not in TARGET_TRANSFORMS:
            raise ValueError(f"преобразование таргета {target_transform!r} не поддержано: {TARGET_TRANSFORMS}")
        if not isinstance(split.y_train, pd.Series):
            raise ValueError("несколько таргетов пока не поддержаны")
        codec, target = _prepare_target(split.y_train, kind, target_transform)
        n_classes = len(codec.classes_) if codec else None
        notes = list(prepare_params(model, kind, params)[1]) if model in ("lightgbm", "xgboost", "catboost") else []
        if n_rounds is None:
            n_rounds, scheme, round_notes = _auto_rounds(
                split.X_train, target, kind, profile, n_classes, model, params, n_jobs, seed, max_rounds,
                learning_rate, None if time_budget is None else time_budget / 2, validation=validation,
            )
            notes += round_notes
            notes.append(f"n_rounds={n_rounds} подобрано ранней остановкой (схема {scheme})")
        pre = TabularPreprocessor()
        X_tr, X_te = pre.fit_transform(split.X_train), pre.transform(split.X_test)
        left = None if time_budget is None else max(0.0, time_budget - (time.time() - started))
        boosting = BoostingModel(
            model, kind, n_classes=n_classes, params=params, n_jobs=n_jobs,
            seed=seed, max_rounds=max_rounds, learning_rate=learning_rate, time_budget=left,
        )
        boosting.fit(X_tr, target, n_rounds=n_rounds)
        raw = boosting.predict(X_te)
        out = {"status": "ok", "model": model, "task_type": kind, "n_rounds": boosting.best_iteration_,
               "stopped_by_time": bool(boosting.stopped_by_time_),  # True — обучение прервано по time_budget
               "seconds": time.time() - started, "notes": notes,
               "warnings": list(pre.warnings_) + list(split.warnings)}
        if kind == "regression":
            out["pred"] = _inverse(raw, kind, target_transform)
        else:
            out.update(proba=raw, classes=list(codec.classes_), pred=codec.decode(raw.argmax(axis=1)))
        return out
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}", "seconds": time.time() - started}


# ---- сдвиг train/test: двухступенчатая схема «фон против остальных» ----------------------------
SHIFT_AUC = 0.8          # adversarial AUC train/test, выше которого включаются механизмы для сдвига
BACKGROUND_SHARE = 0.3   # самый частый класс с такой долей train считается «фоном» (Normal среди атак)


def _time_left(deadline, phase: str):
    """Остаток общего бюджета; проверки кооперативные, между операциями и итерациями модели."""
    if deadline is None:
        return None
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError(f"бюджет времени исчерпан: {phase}")
    return left


def shift_check(X_train: pd.DataFrame, X_test: pd.DataFrame, seed: int = 0, n_jobs: Optional[int] = None,
                max_rows: int = 100_000, time_budget: Optional[float] = None) -> dict:
    """Насколько test отличается от train: AUC LightGBM, отличающего строку test от строки train (3 части, вне фолда).
    Успех: status='ok', complete=True, auc/shifted. Около 0.5 — случайное разбиение; выше SHIFT_AUC — сильный сдвиг.
    При исчерпании time_budget: skipped, complete=False, auc=None; частичная проверка не считается оценкой сдвига."""
    started = time.monotonic()
    deadline = None if time_budget is None else started + time_budget
    try:
        _time_left(deadline, "до проверки сдвига")
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        rng = np.random.default_rng(seed)

        def cap(frame):
            return frame if len(frame) <= max_rows else frame.iloc[np.sort(rng.choice(len(frame), max_rows, replace=False))]

        a, b = cap(X_train), cap(X_test)
        _time_left(deadline, "после выбора строк для проверки сдвига")
        pre = TabularPreprocessor().fit(a)
        _time_left(deadline, "после подготовки признаков проверки сдвига")
        both = pd.concat([pre.transform(a), pre.transform(b)], ignore_index=True)
        _time_left(deadline, "после преобразования признаков проверки сдвига")
        z = np.r_[np.zeros(len(a)), np.ones(len(b))]
        p = np.zeros(len(both))

        def stop_on_time(env):
            _time_left(deadline, "во время обучения проверки сдвига")

        for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(both, z):
            _time_left(deadline, "перед очередной частью проверки сдвига")
            m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=63, verbose=-1, random_state=seed,
                                   n_jobs=n_jobs or default_n_jobs())
            fit_args = {"callbacks": [stop_on_time]} if deadline is not None else {}
            m.fit(both.iloc[tr], z[tr], **fit_args)
            _time_left(deadline, "после обучения проверки сдвига")
            p[va] = m.predict_proba(both.iloc[va])[:, 1]
            _time_left(deadline, "после предсказания проверки сдвига")
        auc = float(roc_auc_score(z, p))
        _time_left(deadline, "после расчёта AUC проверки сдвига")
        return {"status": "ok", "complete": True, "auc": round(auc, 4), "shifted": auc > SHIFT_AUC,
                "seconds": time.monotonic() - started}
    except TimeoutError as error:
        return {"status": "skipped", "complete": False, "auc": None, "shifted": False,
                "reason": str(error), "seconds": time.monotonic() - started}


def _em_share(p: np.ndarray, prior: float, iters: int = 200) -> float:
    """Доля класса на test по вероятностям бинарной модели, обученной при доле prior (EM, Saerens et al.)."""
    q = prior
    for _ in range(iters):
        adj = q / prior * p
        adj = adj / (adj + (1 - q) / (1 - prior) * (1 - p))
        q_new = float(adj.mean())
        if abs(q_new - q) < 1e-7:
            break
        q = q_new
    return q


def _background_mask(p: np.ndarray, share: float) -> np.ndarray:
    """Ровно round(share*n) фоновых строк; при ties порядок исходных строк служит стабильным разрешением."""
    n = len(p)
    k = int(np.clip(np.rint(share * n), 0, n))
    mask = np.zeros(n, dtype=bool)
    if k:
        mask[np.argsort(-np.asarray(p), kind="stable")[:k]] = True
    return mask


def fit_predict_two_stage(split: SplitData, profile: dict, models=("lightgbm", "xgboost"), learning_rate: float = 0.1,
                          n_jobs: Optional[int] = None, seed: int = 0, time_budget: Optional[float] = None,
                          check: Optional[dict] = None) -> dict:
    """Многоклассовая задача со сдвигом train/test и «фоновым» классом: сначала «фон против остальных», потом класс.

    Только для Macro F1 по меткам. Включается, если классов не меньше трёх, самый частый класс занимает >= BACKGROUND_SHARE train и adversarial
    AUC train/test > SHIFT_AUC (check — результат shift_check, если уже посчитан); иначе status='skipped' и агент работает
    как обычно. Для каждой модели из models: бинарная «фон / не фон» и мультикласс без фона (train_eval -> rounds_from ->
    fit_predict), вероятности смешиваются поровну, веса классов — choose_class_weights по смешанному OOF. Доля фона на
    test — EM по бинарной смеси; фоном считаются строки с наибольшей вероятностью в этой доле. Возвращает pred (исходные
    метки), background, share_train, share_test, auc, notes. Замер (public UNSW-NB15, смесь LightGBM+XGBoost, 3 seed):
    Macro F1 0.5350 -> 0.5457, лучше на каждом seed."""
    started = time.monotonic()
    deadline = None if time_budget is None else started + time_budget
    notes = []

    def skipped(reason, **info):
        return {"status": "skipped", "applied": False, "reason": reason, "notes": list(notes),
                "seconds": time.monotonic() - started, **info}

    try:
        _time_left(deadline, "до двухступенчатой схемы")
        y = split.y_train
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                return skipped("несколько таргетов")
            y = y.iloc[:, 0]
        if _resolve_task(profile, None) != "classification":
            return skipped("не классификация")
        primary = (profile.get("metric") or {}).get("primary")
        if primary is None:
            notes.append("основная метрика не указана: двухступенчатая схема предназначена только для Macro F1")
            return skipped(notes[-1])
        try:
            metric = ml_metrics.get_metric(primary)
        except ml_metrics.MetricError:
            return skipped(f"метрика {primary!r} не поддержана: двухступенчатая схема предназначена только для Macro F1")
        if metric.name != "f1_macro":
            return skipped(f"метрика {metric.name!r}: двухступенчатая схема предназначена только для Macro F1")
        counts = y.value_counts(normalize=True)
        if len(counts) < 3 or counts.iloc[0] < BACKGROUND_SHARE:
            return skipped(f"нет фонового класса: классов {len(counts)}, доля самого частого {counts.iloc[0]:.3f}")
        left = _time_left(deadline, "перед проверкой сдвига")
        check = check or shift_check(split.X_train, split.X_test, seed=seed, n_jobs=n_jobs, time_budget=left)
        if check.get("complete") is False or check.get("status") in ("skipped", "error"):
            return skipped(check.get("reason") or check.get("error") or "проверка сдвига не завершена", auc=check.get("auc"))
        _time_left(deadline, "после проверки сдвига")
        if not check["shifted"]:
            return skipped(f"сдвига train/test нет: adversarial AUC {check['auc']} <= {SHIFT_AUC}", auc=check["auc"])
        background, share_train = counts.index[0], float(counts.iloc[0])
        is_bg = (y == background).to_numpy()
        binary = replace(split, y_train=pd.Series(np.where(is_bg, "bg", "rest"), index=y.index))
        attack = replace(split, X_train=split.X_train[~is_bg], y_train=y[~is_bg])
        task = profile.get("task") or {}
        prof_bin = {**profile, "task": {**task, "type": "binary"}}
        prof_att = {**profile, "task": {**task, "type": "multiclass" if (~is_bg).any() and y[~is_bg].nunique() > 2 else "binary"}}
        _time_left(deadline, "после подготовки данных двухступенчатой схемы")
        stages_left = 4 * len(models)
        common = dict(task_type="classification", learning_rate=learning_rate, n_jobs=n_jobs, seed=seed)

        def stage_budget():
            left = _time_left(deadline, "перед следующим обучением двухступенчатой схемы")
            return None if left is None else left / stages_left

        p_bg, att_proba, att_oof, classes = [], [], [], None
        for model in models:
            ev = train_eval(binary, prof_bin, model=model, time_budget=stage_budget(), **common)
            stages_left -= 1
            _time_left(deadline, "после оценки фоновой модели")
            if ev.status != "ok":
                raise RuntimeError(f"{model}, фон / не фон: {ev.error}")
            out = fit_predict(binary, prof_bin, model=model, n_rounds=rounds_from(ev), time_budget=stage_budget(), **common)
            stages_left -= 1
            _time_left(deadline, "после предсказания фоновой модели")
            if out["status"] != "ok":
                raise RuntimeError(f"{model}, фон / не фон: {out['error']}")
            p_bg.append(np.asarray(out["proba"])[:, [str(c) for c in out["classes"]].index("bg")])
            ev = train_eval(attack, prof_att, model=model, time_budget=stage_budget(), **common)
            stages_left -= 1
            _time_left(deadline, "после оценки модели без фона")
            if ev.status != "ok":
                raise RuntimeError(f"{model}, без фона: {ev.error}")
            out = fit_predict(attack, prof_att, model=model, n_rounds=rounds_from(ev), time_budget=stage_budget(), **common)
            stages_left -= 1
            _time_left(deadline, "после предсказания модели без фона")
            if out["status"] != "ok":
                raise RuntimeError(f"{model}, без фона: {out['error']}")
            classes = list(out["classes"]) if classes is None else classes
            att_proba.append(np.asarray(out["proba"]))
            order = np.argsort(np.asarray(ev.oof_index))
            att_oof.append((np.asarray(ev.oof_index)[order], np.asarray(ev.oof_pred)[order]))
        p = np.mean(p_bg, axis=0)
        proba = np.mean(att_proba, axis=0)
        idx = att_oof[0][0]
        for other, _ in att_oof[1:]:
            idx = np.intersect1d(idx, other)
        oof = np.mean([pred[np.searchsorted(i, idx)] for i, pred in att_oof], axis=0)
        y_att = attack.y_train.reset_index(drop=True)
        codes = LabelCodec().fit(y_att).encode(y_att.iloc[idx])
        _time_left(deadline, "перед подбором весов классов")
        choice = ml_postprocess.choose_class_weights(oof, codes, groups=ml_validation.row_groups(attack.X_train.iloc[idx]))
        _time_left(deadline, "после подбора весов классов")
        share = _em_share(p, share_train)
        _time_left(deadline, "после оценки доли фона")
        is_test_bg = _background_mask(p, share)
        labels = np.asarray(classes, dtype=object)
        pred = np.where(is_test_bg, background, labels[(proba * choice["weights"]).argmax(1)])
        _time_left(deadline, "после постобработки предсказаний")
        notes.append(f"двухступенчатая схема: фон {background!r}, доля train {share_train:.3f}, оценка на test {share:.3f}, "
                     f"adversarial AUC {check['auc']}, модели {list(models)}")
        return {"status": "ok", "applied": True, "pred": pred, "background": background, "share_train": share_train,
                "share_test": share, "auc": check["auc"], "class_weights_applied": bool(choice["applied"]),
                "models": list(models), "notes": notes, "seconds": time.monotonic() - started}
    except TimeoutError as error:
        return skipped(str(error))
    except Exception as error:  # ожидаемые сбои — статусом, как в train_eval / fit_predict
        return {"status": "error", "applied": False, "error": f"{type(error).__name__}: {error}",
                "trace": traceback.format_exc(limit=3), "seconds": time.monotonic() - started}

