"""Единый интерфейс к LightGBM, XGBoost и CatBoost для регрессии и классификации.

Признаки приходят из TabularPreprocessor: числа с пропусками и категории в dtype category.
Классификация ждёт целые метки 0..K-1 (LabelCodec) и возвращает вероятности (n, K).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd

import ml_metrics
from ml_postprocess import macro_f1_codes

MODELS = ("lightgbm", "xgboost", "catboost")
TASKS = ("regression", "classification")
_MIN_PROBA = 1e-4  # вероятность класса, которого не было в обучающей части


class ModelError(ValueError):
    """Неизвестная модель или несовместимые параметры."""


# Число раундов и раннюю остановку задаёт код (train_eval подбирает, fit_predict получает n_rounds):
# синонимы в params роняли бы fit_predict или молча урезали модель.
_ROUND_PARAMS = {
    "n_estimators", "num_boost_round", "num_iterations", "num_iteration", "num_trees", "num_tree",
    "num_round", "num_rounds", "nrounds", "n_iter", "max_iter", "iterations", "max_iterations",
    "early_stopping", "early_stopping_round", "early_stopping_rounds", "n_iter_no_change", "od_wait", "od_type",
}
# CatBoost падает на параметрах LightGBM/XGBoost: известные переводим, остальные убираем
_CATBOOST_RENAME = {"max_depth": "depth", "colsample_bytree": "rsm", "feature_fraction": "rsm",
                    "colsample_bylevel": "rsm", "reg_lambda": "l2_leaf_reg", "lambda_l2": "l2_leaf_reg"}
_CATBOOST_DROP = {"num_leaves", "min_child_samples", "min_data_in_leaf", "min_child_weight", "gamma",
                  "reg_alpha", "lambda_l1", "subsample_freq", "bagging_freq", "colsample_bynode",
                  "min_split_gain", "is_unbalance", "max_bin_by_feature"}


# Метрика задачи -> встроенная метрика библиотеки для ранней остановки. Чего здесь нет,
# LightGBM и XGBoost считают своей функцией, а CatBoost оставляет свою метрику по умолчанию.
_NATIVE = {
    "lightgbm": {
        "regression": {"rmse": "l2", "mse": "l2", "r2": "l2", "mae": "l1"},
        "binary": {"logloss": "binary_logloss", "roc_auc": "auc", "average_precision": "average_precision",
                   "accuracy": "binary_error"},
        "multiclass": {"logloss": "multi_logloss", "roc_auc": "auc_mu", "accuracy": "multi_error"},
    },
    "xgboost": {
        "regression": {"rmse": "rmse", "mse": "rmse", "r2": "rmse", "mae": "mae"},
        "binary": {"logloss": "logloss", "roc_auc": "auc", "average_precision": "aucpr", "accuracy": "error"},
        "multiclass": {"logloss": "mlogloss", "roc_auc": "auc", "accuracy": "merror"},
    },
    "catboost": {
        "regression": {"rmse": "RMSE", "mse": "RMSE", "r2": "R2", "mae": "MAE"},
        "binary": {"logloss": "Logloss", "roc_auc": "AUC", "average_precision": "PRAUC", "accuracy": "Accuracy",
                   "f1_macro": "TotalF1:average=Macro", "f1_weighted": "TotalF1:average=Weighted",
                   "f1_micro": "TotalF1:average=Micro"},
        "multiclass": {"logloss": "MultiClass", "roc_auc": "AUC", "accuracy": "Accuracy",
                       "f1_macro": "TotalF1:average=Macro", "f1_weighted": "TotalF1:average=Weighted",
                       "f1_micro": "TotalF1:average=Micro"},
    },
}
# Метрики, которые меняются ступенями (доля верных ответов, F1): лучшая итерация по ним на валидации — во
# многом удача, OOF завышается без выигрыша на скрытых данных (замер Ивана 25.10, 3 сида: CatBoost TotalF1
# CV +0.003…+0.012 при скрытой −0.006…−0.002; LightGBM multi_error CV +0.014 при скрытой +0.002), а агент
# выбирает модели и веса смеси по OOF. Поэтому по ним остановка — только при custom_eval=True.
_STEP_METRICS = {"accuracy", "f1_macro", "f1_weighted", "f1_micro"}
# встроенный многоклассовый AUC библиотек устроен иначе, чем наш one-vs-rest: это приближение,
# зато без расчёта AUC по каждому классу на каждой итерации
_APPROXIMATE = {"auc_mu", ("xgboost", "multiclass", "roc_auc"), ("catboost", "multiclass", "roc_auc")}


def _proba_matrix(pred) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    return np.column_stack([1.0 - pred, pred]) if pred.ndim == 1 else pred


def _score_function(metric: ml_metrics.Metric, kind: str, n_classes: int):
    """(y_true, предсказание библиотеки) -> значение метрики задачи; при сбое метрики — худшее значение."""
    worst = -1e9 if metric.higher_is_better else 1e9

    def score(y_true, pred):
        try:
            y_true = np.asarray(y_true)
            if kind == "regression":
                return float(metric(y_true, np.asarray(pred, dtype=float).ravel()))
            proba = _proba_matrix(pred)
            codes = y_true.astype("int64")
            if metric.name == "f1_macro":
                return macro_f1_codes(codes, proba.argmax(axis=1), proba.shape[1])  # быстрее sklearn
            if metric.needs == "labels":
                return float(metric(codes, proba.argmax(axis=1)))
            return float(metric(codes, proba, labels=list(range(proba.shape[1]))))
        except (ml_metrics.MetricError, ValueError):
            return worst
    return score


def prepare_params(name: str, task: str, params: Optional[dict]):
    """Параметры пользователя для библиотеки name: (чистые params, заметки об изменениях)."""
    clean, notes = {}, []
    for key, value in dict(params or {}).items():
        if key in _ROUND_PARAMS:
            notes.append(f"параметр {key!r} убран: число раундов и раннюю остановку задаёт код")
        elif key in ("boosting_type", "boosting", "boost", "booster") and str(value).lower() == "dart":
            notes.append(f"{key}='dart' убран: в этом режиме не работает ранняя остановка и уходит весь бюджет")
        elif name == "catboost" and key in _CATBOOST_DROP:
            notes.append(f"параметр {key!r} не поддерживается CatBoost и убран")
        elif name == "catboost" and key in _CATBOOST_RENAME:
            clean[_CATBOOST_RENAME[key]] = value
            notes.append(f"параметр {key!r} для CatBoost передан как {_CATBOOST_RENAME[key]!r}")
        elif name == "catboost" and key == "class_weight":
            if task == "classification" and str(value).lower() == "balanced":
                clean["auto_class_weights"] = "Balanced"
                notes.append("class_weight='balanced' для CatBoost передан как auto_class_weights='Balanced'")
            else:
                notes.append(f"параметр class_weight={value!r} для CatBoost не поддержан и убран")
        else:
            clean[key] = value
    return clean, notes


def default_n_jobs() -> int:
    """Потоки по умолчанию: переменная ML_N_JOBS или не больше 8 (лимит платформы)."""
    env = os.environ.get("ML_N_JOBS")
    return max(1, int(env)) if env else max(1, min(8, os.cpu_count() or 1))


class BoostingModel:
    def __init__(self, name: str, task: str, n_classes: Optional[int] = None, params: Optional[dict] = None,
                 n_jobs: Optional[int] = None, seed: int = 0, max_rounds: int = 3000,
                 learning_rate: float = 0.05, early_stopping_rounds: int = 100,
                 time_budget: Optional[float] = None, eval_metric=None, custom_eval: bool = False):
        if name not in MODELS:
            raise ModelError(f"неизвестная модель {name!r}; доступны: {MODELS}")
        if task not in TASKS:
            raise ModelError(f"неизвестный тип задачи {task!r}; доступны: {TASKS}")
        if task == "classification" and (n_classes is None or n_classes < 2):
            raise ModelError("для классификации нужно n_classes >= 2")
        self.name, self.task, self.n_classes = name, task, n_classes
        self.params, self.param_notes_ = prepare_params(name, task, params)
        self.n_jobs = n_jobs or default_n_jobs()
        self.seed, self.max_rounds, self.learning_rate = seed, max_rounds, learning_rate
        self.early_stopping_rounds, self.time_budget = early_stopping_rounds, time_budget
        # метрика задачи (имя или ml_metrics.Metric) для ранней остановки; None — метрика библиотеки
        self.eval_metric = ml_metrics.get_metric(eval_metric) if eval_metric is not None else None
        # своя функция метрики на каждой итерации: на публичной классификации +0.003 Macro F1 на скрытой
        # части при обучении на 40–60% дольше, поэтому по умолчанию выключена
        self.custom_eval = custom_eval

    # ---- публичный интерфейс -------------------------------------------------
    def fit(self, X: pd.DataFrame, y, X_valid=None, y_valid=None, n_rounds: Optional[int] = None):
        """С валидацией — ранняя остановка и best_iteration_; без неё — n_rounds (или max_rounds)."""
        y = np.asarray(y)
        self._check_labels(y)
        has_valid = X_valid is not None and y_valid is not None
        if self.task == "classification":
            # библиотеки требуют подряд идущие классы 0..k-1: если в fit не хватает класса, переиндексируем
            self._present = np.unique(y)
            if len(self._present) < 2:
                raise ModelError("для классификации в обучающих данных нужно хотя бы два класса")
            y = np.searchsorted(self._present, y)
            if has_valid:
                y_valid = np.asarray(y_valid)
                keep = np.isin(y_valid, self._present)
                X_valid, y_valid = X_valid.loc[keep] if hasattr(X_valid, "loc") else X_valid[keep], y_valid[keep]
                y_valid = np.searchsorted(self._present, y_valid)
                has_valid = len(y_valid) > 0
        rounds = n_rounds or self.max_rounds
        deadline = time.time() + self.time_budget if self.time_budget is not None else None
        started = time.time()
        self._time_stopped = False  # ставят callbacks, только когда действительно прервали обучение
        self._stop_native, self._stop_custom = self._stopping_metric(has_valid)
        getattr(self, f"_fit_{self.name}")(X, y, X_valid if has_valid else None,
                                           np.asarray(y_valid) if has_valid else None, rounds, deadline)
        self.fit_seconds_ = time.time() - started
        self.stopped_by_time_ = self._time_stopped
        return self

    def _stopping_metric(self, has_valid: bool):
        """(встроенная метрика | None, своя функция | None) для ранней остановки; заметка — в eval_note_."""
        metric = self.eval_metric
        if not has_valid or metric is None:
            self.eval_note_ = None
            return None, None
        kind = "regression" if self.task == "regression" else ("binary" if len(self._present) == 2 else "multiclass")
        if metric.task != ("regression" if kind == "regression" else "classification"):
            self.eval_note_ = f"метрика {metric.name} не подходит к задаче: остановка по метрике библиотеки"
            return None, None
        native = _NATIVE[self.name][kind].get(metric.name)
        if metric.name in _STEP_METRICS and not self.custom_eval:
            self.eval_note_ = (f"{metric.name} меняется ступенями: остановка по метрике библиотеки, ранняя остановка "
                               "по ней завышает CV без выигрыша на скрытых данных (включается custom_stopping=True)")
            return None, None
        if native:
            approx = native in _APPROXIMATE or (self.name, kind, metric.name) in _APPROXIMATE
            self.eval_note_ = f"ранняя остановка по {native}" + (f" (приближение {metric.name})" if approx else "")
            return native, None
        if self.name == "catboost":
            self.eval_note_ = f"у CatBoost нет метрики {metric.name}: остановка по его метрике по умолчанию"
            return None, None
        if not self.custom_eval:
            self.eval_note_ = (f"встроенной метрики {metric.name} у {self.name} нет: остановка по метрике библиотеки "
                               "(своя функция включается custom_stopping=True)")
            return None, None
        self.eval_note_ = f"ранняя остановка по {metric.name} (своя функция)"
        return None, _score_function(metric, kind, len(self._present) if self.task != "regression" else 0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Регрессия — значения (n,); классификация — вероятности (n, K)."""
        if not hasattr(self, "model_"):
            raise ModelError("сначала вызовите fit()")
        if self.task == "regression":
            return np.asarray(self._predict_raw(X), dtype=float).ravel()
        proba = np.asarray(self._predict_raw(X), dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        out = np.zeros((len(X), self.n_classes))
        out[:, self._present] = proba
        if len(self._present) < self.n_classes:
            # класс, не попавший в fit, получает малую вероятность вместо нуля: при нуле logloss на его
            # строках около 34 и портит сравнение моделей; выбор класса (argmax) от этого не меняется
            out = np.clip(out, _MIN_PROBA, None)
            out /= out.sum(axis=1, keepdims=True)
        return out

    # ---- общее --------------------------------------------------------------
    def _check_labels(self, y):
        if self.task == "classification":
            if y.dtype.kind not in "iu" or y.min() < 0 or y.max() >= self.n_classes:
                raise ModelError(f"метки классификации должны быть целыми от 0 до {self.n_classes - 1}")
        elif not np.isfinite(y.astype(float)).all():
            raise ModelError("в таргете регрессии есть NaN или inf")

    def _predict_raw(self, X):
        X = self._prepare(X)
        return self.model_.predict(X) if self.task == "regression" else self.model_.predict_proba(X)

    def _prepare(self, X):
        return X

    # ---- LightGBM -----------------------------------------------------------
    def _fit_lightgbm(self, X, y, Xv, yv, rounds, deadline):
        import lightgbm as lgb

        params = dict(
            n_estimators=rounds, learning_rate=self.learning_rate, num_leaves=63, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=self.n_jobs, random_state=self.seed, verbose=-1,
        )
        params.update(self.params)
        fit_args = {}
        if "metric" not in self.params:  # явно заданная метрика пользователя важнее
            if self._stop_native:
                params["metric"] = self._stop_native
            elif self._stop_custom:
                score, higher = self._stop_custom, self.eval_metric.higher_is_better
                name = self.eval_metric.name
                params["metric"] = "None"  # только наша метрика, иначе остановка смотрит и на встроенную
                fit_args["eval_metric"] = lambda y_true, y_pred: (name, score(y_true, y_pred), higher)
        callbacks = []
        if Xv is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        if deadline is not None:
            best = {"iteration": None, "score": None, "results": []}

            def stop_on_time(env):
                # запоминаем лучшую итерацию по первой метрике валидации, чтобы при остановке
                # по времени вернуть её, а не последнюю
                results = env.evaluation_result_list or []
                if results:
                    _, _, score, higher_is_better = results[0][:4]
                    better = best["score"] is None or (score > best["score"] if higher_is_better else score < best["score"])
                    if better:
                        best.update(iteration=env.iteration, score=score, results=list(results))
                if time.time() >= deadline:
                    self._time_stopped = True
                    stop_at = env.iteration if best["iteration"] is None else best["iteration"]
                    raise lgb.callback.EarlyStopException(stop_at, best["results"] or results)
            stop_on_time.order = 30
            callbacks.append(stop_on_time)
        cls = lgb.LGBMRegressor if self.task == "regression" else lgb.LGBMClassifier
        self.model_ = cls(**params)
        fit_args["callbacks"] = callbacks
        if Xv is not None:
            fit_args["eval_set"] = [(Xv, yv)]
        self.model_.fit(X, y, **fit_args)
        self.best_iteration_ = int(self.model_.best_iteration_ or self.model_.booster_.current_iteration())

    # ---- XGBoost ------------------------------------------------------------
    def _fit_xgboost(self, X, y, Xv, yv, rounds, deadline):
        import xgboost as xgb

        callbacks, owner = [], self
        if deadline is not None:
            class StopOnTime(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    if time.time() >= deadline:
                        owner._time_stopped = True
                        return True
                    return False
            callbacks.append(StopOnTime())
        params = dict(
            n_estimators=rounds, learning_rate=self.learning_rate, max_depth=7, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, tree_method="hist",
            # «категория против остальных» вместо разбиения множества категорий: при разбиении новые
            # в test категории (у нас — пропуск) уходили в случайную ветку, и на спрятанном году
            # регрессии прогнозы доходили до 1576 при реальных 37–200 (RMSE 154 против 42 с этой настройкой)
            enable_categorical=True, max_cat_to_onehot=256,
            n_jobs=self.n_jobs, random_state=self.seed, verbosity=0,
            callbacks=callbacks or None,
        )
        if Xv is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds
            if self._stop_native:
                params["eval_metric"] = self._stop_native
            elif self._stop_custom:
                score, higher = self._stop_custom, self.eval_metric.higher_is_better

                def task_metric(y_true, y_pred):
                    # XGBoost останавливается по последней метрике и считает, что её надо уменьшать
                    value = score(y_true, y_pred)
                    return -value if higher else value
                params["eval_metric"] = task_metric
        params.update(self.params)
        cls = xgb.XGBRegressor if self.task == "regression" else xgb.XGBClassifier
        self.model_ = cls(**params)
        self.model_.fit(X, y, eval_set=[(Xv, yv)] if Xv is not None else None, verbose=False)
        best = getattr(self.model_, "best_iteration", None)
        self.best_iteration_ = int(best + 1) if best is not None else int(self.model_.get_booster().num_boosted_rounds())

    # ---- CatBoost -----------------------------------------------------------
    def _fit_catboost(self, X, y, Xv, yv, rounds, deadline):
        from catboost import CatBoostClassifier, CatBoostRegressor

        self._cat_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        params = dict(
            iterations=rounds, learning_rate=self.learning_rate, depth=6, thread_count=self.n_jobs,
            random_seed=self.seed, verbose=False, allow_writing_files=False,
        )
        if Xv is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds
            # явно: CatBoost сам выключает use_best_model, например при постоянном таргете валидации, и тогда
            # предсказывал всеми деревьями, а сообщал лучшую итерацию (RMSE 2.84 против 0.29 у лучшего префикса)
            params["use_best_model"] = True
            if self._stop_native:
                native = self._stop_native
                weighted = any(k in self.params for k in ("auto_class_weights", "class_weights", "scale_pos_weight"))
                if weighted and native.startswith(("TotalF1", "Accuracy")):
                    # метрика задачи невзвешенная, а при весах классов CatBoost по умолчанию взвешивает и её;
                    # без весов параметр use_weights CatBoost не принимает
                    native += (";" if ":" in native else ":") + "use_weights=false"
                params["eval_metric"] = native
        params.update(self.params)
        cls = CatBoostRegressor if self.task == "regression" else CatBoostClassifier
        self.model_ = cls(**params)
        fit_args = {"cat_features": self._cat_cols}
        if Xv is not None:
            fit_args["eval_set"] = (self._prepare(Xv), yv)  # с eval_set CatBoost оставляет лучшую итерацию
        if deadline is not None:
            owner = self

            class StopOnTime:
                def after_iteration(self, info):
                    if time.time() >= deadline:
                        owner._time_stopped = True
                        return False  # False — остановить обучение
                    return True
            fit_args["callbacks"] = [StopOnTime()]
        self.model_.fit(self._prepare(X), y, **fit_args)
        best = self.model_.get_best_iteration()
        self.best_iteration_ = int(best + 1) if best is not None else int(self.model_.tree_count_)

    def _prepare(self, X):
        if self.name != "catboost" or not getattr(self, "_cat_cols", None):
            return X
        X = X.copy()
        for col in self._cat_cols:
            X[col] = X[col].astype(object).where(X[col].notna(), "__nan__").astype(str)
        return X
