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

MODELS = ("lightgbm", "xgboost", "catboost")
TASKS = ("regression", "classification")


class ModelError(ValueError):
    """Неизвестная модель или несовместимые параметры."""


def default_n_jobs() -> int:
    """Потоки по умолчанию: переменная ML_N_JOBS или не больше 8 (лимит платформы)."""
    env = os.environ.get("ML_N_JOBS")
    return max(1, int(env)) if env else max(1, min(8, os.cpu_count() or 1))


class BoostingModel:
    def __init__(self, name: str, task: str, n_classes: Optional[int] = None, params: Optional[dict] = None,
                 n_jobs: Optional[int] = None, seed: int = 0, max_rounds: int = 3000,
                 learning_rate: float = 0.05, early_stopping_rounds: int = 100,
                 time_budget: Optional[float] = None):
        if name not in MODELS:
            raise ModelError(f"неизвестная модель {name!r}; доступны: {MODELS}")
        if task not in TASKS:
            raise ModelError(f"неизвестный тип задачи {task!r}; доступны: {TASKS}")
        if task == "classification" and (n_classes is None or n_classes < 2):
            raise ModelError("для классификации нужно n_classes >= 2")
        self.name, self.task, self.n_classes = name, task, n_classes
        self.params = dict(params or {})
        self.n_jobs = n_jobs or default_n_jobs()
        self.seed, self.max_rounds, self.learning_rate = seed, max_rounds, learning_rate
        self.early_stopping_rounds, self.time_budget = early_stopping_rounds, time_budget

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
        getattr(self, f"_fit_{self.name}")(X, y, X_valid if has_valid else None,
                                           np.asarray(y_valid) if has_valid else None, rounds, deadline)
        self.fit_seconds_ = time.time() - started
        self.stopped_by_time_ = self._time_stopped
        return self

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
        out[:, self._present] = proba  # класс, не попавший в fit, получает вероятность 0
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
        fit_args = {"callbacks": callbacks}
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
            enable_categorical=True, n_jobs=self.n_jobs, random_state=self.seed, verbosity=0,
            callbacks=callbacks or None,
        )
        if Xv is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds
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
