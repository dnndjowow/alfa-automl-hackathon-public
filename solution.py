"""Точка входа платформы: python3 solution.py task.md (из каталога задачи).

Порядок: страховочный ответ -> LLM-агент управляет экспериментами -> код гарантирует, что
в submission.csv лежит лучший обученный запуск. Если LLM недоступна или не довела дело до
обученной модели, тот же набор инструментов работает по простой фиксированной политике.

Переменные окружения:
  OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL — выданная модель (обязательны для агента);
  AGENT_TIME_LIMIT — общий лимит секунд (по умолчанию 10800 = 3 часа платформы);
  AGENT_SAFETY_S — запас до лимита на запись ответа и выход (по умолчанию 900);
  AGENT_MAX_STEPS — максимум шагов диалога с моделью (по умолчанию 40);
  AGENT_LLM=off — не обращаться к LLM (офлайн-проверки, чтобы не тратить общий бюджет ключа).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

START = time.monotonic()
# модули решения лежат рядом с solution.py: импорт не должен зависеть от способа запуска
sys.path.insert(0, str(Path(__file__).resolve().parent))


def log(message: str) -> None:
    elapsed = time.monotonic() - START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def load_local_env() -> None:
    """Локальная разработка: .env рядом с solution.py. На платформе файла нет, переменные уже заданы."""
    path = Path(__file__).resolve().parent / ".env"
    if path.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(path, override=False)
        except Exception as error:
            log(f".env не прочитан: {type(error).__name__}")


def fallback_policy(session) -> None:
    """Без LLM: эвристическая постановка (или уже выбранная LLM), одна сильная база, финальное обучение."""
    if session.setup is None:
        suggested = session.overview()["suggested"]
        targets, task_type = suggested["target_cols"], suggested["task_type"]
        if not targets:
            log("[fallback] таргет не найден")
            return
        if len(targets) > 1 or task_type not in ("regression", "binary", "multiclass"):
            import ml_schema
            # несколько таргетов: тип по первому; учатся таргеты того же типа, остальные получат константу
            # текст условия общий и не говорит, к какому таргету относится: для остальных — только данные
            kinds = {t: ml_schema.suggest_task_type(session.data.train[t], session.data.text if i == 0 else "")["type"]
                     for i, t in enumerate(targets)}
            task_type = kinds[targets[0]]
            targets = [t for t in targets if kinds[t] == task_type]
        if task_type not in ("regression", "binary", "multiclass"):
            log(f"[fallback] тип задачи {task_type!r} не поддержан")
            return
        metric = suggested["metric"]
        family = "regression" if task_type == "regression" else "classification"
        if metric is None or metric_family(metric) != family:
            metric = ("rmse" if family == "regression"
                      else "roc_auc" if task_type == "binary" and session.sample_suggests_proba() else "f1_macro")
        result = session.call("set_task", {"target_cols": targets, "id_cols": suggested["id_cols"],
                                           "task_type": task_type, "metric": metric, "reason": "fallback"})
        if not result.get("ok") and len(targets) > 1:
            result = session.call("set_task", {"target_cols": targets[:1], "id_cols": suggested["id_cols"],
                                               "task_type": task_type, "metric": metric, "reason": "fallback"})
        if not result.get("ok"):
            log(f"[fallback] set_task: {result.get('error')}")
            return
    improve(session)


def improve(session) -> None:
    """Без LLM: база, затем несколько проверенных на табличных задачах шагов и смесь — пока хватает времени."""
    setup = session.setup
    regression = setup["task_type"] == "regression"
    log_metric = setup["metric"] in ("rmsle", "msle")
    y = next(iter(setup["splits"].values())).y_train
    nonneg_skewed = regression and float(y.min()) >= 0 and abs(float(y.skew())) > 0.75
    transform = "log1p" if log_metric else None
    steps = [{"model": "lightgbm", "target_transform": transform, "hypothesis": "fallback: база"}]
    if nonneg_skewed and not log_metric:
        steps.append({"model": "lightgbm", "target_transform": "log1p", "hypothesis": "fallback: скошенный таргет"})
    objective = {"mae": "l1", "mape": "mape"}.get(setup["metric"])
    if objective:   # метрика по модулю ошибки: учим ту же функцию потерь
        steps.append({"model": "lightgbm", "params": {"objective": objective}, "target_transform": transform,
                      "hypothesis": f"fallback: objective {objective} под метрику {setup['metric']}"})
    steps += [{"model": "catboost", "hypothesis": "fallback: вторая модель"},
              {"model": "xgboost", "hypothesis": "fallback: третья модель"}]
    last = 0.0
    auto_features = True
    for step in steps:
        if step["model"] != "lightgbm" or step is not steps[0]:
            best = session.best_run()
            if best and step.get("target_transform") is None and regression:
                step["target_transform"] = session.runs[best]["config"]["target_transform"]
            reserve = (session.estimate_final_seconds(best) if best else 60.0) * 4 + 120
            if session.time_left() < reserve + 2 * last:
                log("[fallback] времени на следующие шаги нет")
                break
        if session._ok_runs(full_only=True) and step is steps[0]:
            continue  # база уже есть (её сделала LLM)
        started = time.monotonic()
        result = session.call("evaluate", step)
        last = time.monotonic() - started
        if not result.get("ok"):
            log(f"[fallback] evaluate {step['model']}: {result.get('error')}")
        if auto_features and session._ok_runs(full_only=True):
            auto_features = False
            add_safe_features(session, last)
    blend(session)
    best = session.best_run()
    if best and session.runs[best]["config"]["model"] == "blend":
        best_single = min(session.runs[best]["config"]["members"], key=lambda m: _rank(session, m))
    else:
        best_single = best
    if best_single:
        config = session.runs[best_single]["config"]
        reserve = session.estimate_final_seconds(best or best_single, 3) + 120
        budget = 0.6 * (session.time_left() - reserve)
        if budget > 120 and os.environ.get("AGENT_FALLBACK_TUNE", "on").lower() != "off":
            result = session.call("tune", {"model": config["model"], "n_trials": 40, "time_budget_s": budget,
                                           "target_transform": config["target_transform"], "hypothesis": "fallback"})
            log(f"[fallback] tune {config['model']}: ok={result.get('ok')} trials={result.get('trials')} "
                f"best={result.get('best', {}).get('score')}")
            blend(session)


def add_safe_features(session, base_seconds) -> None:
    """Без LLM: частоты категорий и число пропусков — дешёвые признаки без смысла задачи; принимаются кодом
    только при устойчивом выигрыше на тех же фолдах."""
    if session.time_left() < 4 * base_seconds + 300:
        return
    X = session.setup["split"].X_train
    categorical = [c for c in X.columns if not pd_numeric(X[c]) and X[c].nunique() > 2][:15]
    specs = [{"op": "count", "col": c} for c in categorical]
    if X.isna().any(axis=1).mean() > 0.05:
        specs.append({"op": "nan_count"})
    if specs:
        result = session.call("add_features", {"features": specs, "hypothesis": "fallback: частоты и пропуски"})
        log(f"[fallback] add_features: accepted={result.get('accepted')} {result.get('error', '')}")
    # целые «коды» (много значений, но не непрерывная величина) как категории: выигрывает CatBoost
    # со статистиками по таргету, поэтому и проверяем CatBoost'ом
    codes = [c for c in X.columns if pd_numeric(X[c]) and is_code_like(X[c])][:15]
    if codes and session.time_left() > 8 * base_seconds + 300:
        result = session.call("add_features", {"features": [{"op": "as_category", "col": c} for c in codes],
                                               "model": "catboost", "hypothesis": "fallback: коды как категории"})
        log(f"[fallback] as_category: accepted={result.get('accepted')} {result.get('error', '')}")


def is_code_like(series) -> bool:
    values = series.dropna()
    if len(values) < 100 or not (values % 1 == 0).all():
        return False
    unique = values.nunique()
    return 20 < unique < 0.5 * len(values)


def pd_numeric(series) -> bool:
    import pandas as pd
    return pd.api.types.is_numeric_dtype(series)


def _rank(session, run_id):
    score = session._score(run_id)
    return -score if session.setup["higher_is_better"] else score


def blend(session) -> None:
    if len(session._ok_runs(full_only=True)) >= 2:
        result = session.call("blend", {"reason": "fallback"})
        log(f"[fallback] blend: ok={result.get('ok')} {result.get('error', '')}")


def polish(session) -> None:
    """Если время осталось: записанный лучший запуск переобучить средним по нескольким сидам."""
    written = session.written or {}
    run_id = written.get("run_id")
    if not run_id or run_id not in session.runs:
        return
    for seeds in (5, 4, 3, 2):
        if seeds <= written.get("seeds", 1):
            return
        if session.estimate_final_seconds(run_id, seeds) <= 0.7 * session.time_left():
            result = session.call("finalize", {"run_id": run_id, "seeds": seeds})
            log(f"[polish] {run_id} x{seeds} сидов: ok={result.get('ok')} {result.get('error', '')}")
            return


def metric_family(metric):
    import agent_tools
    import ml_metrics
    import re
    if re.sub(r"[^a-z0-9]", "", str(metric).lower()) in agent_tools._BINARY_METRICS:
        return "classification"   # бинарная f1/precision/recall: считает агент по положительному классу
    try:
        return agent_tools.resolve_metric(metric).task
    except ml_metrics.MetricError:
        return None


def ensure_final(session) -> None:
    """После диалога: если лучший запуск честно лучше записанного и время есть — обучить и записать его."""
    if session.setup is None:
        return
    best = session.best_run()
    if best is None or (session.written or {}).get("run_id") == best:
        return
    if not session._better_than_written(best, automatic=True):
        return
    if not session.can_finalize(best):
        log(f"[final] на финальное обучение {best} не хватает времени; остаётся {session.written}")
        return
    result = session.call("finalize", {"run_id": best})
    log(f"[final] finalize {best}: ok={result.get('ok')} {result.get('error', '')}")


def guarded(step, *args) -> None:
    try:
        step(*args)
    except Exception as error:  # ни один шаг не должен лишить нас уже записанного ответа
        log(f"[{step.__name__}] сбой: {type(error).__name__}: {str(error)[:300]}")


def main(argv) -> int:
    task_md = Path(argv[1] if len(argv) > 1 else "task.md").resolve()
    load_local_env()
    limit = float(os.environ.get("AGENT_TIME_LIMIT", 10800))
    safety = float(os.environ.get("AGENT_SAFETY_S", 900))
    deadline = START + limit - safety
    log(f"задача {task_md}; бюджет {limit:.0f}s, запас {safety:.0f}s")
    submission = task_md.parent / "submission.csv"

    import agent_tools
    import agent_llm

    try:
        session = agent_tools.Session(task_md, deadline=deadline, log=log, final_slack=0.5 * safety)
    except Exception as error:
        log(f"данные не загружены: {type(error).__name__}: {str(error)[:300]}")
        copied = agent_tools.copy_sample_submission(task_md)
        log(f"последний рубеж: {'скопирован шаблон ответа' if copied else 'шаблона нет'}")
        return 0 if submission.exists() else 1
    session.write_constant()

    try:
        if os.environ.get("AGENT_LLM", "on").lower() == "off":
            raise agent_llm.LLMUnavailable("AGENT_LLM=off: LLM отключена для офлайн-проверки", fatal=True)
        chat = agent_llm.OpenAIChat()
        log(f"модель {chat.model}")
        summary = agent_llm.run_agent(session, chat, max_steps=int(os.environ.get("AGENT_MAX_STEPS", 40)), log=log)
        log(f"[agent] шагов {summary['steps']}, finish: {summary['finished']}, usage: {summary['usage']}")
    except agent_llm.LLMUnavailable as error:
        log(f"[agent] LLM недоступна ({error}); дальше запасная политика")
    except Exception as error:  # ошибка агента не должна лишить нас ответа
        log(f"[agent] сбой: {type(error).__name__}: {str(error)[:300]}")

    if session.setup is None or not session._ok_runs(full_only=True):
        log("[agent] обученной модели нет: запасная политика")
        guarded(fallback_policy, session)
    guarded(ensure_final, session)
    guarded(polish, session)
    if not submission.exists():
        log("submission.csv не создан")
        return 1
    log(f"готово: записан {session.written}; запусков {len(session.runs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
