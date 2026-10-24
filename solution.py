"""Точка входа платформы: python3 solution.py task.md (из каталога задачи).

Порядок: страховочный ответ -> LLM-агент управляет экспериментами -> код гарантирует, что
в submission.csv лежит лучший обученный запуск. Если LLM недоступна или не довела дело до
обученной модели, тот же набор инструментов работает по простой фиксированной политике.

Переменные окружения:
  OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL — выданная модель (обязательны для агента);
  AGENT_TIME_LIMIT — общий лимит секунд (по умолчанию 10800 = 3 часа платформы);
  AGENT_SAFETY_S — запас до лимита на запись ответа и выход (по умолчанию 900);
  AGENT_MAX_STEPS — максимум шагов диалога с моделью (по умолчанию 40).
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
        if task_type not in ("regression", "binary", "multiclass"):
            import ml_schema
            # несколько таргетов: тип по первому, учатся все по очереди
            task_type = ml_schema.suggest_task_type(session.data.train[targets[0]], session.data.text)["type"]
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
    if not session._ok_runs(full_only=True):
        log_metric = session.setup["metric"] in ("rmsle", "msle")
        result = session.call("evaluate", {"model": "lightgbm", "hypothesis": "fallback: базовая модель",
                                           "target_transform": "log1p" if log_metric else None})
        if not result.get("ok"):
            log(f"[fallback] evaluate: {result.get('error')}")


def metric_family(metric):
    import agent_tools
    import ml_metrics
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
    if not submission.exists():
        log("submission.csv не создан")
        return 1
    log(f"готово: записан {session.written}; запусков {len(session.runs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
