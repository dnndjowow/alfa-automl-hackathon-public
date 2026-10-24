"""LLM-контроллер: выданная модель выбирает инструменты, код их исполняет и возвращает результат.

Цикл: сообщение модели -> вызовы инструментов (function calling) -> JSON-результаты обратно в
диалог -> следующее решение модели, пока она не вызовет finish или не кончится время/шаги.
Если сервер не поддерживает tools или модель пишет вызов текстом, вызов читается из JSON в ответе.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Optional

import agent_tools

SYSTEM_PROMPT = """Ты — автономный ML-инженер. Решаешь табличную задачу только через инструменты.

Порядок работы:
1. Прочитай условие (task.md) и профиль данных. Вызови set_task: таргет, id, тип задачи и метрику
   бери из условия; подсказки профиля — эвристика, проверь их по тексту. Если в условии метрика,
   которой нет в списке, выбери самую близкую из доступных и объясни в reason. Если ответ —
   вероятность (AUC, logloss, «вероятность»), укажи prediction_format=proba, а для бинарной задачи —
   positive_class: класс, вероятность которого просят. Если таргетов несколько, передай все.
   Для RMSLE/MSLE обычно помогает target_transform=log1p.
2. Вызови evaluate для простой сильной базы (lightgbm с параметрами по умолчанию).
3. Дальше каждый эксперимент — проверка гипотезы по результатам прошлых: другая модель
   (lightgbm / xgboost / catboost), learning_rate, регуляризация, глубина, log1p для скошенного
   неотрицательного таргета регрессии, другая схема валидации, если текущая не отражает test.
   Пиши гипотезу в поле hypothesis. Сравнивай только запуски на одной схеме валидации.
4. Признаки: по смыслу task.md предложи через add_features 5–20 осмысленных признаков (отношения вроде
   цены за единицу площади, разности, отклонение от среднего по группе, частоты категорий). Код примет
   их, только если они устойчиво помогают. После принятия заново сравни модели.
5. Когда есть 2–4 удачных запуска разных моделей на одной схеме, вызови blend: смесь по OOF обычно
   лучше любой одиночной модели. Лучший запуск или смесь обучи на всём train через finalize (он запишет
   submission.csv). Первый удачный запуск код записывает сам — это страховка, улучшай её.
6. Цель — максимальная метрика на скрытом test. Используй бюджет времени: пока time_left_s больше
   примерно трёх полных экспериментов плюс финала, продолжай улучшать — tune для лучшей модели,
   другие модели, log1p, смесь. Вызывай finish, когда время на исходе или три попытки подряд не
   улучшили лучший результат. Финал лучшего запуска или смеси делай с seeds=3–5, если хватает времени.

Следи за time_left_s в ответах инструментов: финальное обучение занимает примерно время одной
части валидации с запасом. Шаги диалога платные: не повторяй одинаковых запусков, крупный перебор
делай одним вызовом tune. Отвечай кратко;
каждое сообщение должно содержать вызов инструмента."""

TEXT_FORMAT = ('Вызов инструмента пиши одним JSON-объектом без пояснений вокруг: '
               '{"tool": "<имя>", "arguments": {...}}. Например: '
               '{"tool": "evaluate", "arguments": {"model": "lightgbm", "hypothesis": "база"}}')

DEFAULT_MODEL = "qwen/qwen3.8-27b"   # из инструкции организаторов трека AutoML
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TOOL_NAMES = {spec["name"] for spec in agent_tools.TOOL_SPECS}


class LLMUnavailable(RuntimeError):
    """Модель недоступна. fatal=True — нет смысла повторять (ключ, модель, адрес)."""

    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


def _arguments_text(arguments) -> str:
    if isinstance(arguments, str):
        return arguments or "{}"
    return json.dumps(arguments or {}, ensure_ascii=False)


class OpenAIChat:
    """Тонкая обёртка над openai: один запрос -> нормализованный ответ с tool_calls."""

    def __init__(self, model: Optional[str] = None, timeout: float = 240.0, retries: int = 4):
        from openai import OpenAI

        # по инструкции организаторов OPENAI_MODEL может быть не задан: прокси сам подставит модель задачи
        self.model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        missing = [v for v in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if not os.environ.get(v)]
        if missing:
            # без адреса клиент ушёл бы на api.openai.com и отправил туда выданный ключ
            raise LLMUnavailable(f"не заданы {', '.join(missing)}", fatal=True)
        # OPENAI_BASE_URL клиент берёт из окружения сам
        self.client = OpenAI(timeout=timeout, max_retries=0)
        self.retries, self.timeout = retries, timeout
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0, "cost": 0.0}
        self.tools_supported = True
        self.time_left = None                 # run_agent подставляет session.time_left

    def _request(self, messages, tools):
        kwargs = {"model": self.model, "messages": messages, "timeout": self._timeout()}
        if self.tools_supported:
            kwargs.update(tools=tools, tool_choice="auto")
        response = self.client.chat.completions.create(**kwargs)
        self.usage["requests"] += 1
        if response.usage:
            self.usage["prompt_tokens"] += response.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += response.usage.completion_tokens or 0
            self.usage["cost"] = round(self.usage["cost"] + float(getattr(response.usage, "cost", 0) or 0), 4)
        if not response.choices:
            raise LLMUnavailable("пустой ответ: нет choices")
        message = response.choices[0].message
        calls = [{"id": c.id or f"call_{i}", "name": c.function.name, "arguments": _arguments_text(c.function.arguments)}
                 for i, c in enumerate(message.tool_calls or []) if c.function is not None]
        return {"content": message.content or "", "tool_calls": calls}

    def _timeout(self) -> float:
        if self.time_left is None:
            return self.timeout
        return max(15.0, min(self.timeout, self.time_left() - 60.0))

    def __call__(self, messages: list, tools: list) -> dict:
        last = None
        for attempt in range(self.retries):
            if self.time_left is not None and self.time_left() < 75:
                break                          # ждать ответа модели уже некогда
            try:
                return self._request(messages, tools)
            except LLMUnavailable as error:
                last = error
            except Exception as error:
                last = error
                status = getattr(error, "status_code", None)
                if status in (401, 403, 404):
                    raise LLMUnavailable(f"{type(error).__name__} {status}: {str(error)[:300]}", fatal=True)
                text = str(error).lower()
                if self.tools_supported and status in (400, 422) and ("tool" in text or "function" in text):
                    # сервер не принимает tools: дальше вызовы инструментов читаются из текста ответа
                    self.tools_supported = False
                    try:
                        return self._request(messages, tools)
                    except Exception as retry_error:
                        last = retry_error
            if attempt < self.retries - 1:
                time.sleep(min(5 * 2 ** attempt, 40))
        raise LLMUnavailable(f"{type(last).__name__}: {str(last)[:300]}")


def parse_text_calls(content: str) -> list:
    """Вызовы инструментов, записанные JSON-ом в тексте: {"tool"|"name": ..., "arguments": {...}},
    в том числе в форме OpenAI {"function": {"name", "arguments"}}; рассуждения <think> пропускаются."""
    text = _THINK.sub("", content or "")
    decoder, calls, pos = json.JSONDecoder(), [], 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            data, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        pos = end
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("function"), dict):
            data = data["function"]
        name = data.get("tool") or data.get("name")
        if not isinstance(name, str) or name not in _TOOL_NAMES:
            continue
        args = data.get("arguments", data.get("args", data.get("parameters", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        calls.append({"id": f"text_call_{len(calls)}", "name": name,
                      "arguments": json.dumps(args, ensure_ascii=False)})
    return calls


def initial_messages(session: agent_tools.Session, tools_in_text: bool = False, task_limit: int = 12000) -> list:
    overview = session.overview()
    task_text = session.data.text
    if len(task_text) > task_limit:
        task_text = task_text[:task_limit] + "\n…(обрезано)"
    system = SYSTEM_PROMPT
    if tools_in_text:
        system += "\n\n" + TEXT_FORMAT + "\nИнструменты:\n" + json.dumps(agent_tools.TOOL_SPECS, ensure_ascii=False)
    user = (
        f"# task.md\n\n{task_text}\n\n# Профиль данных (автоматически)\n\n{overview['profile_text']}\n\n"
        f"# Подсказки и доступные значения\n\n"
        + json.dumps({k: overview[k] for k in ("suggested", "available", "extra_csv", "time_left_s")}, ensure_ascii=False)
        + f"\n\nВсе столбцы train: {overview['all_train_columns'][:300]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def runs_table(session: agent_tools.Session) -> str:
    """Короткая сводка всех запусков — заменяет старые сообщения при сжатии истории."""
    rows = []
    for run_id, run in session.runs.items():
        c, r = run["config"], run["result"]
        rows.append(f"{run_id}: {c['model']} lr={c['learning_rate']} params={c['params']} "
                    f"transform={c['target_transform']} subsample={c['subsample']} -> {r['status']} "
                    f"score={r.get('score')} {round(r.get('seconds', 0))}s"
                    + (" (текущая схема)" if run["key"] == session.plan_key else ""))
    setup = session.setup
    head = (f"Постановка: таргет {list(setup['schema'].target_cols)}, {setup['task_type']}, метрика {setup['metric']}, "
            f"валидация {session.plan.get('strategy')}. " if setup else "Постановка ещё не задана. ")
    return head + f"Записан ответ: {session.written}. Запуски:\n" + ("\n".join(rows) or "нет")


def compact(messages: list, session: agent_tools.Session, keep: int = 14, base_first: Optional[dict] = None) -> list:
    """Оставить system, исходное первое сообщение со свежей сводкой запусков и хвост диалога с целыми
    парами вызов-ответ. Сводка строится от исходного сообщения, иначе копии накапливаются."""
    if len(messages) <= keep + 2:
        return messages
    cut = len(messages) - keep
    while cut < len(messages) and messages[cut]["role"] != "assistant":
        cut += 1
    if cut >= len(messages):
        return messages
    first = dict(base_first or messages[1])
    first["content"] = first["content"] + "\n\n# Что уже сделано (сводка ранних шагов)\n\n" + runs_table(session)
    return [messages[0], first] + messages[cut:]


def run_agent(session: agent_tools.Session, chat: Callable, max_steps: int = 40, log=print,
              min_time_s: float = 90.0, max_failures: int = 4) -> dict:
    """Диалог с моделью до finish / лимита шагов / времени. Возвращает сводку."""
    tools = agent_tools.openai_tools()
    tools_in_text = getattr(chat, "tools_supported", True) is False
    messages = initial_messages(session, tools_in_text)
    base_first = messages[1]
    if hasattr(chat, "time_left"):
        chat.time_left = session.time_left   # клиент не ждёт ответа дольше, чем осталось времени
    idle = steps = failures = bad_calls = 0
    while steps < max_steps and session.finished is None:
        if session.time_left() < min_time_s:
            log("[agent] время на исходе: выходим из диалога")
            break
        messages = compact(messages, session, base_first=base_first)
        if not tools_in_text and getattr(chat, "tools_supported", True) is False:
            tools_in_text = True              # клиент перешёл на текстовые вызовы: даём формат в system
            messages[0] = initial_messages(session, True)[0]
        try:
            reply = chat(messages, tools)
        except LLMUnavailable as error:
            failures += 1
            if error.fatal or failures >= max_failures or session.time_left() < min_time_s + 120:
                raise
            log(f"[agent] LLM временно недоступна ({error}); пауза и повтор {failures}/{max_failures - 1}")
            time.sleep(60)
            continue
        failures = 0
        steps += 1
        content = reply.get("content") or ""
        calls = reply.get("tool_calls") or parse_text_calls(content)
        if content:
            log(f"[llm] {_THINK.sub('', content)[:400]}")
        if not calls:
            idle += 1
            if content.strip():
                messages.append({"role": "assistant", "content": content})
            if idle >= 3:
                log("[agent] модель трижды ответила без вызова инструмента: завершаем")
                break
            if not tools_in_text:
                tools_in_text = True  # модель не вызывает tools: даём формат вызова текстом
                messages[0] = initial_messages(session, True)[0]
            messages.append({"role": "user", "content": "Сделай следующий шаг вызовом инструмента. " + TEXT_FORMAT})
            continue
        idle = 0
        native = bool(reply.get("tool_calls"))
        if native:
            messages.append({"role": "assistant", "content": content or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls]})
        else:
            messages.append({"role": "assistant", "content": content})
        for call in calls:
            try:
                args = json.loads(call["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise TypeError("аргументы не объект")
            except (json.JSONDecodeError, TypeError) as error:
                result = {"ok": False, "error": f"аргументы не JSON-объект: {error}"}
            else:
                log(f"[tool] {call['name']} {json.dumps(args, ensure_ascii=False)[:300]}")
                result = session.call(call["name"], args)
            bad_calls = 0 if result.get("ok") else bad_calls + 1
            text = json.dumps(result, ensure_ascii=False)
            if native:
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
            else:
                messages.append({"role": "user", "content": f"Результат {call['name']}: {text}"})
        if bad_calls >= 8:
            log("[agent] восемь неудачных вызовов подряд: модель зациклилась, завершаем диалог")
            break
        if bad_calls >= 4:
            # модель повторяет ошибку: напоминаем точную схему аргументов
            messages.append({"role": "user", "content": "Последние вызовы не удались. Используй ровно эти имена "
                             "аргументов: " + json.dumps({sp["name"]: list(sp["parameters"]["properties"])
                                                         for sp in agent_tools.TOOL_SPECS}, ensure_ascii=False)})
    return {"steps": steps, "finished": session.finished, "usage": getattr(chat, "usage", None)}
