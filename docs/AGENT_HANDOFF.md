# Агент Ивана (A01): запуск, общий вызов и передача Юре и Егору

Ветка `ivan/a01-llm-agent`. В ней ML-ядро Юры (`origin/yuri/a02-ml-core` @ 8ce4f47, влито merge-коммитом) и агент Ивана поверх него. Документ для агентов и людей, которые продолжают работу Юры (A07/A11) и Егора (A08/A13).

## Что где

| Файл | Владелец | Назначение |
| --- | --- | --- |
| `solution.py` | Иван | Точка входа `python3 solution.py task.md`: дедлайн, страховочный ответ, LLM-агент, запасная политика, финальная гарантия ответа |
| `agent_tools.py` | Иван | `Session`: инструменты для LLM (`set_task`, `evaluate`, `blend`, `finalize`, `finish`), выбор лучшего запуска по OOF, смесь моделей, сборка и атомарная запись `submission.csv` |
| `agent_llm.py` | Иван | Клиент OpenAI-совместимого API, промпт, цикл function calling, разбор вызовов из текста, сжатие истории |
| `ml_*.py` | Юра | Загрузка, профиль, схема, предобработка, валидация, метрики, модели, `train_eval`/`fit_predict` |
| `tests/test_agent.py` | Иван | 30 тестов агента. LLM заменена сценарием `ScriptedChat`, ключ не нужен |
| `tests/external_tasks.py` | Иван | 12 незнакомых задач из OpenML, scikit-learn и зеркал Kaggle в формате платформы и оценка по скрытым ответам. В ZIP не входит |

Модули лежат плоско в корне: платформа копирует только корневые `.py`. `mcp` не используется. Инструменты вызываются в том же процессе через `openai` function calling.

## Запуск

```bash
# окружение: Python 3.12 + reference/requirements.txt (на macOS ещё brew install libomp)
cd data/public/regression_case && python3 /path/to/repo/solution.py task.md
.venv/bin/python -W ignore -m unittest discover -s tests -t .   # все тесты
```

Переменные окружения:

| Переменная | Значение |
| --- | --- |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | Выданная модель. Без них агент работает в запасной политике без LLM. Локально можно положить в `.env` рядом с `solution.py`: файл исключён из Git и в ZIP не попадает |
| `AGENT_TIME_LIMIT` | Общий бюджет, по умолчанию 10800 с (лимит платформы). Для коротких проверок 1200–1800 |
| `AGENT_SAFETY_S` | Запас до лимита, по умолчанию 900 с. Половину запаса может занять финальное обучение |
| `AGENT_MAX_STEPS` | Шагов диалога с LLM, по умолчанию 40 |

Поток `solution.py`:

1. Загрузка и профиль через `ml_io`/`ml_profile`. Если загрузка упала, копируется `sample_submission`.
2. **Страховочный ответ**: медиана, частый класс или доли классов.
3. LLM видит `task.md` и профиль. Она выбирает таргет, id, тип, метрику и формат ответа (`set_task`), затем ставит эксперименты (`evaluate`) с гипотезой и останавливается сама (`finish`).
4. Первый успешный полный `evaluate` код сам обучает на всём train и записывает: **ранний настоящий ответ**.
5. Ответ заменяется только запуском, который лучше на той же схеме валидации. Запуски на подвыборке в финал не идут.
6. Если обученной модели нет, срабатывает запасная политика (эвристика + LightGBM). Причины: LLM недоступна, не вызвала инструменты или упала. После этого `ensure_final` обучает лучший запуск, если он лучше записанного и успевает по времени.

## Общий вызов Иван ↔ Юра

Агент использует ML-ядро так (`agent_tools.Session`):

```python
data    = ml_io.load_task(task_md)
profile = ml_profile.build_profile(data)
schema  = ml_schema.resolve_schema(profile["schema"], id_cols=ids, target_cols=[target])  # выбор LLM
split   = ml_data.split_task(data, schema)
plan    = ml_validation.suggest_validation(profile_eff, split.X_train)   # profile_eff: тип/метрика от LLM
ev      = ml_tools.train_eval(split, profile_eff, model=..., metric=..., validation=plan, params=...,
                              task_type=..., target_transform=..., time_budget=..., subsample=...)
out     = ml_tools.fit_predict(split, profile_eff, model=..., n_rounds=ml_tools.rounds_from(ev), ...)
```

На что агент опирается. Менять эти места только с уведомлением Ивана:

- `EvalResult.to_dict()`: ключи `status`, `score`, `fold_scores`, `best_iterations`, `seconds`, `validation`, `notes`, `error`. Плюс `oof_index`/`oof_pred` для весов классов.
- `fit_predict` → `status`, `pred`, `proba`, `classes`, `n_rounds`, `notes`, `warnings`.
- `profile_eff` — копия профиля, в которой `task.type` и `metric.primary` заменены выбором LLM. Это обходит то, что `fit_predict` берёт метрику только из профиля.
- `ml_postprocess.choose_class_weights(oof_pred, codes, groups=row_groups(...))` применяется только для `f1_macro`.

### Просьбы к Юре (A07/A11, не блокируют)

Полный список дефектов ядра с воспроизведениями и приоритетами — [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md), раздел 1 (Y1–Y15). До заморозки желательно закрыть Y1–Y8. Ниже — то, что касается общего вызова.

1. `ml_tools.fit_predict` должен принимать `metric` или перестать проверять метрику профиля (`_resolve_task_and_metric(profile, task_type, None)`). Сейчас агент обходит это через `profile_eff`.
2. `fit_predict` должен отдавать `stopped_by_time`. В `_auto_rounds` нужен try/except по фолдам и учёт переданной схемы валидации.
3. `ml_validation`: группировка по столбцу (`plan["column"]` для `group`). Сейчас `group` объединяет только одинаковые строки. Агент запрещает LLM передавать `column` для `group`, чтобы её решение не игнорировалось молча.
4. `ml_io.find_task_files`: `task.md` упоминает `` `submission.csv` ``. При повторном запуске в том же каталоге наш ответ даёт warning «роль файла submission.csv не распознана». Из-за этого `tests.test_io.PublicTasksLoadingTest` падает, если ответ уже лежит рядом. На платформе запуск первый, поэтому это мелочь.
5. Несколько таргетов: `train_eval`/`fit_predict` принимают один. Агент обучает каждый таргет отдельно одной конфигурацией (`Session.setup["splits"]`); score — среднее. Нативная поддержка в ядре не обязательна.
7. Временной сплит по составному времени (`year` + `month`): в `bike_demand` test — последние месяцы, но каждое значение `year` и `month` по отдельности есть в train. Поэтому `suggest_validation` выбирает `kfold`, и оценка выходит оптимистичной. Нужна схема temporal по упорядочивающему ключу из нескольких столбцов или по номеру строки.
8. Свои метрики агента (`agent_tools.EXTRA_METRICS`: rmsle, msle, mape, smape) передаются в `train_eval` объектом `ml_metrics.Metric`. Важно, чтобы `get_metric` и дальше возвращал `Metric` как есть.
6. Единый тип `classes` в `EvalResult` (str) и `fit_predict` (исходные метки). Агент сопоставляет их сам.

## Для Егора: сборка, проверка, сабмит

Замечания к `tools/check_environment.py` и `tools/build_submission.py` — [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md), раздел 2 (E1–E3). Главное — E1: `--api` не проверяет вызов инструментов (function calling), на котором работает агент.

Сборка проверена инструментом из `origin/codex/a03-release-tools` (`tools/build_submission.py`):

```bash
python tools/build_submission.py --root . --output artifacts/solution.zip
python tools/build_submission.py --root . --check artifacts/solution.zip --extract-to tmp/clean-<N>
# в tmp/clean-<N>: task.md + данные по путям из условия, затем
python3 solution.py task.md
```

- **Состав ZIP (55 КБ, 13 файлов):** `solution.py`, `agent_llm.py`, `agent_tools.py`, `ml_data.py`, `ml_io.py`, `ml_metrics.py`, `ml_models.py`, `ml_postprocess.py`, `ml_profile.py`, `ml_schema.py`, `ml_tools.py`, `ml_validation.py`, `requirements.txt`.
- **Лишние файлы в корне:** все корневые `.py` попадают в ZIP, поэтому не класть туда экспериментальные скрипты.
- **Импорты:** `solution.py` сам добавляет свой каталог в `sys.path`. Импорт работает и при `python -I`.
- **Ключ API:** в коде ключей нет, только чтение из окружения.
- **Вывод:** `stdout` — строки вида `[  12.3s] ...`. По ним видно, что делала LLM (`[llm]`, `[tool]`) и какой ответ записан (`[finalize rN] записан`).
- **Выбор private-версии:** сначала отправить версию, проверенную с настоящей LLM. Платформа выбирает отправку с лучшим public score; запасная политика без LLM как кандидат не подходит (`docs/SUBMISSIONS.md`).

## Что проверено (24.10.2026, macOS, Python 3.12.15)

| Проверка | Результат |
| --- | --- |
| `unittest discover -s tests -t .` | 190 тестов; 188 ok. 2 падают только при лежащем рядом `submission.csv`, см. просьбу 4 |
| Регрессия, запасная политика | temporal-валидация, RMSE 50.2, 11 с, 22 449 строк, порядок id = test |
| Классификация, запасная политика | group-валидация, Macro F1 0.608, веса классов +0.035 на отложенной части, 110 с |
| ZIP из чистого каталога, `python -I` | обе задачи дают корректный `submission.csv` |
| Путь LLM, сценарий вместо модели, регрессия | lightgbm 50.2 → log1p 48.7 → catboost 49.8 → xgboost 52.2; записан лучший (log1p) |
| Путь LLM, сценарий вместо модели, классификация | подвыборка в финал не попала; catboost, урезанный бюджетом 300 с, дал 0.521 |

Незнакомые задачи (`python tests/external_tasks.py build && python tests/external_tasks.py run`), запасная политика без LLM, лимит 900 с на задачу. Столбец «v1» — одна модель LightGBM, «v2» — текущая версия: база, `log1p` для скошенного таргета, CatBoost, XGBoost и смесь по OOF.

| Задача | Что проверяет | Метрика | v1 | v2 | Константа |
| --- | --- | --- | --- | --- | --- |
| adult_income | нет id в test, `row_id` из шаблона, вероятность `>50K` | ROC AUC ↑ | 0.928 | 0.929 | 0.5 |
| credit_risk | вероятность класса `bad` (не второго по алфавиту) | ROC AUC ↑ | 0.832 | 0.825 | 0.5 |
| california_housing | файлы рядом с task.md | RMSE ↓ | 43 495 | 42 675 | 118 563 |
| bike_demand | RMSLE, прогноз вперёд | RMSLE ↓ | 0.374 | 0.357 | 1.477 |
| covertype_proba | 7 столбцов вероятностей `cover_type_k` | logloss ↓ | 0.274 | 0.275 | 1.205 |
| wine_quality | целый таргет, но регрессия по условию | MAE ↓ | 0.460 | 0.463 | 0.629 |
| energy_two_targets | два таргета | средний RMSE ↓ | 1.03 | 0.689 | 9.84 |
| diabetes_small | 300 строк train | MAE ↓ | 45.5 | 44.9 | 60.1 |
| kaggle_house_prices | 80 признаков, RMSE логарифмов | RMSE log ↓ | 0.130 | 0.116 | 0.386 |
| kaggle_titanic | метки 0/1, текстовые столбцы | Accuracy ↑ | 0.821 | 0.818 | 0.617 |
| kaggle_telco_churn | id-строки, `TotalCharges` текстом | ROC AUC ↑ | 0.841 | 0.850 | 0.5 |
| kaggle_insurance | скошенный таргет | RMSE ↓ | 4 298 | 4 087 | 13 427 |

- Формат ответа совпадает с шаблоном во всех 12 задачах.
- В v2 девять задач лучше. Три немного хуже, но это маленькие test (300–480 строк) и разница в пределах шума.
- На covertype за 900 с успела только база. На платформе (3 часа) запасная политика дойдёт до смеси.
- Найденные ошибки и что с ними сделано — в [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md).

**Не проверено:** настоящая выданная LLM (переменных API на машине нет), Linux-контейнер платформы, полный 3-часовой бюджет.

## Известные ограничения

- Ранжирование и задачи без стандартного y не поддержаны (A10).
- Несколько таргетов допустимы только одного типа и учатся одной конфигурацией.
- Метрики: 11 из `ml_metrics`, плюс rmsle, msle, mape и smape в агенте. Gini приравнен к ROC AUC. Незнакомую метрику LLM заменяет ближайшей и объясняет в `reason`.
- Временной сплит возможен только по одному числовому столбцу (см. просьбу 7).
- Модели — LightGBM / XGBoost / CatBoost без собственной генерации признаков.
- Жёсткого прерывания зависшего обучения нет. Защита — ранний ответ на диске и кооперативный `time_budget` в бустинге.
