# Порядок выполнения: 23–25 октября 2026

**Старт: 23.10 в 20:00. Финальное решение: 25.10 в 20:00. Всё время по МСК (UTC+3).**

Здесь 44 рабочие карточки в порядке начала. На [доске](https://github.com/users/selecteg12/projects/2) открыть вкладку «Порядок выполнения»: те же строки, статусы и сроки. Равное время у разных участников означает параллельную работу; номер S — удобный порядок чтения. Строгая очередность задаётся колонкой «После» и зависимостями внутри карточки. Один человек выполняет свои карточки последовательно.

## Контрольные точки

| Когда МСК | Что должно быть готово | Кто подтверждает |
| --- | --- | --- |
| 23.10 21:00 | Контракт и доступы, все начали первые карточки | Иван + Егор |
| 24.10 01:00 | Загрузчик, предобработка, профиль, LLM-контроллер на заглушках, окружение/упаковка | Иван + Юра + Егор |
| 24.10 01:00–09:00 | Перерыв; длинные новые обязательные задачи не назначаем | Вся команда |
| 24.10 14:00 | Обе публичные задачи проходят от условия до корректного ответа | Иван + Юра |
| 24.10 15:00 | Первый рабочий сабмит отправлен; есть запись и резерв | Егор |
| 24.10 17:30 | Исследовательский цикл, лучший результат, бюджеты времени | Иван |
| 24.10 23:00 | Проверки универсальности и кандидаты улучшений сохранены | Иван + Юра |
| 24.10 23:00–25.10 08:00 | Ночной перерыв | Вся команда |
| 25.10 10:00 | Заморозка новых функций, фиксируем SHA и конфигурацию | Иван + Егор |
| 25.10 10:00–16:00 | Два полных прогона из чистого архива; по 3 ч на задачу | Егор, отчёт Максима |
| 25.10 17:00 | Техническая приёмка завершена, отправляем финал | Иван + Егор |
| 25.10 19:30 | Проверены выбранная версия и статус сдачи | Егор |
| 25.10 20:00 | Жёсткий дедлайн | Егор |

Длительные автоматические запуски не равны активной работе человека. С 10:00 25 октября разработчики доступны для блокирующих исправлений; любое такое изменение требует проверки затронутого пути. Если полного повторения не успеваем, выбираем уже проверенный резерв. Очередь платформы не имеет гарантированной длительности.

## Все шаги по времени

| № | Карточка | Кто | Начать МСК | Сдать до МСК | После |
| --- | --- | --- | --- | --- | --- |
| 01 | [S01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S01.md) Согласовать контракт задачи и результатов ML-инструментов | Иван | 23.10 20:00 | 23.10 21:00 | — |
| 02 | [S02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S02.md) Реализовать загрузку файлов и сохранение порядка строк | Юра | 23.10 20:00 | 23.10 22:00 | — |
| 03 | [S03](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S03.md) Запустить работу команды и снять вопросы по доступам | Егор / PM | 23.10 20:00 | 23.10 21:00 | — |
| 04 | [S04](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S04.md) Собрать навигатор по материалам и комплектам данных | Артём | 23.10 20:00 | 23.10 21:00 | — |
| 05 | [S05](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S05.md) Проверить ссылки и подготовить шаблон отчёта о запуске | Максим | 23.10 20:00 | 23.10 21:00 | — |
| 06 | [S06](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S06.md) Научить агент читать описание новой задачи | Иван | 23.10 21:00 | 23.10 22:00 | S01 |
| 07 | [S07](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S07.md) Проверить Python, зависимости и доступ к API | Егор / PM | 23.10 21:00 | 23.10 23:00 | S03 |
| 08 | [S08](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S08.md) Провести EDA регрессии и записать проверенные наблюдения | Артём | 23.10 21:00 | 23.10 23:00 | S04 |
| 09 | [S09](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S09.md) Подключить LLM и проверить структурированный ответ | Иван | 23.10 22:00 | 23.10 23:00 | S06, S03 |
| 10 | [S10](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S10.md) Сделать предобработку числовых и категориальных признаков | Юра | 23.10 22:00 | 24.10 00:00 | S02, S01 |
| 11 | [S11](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S11.md) Собрать контроллер инструментов на заглушках | Иван | 23.10 23:00 | 24.10 00:00 | S09 |
| 12 | [S12](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S12.md) Подготовить сборку и аудит ZIP решения | Егор / PM | 23.10 23:00 | 24.10 00:00 | S07 |
| 13 | [S13](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S13.md) Провести EDA классификации и оценить дисбаланс | Артём | 23.10 23:00 | 24.10 01:00 | S08 |
| 14 | [S14](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S14.md) Повторить подготовку окружения по инструкции | Максим | 23.10 23:00 | 24.10 01:00 | S07, S05 |
| 15 | [S15](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S15.md) Сформировать компактный автоматический профиль данных для LLM | Юра | 24.10 00:00 | 24.10 01:00 | S10 |
| 16 | [S16](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S16.md) Провести первую интеграционную сверку | Егор / PM | 24.10 00:00 | 24.10 01:00 | S11, S10, S12 |
| 17 | [S17](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S17.md) Выбрать базовые ML-конфигурации и план экспериментов | Иван | 24.10 09:00 | 24.10 11:00 | S06, S11 |
| 18 | [S18](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S18.md) Реализовать разбиение данных и вычисление метрик | Юра | 24.10 09:00 | 24.10 10:00 | S15, S01 |
| 19 | [S19](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S19.md) Свести EDA в короткий список решений и рисков | Артём | 24.10 09:00 | 24.10 10:00 | S13 |
| 20 | [S20](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S20.md) Обучить первую регрессионную модель | Юра | 24.10 10:00 | 24.10 11:00 | S18 |
| 21 | [S21](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S21.md) Обучить первую модель классификации | Юра | 24.10 11:00 | 24.10 12:00 | S20 |
| 22 | [S22](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S22.md) Подключить реальные ML-инструменты к LLM-контроллеру | Иван | 24.10 12:00 | 24.10 14:00 | S11, S06, S17, S21, S15 |
| 23 | [S23](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S23.md) Реализовать запись и валидацию submission.csv | Юра | 24.10 12:00 | 24.10 13:00 | S21 |
| 24 | [S24](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S24.md) Соединить полный запуск и передать первую рабочую версию | Юра | 24.10 13:00 | 24.10 14:00 | S23, S11 |
| 25 | [S25](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S25.md) Сохранять историю экспериментов и лучший результат | Иван | 24.10 14:00 | 24.10 15:30 | S22, S24 |
| 26 | [S26](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S26.md) Собрать и отправить первый рабочий сабмит | Егор / PM | 24.10 14:00 | 24.10 15:00 | S22, S24, S12 |
| 27 | [S27](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S27.md) Сверить решение с требованиями и собрать доказательства | Артём | 24.10 14:00 | 24.10 16:00 | S19, S22, S24 |
| 28 | [S28](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S28.md) Независимо повторить обе публичные команды | Максим | 24.10 14:00 | 24.10 16:00 | S14, S24 |
| 29 | [S29](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S29.md) Замкнуть AutoResearch и ограничить время выполнения | Иван | 24.10 15:30 | 24.10 17:30 | S25 |
| 30 | [S30](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S30.md) Проверить новые схемы входа и ответа | Иван | 24.10 17:30 | 24.10 19:00 | S29 |
| 31 | [S31](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S31.md) Проверить сбои экспериментов и сохранение раннего ответа | Юра | 24.10 17:30 | 24.10 19:00 | S29, S24 |
| 32 | [S32](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S32.md) Проверить метрики, редкие случаи и границы поддержки | Иван | 24.10 19:00 | 24.10 20:00 | S30 |
| 33 | [S33](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S33.md) Ограничить ресурсы и убрать лишние расходы pipeline | Юра | 24.10 19:00 | 24.10 21:00 | S31 |
| 34 | [S34](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S34.md) Повторить проверки ошибок и обновить список багов | Максим | 24.10 19:00 | 24.10 21:00 | S31, S28 |
| 35 | [S35](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S35.md) Сравнить варианты регрессии на одной валидации | Иван | 24.10 20:00 | 24.10 21:30 | S32 |
| 36 | [S36](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S36.md) Сравнить варианты классификации с учётом редких классов | Иван | 24.10 21:30 | 24.10 23:00 | S35 |
| 37 | [S37](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S37.md) Выбрать финальную конфигурацию и заморозить функции | Иван | 25.10 08:00 | 25.10 10:00 | S36 |
| 38 | [S38](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S38.md) Исправить критические баги и проверить финальный pipeline | Юра | 25.10 08:00 | 25.10 10:00 | S32, S33 |
| 39 | [S39](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S39.md) Подготовить понятное описание результата и материалы сдачи | Артём | 25.10 09:00 | 25.10 11:00 | S27, S24 |
| 40 | [S40](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S40.md) Провести полный прогон из чистого архива | Егор / PM | 25.10 10:00 | 25.10 16:00 | S37, S38, S12 |
| 41 | [S41](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S41.md) Проверить состав финального архива по чек-листу | Максим | 25.10 10:00 | 25.10 11:00 | S37, S12 |
| 42 | [S42](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S42.md) Собрать результаты полных прогонов и финальный список багов | Максим | 25.10 11:00 | 25.10 16:00 | S41, S34 |
| 43 | [S43](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S43.md) Подтвердить техническую готовность финального архива | Иван | 25.10 16:00 | 25.10 17:00 | S40, S37, S38 |
| 44 | [S44](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S44.md) Отправить финал и проверить, что сдача зафиксирована | Егор / PM | 25.10 17:00 | 25.10 19:30 | S43 |

## Личный маршрут

### Иван: 14 карточек, около 20 ч активной работы

1. [S01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S01.md) — Согласовать контракт задачи и результатов ML-инструментов. **До 23.10 21:00 МСК.**
2. [S06](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S06.md) — Научить агент читать описание новой задачи. **До 23.10 22:00 МСК.**
3. [S09](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S09.md) — Подключить LLM и проверить структурированный ответ. **До 23.10 23:00 МСК.**
4. [S11](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S11.md) — Собрать контроллер инструментов на заглушках. **До 24.10 00:00 МСК.**
5. [S17](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S17.md) — Выбрать базовые ML-конфигурации и план экспериментов. **До 24.10 11:00 МСК.**
6. [S22](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S22.md) — Подключить реальные ML-инструменты к LLM-контроллеру. **До 24.10 14:00 МСК.**
7. [S25](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S25.md) — Сохранять историю экспериментов и лучший результат. **До 24.10 15:30 МСК.**
8. [S29](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S29.md) — Замкнуть AutoResearch и ограничить время выполнения. **До 24.10 17:30 МСК.**
9. [S30](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S30.md) — Проверить новые схемы входа и ответа. **До 24.10 19:00 МСК.**
10. [S32](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S32.md) — Проверить метрики, редкие случаи и границы поддержки. **До 24.10 20:00 МСК.**
11. [S35](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S35.md) — Сравнить варианты регрессии на одной валидации. **До 24.10 21:30 МСК.**
12. [S36](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S36.md) — Сравнить варианты классификации с учётом редких классов. **До 24.10 23:00 МСК.**
13. [S37](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S37.md) — Выбрать финальную конфигурацию и заморозить функции. **До 25.10 10:00 МСК.**
14. [S43](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S43.md) — Подтвердить техническую готовность финального архива. **До 25.10 17:00 МСК.**

### Юра: 11 карточек, около 15.5 ч активной работы

1. [S02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S02.md) — Реализовать загрузку файлов и сохранение порядка строк. **До 23.10 22:00 МСК.**
2. [S10](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S10.md) — Сделать предобработку числовых и категориальных признаков. **До 24.10 00:00 МСК.**
3. [S15](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S15.md) — Сформировать компактный автоматический профиль данных для LLM. **До 24.10 01:00 МСК.**
4. [S18](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S18.md) — Реализовать разбиение данных и вычисление метрик. **До 24.10 10:00 МСК.**
5. [S20](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S20.md) — Обучить первую регрессионную модель. **До 24.10 11:00 МСК.**
6. [S21](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S21.md) — Обучить первую модель классификации. **До 24.10 12:00 МСК.**
7. [S23](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S23.md) — Реализовать запись и валидацию submission.csv. **До 24.10 13:00 МСК.**
8. [S24](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S24.md) — Соединить полный запуск и передать первую рабочую версию. **До 24.10 14:00 МСК.**
9. [S31](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S31.md) — Проверить сбои экспериментов и сохранение раннего ответа. **До 24.10 19:00 МСК.**
10. [S33](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S33.md) — Ограничить ресурсы и убрать лишние расходы pipeline. **До 24.10 21:00 МСК.**
11. [S38](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S38.md) — Исправить критические баги и проверить финальный pipeline. **До 25.10 10:00 МСК.**

### Егор / PM: 7 карточек, около 8.5 ч активной работы

1. [S03](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S03.md) — Запустить работу команды и снять вопросы по доступам. **До 23.10 21:00 МСК.**
2. [S07](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S07.md) — Проверить Python, зависимости и доступ к API. **До 23.10 23:00 МСК.**
3. [S12](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S12.md) — Подготовить сборку и аудит ZIP решения. **До 24.10 00:00 МСК.**
4. [S16](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S16.md) — Провести первую интеграционную сверку. **До 24.10 01:00 МСК.**
5. [S26](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S26.md) — Собрать и отправить первый рабочий сабмит. **До 24.10 15:00 МСК.**
6. [S40](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S40.md) — Провести полный прогон из чистого архива. **До 25.10 16:00 МСК.**
7. [S44](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S44.md) — Отправить финал и проверить, что сдача зафиксирована. **До 25.10 19:30 МСК.**

### Артём: 6 карточек, около 9.5 ч активной работы

1. [S04](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S04.md) — Собрать навигатор по материалам и комплектам данных. **До 23.10 21:00 МСК.**
2. [S08](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S08.md) — Провести EDA регрессии и записать проверенные наблюдения. **До 23.10 23:00 МСК.**
3. [S13](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S13.md) — Провести EDA классификации и оценить дисбаланс. **До 24.10 01:00 МСК.**
4. [S19](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S19.md) — Свести EDA в короткий список решений и рисков. **До 24.10 10:00 МСК.**
5. [S27](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S27.md) — Сверить решение с требованиями и собрать доказательства. **До 24.10 16:00 МСК.**
6. [S39](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S39.md) — Подготовить понятное описание результата и материалы сдачи. **До 25.10 11:00 МСК.**

### Максим: 6 карточек, около 7.5 ч активной работы

1. [S05](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S05.md) — Проверить ссылки и подготовить шаблон отчёта о запуске. **До 23.10 21:00 МСК.**
2. [S14](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S14.md) — Повторить подготовку окружения по инструкции. **До 24.10 01:00 МСК.**
3. [S28](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S28.md) — Независимо повторить обе публичные команды. **До 24.10 16:00 МСК.**
4. [S34](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S34.md) — Повторить проверки ошибок и обновить список багов. **До 24.10 21:00 МСК.**
5. [S41](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S41.md) — Проверить состав финального архива по чек-листу. **До 25.10 11:00 МСК.**
6. [S42](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/steps/S42.md) — Собрать результаты полных прогонов и финальный список багов. **До 25.10 16:00 МСК.**

## Если задача опаздывает

1. Сразу отметить blocked, написать конкретную причину, что уже работает, кто нужен и новое время прогноза. Самостоятельно не сдвигать сроки зависимых карточек.
2. Егор и Иван выбирают: сократить объём, снять P1, временно упростить инструмент или передать конкретный блокер. Цель первого сабмита — рабочий полный путь.
3. EDA и дополнительные отчёты Артёма и проверки Максима не задерживают первый запуск Ивана и Юры. При опоздании вспомогательных проверок их минимальную часть выполняет PM.
4. После заморозки исправляем блокеры и ошибки формата/запуска; новые функции откладываем. Артём помогает с материалами, Максим — с воспроизведением и фактами.
