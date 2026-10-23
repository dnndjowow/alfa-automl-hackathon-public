# Проверка соответствия плану кейса — 23.10.2026

Проверены все 44 прежние рабочие карточки по PDF кейса (с. 5–8, 11–19), обоим task.md и 55 ответам организаторов. Результат: технический замысел соответствует кейсу, но план был перегружен отдельными карточками и формальными результатами. Он заменён 14 самостоятельными блоками. Это аудит плана, не подтверждение работоспособности ещё не написанного решения.

## Что действительно требуется

| Требование | Где выполняется |
| --- | --- |
| Самостоятельно прочитать описание/данные и определить постановку | [A01 · #57](https://github.com/selecteg12/alfa-automl-hackathon/issues/57), [A02 · #58](https://github.com/selecteg12/alfa-automl-hackathon/issues/58) |
| Реальные решения LLM, готовые инструменты допустимы (FAQ 3–7, 52) | [A01 · #57](https://github.com/selecteg12/alfa-automl-hackathon/issues/57), [A06 · #62](https://github.com/selecteg12/alfa-automl-hackathon/issues/62) |
| Эксперименты, оценка, дальнейшее решение/остановка (FAQ 24–26) | [A06 · #62](https://github.com/selecteg12/alfa-automl-hackathon/issues/62) |
| Новые табличные задачи/метрики/выходы, контекст валидации (FAQ 16–22, 30, 55) | [A10 · #66](https://github.com/selecteg12/alfa-automl-hackathon/issues/66), [A07 · #63](https://github.com/selecteg12/alfa-automl-hackathon/issues/63) |
| Корректный ответ, интерфейс запуска и ресурсы | [A02 · #58](https://github.com/selecteg12/alfa-automl-hackathon/issues/58), [A03 · #59](https://github.com/selecteg12/alfa-automl-hackathon/issues/59), [A07 · #63](https://github.com/selecteg12/alfa-automl-hackathon/issues/63) |
| Запуск из чистой сдаваемой версии | [A13 · #69](https://github.com/selecteg12/alfa-automl-hackathon/issues/69) |
| Учёт автоматического выбора лучшей public-отправки (FAQ 40–41) | [A08 · #64](https://github.com/selecteg12/alfa-automl-hackathon/issues/64), [A12 · #68](https://github.com/selecteg12/alfa-automl-hackathon/issues/68), [A13 · #69](https://github.com/selecteg12/alfa-automl-hackathon/issues/69) |

Основной критерий — private score; при сопоставимом качестве учитываются токены, затем время (FAQ 46–50). Дополнительные отчёты/история экспериментов для сдачи не требуются (FAQ 8, 27), но минимальное состояние полезно самому агенту. Количество инструментов/агентов отдельно не оценивается (FAQ 2, 48).

## Исправленные перекосы

- Первая задача Ивана теперь даёт работающий агентный запуск. С Юрой нужна короткая договорённость о вызове; отдельные TaskSpec/ExperimentConfig/ExperimentResult как обязательная архитектура сняты.
- 44 мелкие карточки и 12 обзорных этапов заменены 14 блоками с промежуточными результатами внутри. Реализацию владелец выбирает сам; отчёт только при результате/блокере.
- Убраны отдельные каталоги материалов, шаблоны отчётов, матрицы и описания архитектуры ради сдачи. Один README и фактическая проверка достаточны для команды.
- Ручной EDA не заменяет автоматический профиль/анализ агента; Артём помогает фактами и проверками новых входов.
- Сохранение истории оставлено в минимальном техническом объёме; отдельный отчёт о каждом эксперименте не нужен.
- Ограничения ресурсов остаются обязательными, дополнительное ускорение выполняется по пользе.
- Первый сабмит должен уже иметь настоящую агентную логику. Обоснованная остановка после одного эксперимента допустима; голый статический pipeline не считается агентом.
- Нельзя считать последний ZIP автоматически финальным: выбор зависит от public score, а универсальность потребуется на private.
- Неподдержанная private-постановка — риск потери баллов; запись об ограничении не заменяет работающий путь её решения.

## Куда вошли все прежние задачи

| Прежняя карточка | Назначение | Текущий блок |
| --- | --- | --- |
| S01 Согласовать контракт задачи и результатов ML-инструментов | Координация | [A01 · #57](https://github.com/selecteg12/alfa-automl-hackathon/issues/57) |
| S02 Реализовать загрузку файлов и сохранение порядка строк | Рабочая часть решения | [A02 · #58](https://github.com/selecteg12/alfa-automl-hackathon/issues/58) |
| S03 Запустить работу команды и снять вопросы по доступам | Координация | [A03 · #59](https://github.com/selecteg12/alfa-automl-hackathon/issues/59) |
| S04 Собрать навигатор по материалам и комплектам данных | Помощь разработке/проверке | [A04 · #60](https://github.com/selecteg12/alfa-automl-hackathon/issues/60) |
| S05 Проверить ссылки и подготовить шаблон отчёта о запуске | Помощь разработке/проверке | [A05 · #61](https://github.com/selecteg12/alfa-automl-hackathon/issues/61) |
| S06 Научить агент читать описание новой задачи | Рабочая часть решения | [A01 · #57](https://github.com/selecteg12/alfa-automl-hackathon/issues/57) |
| S07 Проверить Python, зависимости и доступ к API | Рабочая часть решения | [A03 · #59](https://github.com/selecteg12/alfa-automl-hackathon/issues/59) |
| S08 Провести EDA регрессии и записать проверенные наблюдения | Помощь разработке/проверке | [A04 · #60](https://github.com/selecteg12/alfa-automl-hackathon/issues/60) |
| S09 Подключить LLM и проверить структурированный ответ | Рабочая часть решения | [A01 · #57](https://github.com/selecteg12/alfa-automl-hackathon/issues/57) |
| S10 Сделать предобработку числовых и категориальных признаков | Рабочая часть решения | [A02 · #58](https://github.com/selecteg12/alfa-automl-hackathon/issues/58) |
| S11 Собрать контроллер инструментов на заглушках | Рабочая часть решения | [A01 · #57](https://github.com/selecteg12/alfa-automl-hackathon/issues/57) |
| S12 Подготовить сборку и аудит ZIP решения | Рабочая часть решения | [A03 · #59](https://github.com/selecteg12/alfa-automl-hackathon/issues/59) |
| S13 Провести EDA классификации и оценить дисбаланс | Помощь разработке/проверке | [A04 · #60](https://github.com/selecteg12/alfa-automl-hackathon/issues/60) |
| S14 Повторить подготовку окружения по инструкции | Помощь разработке/проверке | [A05 · #61](https://github.com/selecteg12/alfa-automl-hackathon/issues/61) |
| S15 Сформировать компактный автоматический профиль данных для LLM | Рабочая часть решения | [A02 · #58](https://github.com/selecteg12/alfa-automl-hackathon/issues/58) |
| S16 Провести первую интеграционную сверку | Координация | [A03 · #59](https://github.com/selecteg12/alfa-automl-hackathon/issues/59) |
| S17 Выбрать базовые ML-конфигурации и план экспериментов | Рабочая часть решения | [A01 · #57](https://github.com/selecteg12/alfa-automl-hackathon/issues/57) |
| S18 Реализовать разбиение данных и вычисление метрик | Рабочая часть решения | [A02 · #58](https://github.com/selecteg12/alfa-automl-hackathon/issues/58) |
| S19 Свести EDA в короткий список решений и рисков | Помощь разработке/проверке | [A04 · #60](https://github.com/selecteg12/alfa-automl-hackathon/issues/60) |
| S20 Обучить первую регрессионную модель | Рабочая часть решения | [A02 · #58](https://github.com/selecteg12/alfa-automl-hackathon/issues/58) |
| S21 Обучить первую модель классификации | Рабочая часть решения | [A02 · #58](https://github.com/selecteg12/alfa-automl-hackathon/issues/58) |
| S22 Подключить реальные ML-инструменты к LLM-контроллеру | Рабочая часть решения | [A01 · #57](https://github.com/selecteg12/alfa-automl-hackathon/issues/57) |
| S23 Реализовать запись и валидацию submission.csv | Рабочая часть решения | [A02 · #58](https://github.com/selecteg12/alfa-automl-hackathon/issues/58) |
| S24 Соединить полный запуск и передать первую рабочую версию | Рабочая часть решения | [A02 · #58](https://github.com/selecteg12/alfa-automl-hackathon/issues/58) |
| S25 Сохранять историю экспериментов и лучший результат | Рабочая часть решения | [A06 · #62](https://github.com/selecteg12/alfa-automl-hackathon/issues/62) |
| S26 Собрать и отправить первый рабочий сабмит | Рабочая часть решения | [A08 · #64](https://github.com/selecteg12/alfa-automl-hackathon/issues/64) |
| S27 Сверить решение с требованиями и собрать доказательства | Помощь разработке/проверке | [A09 · #65](https://github.com/selecteg12/alfa-automl-hackathon/issues/65) |
| S28 Независимо повторить обе публичные команды | Помощь разработке/проверке | [A05 · #61](https://github.com/selecteg12/alfa-automl-hackathon/issues/61) |
| S29 Замкнуть AutoResearch и ограничить время выполнения | Рабочая часть решения | [A06 · #62](https://github.com/selecteg12/alfa-automl-hackathon/issues/62) |
| S30 Проверить новые схемы входа и ответа | Рабочая часть решения | [A10 · #66](https://github.com/selecteg12/alfa-automl-hackathon/issues/66) |
| S31 Проверить сбои экспериментов и сохранение раннего ответа | Рабочая часть решения | [A07 · #63](https://github.com/selecteg12/alfa-automl-hackathon/issues/63) |
| S32 Проверить метрики, редкие случаи и границы поддержки | Рабочая часть решения | [A10 · #66](https://github.com/selecteg12/alfa-automl-hackathon/issues/66) |
| S33 Ограничить ресурсы и убрать лишние расходы pipeline | Рабочая часть решения | [A07 · #63](https://github.com/selecteg12/alfa-automl-hackathon/issues/63) |
| S34 Повторить проверки ошибок и обновить список багов | Помощь разработке/проверке | [A05 · #61](https://github.com/selecteg12/alfa-automl-hackathon/issues/61) |
| S35 Сравнить варианты регрессии на одной валидации | Рабочая часть решения | [A12 · #68](https://github.com/selecteg12/alfa-automl-hackathon/issues/68) |
| S36 Сравнить варианты классификации с учётом редких классов | Рабочая часть решения | [A12 · #68](https://github.com/selecteg12/alfa-automl-hackathon/issues/68) |
| S37 Выбрать финальную конфигурацию и заморозить функции | Рабочая часть решения | [A12 · #68](https://github.com/selecteg12/alfa-automl-hackathon/issues/68) |
| S38 Исправить критические баги и проверить финальный pipeline | Рабочая часть решения | [A11 · #67](https://github.com/selecteg12/alfa-automl-hackathon/issues/67) |
| S39 Подготовить понятное описание результата и материалы сдачи | Помощь разработке/проверке | [A09 · #65](https://github.com/selecteg12/alfa-automl-hackathon/issues/65) |
| S40 Провести полный прогон из чистого архива | Рабочая часть решения | [A13 · #69](https://github.com/selecteg12/alfa-automl-hackathon/issues/69) |
| S41 Проверить состав финального архива по чек-листу | Помощь разработке/проверке | [A14 · #70](https://github.com/selecteg12/alfa-automl-hackathon/issues/70) |
| S42 Собрать результаты полных прогонов и финальный список багов | Помощь разработке/проверке | [A14 · #70](https://github.com/selecteg12/alfa-automl-hackathon/issues/70) |
| S43 Подтвердить техническую готовность финального архива | Рабочая часть решения | [A12 · #68](https://github.com/selecteg12/alfa-automl-hackathon/issues/68) |
| S44 Отправить финал и проверить, что сдача зафиксирована | Рабочая часть решения | [A13 · #69](https://github.com/selecteg12/alfa-automl-hackathon/issues/69) |

## Что делать Ивану сейчас

Открыть [A01 · #57](https://github.com/selecteg12/alfa-automl-hackathon/issues/57). За короткую сверку с Юрой определить один внутренний вызов, затем делать код: чтение task.md, подключение выданной LLM, исполняемый вызов инструмента. К 23:00 — рабочая цепочка на заглушке; к 24.10 14:00 — реальные инструменты и два корректных ответа. Сам по себе документ об архитектуре баллов не даёт.

Источники: [кейс PDF](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/reference/Alfa_AutoML.pdf), [FAQ](https://docs.google.com/document/d/1Y1GZ81NkgSCnRv084xWOWokPojTIKz-v8XJ6vkTGTxU/edit), [публичные task.md](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/PUBLIC_TASKS.md).
