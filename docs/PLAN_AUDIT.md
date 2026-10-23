# Проверка соответствия плану кейса — 23.10.2026

Проверены все 44 прежние рабочие карточки по PDF кейса (с. 5–8, 11–19), обоим task.md и 55 ответам организаторов. Результат: технический замысел соответствует кейсу, но план был перегружен отдельными карточками и формальными результатами. Он заменён 14 самостоятельными блоками. Это аудит плана, не подтверждение работоспособности ещё не написанного решения.

## Что действительно требуется

| Требование | Где выполняется |
| --- | --- |
| Самостоятельно прочитать описание/данные и определить постановку | [A01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A01.md), [A02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A02.md) |
| Реальные решения LLM, готовые инструменты допустимы (FAQ 3–7, 52) | [A01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A01.md), [A06](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A06.md) |
| Эксперименты, оценка, дальнейшее решение/остановка (FAQ 24–26) | [A06](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A06.md) |
| Новые табличные задачи/метрики/выходы, контекст валидации (FAQ 16–22, 30, 55) | [A10](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A10.md), [A07](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A07.md) |
| Корректный ответ, интерфейс запуска и ресурсы | [A02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A02.md), [A03](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A03.md), [A07](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A07.md) |
| Запуск из чистой сдаваемой версии | [A13](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A13.md) |
| Учёт автоматического выбора лучшей public-отправки (FAQ 40–41) | [A08](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A08.md), [A12](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A12.md), [A13](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A13.md) |

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
| S01 Согласовать контракт задачи и результатов ML-инструментов | Координация | [A01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A01.md) |
| S02 Реализовать загрузку файлов и сохранение порядка строк | Рабочая часть решения | [A02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A02.md) |
| S03 Запустить работу команды и снять вопросы по доступам | Координация | [A03](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A03.md) |
| S04 Собрать навигатор по материалам и комплектам данных | Помощь разработке/проверке | [A04](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A04.md) |
| S05 Проверить ссылки и подготовить шаблон отчёта о запуске | Помощь разработке/проверке | [A05](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A05.md) |
| S06 Научить агент читать описание новой задачи | Рабочая часть решения | [A01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A01.md) |
| S07 Проверить Python, зависимости и доступ к API | Рабочая часть решения | [A03](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A03.md) |
| S08 Провести EDA регрессии и записать проверенные наблюдения | Помощь разработке/проверке | [A04](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A04.md) |
| S09 Подключить LLM и проверить структурированный ответ | Рабочая часть решения | [A01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A01.md) |
| S10 Сделать предобработку числовых и категориальных признаков | Рабочая часть решения | [A02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A02.md) |
| S11 Собрать контроллер инструментов на заглушках | Рабочая часть решения | [A01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A01.md) |
| S12 Подготовить сборку и аудит ZIP решения | Рабочая часть решения | [A03](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A03.md) |
| S13 Провести EDA классификации и оценить дисбаланс | Помощь разработке/проверке | [A04](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A04.md) |
| S14 Повторить подготовку окружения по инструкции | Помощь разработке/проверке | [A05](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A05.md) |
| S15 Сформировать компактный автоматический профиль данных для LLM | Рабочая часть решения | [A02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A02.md) |
| S16 Провести первую интеграционную сверку | Координация | [A03](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A03.md) |
| S17 Выбрать базовые ML-конфигурации и план экспериментов | Рабочая часть решения | [A01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A01.md) |
| S18 Реализовать разбиение данных и вычисление метрик | Рабочая часть решения | [A02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A02.md) |
| S19 Свести EDA в короткий список решений и рисков | Помощь разработке/проверке | [A04](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A04.md) |
| S20 Обучить первую регрессионную модель | Рабочая часть решения | [A02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A02.md) |
| S21 Обучить первую модель классификации | Рабочая часть решения | [A02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A02.md) |
| S22 Подключить реальные ML-инструменты к LLM-контроллеру | Рабочая часть решения | [A01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A01.md) |
| S23 Реализовать запись и валидацию submission.csv | Рабочая часть решения | [A02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A02.md) |
| S24 Соединить полный запуск и передать первую рабочую версию | Рабочая часть решения | [A02](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A02.md) |
| S25 Сохранять историю экспериментов и лучший результат | Рабочая часть решения | [A06](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A06.md) |
| S26 Собрать и отправить первый рабочий сабмит | Рабочая часть решения | [A08](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A08.md) |
| S27 Сверить решение с требованиями и собрать доказательства | Помощь разработке/проверке | [A09](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A09.md) |
| S28 Независимо повторить обе публичные команды | Помощь разработке/проверке | [A05](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A05.md) |
| S29 Замкнуть AutoResearch и ограничить время выполнения | Рабочая часть решения | [A06](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A06.md) |
| S30 Проверить новые схемы входа и ответа | Рабочая часть решения | [A10](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A10.md) |
| S31 Проверить сбои экспериментов и сохранение раннего ответа | Рабочая часть решения | [A07](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A07.md) |
| S32 Проверить метрики, редкие случаи и границы поддержки | Рабочая часть решения | [A10](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A10.md) |
| S33 Ограничить ресурсы и убрать лишние расходы pipeline | Рабочая часть решения | [A07](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A07.md) |
| S34 Повторить проверки ошибок и обновить список багов | Помощь разработке/проверке | [A05](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A05.md) |
| S35 Сравнить варианты регрессии на одной валидации | Рабочая часть решения | [A12](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A12.md) |
| S36 Сравнить варианты классификации с учётом редких классов | Рабочая часть решения | [A12](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A12.md) |
| S37 Выбрать финальную конфигурацию и заморозить функции | Рабочая часть решения | [A12](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A12.md) |
| S38 Исправить критические баги и проверить финальный pipeline | Рабочая часть решения | [A11](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A11.md) |
| S39 Подготовить понятное описание результата и материалы сдачи | Помощь разработке/проверке | [A09](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A09.md) |
| S40 Провести полный прогон из чистого архива | Рабочая часть решения | [A13](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A13.md) |
| S41 Проверить состав финального архива по чек-листу | Помощь разработке/проверке | [A14](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A14.md) |
| S42 Собрать результаты полных прогонов и финальный список багов | Помощь разработке/проверке | [A14](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A14.md) |
| S43 Подтвердить техническую готовность финального архива | Рабочая часть решения | [A12](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A12.md) |
| S44 Отправить финал и проверить, что сдача зафиксирована | Рабочая часть решения | [A13](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A13.md) |

## Что делать Ивану сейчас

Открыть [A01](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/tasks/current/A01.md). За короткую сверку с Юрой определить один внутренний вызов, затем делать код: чтение task.md, подключение выданной LLM, исполняемый вызов инструмента. К 23:00 — рабочая цепочка на заглушке; к 24.10 14:00 — реальные инструменты и два корректных ответа. Сам по себе документ об архитектуре баллов не даёт.

Источники: [кейс PDF](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/reference/Alfa_AutoML.pdf), [FAQ](https://docs.google.com/document/d/1Y1GZ81NkgSCnRv084xWOWokPojTIKz-v8XJ6vkTGTxU/edit), [публичные task.md](https://github.com/selecteg12/alfa-automl-hackathon/blob/main/docs/PUBLIC_TASKS.md).
