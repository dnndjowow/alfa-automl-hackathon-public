# Локальные данные для публичных задач

CSV скачиваются из папок организаторов. Они и ответы локальных запусков находятся в data/, который исключён из Git.

## Источники

- [Регрессия: все пять файлов](https://drive.google.com/drive/folders/1Wv-FZQHIRawf71S9Rfs4GkAEvGoayTUx).
- [Классификация: все пять файлов](https://drive.google.com/drive/folders/1d89FU5gTbwpNfPIh78GbZO7fPzj806u8).
- Проверенные копии task.md и data_dictionary.csv находятся в reference/public_tasks/; при новом обновлении свериться с источником.

## Порядок подготовки

1. В каждую папку организаторов зайти и скачать train.csv, test.csv, sample_submission.csv, data_dictionary.csv и task.md. CSV из классификации и регрессии не смешивать.
2. Создать два разных корня задач в data/public/. Файл task.md положить в корень соответствующей задачи.
3. Остальные файлы разместить по относительным путям из task.md. Google Drive показывает их в плоской папке; это не означает, что в запуске они должны лежать рядом с task.md.
4. Сверить имена файлов, UTF-8, запятую как разделитель и отсутствие случайного добавленного расширения .txt.
5. Сохранить исходные файлы без исправлений. Проверки изменённой схемы делать на отдельной маленькой локальной копии.

## Пример размещения

```text
data/public/
  regression_case/
    task.md
    regression/participants/
      train.csv
      test.csv
      data_dictionary.csv
      sample_submission.csv
  multiclass_case/
    task.md
    multiclass/participants/
      train.csv
      test.csv
      data_dictionary.csv
      sample_submission.csv
```

Этот пример воспроизводит пути двух прочитанных публичных task.md. Private-пути могут отличаться; агент разрешает их относительно расположения своего task.md.

После реализации точки входа команда по договорённости команды может выглядеть так:

```text
python solution.py data/public/regression_case/task.md
python solution.py data/public/multiclass_case/task.md
```

Это пример ожидаемого интерфейса, не подтверждение готовности solution.py. Фактические команды и окружение фиксирует PM в RUNBOOK.

## Ожидаемые ответы

- Регрессия: data/public/regression_case/submission.csv.
- Классификация: data/public/multiclass_case/submission.csv.

У каждого ответа собственный каталог task.md; один запуск не должен выдавать старый файл другого. Предсказания и исходные CSV не коммитить.

