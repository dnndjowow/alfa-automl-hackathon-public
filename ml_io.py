"""Поиск и загрузка файлов задачи по task.md.

Файлы ищутся по путям в обратных кавычках внутри task.md, а при их отсутствии —
по имени файла в каталоге задачи. Роль файла определяется по имени
(train, test, sample_submission, data_dictionary); остальные CSV попадают
в extra_csv. Имена таргета и id здесь не используются.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

ROLES = ("train", "test", "sample_submission", "data_dictionary")

_CSV_REF = re.compile(r"`([^`\n]+?\.(?:csv|tsv))`", re.IGNORECASE)
_TABLE_SUFFIXES = (".csv", ".tsv")
# метки пропуска, которые встречаются в числовых столбцах чужих датасетов
_MISSING_TOKENS = {"na", "n/a", "nan", "null", "none", "?", "-", "--", "#n/a"}
_NUMERIC_SHARE = 0.95
_JUNK_ROWS, _JUNK_SHARE, _JUNK_VALUES = 2, 0.001, 2  # единичный мусор в числовом столбце: строк и разных значений
_SKIP_DIRS = {"__pycache__", "node_modules", "venv", "site-packages"}
_MAX_DEPTH = 4


class TaskLoadError(RuntimeError):
    """task.md не найден или не читается."""


@dataclass
class TaskFiles:
    task_md: Path
    task_dir: Path
    paths: dict = field(default_factory=dict)  # роль -> Path | None
    train_labels: Optional[Path] = None        # метки train отдельным файлом (train_labels.csv, y_train.csv)
    extra_csv: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class TaskData:
    files: TaskFiles
    text: str
    train: Optional[pd.DataFrame] = None
    test: Optional[pd.DataFrame] = None
    sample_submission: Optional[pd.DataFrame] = None
    data_dictionary: Optional[pd.DataFrame] = None


def read_task_text(task_md) -> str:
    """Текст task.md: UTF-8 (с BOM или без), переводы строк приведены к \\n."""
    try:
        text = Path(task_md).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise TaskLoadError(f"не удалось прочитать {task_md}: {error}") from error
    return text.replace("\r\n", "\n").replace("\r", "\n")


_OUTPUT_TOKENS = {"pred", "preds", "prediction", "predictions", "submission", "oof", "output", "result", "results"}
_DICT_TOKENS = {"dict", "dictionary", "description", "descriptions"}


_TRAIN_TOKENS = {"train", "training", "trainset"}
_TEST_TOKENS = {"test", "testing", "testset"}
_LABEL_TOKENS = {"label", "labels", "target", "targets", "y"}
_TEMPLATE_TOKENS = {"sample", "example", "template"}


def classify_csv(path) -> Optional[str]:
    """Роль файла по словам в имени; None, если имя не распознано (в том числе submission.csv).

    Сравниваются целые слова, а не подстроки: predictions.csv не словарь, train_subsample.csv
    не sample_submission. Файлы с признаками выходных данных (pred, oof, output) роли не получают.
    Шаблон ответа — слово submission рядом с sample / example / template. Метки train отдельным
    файлом (train_labels.csv, y_train.csv) получают роль "train_labels".
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", Path(path).stem.lower()) if t]
    words = set(tokens)
    if any(t.startswith("sub") for t in tokens) and _TEMPLATE_TOKENS & words:
        return "sample_submission"
    if "".join(tokens) in ("samplesubmission", "samplesub"):
        return "sample_submission"
    if _OUTPUT_TOKENS & words:
        return None
    if _DICT_TOKENS & words:
        return "data_dictionary"
    if _TRAIN_TOKENS & words:
        return "train_labels" if _LABEL_TOKENS & words else "train"
    if _TEST_TOKENS & words and not _LABEL_TOKENS & words:
        return "test"
    return None


def display_path(path: Path, task_dir: Path) -> str:
    """Путь относительно каталога задачи, а для файла вне его — абсолютный."""
    try:
        return str(Path(path).relative_to(task_dir))
    except ValueError:
        return str(path)


def _rank(path: Path, task_dir: Path):
    """Ближе к task.md — лучше; файлы вне каталога задачи идут после файлов внутри."""
    try:
        return (0, len(path.relative_to(task_dir).parts), str(path))
    except ValueError:
        return (1, len(path.parts), str(path))


def referenced_csv_paths(text: str, task_dir: Path) -> list:
    """Существующие CSV, упомянутые в task.md в обратных кавычках (пути от task_dir)."""
    found = []
    for raw in _CSV_REF.findall(text):
        path = (task_dir / raw.strip().replace("\\", "/")).resolve()
        if path.is_file() and path not in found:
            found.append(path)
    return found


def _walk_csv(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS
        ) if depth < _MAX_DEPTH else []
        for name in sorted(filenames):
            if name.lower().endswith(_TABLE_SUFFIXES):
                yield (Path(dirpath) / name).resolve()


def find_task_files(task_md) -> TaskFiles:
    task_md = Path(task_md).resolve()
    task_dir = task_md.parent
    files = TaskFiles(task_md=task_md, task_dir=task_dir, paths={role: None for role in ROLES})
    refs = referenced_csv_paths(read_task_text(task_md), task_dir)

    candidates = {role: [] for role in ROLES + ("train_labels",)}
    # task.md обычно упоминает и будущий submission.csv: это наш ответ, а не входной файл
    refs = [p for p in refs if p.name.lower() != "submission.csv"]
    unrecognised = []
    for path in refs:
        role = classify_csv(path)
        if role:
            candidates[role].append(path)
        elif path not in unrecognised:
            unrecognised.append(path)

    missing = {role for role, found in candidates.items() if not found}
    for path in _walk_csv(task_dir):
        role = classify_csv(path)
        if role in missing:
            candidates[role].append(path)
        elif role is None and path not in refs and path.name.lower() != "submission.csv":
            if path not in unrecognised:
                unrecognised.append(path)

    for role, options in candidates.items():
        if not options:
            if role != "train_labels":
                files.warnings.append(f"файл роли {role!r} не найден")
            continue
        options = sorted(options, key=lambda p: _rank(p, task_dir))
        if role == "train_labels":
            files.train_labels = options[0]
        else:
            files.paths[role] = options[0]
        if len(options) > 1:
            files.warnings.append(
                f"для роли {role!r} несколько файлов, выбран {display_path(options[0], task_dir)}; "
                "остальные — в extra_csv"
            )
            files.extra_csv.extend(p for p in options[1:] if p not in files.extra_csv)
    if files.paths.get("train") is None and files.train_labels is not None:
        # только train_labels.csv / train_target.csv: это и есть train (с признаками и таргетом)
        files.paths["train"], files.train_labels = files.train_labels, None
        files.warnings = [w for w in files.warnings if w != "файл роли 'train' не найден"]
        files.warnings.append(f"отдельного train нет: как train используется {display_path(files.paths['train'], task_dir)}")
    for path in unrecognised:
        files.extra_csv.append(path)
        files.warnings.append(f"роль файла {display_path(path, task_dir)} не распознана")
    return files


def _attach_labels(data: "TaskData") -> None:
    """Присоединить метки из отдельного файла к train: по общим столбцам (id) или по порядку строк."""
    files = data.files
    labels = read_csv(files.train_labels)
    name = display_path(files.train_labels, files.task_dir)
    common = [c for c in labels.columns if c in data.train.columns]
    new = [c for c in labels.columns if c not in data.train.columns]
    if not new:
        files.warnings.append(f"в {name} нет новых столбцов: метки не присоединены")
    elif common and not labels[common].duplicated().any() and not data.train[common].duplicated().any():
        data.train = data.train.merge(labels[common + new], on=common, how="left")
        files.warnings.append(f"метки {new} из {name} присоединены к train по {common}")
    elif len(labels) == len(data.train):
        data.train = pd.concat([data.train.reset_index(drop=True), labels[new].reset_index(drop=True)], axis=1)
        files.warnings.append(f"метки {new} из {name} присоединены к train по порядку строк")
    else:
        files.warnings.append(f"метки из {name} не удалось сопоставить с train: файл в extra_csv")
        files.extra_csv.append(files.train_labels)


def read_csv(path, text_columns=()) -> pd.DataFrame:
    """Таблица по правилам кейса: UTF-8, запятая, точка; пропуск — пустая ячейка.

    Запасные варианты для чужих файлов: кодировка cp1251, если UTF-8 не читается; .tsv — табуляция;
    если всё прочиталось одним столбцом с «;» или табуляцией в заголовке — разделитель определяется
    автоматически. text_columns читаются как текст без преобразования в числа (коды «00123»).
    """
    path = Path(path)
    options = dict(keep_default_na=False, na_values=[""], dtype={c: str for c in text_columns} or None)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    frame, encoding = None, None
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            frame = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False, **options)
            break
        except UnicodeDecodeError:
            continue
    if frame is None:
        raise TaskLoadError(f"{path.name}: не удалось прочитать ни в UTF-8, ни в cp1251")
    if frame.shape[1] == 1 and any(mark in str(frame.columns[0]) for mark in (";", "\t", "|")):
        frame = pd.read_csv(path, sep=None, engine="python", encoding=encoding, **options)
    return frame


def numeric_text_columns(frame: pd.DataFrame) -> list:
    """Текстовые столбцы, которые на самом деле числовые: не меньше 95% непустых значений — числа,
    остальное — метки пропуска («NA», «?», «null», пробелы) и не больше max(2, 0.1%) строк единичного
    мусора с одним-двумя значениями. Больше нечисловых строк — это законная категория (например,
    редкий класс «A» в таргете из «1»/«2»/«3»), и столбец остаётся текстом."""
    found = []
    for col in frame.columns:
        values = frame[col]
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            continue
        text = values.dropna().astype(str).str.strip()
        rest = text[~(text.eq("") | text.str.lower().isin(_MISSING_TOKENS))]
        if not len(rest):
            continue
        bad = rest[pd.to_numeric(rest, errors="coerce").isna()]
        if (1 - len(bad) / len(rest) >= _NUMERIC_SHARE and len(bad) <= max(_JUNK_ROWS, _JUNK_SHARE * len(rest))
                and bad.nunique() <= _JUNK_VALUES):
            found.append(col)
    return found


def _to_numeric(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().where(values.notna())
    missing = text.isna() | text.eq("") | text.str.lower().isin(_MISSING_TOKENS)
    return pd.to_numeric(text.where(~missing), errors="coerce")


def _coerce(frame: pd.DataFrame, columns) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns and not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = _to_numeric(frame[col])
    return frame


def load_task(task_md) -> TaskData:
    """Найти и прочитать все распознанные файлы задачи. Отсутствующие роли остаются None.

    Типы столбцов задаёт train:
    - столбец, который почти весь из чисел и лишь местами содержит «NA», «?», «null» или пробелы,
      переводится в числа (в train и test), а такие метки становятся пропусками. По условию кейса
      пропуск — пустая ячейка; правило касается только явно числовых столбцов, текстовые
      категории (в том числе значение «NA») не трогаются;
    - если в train столбец текстовый, а в test он прочитался бы числами («00123» -> 123),
      test перечитывается с этим столбцом как текстом.
    """
    files = find_task_files(task_md)
    data = TaskData(files=files, text=read_task_text(files.task_md))
    for role, path in files.paths.items():
        if path is not None:
            setattr(data, role, read_csv(path))
    if data.train is None:
        return data
    if files.train_labels is not None:
        _attach_labels(data)
    numeric = numeric_text_columns(data.train)
    if data.test is not None:
        numeric += [c for c in data.test.columns
                    if c in data.train.columns and c not in numeric
                    and pd.api.types.is_numeric_dtype(data.train[c])
                    and not pd.api.types.is_numeric_dtype(data.test[c])]
    if numeric:
        data.train = _coerce(data.train, numeric)
        files.warnings.append(f"столбцы {numeric} переведены в числа: метки вроде 'NA', '?', 'null' — пропуски")
    if data.test is not None:
        text_in_train = [
            c for c in data.test.columns
            if c in data.train.columns
            and not pd.api.types.is_numeric_dtype(data.train[c])
            and pd.api.types.is_numeric_dtype(data.test[c])
        ]
        if text_in_train:
            data.test = read_csv(files.paths["test"], text_columns=text_in_train)
            files.warnings.append(f"столбцы {text_in_train} в test прочитаны как текст, как в train")
        data.test = _coerce(data.test, numeric)
    return data
