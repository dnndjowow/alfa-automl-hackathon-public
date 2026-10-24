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

_CSV_REF = re.compile(r"`([^`\n]+?\.csv)`", re.IGNORECASE)
_SKIP_DIRS = {"__pycache__", "node_modules", "venv", "site-packages"}
_MAX_DEPTH = 4


class TaskLoadError(RuntimeError):
    """task.md не найден или не читается."""


@dataclass
class TaskFiles:
    task_md: Path
    task_dir: Path
    paths: dict = field(default_factory=dict)  # роль -> Path | None
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


def classify_csv(path) -> Optional[str]:
    """Роль файла по словам в имени; None, если имя не распознано (в том числе submission.csv).

    Сравниваются целые слова, а не подстроки: predictions.csv не словарь, train_subsample.csv
    не sample_submission. Файлы с признаками выходных данных (pred, oof, output) роли не получают.
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", Path(path).stem.lower()) if t]
    if "sample" in tokens and any(t.startswith("sub") for t in tokens):
        return "sample_submission"
    if "".join(tokens) in ("samplesubmission", "samplesub"):
        return "sample_submission"
    if _OUTPUT_TOKENS & set(tokens):
        return None
    if _DICT_TOKENS & set(tokens):
        return "data_dictionary"
    if "train" in tokens:
        return "train"
    if "test" in tokens:
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
            if name.lower().endswith(".csv"):
                yield (Path(dirpath) / name).resolve()


def find_task_files(task_md) -> TaskFiles:
    task_md = Path(task_md).resolve()
    task_dir = task_md.parent
    files = TaskFiles(task_md=task_md, task_dir=task_dir, paths={role: None for role in ROLES})
    refs = referenced_csv_paths(read_task_text(task_md), task_dir)

    candidates = {role: [] for role in ROLES}
    # task.md обычно упоминает и будущий submission.csv: это наш ответ, а не входной файл
    refs = [p for p in refs if p.name.lower() != "submission.csv"]
    for path in refs:
        role = classify_csv(path)
        if role:
            candidates[role].append(path)
        elif path not in files.extra_csv:
            files.extra_csv.append(path)

    missing = {role for role in ROLES if not candidates[role]}
    for path in _walk_csv(task_dir):
        role = classify_csv(path)
        if role in missing:
            candidates[role].append(path)
        elif role is None and path not in refs and path.name.lower() != "submission.csv":
            if path not in files.extra_csv:
                files.extra_csv.append(path)

    for role, options in candidates.items():
        if not options:
            files.warnings.append(f"файл роли {role!r} не найден")
            continue
        options = sorted(options, key=lambda p: _rank(p, task_dir))
        files.paths[role] = options[0]
        if len(options) > 1:
            files.warnings.append(
                f"для роли {role!r} несколько файлов, выбран {display_path(options[0], task_dir)}"
            )
    for path in files.extra_csv:
        files.warnings.append(f"роль файла {display_path(path, task_dir)} не распознана")
    return files


def read_csv(path, text_columns=()) -> pd.DataFrame:
    """CSV по правилам кейса: UTF-8, запятая, точка; пропуск — только пустая ячейка.

    Строки вроде «NA», «None», «null» остаются значениями (это могут быть категории).
    text_columns читаются как текст без преобразования в числа (например, коды «00123»).
    """
    return pd.read_csv(
        path, encoding="utf-8-sig", keep_default_na=False, na_values=[""], low_memory=False,
        dtype={c: str for c in text_columns} or None,
    )


def load_task(task_md) -> TaskData:
    """Найти и прочитать все распознанные файлы задачи. Отсутствующие роли остаются None.

    Тип столбца задаёт train: если там столбец текстовый, а в test он прочитался бы числами
    («00123» -> 123), test перечитывается с этим столбцом как текстом.
    """
    files = find_task_files(task_md)
    data = TaskData(files=files, text=read_task_text(files.task_md))
    for role, path in files.paths.items():
        if path is not None:
            setattr(data, role, read_csv(path))
    if data.train is not None and data.test is not None:
        text_in_train = [
            c for c in data.test.columns
            if c in data.train.columns
            and not pd.api.types.is_numeric_dtype(data.train[c])
            and pd.api.types.is_numeric_dtype(data.test[c])
        ]
        if text_in_train:
            data.test = read_csv(files.paths["test"], text_columns=text_in_train)
            files.warnings.append(f"столбцы {text_in_train} в test прочитаны как текст, как в train")
    return data
