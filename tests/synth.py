"""Генератор маленьких синтетических задач для тестов (в ZIP не попадает).

make_task() создаёт каталог задачи: task.md, train/test/sample_submission/
data_dictionary и возвращает путь к task.md. Параметры нужны, чтобы проверять
не только две публичные задачи: другие имена таргета и id, строковые метки,
пропуски, новые категории в test, CRLF и BOM в task.md, разные раскладки файлов.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

CLASS_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon"]


class SynthTask(NamedTuple):
    task_md: Path
    truth: pd.DataFrame  # скрытые ответы test (id, target); в файлы задачи не пишутся


def make_frame(n, rng, kind, n_classes, target, nan_frac, extra_category=False):
    cats = ["red", "green", "blue"] + (["violet"] if extra_category else [])
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.integers(0, 100, size=n).astype(float),
            "flag": rng.integers(0, 2, size=n),
            "cat": rng.choice(cats, size=n),
            "grp": rng.choice([f"g{i}" for i in range(25)], size=n),
        }
    )
    effect = df["cat"].map({"red": 0.0, "green": 1.0, "blue": -1.0, "violet": 2.0}).to_numpy()
    if kind == "regression":
        df[target] = 2 * df["a"] - df["b"] + effect + rng.normal(scale=0.3, size=n)
    else:
        scores = np.column_stack(
            [(k + 1) * df["a"] * (-1) ** k + (k - 1) * df["b"] + effect * (k % 2) for k in range(n_classes)]
        ) + rng.normal(scale=0.5, size=(n, n_classes))
        df[target] = np.array(CLASS_NAMES[:n_classes])[scores.argmax(axis=1)]
    if nan_frac:
        for column in ("a", "c", "cat"):
            df.loc[rng.random(n) < nan_frac, column] = np.nan
    return df


def make_task(
    root,
    kind="regression",
    n_train=300,
    n_test=120,
    seed=0,
    target="y",
    id_col="id",
    n_classes=3,
    nan_frac=0.05,
    unseen_in_test=True,
    layout="nested",
    crlf=False,
    bom=False,
    extra_files=(),
):
    """kind: regression | binary | multiclass. layout: nested | flat. Возвращает SynthTask."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    n_classes = 2 if kind == "binary" else n_classes
    train = make_frame(n_train, rng, kind, n_classes, target, nan_frac)
    test = make_frame(n_test, rng, kind, n_classes, target, nan_frac, extra_category=unseen_in_test)
    test_target = test.pop(target)  # скрытые ответы в файлы не пишутся

    # id не 0..N-1 и не подряд — как в публичной регрессии
    train.insert(0, id_col, rng.permutation(np.arange(n_train * 3))[:n_train] * 2 + 10)
    test.insert(0, id_col, np.sort(rng.permutation(np.arange(n_test * 3))[:n_test]) * 2 + 10_001)
    placeholder = 0.0 if kind == "regression" else CLASS_NAMES[0]
    sample = pd.DataFrame({id_col: test[id_col], target: placeholder})
    dictionary = pd.DataFrame(
        {
            "feature": [c for c in train.columns],
            "role": ["id" if c == id_col else "target" if c == target else "feature" for c in train.columns],
            "dtype": ["category" if train[c].dtype == object else "numeric" for c in train.columns],
        }
    )

    sub = Path("data/participants") if layout == "nested" else Path(".")
    folder = root / sub
    folder.mkdir(parents=True, exist_ok=True)
    files = {"train": train, "test": test, "sample_submission": sample, "data_dictionary": dictionary}
    for stem, frame in files.items():
        frame.to_csv(folder / f"{stem}.csv", index=False)
    for name in extra_files:
        (folder / name).write_text("x,y\n1,2\n", encoding="utf-8")

    refs = "\n".join(f"- `{(sub / f'{stem}.csv').as_posix()}` — файл {stem}." for stem in files)
    text = (
        f"# Синтетическая задача ({kind})\n\n"
        f"Предскажите `{target}` для каждого `{id_col}` из test.\n\n"
        f"## Данные\n\nПути указаны относительно каталога этого файла:\n\n{refs}\n\n"
        f"## Формат результата\n\nФайл `submission.csv` рядом с task.md, колонки `{id_col},{target}`.\n"
    )
    if crlf:
        text = text.replace("\n", "\r\n")
    task_md = root / "task.md"
    task_md.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    truth = pd.DataFrame({id_col: test[id_col], target: test_target})
    return SynthTask(task_md, truth)
