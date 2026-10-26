"""Скрытые ответы public-задач из открытых исходников — чтобы мерить решение так же, как платформа.

Public-задачи — открытые датасеты: WiDS Datathon 2022 (site_eui) и UNSW-NB15 (attack_cat). Скрипт скачивает
оригиналы, сопоставляет с test и пишет data/public_truth/{regression,multiclass}_truth.csv (в git не попадают).

    python tests/public_truth.py            # собрать ответы (нужен data/public/*_case из кабинета)
    python tests/public_truth.py score DIR  # оценить DIR/regression/submission.csv и DIR/multiclass/submission.csv

UNSW: у ~10% строк test есть полные дубликаты признаков с разными метками; метки в такой группе раздаются
случайно в пропорции исходной группы (модель всё равно отвечает одинаково на одинаковые строки).
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data" / "public"
OUT = ROOT / "data" / "public_truth"
WIDS = "https://raw.githubusercontent.com/eminaomasa/energy_usage_prediction/HEAD/Data/train.csv"
UNSW = "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/train.csv"   # официальный testing-set


def fetch(url, name):
    path = OUT / name
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    wids = pd.read_csv(fetch(WIDS, "wids_train.csv"))
    test = pd.read_csv(PUBLIC / "regression_case/regression/participants/test.csv")
    merged = test[["id", "floor_area"]].merge(wids[["id", "floor_area", "site_eui"]], on="id", suffixes=("", "_src"))
    assert len(merged) == len(test) and np.allclose(merged.floor_area, merged.floor_area_src)
    merged[["id", "site_eui"]].to_csv(OUT / "regression_truth.csv", index=False)

    src = pd.read_csv(fetch(UNSW, "unsw_testing_set.csv"), encoding="utf-8-sig")
    test = pd.read_csv(PUBLIC / "multiclass_case/multiclass/participants/test.csv")
    feats = [c for c in test.columns if c != "id"]

    def key(frame):
        part = frame[feats].copy()
        for col in feats:
            if part[col].dtype.kind == "f":
                part[col] = part[col].round(5)
        return part.astype(str).agg("|".join, axis=1)

    test["k"], src["k"] = key(test), key(src)
    rng = np.random.default_rng(0)
    pools = {k: list(rng.permutation(g.attack_cat.values)) for k, g in src.groupby("k")}
    test["attack_cat"] = [pools[k].pop() for k in test.k]
    test[["id", "attack_cat"]].to_csv(OUT / "multiclass_truth.csv", index=False)
    print("ok:", OUT)


def check_submission(truth, pred, col):
    """Проблемы ответа: id должны быть уникальны и совпадать с truth, столбец ответа — без пропусков.
    Пусто — ответ полный; иначе метрику не считаем (ревью 26.10, R8: дубли id подменяли пропущенные строки)."""
    problems = []
    if "id" not in pred.columns or col not in pred.columns:
        return [f"нет столбцов id/{col}: {list(pred.columns)}"]
    if pred["id"].duplicated().any():
        problems.append(f"дубли id: {int(pred['id'].duplicated().sum())}")
    missing, extra = set(truth["id"]) - set(pred["id"]), set(pred["id"]) - set(truth["id"])
    if missing or extra:
        problems.append(f"id нет в ответе: {len(missing)}, лишних id: {len(extra)}")
    if pred[col].isna().any():
        problems.append(f"пропуски в {col}: {int(pred[col].isna().sum())}")
    return problems


def score(folder):
    from sklearn.metrics import f1_score
    for task, col in (("regression", "site_eui"), ("multiclass", "attack_cat")):
        path = Path(folder) / task / "submission.csv"
        if not path.exists():
            print(task, "нет", path)
            continue
        truth, pred = pd.read_csv(OUT / f"{task}_truth.csv"), pd.read_csv(path)
        problems = check_submission(truth, pred, col)
        if problems:
            print(task, "ответ неполный, метрика не считается:", "; ".join(problems))
            continue
        m = truth.merge(pred, on="id", suffixes=("", "_p"), validate="one_to_one")
        if task == "regression":
            print(task, "RMSE", round(float(np.sqrt(((m[col] - m[col + "_p"]) ** 2).mean())), 4))
        else:
            print(task, "macro F1", round(f1_score(m[col].astype(str), m[col + "_p"].astype(str), average="macro"), 4))


if __name__ == "__main__":
    score(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "score" else build()
