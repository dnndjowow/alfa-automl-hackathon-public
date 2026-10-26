# ТЗ для Ивана: офлайн-сравнение `feedaa8` и `f350a0c` на public — почему 24.25 < 24.94

От Юры · 26.10.2026, 14:40 МСК · у Юры сейчас нет доступа к PC

## Зачем

| Отправка | Что внутри | Балл |
| --- | --- | --- |
| `feedaa8` (01:15) | без двухступенчатой схемы | **24.94** — лучшая, идёт в private |
| `f350a0c` (10:54) | + двухступенчатая схема (F20–F24) + твои F01/F02/F06/F07/F12 | **24.25** (−0.69) |

Отправку `f350a0c` делали без прогонов на данных, поэтому мы не знаем, где потеря:
- в мультиклассе (схема или резерв 20% времени);
- в регрессии (твои правки F01/F06/F07/F12);
- или это разброс LLM.

На 26.10 осталась **1 попытка** — не тратим её вслепую.

**Что уже проверено (Юра, Mac):**
- `solution_f350a0c.zip` из кабинета: все 12 `.py` побайтно = коммит `f350a0c`, ядро = `b119dd6`, кроме `ml_io.py` (твоя правка F02). SHA256 полностью — `a0438309e650112893c4d1bbfd66ffe2c36054f469820c9f5fa02022332c2aa8`; в `SUBMISSIONS.md` записан префикс `a0438309e6504128` — расходится после 12-го знака, похоже на опечатку (размер 101 023 совпадает). Поправь, пожалуйста, запись.
- Твоя правка F02 в `ml_io.py` на чтение public **не влияет**: отпечатки train те же, что у ядра `b119dd6`.
- Скрипт ниже проверен на Mac коротким прогоном (бюджет 300 с, мультикласс): оба ZIP — exit 0, Macro F1 0.5300 у обоих. При таком бюджете схема не включается (резерву нужно ≥ 300 с остатка), поэтому для ответа нужен полный бюджет.

## Что сделать

Прогнать **оба ZIP из кабинета** на public-мультиклассе и public-регрессии так же, как платформа (`python -I solution.py task.md` из распакованного ZIP), но **без LLM**. Без LLM прогон детерминирован, и разница будет только от кода. Оценка — по скрытым ответам (`data/public_truth/`, у тебя есть).

| # | Прогон | Бюджет | Время |
| --- | --- | --- | --- |
| 1 | мультикласс `feedaa8` | `AGENT_TIME_LIMIT=2400`, `AGENT_SAFETY_S=300`, `ML_N_JOBS=8` | ≈ 40 мин |
| 2 | мультикласс `f350a0c` | то же | ≈ 40 мин |
| 3 | регрессия `feedaa8` | то же | ≈ 40 мин |
| 4 | регрессия `f350a0c` | то же | ≈ 40 мин |

Если есть сервер, мультикласс и регрессию можно гнать параллельно на разных машинах.

**Команда** (скрипт — ниже целиком; сохрани его как **`tests/zip_compare.py`**: он ищет `data/` на уровень выше своей папки. Не коммить):

```bash
python tests/zip_compare.py --zip feedaa8=artifacts/solution_feedaa8.zip --zip f350a0c=artifacts/solution_f350a0c.zip --label ivan
# только мультикласс: --tasks multiclass ; быстрая проверка: --time-limit 300 --safety 60
```

Результат:
- таблица `output/zip_compare_ivan.md`, перезаписывается после каждого прогона;
- журналы — `output/zip_runs_ivan/<задача>_<zip>.log`.

## Что прислать сюда (`docs/STATUS_YURI.md`)

1. Таблица из `zip_compare_ivan.md`: задача, ZIP, метрика, значение, строки `[two_stage]`, итог агента.
2. Из журнала мультикласса `f350a0c`: строки `[two_stage] …` целиком (сколько отложено, включилась ли, доля фона, время).
3. Вывод: в какой задаче разница и в какую сторону.

**Как читать результат:**

| Офлайн | Что значит | Последняя попытка 26.10 |
| --- | --- | --- |
| `f350a0c` ≥ `feedaa8` в обеих задачах | падение на платформе — разброс LLM, а не код | можно отправить `f350a0c` ещё раз или лучшую версию — по твоему решению |
| `f350a0c` хуже в мультиклассе | схема или резерв вредят в полном агенте → разбираю я | не отправлять со схемой; при желании — версия с `AGENT_TWO_STAGE` выключенным в коде |
| `f350a0c` хуже в регрессии | причина в F01/F06/F07/F12 → разбираешь ты | не отправлять, пока не найдена причина |

## Скрипт `tests/zip_compare.py`

```python
"""Офлайн-сравнение отправленных ZIP на public-задачах: как на платформе, но без LLM, с оценкой по скрытым ответам.

Для каждого ZIP и задачи: ZIP распаковывается в новый пустой каталог, задача копируется в новый каталог,
запускается `python -I <zip>/solution.py task.md` (из каталога задачи, AGENT_LLM=off), ответ оценивается по
data/public_truth/{regression,multiclass}_truth.csv. Из журнала берутся строки [two_stage] и итоговое «готово: записан …».
Отчёт перезаписывается после каждого прогона.

Запуск из корня репозитория (Windows):
  .venv\\Scripts\\python output\\pc_zip_compare.py --zip feedaa8=C:\\path\\solution_feedaa8.zip --zip f350a0c=C:\\path\\solution_f350a0c.zip
  ... --tasks multiclass --time-limit 600 --label smoke   # быстрая проверка
Результат: output/zip_compare_<label>.md и журналы в output/zip_runs_<label>/.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--zip", action="append", required=True, help="метка=путь к ZIP, можно несколько раз")
parser.add_argument("--tasks", default="multiclass,regression")
parser.add_argument("--time-limit", type=int, default=2400, help="AGENT_TIME_LIMIT, с")
parser.add_argument("--safety", type=int, default=300, help="AGENT_SAFETY_S, с")
parser.add_argument("--n-jobs", type=int, default=8)
parser.add_argument("--label", default="pc")
args = parser.parse_args()
CASES = {"multiclass": ("multiclass_case", "attack_cat"), "regression": ("regression_case", "site_eui")}
RUNS = ROOT / "output" / f"zip_runs_{args.label}"
RUNS.mkdir(parents=True, exist_ok=True)
rows = []


def score(task, submission):
    case, col = CASES[task]
    truth = pd.read_csv(ROOT / "data" / "public_truth" / f"{task}_truth.csv")
    pred = pd.read_csv(submission)
    merged = truth.merge(pred, on="id", suffixes=("", "_p"))
    problems = []
    if len(pred) != len(truth) or len(merged) != len(truth):
        problems.append(f"строк {len(pred)}, совпало id {len(merged)} из {len(truth)}")
    if task == "regression":
        value = float(np.sqrt(((merged[col] - merged[col + "_p"]) ** 2).mean()))
        return "RMSE", value, problems
    value = float(f1_score(merged[col].astype(str), merged[col + "_p"].astype(str), average="macro"))
    return "Macro F1", value, problems


def report():
    if not rows:
        return
    table = pd.DataFrame(rows)
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    (ROOT / "output" / f"zip_compare_{args.label}.md").write_text(
        f"# Офлайн-сравнение ZIP на public ({args.label})\n\nAGENT_LLM=off, AGENT_TIME_LIMIT={args.time_limit}, "
        f"AGENT_SAFETY_S={args.safety}, ML_N_JOBS={args.n_jobs}, `python -I solution.py task.md` из распакованного ZIP.\n"
        "RMSE — меньше лучше, Macro F1 — больше лучше.\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


for task in args.tasks.split(","):
    case, _ = CASES[task]
    for item in args.zip:
        label, path = item.split("=", 1)
        zdir, tdir = RUNS / f"zip_{label}", RUNS / f"{task}_{label}"
        for d in (zdir, tdir):
            if d.exists():
                shutil.rmtree(d)
        zdir.mkdir()
        with zipfile.ZipFile(path) as z:
            z.extractall(zdir)
        shutil.copytree(ROOT / "data" / "public" / case, tdir)
        (tdir / "submission.csv").unlink(missing_ok=True)
        env = {**os.environ, "AGENT_LLM": "off", "AGENT_TIME_LIMIT": str(args.time_limit),
               "AGENT_SAFETY_S": str(args.safety), "ML_N_JOBS": str(args.n_jobs), "PYTHONIOENCODING": "utf-8"}
        log_path = RUNS / f"{task}_{label}.log"
        print(f"{time.strftime('%H:%M:%S')} старт: {task} / {label}", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, "-I", str(zdir / "solution.py"), "task.md"], cwd=tdir,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        seconds = round(time.time() - t0)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        two_stage = " ‖ ".join(line.split("] ", 1)[-1][:160] for line in text.splitlines() if "[two_stage]" in line) or "—"
        done = next((line.split("] ", 1)[-1][:200] for line in reversed(text.splitlines()) if "готово:" in line), "—")
        row = {"задача": task, "ZIP": label, "код выхода": code, "секунд": seconds}
        submission = tdir / "submission.csv"
        if submission.exists():
            metric, value, problems = score(task, submission)
            row.update({"метрика": metric, "значение": value, "проблемы": "; ".join(problems) or "—"})
        else:
            row.update({"метрика": "—", "значение": None, "проблемы": "submission.csv не создан"})
        row.update({"two_stage в журнале": two_stage, "итог агента": done})
        rows.append(row)
        print({k: row[k] for k in ("задача", "ZIP", "код выхода", "секунд", "метрика", "значение")}, flush=True)
        report()
print(f"отчёт: output/zip_compare_{args.label}.md")
```
