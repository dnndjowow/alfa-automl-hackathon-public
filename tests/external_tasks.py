"""Незнакомые задачи из открытых датасетов (OpenML, scikit-learn, зеркала Kaggle) для проверки переноса (A10).

Каждая задача собирается в формате платформы: task.md, train/test без ответов, sample_submission,
data_dictionary. Скрытые ответы test лежат отдельно (data/external/_truth) и агенту не видны.
Задачи проверяют то, чего нет в публичных: вероятности, классы-строки с неочевидным положительным
классом, столбцы вероятностей по классам, отсутствие id в test, несколько таргетов, временной
сплит, неподдержанную метрику, целый таргет регрессии, маленькие данные, плоскую раскладку.

  python tests/external_tasks.py build             # скачать и собрать задачи
  python tests/external_tasks.py run [имя ...]     # запустить solution.py без LLM и оценить по скрытым ответам
  python tests/external_tasks.py run --llm [имя]   # то же с настоящей LLM (тратит общий бюджет ключа)

Данные и ответы в Git не попадают (data/ в .gitignore). Нужен интернет для первой загрузки.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
# свой каталог на эксперимент: параллельные прогоны не должны перезаписывать задачи и скрытые ответы друг друга
OUT = Path(os.environ.get("EXTERNAL_OUT", ROOT / "data" / "external"))
TRUTH = OUT / "_truth"
SEED = int(os.environ.get("EXTERNAL_SEED", 0))   # другой сид — другое разбиение train/test


def openml(name, version):
    from sklearn.datasets import fetch_openml
    return fetch_openml(name=name, version=version, as_frame=True, parser="auto").frame


def _dictionary(frame, roles):
    rows = []
    for col in frame.columns:
        kind = "numeric" if pd.api.types.is_numeric_dtype(frame[col]) else "category"
        rows.append({"feature": col, "role": roles.get(col, "feature"), "dtype": kind, "unit": "", "description_ru": ""})
    return pd.DataFrame(rows)


def write_task(name, train, test, sample, truth, text, roles, layout="nested", meta=None):
    task_dir = OUT / name
    folder = task_dir / ("data" if layout == "nested" else ".")
    folder.mkdir(parents=True, exist_ok=True)
    prefix = "data/" if layout == "nested" else ""
    train.to_csv(folder / "train.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    sample.to_csv(folder / "sample_submission.csv", index=False)
    _dictionary(train, roles).to_csv(folder / "data_dictionary.csv", index=False)
    (task_dir / "task.md").write_text(text.replace("{p}", prefix), encoding="utf-8")
    TRUTH.mkdir(parents=True, exist_ok=True)
    truth.to_csv(TRUTH / f"{name}.csv", index=False)
    (TRUTH / f"{name}.json").write_text(json.dumps(meta or {}, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: train {train.shape}, test {test.shape}, ответ {list(sample.columns)}")


def files_block(extra=""):
    return ("## Данные\n\nПути указаны относительно каталога этого файла:\n\n"
            "- `{p}train.csv` — обучающая выборка с таргетом;\n- `{p}test.csv` — признаки без таргета;\n"
            "- `{p}sample_submission.csv` — шаблон ответа, значения в нём не являются ответами;\n"
            "- `{p}data_dictionary.csv` — описание полей.\n" + extra)


def build_adult():
    df = openml("adult", 2).rename(columns={"class": "income"})
    df["income"] = df["income"].astype(str)
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["income"])
    test = test.reset_index(drop=True)
    truth = pd.DataFrame({"row_id": range(len(test)), "income": (test.pop("income") == ">50K").astype(int)})
    sample = pd.DataFrame({"row_id": range(len(test)), "income": 0.5})
    text = ("# Доход взрослого населения\n\nПо анкетным данным переписи оцените вероятность того, что годовой доход "
            "человека превышает 50 тысяч долларов (`income` = `>50K`; иначе `<=50K`).\n\n" + files_block() +
            "\nВ `test.csv` нет идентификатора: строки нумеруются с нуля в порядке файла (`row_id`).\n\n"
            "## Метрика\n\nОсновная метрика — **ROC AUC** по вероятности класса `>50K`. Больше — лучше.\n\n"
            "## Формат результата\n\n`submission.csv` рядом с task.md, столбцы `row_id,income`, где `income` — "
            "вероятность `>50K` от 0 до 1, порядок строк как в `test.csv`.\n")
    write_task("adult_income", train, test, sample, truth, text, {"income": "target"},
               meta={"metric": "roc_auc", "target": "income", "id": "row_id"})


def build_credit():
    df = openml("credit-g", 1).rename(columns={"class": "risk"})
    df["risk"] = df["risk"].astype(str)
    df.insert(0, "applicant_id", np.arange(1001, 1001 + len(df)))
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["risk"])
    test = test.reset_index(drop=True)
    truth = pd.DataFrame({"applicant_id": test["applicant_id"], "risk": (test.pop("risk") == "bad").astype(int)})
    sample = pd.DataFrame({"applicant_id": test["applicant_id"], "risk": 0.5})
    text = ("# Кредитный риск\n\nПо анкете заёмщика оцените **вероятность того, что он окажется плохим заёмщиком** "
            "(`risk` = `bad`; хорошие — `good`).\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **ROC AUC** по вероятности класса `bad`.\n\n"
            "## Формат результата\n\n`submission.csv`: `applicant_id,risk`, где `risk` — вероятность `bad`.\n")
    write_task("credit_risk", train, test, sample, truth, text, {"risk": "target", "applicant_id": "id"},
               meta={"metric": "roc_auc", "target": "risk", "id": "applicant_id"})


def build_california():
    from sklearn.datasets import fetch_california_housing
    df = fetch_california_housing(as_frame=True).frame
    df = df.rename(columns={"MedHouseVal": "median_house_value"})
    df["median_house_value"] = (df["median_house_value"] * 100000).round(0)
    df.insert(0, "Id", np.arange(len(df)))
    train, test = train_test_split(df, test_size=0.25, random_state=SEED)
    truth = test[["Id", "median_house_value"]].reset_index(drop=True)
    test = test.drop(columns="median_house_value").reset_index(drop=True)
    sample = pd.DataFrame({"Id": test["Id"], "median_house_value": 0.0})
    text = ("# Стоимость жилья в Калифорнии\n\nПредскажите медианную стоимость дома в районе "
            "(`median_house_value`, доллары) по характеристикам района.\n\n" + files_block() +
            "\n## Метрика\n\nКачество оценивается по **RMSE**, меньше — лучше.\n\n"
            "## Формат результата\n\n`submission.csv`: `Id,median_house_value`.\n")
    write_task("california_housing", train, test, sample, truth, text, {"Id": "id", "median_house_value": "target"},
               layout="flat", meta={"metric": "rmse", "target": "median_house_value", "id": "Id"})


def build_bike():
    df = openml("Bike_Sharing_Demand", 2)
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
    df.insert(0, "rental_id", np.arange(len(df)))
    year, month = pd.to_numeric(df["year"]), pd.to_numeric(df["month"])
    is_test = (year == year.max()) & (month >= 9)
    train, test = df[~is_test], df[is_test].reset_index(drop=True)
    truth = test[["rental_id", "count"]].copy()
    test = test.drop(columns="count")
    sample = pd.DataFrame({"rental_id": test["rental_id"], "count": 0})
    text = ("# Спрос на прокат велосипедов\n\nПо календарю и погоде предскажите число аренд за час (`count`). "
            "Обучающие данные — более ранний период, тестовые — последние месяцы наблюдений: "
            "это прогноз вперёд по времени.\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **RMSLE**: `sqrt(mean((log1p(y_pred) - log1p(y_true))**2))`. "
            "Меньше — лучше. Предсказания должны быть неотрицательными.\n\n"
            "## Формат результата\n\n`submission.csv`: `rental_id,count`.\n")
    write_task("bike_demand", train, test, sample, truth, text, {"rental_id": "id", "count": "target"},
               meta={"metric": "rmsle", "target": "count", "id": "rental_id"})


def build_covertype():
    df = openml("covertype", 3).rename(columns={"class": "cover_type"})
    df["cover_type"] = pd.to_numeric(df["cover_type"].astype(str)).astype(int)
    df = df.sample(60000, random_state=SEED)
    df.insert(0, "id", np.arange(len(df)))
    train, test = train_test_split(df, test_size=15000, random_state=SEED, stratify=df["cover_type"])
    test = test.reset_index(drop=True)
    truth = test[["id", "cover_type"]].copy()
    test = test.drop(columns="cover_type")
    classes = sorted(df["cover_type"].unique())
    sample = pd.DataFrame({"id": test["id"], **{f"cover_type_{c}": 1 / len(classes) for c in classes}})
    text = ("# Тип лесного покрова\n\nПо картографическим признакам участка определите тип лесного покрова "
            f"(`cover_type`, целые коды {classes}).\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **многоклассовый logloss** по вероятностям классов. Меньше — лучше.\n\n"
            "## Формат результата\n\n`submission.csv`: `id` и по столбцу вероятности на каждый класс "
            f"(`cover_type_1` … `cover_type_{classes[-1]}`); сумма вероятностей в строке равна 1.\n")
    write_task("covertype_proba", train, test, sample, truth, text, {"id": "id", "cover_type": "target"},
               meta={"metric": "logloss", "target": "cover_type", "id": "id", "classes": [int(c) for c in classes]})


def build_wine():
    df = openml("wine-quality-red", 1).rename(columns={"class": "quality"})
    df["quality"] = pd.to_numeric(df["quality"].astype(str)).astype(int)
    df.insert(0, "wine_id", np.arange(len(df)))
    train, test = train_test_split(df, test_size=0.3, random_state=SEED)
    truth = test[["wine_id", "quality"]].reset_index(drop=True)
    test = test.drop(columns="quality").reset_index(drop=True)
    sample = pd.DataFrame({"wine_id": test["wine_id"], "quality": 0.0})
    text = ("# Качество красного вина\n\nПо физико-химическим показателям оцените экспертную оценку качества "
            "(`quality`, от 0 до 10). Это задача **регрессии**: допускаются дробные предсказания.\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **MAE**, меньше — лучше.\n\n"
            "## Формат результата\n\n`submission.csv`: `wine_id,quality`.\n")
    write_task("wine_quality", train, test, sample, truth, text, {"wine_id": "id", "quality": "target"},
               meta={"metric": "mae", "target": "quality", "id": "wine_id"})


def build_energy():
    df = openml("energy-efficiency", 1)
    names = ["relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height",
             "orientation", "glazing_area", "glazing_area_distribution"]
    df = df.rename(columns={**{f"V{i + 1}": n for i, n in enumerate(names)}, "y1": "heating_load", "y2": "cooling_load"})
    df.insert(0, "building_id", np.arange(len(df)))
    train, test = train_test_split(df, test_size=0.3, random_state=SEED)
    truth = test[["building_id", "heating_load", "cooling_load"]].reset_index(drop=True)
    test = test.drop(columns=["heating_load", "cooling_load"]).reset_index(drop=True)
    sample = pd.DataFrame({"building_id": test["building_id"], "heating_load": 0.0, "cooling_load": 0.0})
    text = ("# Энергоэффективность зданий: две нагрузки\n\nПо форме здания предскажите **две** величины: "
            "нагрузку на отопление `heating_load` и на охлаждение `cooling_load`.\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **RMSE**, усреднённая по двум таргетам. Меньше — лучше.\n\n"
            "## Формат результата\n\n`submission.csv`: `building_id,heating_load,cooling_load`.\n")
    write_task("energy_two_targets", train, test, sample, truth, text,
               {"building_id": "id", "heating_load": "target", "cooling_load": "target"},
               meta={"metric": "rmse_multi", "target": ["heating_load", "cooling_load"], "id": "building_id"})


def build_diabetes():
    from sklearn.datasets import load_diabetes
    df = load_diabetes(as_frame=True, scaled=False).frame.rename(columns={"target": "progression"})
    df.insert(0, "patient_id", np.arange(100, 100 + len(df)))
    train, test = train_test_split(df, test_size=142, random_state=SEED)
    truth = test[["patient_id", "progression"]].reset_index(drop=True)
    test = test.drop(columns="progression").reset_index(drop=True)
    sample = pd.DataFrame({"patient_id": test["patient_id"], "progression": 0.0})
    text = ("# Прогрессирование диабета\n\nПо показателям пациента предскажите количественную меру прогрессирования "
            "болезни через год (`progression`).\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **MAE**, меньше — лучше.\n\n"
            "## Формат результата\n\n`submission.csv`: `patient_id,progression`.\n")
    write_task("diabetes_small", train, test, sample, truth, text, {"patient_id": "id", "progression": "target"},
               meta={"metric": "mae", "target": "progression", "id": "patient_id"})


# ---- датасеты Kaggle (через открытые зеркала: OpenML и GitHub; логин Kaggle не нужен) -------------
def build_kaggle_house():
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=42165, as_frame=True, parser="auto").frame   # Kaggle House Prices, train.csv
    train, test = train_test_split(df, test_size=0.25, random_state=SEED)
    truth = test[["Id", "SalePrice"]].reset_index(drop=True)
    test = test.drop(columns="SalePrice").reset_index(drop=True)
    sample = pd.DataFrame({"Id": test["Id"], "SalePrice": 180000.0})
    text = ("# House Prices (Kaggle)\n\nПредскажите цену продажи дома `SalePrice` по 79 характеристикам.\n\n"
            + files_block() + "\n## Метрика\n\nКачество — **RMSE между логарифмами** предсказанной и настоящей цены "
            "(ошибки на дешёвых и дорогих домах весят одинаково). Меньше — лучше.\n\n"
            "## Формат результата\n\n`submission.csv`: `Id,SalePrice`.\n")
    write_task("kaggle_house_prices", train, test, sample, truth, text, {"Id": "id", "SalePrice": "target"},
               meta={"metric": "rmse_log", "target": "SalePrice", "id": "Id"})


def build_kaggle_titanic():
    df = openml("titanic", 1).drop(columns=["boat", "body", "home.dest"])   # как в Kaggle: без утечек
    df = df.rename(columns={"pclass": "Pclass", "survived": "Survived", "name": "Name", "sex": "Sex", "age": "Age",
                            "sibsp": "SibSp", "parch": "Parch", "ticket": "Ticket", "fare": "Fare",
                            "cabin": "Cabin", "embarked": "Embarked"})
    df["Survived"] = pd.to_numeric(df["Survived"].astype(str)).astype(int)
    df.insert(0, "PassengerId", np.arange(1, len(df) + 1))
    train, test = train_test_split(df, test_size=418, random_state=SEED, stratify=df["Survived"])
    truth = test[["PassengerId", "Survived"]].reset_index(drop=True)
    test = test.drop(columns="Survived").reset_index(drop=True)
    sample = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": 0})
    text = ("# Titanic (Kaggle)\n\nОпределите, выжил ли пассажир (`Survived`: 1 — да, 0 — нет).\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **Accuracy**, больше — лучше.\n\n"
            "## Формат результата\n\n`submission.csv`: `PassengerId,Survived`, где `Survived` — 0 или 1.\n")
    write_task("kaggle_titanic", train, test, sample, truth, text, {"PassengerId": "id", "Survived": "target"},
               meta={"metric": "accuracy", "target": "Survived", "id": "PassengerId"})


def build_kaggle_telco():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url, dtype=str, keep_default_na=False)   # как в оригинале: TotalCharges с пробелами
    for col in ("tenure", "MonthlyCharges", "SeniorCitizen"):
        df[col] = pd.to_numeric(df[col])
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["Churn"])
    test = test.reset_index(drop=True)
    truth = pd.DataFrame({"customerID": test["customerID"], "Churn": (test.pop("Churn") == "Yes").astype(int)})
    sample = pd.DataFrame({"customerID": test["customerID"], "Churn": 0.5})
    text = ("# Отток клиентов телеком-оператора (Kaggle Telco Customer Churn)\n\nОцените **вероятность ухода "
            "клиента** (`Churn` = `Yes`).\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **ROC AUC** по вероятности `Yes`.\n\n"
            "## Формат результата\n\n`submission.csv`: `customerID,Churn`, `Churn` — вероятность ухода.\n")
    write_task("kaggle_telco_churn", train, test, sample, truth, text, {"customerID": "id", "Churn": "target"},
               meta={"metric": "roc_auc", "target": "Churn", "id": "customerID"})


def build_kaggle_insurance():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    df.insert(0, "policy_id", np.arange(5000, 5000 + len(df)))
    train, test = train_test_split(df, test_size=0.3, random_state=SEED)
    truth = test[["policy_id", "charges"]].reset_index(drop=True)
    test = test.drop(columns="charges").reset_index(drop=True)
    sample = pd.DataFrame({"policy_id": test["policy_id"], "charges": 0.0})
    text = ("# Медицинская страховка (Kaggle Medical Cost Personal)\n\nПредскажите годовые медицинские расходы "
            "`charges` по возрасту, ИМТ, курению и региону.\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **RMSE**, меньше — лучше.\n\n"
            "## Формат результата\n\n`submission.csv`: `policy_id,charges`.\n")
    write_task("kaggle_insurance", train, test, sample, truth, text, {"policy_id": "id", "charges": "target"},
               meta={"metric": "rmse", "target": "charges", "id": "policy_id"})


# ---- реальные «тяжёлые» данные: фрод, атаки, банковский скоринг (OpenML) ---------------------------
def build_fraud():
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=1597, as_frame=True, parser="auto").frame.rename(columns={"Class": "is_fraud"})
    df["is_fraud"] = pd.to_numeric(df["is_fraud"].astype(str)).astype(int)
    # в версии OpenML нет столбца Time, но строки идут в исходном порядке по времени (как в CSV Kaggle)
    df = df.reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(len(df)))
    cut = int(len(df) * 0.7)                                   # прогноз вперёд: test — последние 30% по времени
    train, test = df.iloc[:cut], df.iloc[cut:].reset_index(drop=True)
    truth = test[["transaction_id", "is_fraud"]].copy()
    test = test.drop(columns="is_fraud")
    sample = pd.DataFrame({"transaction_id": test["transaction_id"], "is_fraud": 0.5})
    text = ("# Мошеннические транзакции по картам\n\nОцените **вероятность того, что транзакция мошенническая** "
            "(`is_fraud` = 1). Признаки V1–V28 — анонимизированные компоненты, `Amount` — сумма. "
            "Строки упорядочены по времени; test — более поздние транзакции, чем train.\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **Average Precision (PR-AUC)** по вероятности мошенничества: "
            "мошенничество редкое (около 0.2%), поэтому ROC AUC не используется.\n\n"
            "## Формат результата\n\n`submission.csv`: `transaction_id,is_fraud`, где `is_fraud` — вероятность от 0 до 1.\n")
    write_task("real_card_fraud", train, test, sample, truth, text, {"transaction_id": "id", "is_fraud": "target"},
               meta={"metric": "average_precision", "target": "is_fraud", "id": "transaction_id"})


def build_kdd():
    df = openml("KDDCup99", 1).rename(columns={"label": "attack_type"})
    df["attack_type"] = df["attack_type"].astype(str).str.rstrip(".")
    df = df.sample(200000, random_state=SEED).reset_index(drop=True)
    df.insert(0, "connection_id", np.arange(len(df)))
    train, test = train_test_split(df, test_size=50000, random_state=SEED)
    test = test.reset_index(drop=True)
    truth = test[["connection_id", "attack_type"]].copy()
    test = test.drop(columns="attack_type")
    sample = pd.DataFrame({"connection_id": test["connection_id"], "attack_type": "normal"})
    text = ("# Типы сетевых атак (KDD Cup 99)\n\nПо параметрам сетевого соединения определите тип атаки "
            "(`attack_type`, `normal` — нет атаки). Классы сильно несбалансированы: есть типы из нескольких строк.\n\n"
            + files_block() + "\n## Метрика\n\nОсновная метрика — **Macro F1** по всем типам атак.\n\n"
            "## Формат результата\n\n`submission.csv`: `connection_id,attack_type` — строковая метка типа.\n")
    write_task("real_kdd_attacks", train, test, sample, truth, text, {"connection_id": "id", "attack_type": "target"},
               meta={"metric": "f1_macro", "target": "attack_type", "id": "connection_id"})


def build_bank():
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=1461, as_frame=True, parser="auto").frame
    names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
             "month", "duration", "campaign", "pdays", "previous", "poutcome"]
    df = df.rename(columns={f"V{i + 1}": n for i, n in enumerate(names)}).rename(columns={"Class": "subscribed"})
    df["subscribed"] = df["subscribed"].astype(str).map({"1": "no", "2": "yes"})
    df = df.drop(columns="duration")                           # длительность звонка известна только после звонка
    df.insert(0, "client_id", np.arange(len(df)))
    cut = int(len(df) * 0.75)                                  # строки упорядочены по дате кампании
    train, test = df.iloc[:cut], df.iloc[cut:].reset_index(drop=True)
    truth = pd.DataFrame({"client_id": test["client_id"], "subscribed": (test.pop("subscribed") == "yes").astype(int)})
    sample = pd.DataFrame({"client_id": test["client_id"], "subscribed": 0.5})
    text = ("# Отклик на банковский депозит\n\nПо данным клиента и прошлых кампаний оцените **вероятность, что клиент "
            "откроет депозит** (`subscribed` = `yes`). Строки упорядочены по времени: test — более поздние звонки.\n\n"
            + files_block() + "\n## Метрика\n\nОсновная метрика — **ROC AUC** по вероятности `yes`.\n\n"
            "## Формат результата\n\n`submission.csv`: `client_id,subscribed` — вероятность от 0 до 1.\n")
    write_task("real_bank_marketing", train, test, sample, truth, text, {"client_id": "id", "subscribed": "target"},
               meta={"metric": "roc_auc", "target": "subscribed", "id": "client_id"})


def build_default():
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=42477, as_frame=True, parser="auto").frame
    names = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
             "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
             "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    df = df.rename(columns={f"x{i + 1}": n for i, n in enumerate(names)}).rename(columns={"y": "default_next_month"})
    df["default_next_month"] = pd.to_numeric(df["default_next_month"].astype(str)).astype(int)
    df.insert(0, "ID", np.arange(1, len(df) + 1))
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["default_next_month"])
    test = test.reset_index(drop=True)
    truth = test[["ID", "default_next_month"]].copy()
    test = test.drop(columns="default_next_month")
    sample = pd.DataFrame({"ID": test["ID"], "default_next_month": 0})
    text = ("# Дефолт по кредитной карте\n\nПредскажите, допустит ли клиент дефолт в следующем месяце "
            "(`default_next_month`: 1 — да, 0 — нет) по лимиту, истории платежей и счетов.\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **F1-score по классу 1** (дефолт). Больше — лучше.\n\n"
            "## Формат результата\n\n`submission.csv`: `ID,default_next_month` — метка 0 или 1.\n")
    write_task("real_credit_default", train, test, sample, truth, text, {"ID": "id", "default_next_month": "target"},
               meta={"metric": "f1_binary", "target": "default_next_month", "id": "ID"})


def build_amazon():
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=4135, as_frame=True, parser="auto").frame.rename(columns={"target": "ACTION"})
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str))           # как на Kaggle: коды категорий записаны числами
    df.insert(0, "id", np.arange(1, len(df) + 1))
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["ACTION"])
    test = test.reset_index(drop=True)
    truth = test[["id", "ACTION"]].copy()
    test = test.drop(columns="ACTION")
    sample = pd.DataFrame({"id": test["id"], "ACTION": 0.5})
    text = ("# Доступ сотрудников к ресурсам (Amazon)\n\nПо ресурсу и роли сотрудника оцените **вероятность одобрения "
            "доступа** (`ACTION` = 1). Все признаки — коды категорий (менеджер, отдел, должность, ресурс).\n\n"
            + files_block() + "\n## Метрика\n\nОсновная метрика — **ROC AUC** по вероятности `ACTION` = 1.\n\n"
            "## Формат результата\n\n`submission.csv`: `id,ACTION` — вероятность от 0 до 1.\n")
    write_task("real_amazon_access", train, test, sample, truth, text, {"id": "id", "ACTION": "target"},
               meta={"metric": "roc_auc", "target": "ACTION", "id": "id"})


def build_porto():
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=42742, as_frame=True, parser="auto").frame
    df["target"] = pd.to_numeric(df["target"].astype(str)).astype(int)
    df = df.sample(200000, random_state=SEED).reset_index(drop=True)
    if "id" not in df.columns:
        df.insert(0, "id", np.arange(len(df)))
    df = df.replace({-1: np.nan})                               # в Porto Seguro -1 означает пропуск
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["target"])
    test = test.reset_index(drop=True)
    truth = test[["id", "target"]].copy()
    test = test.drop(columns="target")
    sample = pd.DataFrame({"id": test["id"], "target": 0.5})
    text = ("# Страховые случаи водителей (Porto Seguro)\n\nОцените **вероятность, что водитель подаст страховое "
            "требование в следующем году** (`target` = 1). Признаки анонимизированы: группы ind/reg/car/calc, суффиксы "
            "`_cat` — категории, `_bin` — бинарные.\n\n" + files_block() +
            "\n## Метрика\n\nОсновная метрика — **нормированный Gini** = 2·AUC − 1 по вероятности.\n\n"
            "## Формат результата\n\n`submission.csv`: `id,target` — вероятность от 0 до 1.\n")
    write_task("real_porto_seguro", train, test, sample, truth, text, {"id": "id", "target": "target"},
               meta={"metric": "gini", "target": "target", "id": "id"})


BUILDERS = {"adult_income": build_adult, "credit_risk": build_credit, "california_housing": build_california,
            "bike_demand": build_bike, "covertype_proba": build_covertype, "wine_quality": build_wine,
            "energy_two_targets": build_energy, "diabetes_small": build_diabetes,
            "kaggle_house_prices": build_kaggle_house, "kaggle_titanic": build_kaggle_titanic,
            "kaggle_telco_churn": build_kaggle_telco, "kaggle_insurance": build_kaggle_insurance,
            "real_card_fraud": build_fraud, "real_kdd_attacks": build_kdd, "real_bank_marketing": build_bank,
            "real_credit_default": build_default, "real_amazon_access": build_amazon, "real_porto_seguro": build_porto}


# ---- оценка ------------------------------------------------------------------
def score(name):
    """(метрика, значение, константный ориентир, проблемы формата)."""
    task_dir = OUT / name
    meta = json.loads((TRUTH / f"{name}.json").read_text(encoding="utf-8"))
    truth = pd.read_csv(TRUTH / f"{name}.csv")
    sample = pd.read_csv(next(task_dir.rglob("sample_submission.csv")))
    path = task_dir / "submission.csv"
    if not path.exists():
        return meta["metric"], None, None, ["нет submission.csv"]
    sub = pd.read_csv(path)
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append(f"столбцы {list(sub.columns)} вместо {list(sample.columns)}")
    if len(sub) != len(truth):
        return meta["metric"], None, None, problems + [f"строк {len(sub)} вместо {len(truth)}"]   # ревью Юры, F19
    id_col = meta["id"]
    if id_col in sub.columns and not (sub[id_col].to_numpy() == truth[id_col].to_numpy()).all():
        problems.append("порядок или значения id не совпадают")
    if sub.isna().any().any():
        problems.append("есть пропуски")
    if problems:
        return meta["metric"], None, None, problems
    metric, target = meta["metric"], meta["target"]
    try:
        if metric in ("roc_auc", "gini"):
            value = skm.roc_auc_score(truth[target], sub[target].astype(float))
            const = 0.5
            if metric == "gini":
                value, const = 2 * value - 1, 0.0
        elif metric == "average_precision":
            value = skm.average_precision_score(truth[target], sub[target].astype(float))
            const = float(truth[target].mean())
        elif metric == "f1_binary":
            value = skm.f1_score(truth[target].astype(int), sub[target].astype(int))
            const = skm.f1_score(truth[target].astype(int), np.ones(len(truth), dtype=int))
        elif metric == "f1_macro":
            value = skm.f1_score(truth[target].astype(str), sub[target].astype(str), average="macro")
            const = skm.f1_score(truth[target].astype(str), [truth[target].mode()[0]] * len(truth), average="macro")
        elif metric == "logloss":
            cols = [f"{target}_{c}" for c in meta["classes"]]
            proba = sub[cols].to_numpy(dtype=float)
            value = skm.log_loss(truth[target], proba, labels=meta["classes"])
            prior = truth[target].value_counts(normalize=True).reindex(meta["classes"]).to_numpy()
            const = skm.log_loss(truth[target], np.tile(prior, (len(truth), 1)), labels=meta["classes"])
        elif metric == "accuracy":
            value = skm.accuracy_score(truth[target].astype(str), sub[target].astype(str))
            const = float(truth[target].value_counts(normalize=True).max())
        elif metric == "rmse_log":
            y, p = truth[target].to_numpy(dtype=float), sub[target].to_numpy(dtype=float)
            value = float(np.sqrt(np.mean((np.log(np.clip(p, 1, None)) - np.log(y)) ** 2)))
            const = float(np.sqrt(np.mean((np.log(np.median(y)) - np.log(y)) ** 2)))
        elif metric == "rmse_multi":
            value = float(np.mean([np.sqrt(np.mean((truth[t] - sub[t]) ** 2)) for t in target]))
            const = float(np.mean([np.sqrt(np.mean((truth[t] - truth[t].mean()) ** 2)) for t in target]))
        else:
            y, p = truth[target].to_numpy(dtype=float), sub[target].to_numpy(dtype=float)
            fn = {"rmse": lambda a, b: np.sqrt(np.mean((a - b) ** 2)), "mae": lambda a, b: np.mean(np.abs(a - b)),
                  "rmsle": lambda a, b: np.sqrt(np.mean((np.log1p(np.clip(b, 0, None)) - np.log1p(a)) ** 2))}[metric]
            value, const = fn(y, p), fn(y, np.full_like(y, np.median(y)))
    except Exception as error:
        return metric, None, None, [f"метрика не считается: {type(error).__name__}: {error}"]
    return metric, float(value), float(const), []


def run(names, time_limit):
    rows = []
    for name in names:
        task_dir = OUT / name
        (task_dir / "submission.csv").unlink(missing_ok=True)
        env = dict(os.environ, AGENT_TIME_LIMIT=str(time_limit), AGENT_SAFETY_S=str(min(300, time_limit // 4)),
                   AGENT_LLM="on" if USE_LLM else "off")
        started = time.time()
        with open(task_dir / "run.log", "w", encoding="utf-8") as log:
            code = subprocess.run([sys.executable, "-W", "ignore", str(SOLUTION), "task.md"],
                                  cwd=task_dir, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        seconds = time.time() - started
        metric, value, const, problems = score(name)
        rows.append({"task": name, "exit": code, "seconds": round(seconds), "metric": metric,
                     "score": None if value is None else round(value, 4),
                     "constant": None if const is None else round(const, 4), "problems": "; ".join(problems)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return pd.DataFrame(rows)


USE_LLM = "--llm" in sys.argv
# замороженная копия кода для долгих сравнений: правки в репозитории не попадают в идущий эксперимент
SOLUTION = Path(os.environ.get("EXTERNAL_SOLUTION", ROOT / "solution.py"))   # по умолчанию без LLM: прогоны не тратят общий бюджет ключа

if __name__ == "__main__":
    argv = [a for a in sys.argv if a != "--llm"]
    command = argv[1] if len(argv) > 1 else "run"
    selected = argv[2:] or list(BUILDERS)
    if command == "build":
        for name in selected:
            BUILDERS[name]()
    else:
        table = run(selected, int(os.environ.get("EXTERNAL_TIME_LIMIT", 1200)))
        print(table.to_string(index=False))
