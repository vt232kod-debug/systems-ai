# -*- coding: utf-8 -*-
"""
Лабораторна робота №6. Наївний класифікатор Байєса.
Завдання 4: байєсівський аналіз набору даних про ціни на квитки
іспанських високошвидкісних залізниць Renfe (renfe_small.csv).

Постановка задачі (методичка її не дає, формулюємо самостійно):
  передбачити ЦІНОВУ КАТЕГОРІЮ квитка (Дешевий / Середній / Дорогий —
  три рівнопотужні квантильні групи) за характеристиками рейсу:
  станція відправлення, станція призначення, тип потяга, клас вагона,
  тариф, тривалість поїздки, година відправлення, день тижня відправлення
  та глибина бронювання.

Виконав: Камінський Олексій Дмитрович, група ВТ-23-2, варіант 3.
"""

from pathlib import Path              # шляхи, незалежні від поточної робочої теки

import matplotlib                      # бібліотека побудови графіків
matplotlib.use("Agg")                  # неінтерактивний бекенд: пишемо у файл
import matplotlib.pyplot as plt        # інтерфейс малювання
import numpy as np                     # числові масиви
import pandas as pd                    # таблиці
from sklearn.model_selection import train_test_split   # поділ на train/test
from sklearn.naive_bayes import GaussianNB, CategoricalNB   # дві версії наївного Байєса
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer  # кодування і дискретизація
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)          # метрики якості

BASE = Path(__file__).resolve().parent  # тека скрипта: і CSV, і рисунки лежать саме тут
RANDOM_STATE = 3                       # зерно генератора = номер варіанта (відтворюваність)
pd.set_option("display.width", 140)    # ширина друку таблиць

# ---------------------------------------------------------------------------
# Крок 1. Завантаження сирих даних.
# ---------------------------------------------------------------------------
print("=" * 78)
print("ЗАВДАННЯ 4. Байєсівський аналіз набору renfe_small.csv")
print("=" * 78)

df = pd.read_csv(BASE / "renfe_small.csv")              # читаємо CSV у DataFrame
print(f"Розмір сирого набору: {df.shape[0]} рядків, {df.shape[1]} стовпців")
print("\nПерші 3 рядки:")
print(df.head(3).to_string(index=False))
print("\nПропуски по стовпцях:")
print(df.isna().sum().to_string())

# ---------------------------------------------------------------------------
# Крок 2. Обробка пропусків.
#   price        — 3082 пропуски; це ЦІЛЬОВА величина, відновити її нічим,
#                  тому такі рядки вилучаємо (імпутація створила б штучний клас).
#   train_class  — 103 пропуски, fare — 103 пропуски. ВАЖЛИВО: нижче програмно
#                  показано, що ці 103 рядки є ПІДМНОЖИНОЮ рядків без ціни,
#                  тобто вони зникають разом із ними. Отже fillna("Unknown") —
#                  це лише ЗАХИСНИЙ код для інших зрізів даних; на цьому наборі
#                  він не заповнює жодної комірки, і мітка "Unknown" у даних
#                  не з'являється. Друкуємо фактичну кількість заповнень.
# ---------------------------------------------------------------------------
n_before = len(df)                                      # запам'ятовуємо початковий розмір
n_price_na = int(df["price"].isna().sum())              # скільки рядків без ціни
n_class_na = int(df["train_class"].isna().sum())        # скільки рядків без класу вагона
n_fare_na = int(df["fare"].isna().sum())                # скільки рядків без тарифу
# перетин: скільки з пропусків категорій припадає саме на рядки без ціни
both_class = int((df["train_class"].isna() & df["price"].isna()).sum())
both_fare = int((df["fare"].isna() & df["price"].isna()).sum())
print(f"\nПропуски train_class: {n_class_na}, з них price теж NaN: {both_class}")
print(f"Пропуски fare: {n_fare_na}, з них price теж NaN: {both_fare}")

df = df.dropna(subset=["price"]).copy()                 # прибираємо рядки без цільової змінної
# рахуємо ПІСЛЯ видалення: саме стільки комірок реально отримає мітку "Unknown"
n_fill_class = int(df["train_class"].isna().sum())      # залишок пропусків класу вагона
n_fill_fare = int(df["fare"].isna().sum())              # залишок пропусків тарифу
df["train_class"] = df["train_class"].fillna("Unknown")  # захисне заповнення класу вагона
df["fare"] = df["fare"].fillna("Unknown")                # захисне заповнення тарифу
print(f"\nВилучено {n_price_na} рядків без ціни ({n_price_na / n_before * 100:.1f} %); "
      f"залишилось {len(df)}")
print(f'Мітку "Unknown" фактично проставлено: train_class {n_fill_class}, '
      f"fare {n_fill_fare} (тобто захисний fillna не спрацював)")
print(f"Пропусків після обробки: {int(df.isna().sum().sum())}")
print("Значення train_class після обробки:")
print(df["train_class"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Крок 3. Інженерія ознак із дат.
# ---------------------------------------------------------------------------
for col in ["insert_date", "start_date", "end_date"]:   # три стовпці з датами
    df[col] = pd.to_datetime(df[col])                   # перетворюємо текст у datetime

# тривалість поїздки у хвилинах
df["duration_min"] = (df["end_date"] - df["start_date"]).dt.total_seconds() / 60
# за скільки діб до рейсу зафіксовано ціну
df["days_ahead"] = (df["start_date"] - df["insert_date"]).dt.total_seconds() / 86400
df["dep_hour"] = df["start_date"].dt.hour               # година відправлення (0..23)
df["dep_weekday"] = df["start_date"].dt.dayofweek       # день тижня (0 = понеділок)

print("\nСтатистика похідних числових ознак:")
print(df[["duration_min", "days_ahead", "dep_hour", "price"]].describe().round(2).to_string())

# ---------------------------------------------------------------------------
# Крок 4. Поділ на навчальну і тестову вибірки — ПЕРЕД будь-якою підгонкою.
# Так межі квантилів цільової змінної і межі кошиків числових ознак не «бачать»
# тестових рядків (немає витоку інформації з тесту). Через це стратифікація за
# класом неможлива: класів ще не існує. На 22716 рядках випадковий поділ і так
# дає практично рівний баланс — нижче це друкується для контролю.
# ---------------------------------------------------------------------------
idx = np.arange(len(df))                                # індекси рядків
idx_tr, idx_te = train_test_split(idx, test_size=0.25,
                                  random_state=RANDOM_STATE)   # 75 % / 25 %
print(f"\nНавчальна вибірка: {len(idx_tr)} | Тестова вибірка: {len(idx_te)}")

# ---------------------------------------------------------------------------
# Крок 5. Формування цільової змінної — три квантильні цінові категорії.
# Квантильний поділ дає приблизно однакову потужність класів, тож accuracy
# випадкового вгадування дорівнює 1/3 і метрики легко інтерпретувати.
# Межі рахуємо ЛИШЕ за навчальною вибіркою і застосовуємо їх до всього набору.
# ---------------------------------------------------------------------------
LABELS = ["Дешевий", "Середній", "Дорогий"]             # назви цінових категорій
price = df["price"].to_numpy(dtype=float)               # вектор цін
_, bins = pd.qcut(price[idx_tr], q=3, labels=LABELS, retbins=True)  # межі за train
edges = bins.astype(float).copy()                       # копія меж під pd.cut
edges[0], edges[-1] = -np.inf, np.inf                   # крайні межі розкриваємо до ±∞
# застосовуємо train-межі до ВСІХ рядків (train і test однаково)
df["price_cat"] = pd.cut(price, bins=edges, labels=LABELS)
y = df["price_cat"].cat.codes.to_numpy()                # цільовий вектор: 0,1,2
print(f"\nМежі квантильних цінових категорій за train (євро): "
      + " | ".join(f"{b:.2f}" for b in bins))
print("Розподіл класів (усі рядки):")
print(df["price_cat"].value_counts().reindex(LABELS).to_string())
print("Розподіл класів у train / test:")
for name, part in (("train", idx_tr), ("test", idx_te)):
    cnt = np.bincount(y[part], minlength=3)             # кількість об'єктів кожного класу
    print(f"  {name}: " + ", ".join(f"{LABELS[j]} {cnt[j]}" for j in range(3)))

# ---------------------------------------------------------------------------
# Крок 6. Кодування ознак.
# OrdinalEncoder підганяємо на всьому наборі СВІДОМО: він не обчислює жодної
# статистики, а лише складає словник категорій «назва -> номер». Якби його
# навчити тільки на train, категорія, що трапляється лише в тесті, отримала б
# код -1, а CategoricalNB вимагає невід'ємних кодів і впав би з помилкою.
# KBinsDiscretizer, навпаки, обчислює квантилі, тому навчається лише на train.
# ---------------------------------------------------------------------------
CAT_FEATURES = ["origin", "destination", "train_type", "train_class", "fare"]  # категоріальні
NUM_FEATURES = ["duration_min", "days_ahead", "dep_hour", "dep_weekday"]       # числові

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)     # кодувальник категорій
X_cat = enc.fit_transform(df[CAT_FEATURES])             # категорії -> цілі числа 0..k-1
X_num = df[NUM_FEATURES].to_numpy(dtype=float)          # числові ознаки як масив

print("\nКількість унікальних значень категоріальних ознак:")
for name, cats in zip(CAT_FEATURES, enc.categories_):   # показуємо потужність кожної ознаки
    print(f"  {name}: {len(cats)}")

# Дискретизація числових ознак у 5 квантильних кошиків — щоб подати їх
# у CategoricalNB разом із «справжніми» категоріями. Межі кошиків — лише за train.
kbins = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile",
                         quantile_method="averaged_inverted_cdf")   # дискретизатор
kbins.fit(X_num[idx_tr])                                # підгонка ЛИШЕ на навчальній вибірці
X_num_binned = kbins.transform(X_num)                   # числові -> номери кошиків 0..4
X_all_cat = np.hstack([X_cat, X_num_binned])            # повна категоріальна матриця ознак

# ---------------------------------------------------------------------------
# Крок 7. Навчання трьох моделей наївного Байєса.
# ---------------------------------------------------------------------------
models = {}                                             # сюди складемо навчені моделі
results = {}                                            # сюди — прогнози і точність

# (1) GaussianNB — лише числові ознаки (неперервні величини, нормальний розподіл)
gnb = GaussianNB()                                      # гаусів наївний Байєс
gnb.fit(X_num[idx_tr], y[idx_tr])                       # навчання
pred_gnb = gnb.predict(X_num[idx_te])                   # прогноз на тесті
models["GaussianNB (числові)"] = gnb                    # зберігаємо модель
results["GaussianNB (числові)"] = pred_gnb              # зберігаємо прогноз

# (2) CategoricalNB — лише категоріальні ознаки, згладжування Лапласа alpha=1
cnb = CategoricalNB(alpha=1.0, force_alpha=True)        # категоріальний наївний Байєс
cnb.fit(X_cat[idx_tr], y[idx_tr])                       # навчання
pred_cnb = cnb.predict(X_cat[idx_te])                   # прогноз
models["CategoricalNB (категоріальні)"] = cnb           # зберігаємо
results["CategoricalNB (категоріальні)"] = pred_cnb     # зберігаємо

# (3) CategoricalNB — категоріальні + дискретизовані числові (повний набір ознак)
cnb_all = CategoricalNB(alpha=1.0, force_alpha=True)    # та сама модель на ширшому наборі
cnb_all.fit(X_all_cat[idx_tr], y[idx_tr])               # навчання
pred_all = cnb_all.predict(X_all_cat[idx_te])           # прогноз
models["CategoricalNB (усі ознаки)"] = cnb_all          # зберігаємо
results["CategoricalNB (усі ознаки)"] = pred_all        # зберігаємо

print("\n" + "=" * 78)
print("РЕЗУЛЬТАТИ НА ТЕСТОВІЙ ВИБІРЦІ")
print("=" * 78)
acc = {}                                                # словник точностей
for name, pred in results.items():                      # по кожній моделі
    acc[name] = accuracy_score(y[idx_te], pred)         # обчислюємо accuracy
    print(f"\n--- {name} ---")
    print(f"Accuracy = {acc[name]:.4f}")
    print(classification_report(y[idx_te], pred, target_names=LABELS, digits=3,
                                zero_division=0))       # повний звіт по класах

best_name = max(acc, key=acc.get)                       # найкраща модель за accuracy
print(f"Базовий рівень (випадкове вгадування 3 класів) = {1 / 3:.4f}")
print(f"Найкраща модель: {best_name} (accuracy = {acc[best_name]:.4f})")

# ---------------------------------------------------------------------------
# Крок 8. Матриці плутанини.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))         # три панелі — по одній на модель
for ax, (name, pred) in zip(axes, results.items()):     # перебираємо моделі
    cm = confusion_matrix(y[idx_te], pred)              # матриця плутанини
    im = ax.imshow(cm, cmap="Blues")                    # теплова карта
    ax.set_xticks(range(3), LABELS, rotation=20)        # підписи прогнозованих класів
    ax.set_yticks(range(3), LABELS)                     # підписи істинних класів
    ax.set_xlabel("Прогноз")                            # підпис осі X
    ax.set_ylabel("Істина")                             # підпис осі Y
    ax.set_title(f"{name}\naccuracy = {acc[name]:.3f}", fontsize=10)  # заголовок
    for i in range(3):                                  # пишемо числа в комірках
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)             # шкала кольору
    print(f"\nМатриця плутанини — {name}:")             # друкуємо її ж у консоль
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())
fig.suptitle("ЛР6, завдання 4. Матриці плутанини для прогнозу цінової категорії Renfe",
             fontweight="bold")
fig.tight_layout()                                      # вирівнювання
fig.savefig(BASE / "fig2_renfe_confusion.png", dpi=150)  # зберігаємо
plt.close(fig)                                          # закриваємо фігуру
print("\nЗбережено рисунок: fig2_renfe_confusion.png")

# ---------------------------------------------------------------------------
# Крок 9. Розвідувальні графіки: розподіл ціни і середня ціна по групах.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))       # три панелі

ax = axes[0]                                            # панель 1 — гістограма ціни
ax.hist(df["price"], bins=60, color="#4c72b0", edgecolor="white")   # розподіл цін
for b in bins[1:-1]:                                    # внутрішні межі квантилів
    ax.axvline(b, color="#c44e52", linestyle="--", linewidth=2,
               label=f"межа {b:.1f} €")
ax.set_xlabel("ціна квитка, €")                         # підпис осі
ax.set_ylabel("кількість квитків")                      # підпис осі
ax.set_title("Розподіл цін і межі трьох категорій")     # заголовок
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)  # легенда у вільному куті
ax.grid(alpha=0.3)                                      # сітка

ax = axes[1]                                            # панель 2 — ціна за типом потяга
top_types = df["train_type"].value_counts().head(8).index           # 8 найчастіших типів
mean_by_type = df[df["train_type"].isin(top_types)].groupby("train_type")["price"].mean()
mean_by_type = mean_by_type.sort_values()               # сортуємо за зростанням
ax.barh(mean_by_type.index, mean_by_type.values, color="#55a868")   # горизонтальні стовпчики
ax.set_xlabel("середня ціна, €")                        # підпис осі
ax.set_title("Середня ціна за типом потяга")            # заголовок
ax.grid(axis="x", alpha=0.3)                            # сітка

ax = axes[2]                                            # панель 3 — частка категорій за класом
MIN_COUNT = 100                                         # поріг: менші класи статистично пусті
cls_counts = df["train_class"].value_counts()           # кількість квитків кожного класу вагона
keep = cls_counts[cls_counts >= MIN_COUNT].head(5).index            # відбір класів для графіка
print(f"\nКласи вагона на рисунку 3 (поріг n >= {MIN_COUNT}): "
      + ", ".join(f"{k} n={cls_counts[k]}" for k in keep))
print("Відкинуто як статистично незначущі: "
      + ", ".join(f"{k} n={v}" for k, v in cls_counts.items() if v < MIN_COUNT))
share = pd.crosstab(df["train_class"], df["price_cat"], normalize="index")[LABELS]  # частки
share = share.loc[keep]                                 # лишаємо тільки відібрані класи
bottom = np.zeros(len(share))                           # накопичувач для стовпчиків
for lab, color in zip(LABELS, ["#8c9ec4", "#dd8452", "#c44e52"]):    # три шари
    ax.bar(share.index, share[lab], bottom=bottom, label=lab, color=color)
    bottom += share[lab].to_numpy()                     # зсуваємо основу наступного шару
for xpos, name in enumerate(share.index):               # підписуємо обсяг вибірки
    ax.text(xpos, 1.01, f"n = {cls_counts[name]}", ha="center", fontsize=8)
ax.set_ylim(0, 1.0)                                     # частки не виходять за 0..1
ax.set_ylabel("частка")                                 # підпис осі
ax.set_title("Структура цінових категорій за класом вагона", pad=30)  # місце під легенду
ax.tick_params(axis="x", rotation=20)                   # повертаємо підписи
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.055),
          ncol=3, frameon=False, fontsize=8)            # легенда ПОЗА полем даних
fig.suptitle("ЛР6, завдання 4. Розвідувальний аналіз набору Renfe", fontweight="bold")
fig.tight_layout()                                      # вирівнювання
fig.savefig(BASE / "fig3_renfe_eda.png", dpi=150)       # зберігаємо
plt.close(fig)                                          # закриваємо
print("Збережено рисунок: fig3_renfe_eda.png")

# Числові підсумки розвідувального аналізу (щоб цифри у звіті мали джерело)
print("\nКрайні значення середньої ціни за типом потяга:")
print(f"  {mean_by_type.index[0]} = {mean_by_type.iloc[0]:.2f} EUR | "
      f"{mean_by_type.index[-1]} = {mean_by_type.iloc[-1]:.2f} EUR")
print("Частка дешевих квитків за класом вагона, %:")
print((share["Дешевий"] * 100).round(1).to_string())

# ---------------------------------------------------------------------------
# Крок 10. Демонстрація прогнозу на кількох конкретних рейсах.
# Рядки короткі (два рядки на приклад), щоб моноширинний блок не переносився у звіті.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("ПРИКЛАДИ ПРОГНОЗУ НАЙКРАЩОЇ МОДЕЛІ")
print("=" * 78)
print("P: апостеріорні ймовірності у порядку Дешевий / Середній / Дорогий")
best_model = models[best_name]                          # беремо найкращу модель
X_best = X_all_cat if "усі" in best_name else (X_cat if "категоріальні" in best_name else X_num)
sample = idx_te[:5]                                     # перші 5 об'єктів тесту
proba = best_model.predict_proba(X_best[sample])        # апостеріорні ймовірності класів
pred_sample = best_model.predict(X_best[sample])        # прогнозовані класи
for k, i in enumerate(sample):                          # друкуємо по кожному прикладу
    row = df.iloc[i]                                   # вихідний рядок даних
    pr = "/".join(f"{proba[k][j]:.2f}" for j in range(3))   # трійка ймовірностей
    print(f"{k + 1}) {row['origin']} -> {row['destination']} | {row['train_type']} | "
          f"ціна = {row['price']:.2f} € | істина = {row['price_cat']}")
    print(f"   {row['train_class']} | {row['fare']} | "
          f"прогноз = {LABELS[pred_sample[k]]} | P: {pr}")
