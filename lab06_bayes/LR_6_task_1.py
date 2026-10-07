# -*- coding: utf-8 -*-
"""
Лабораторна робота №6. Наївний класифікатор Байєса.
Завдання 2-3: набір даних Play Tennis, частотні таблиці, таблиці правдоподібності
та прогноз для ВАРІАНТА 3: Outlook = Sunny, Humidity = High, Wind = Weak.

Виконав: Камінський Олексій Дмитрович, група ВТ-23-2, варіант 3.
"""

from pathlib import Path            # робота зі шляхами незалежно від поточної теки

import matplotlib                      # бібліотека побудови графіків
matplotlib.use("Agg")                  # неінтерактивний бекенд: малюємо у файл, без вікна
import matplotlib.pyplot as plt        # інтерфейс для малювання рисунків
import numpy as np                     # числові масиви
import pandas as pd                    # таблиці (DataFrame)
from sklearn.naive_bayes import CategoricalNB   # наївний Байєс для категоріальних ознак
from sklearn.preprocessing import OrdinalEncoder  # кодування текстових категорій у цілі числа

BASE = Path(__file__).resolve().parent  # тека самого скрипта: рисунки пишемо саме сюди
pd.set_option("display.width", 120)    # ширина друку таблиць у консолі

# ---------------------------------------------------------------------------
# Крок 0. Набір даних Play Tennis (14 рядків).
# У методичці таблиця подана КАРТИНКОЮ і стовпця Temperature у ній немає —
# наведено лише Day, Outlook, Humidity, Wind, Play. Нижче дані повністю
# збігаються з картинкою методички, а стовпець Temperature доданий із
# класичного набору Mitchell'а для повноти частотних таблиць (на прогноз
# для нашого варіанта він не впливає, бо в умові варіанта не згадується).
# ---------------------------------------------------------------------------
data = [
    ("D1",  "Sunny",    "Hot",  "High",   "Weak",   "No"),   # рядок 1 картинки
    ("D2",  "Sunny",    "Hot",  "High",   "Strong", "No"),   # рядок 2
    ("D3",  "Overcast", "Hot",  "High",   "Weak",   "Yes"),  # рядок 3
    ("D4",  "Rain",     "Mild", "High",   "Weak",   "Yes"),  # рядок 4
    ("D5",  "Rain",     "Cool", "Normal", "Weak",   "Yes"),  # рядок 5
    ("D6",  "Rain",     "Cool", "Normal", "Strong", "No"),   # рядок 6
    ("D7",  "Overcast", "Cool", "Normal", "Strong", "Yes"),  # рядок 7
    ("D8",  "Sunny",    "Mild", "High",   "Weak",   "No"),   # рядок 8
    ("D9",  "Sunny",    "Cool", "Normal", "Weak",   "Yes"),  # рядок 9
    ("D10", "Rain",     "Mild", "Normal", "Weak",   "Yes"),  # рядок 10
    ("D11", "Sunny",    "Mild", "Normal", "Strong", "Yes"),  # рядок 11
    ("D12", "Overcast", "Mild", "High",   "Strong", "Yes"),  # рядок 12
    ("D13", "Overcast", "Hot",  "Normal", "Weak",   "Yes"),  # рядок 13
    ("D14", "Rain",     "Mild", "High",   "Strong", "No"),   # рядок 14
]
df = pd.DataFrame(data, columns=["Day", "Outlook", "Temperature", "Humidity", "Wind", "Play"])

print("=" * 78)
print("ЗАВДАННЯ 2. Набір даних Play Tennis")
print("=" * 78)
print(df.to_string(index=False))                   # друкуємо весь набір даних
print(f"\nУсього рядків: {len(df)}")               # контроль: має бути 14
print("Розподіл цільової змінної Play:")
print(df["Play"].value_counts().to_string())       # скільки Yes і скільки No

FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]   # список ознак
TARGET = "Play"                                             # цільова змінна
CLASSES = ["Yes", "No"]                                     # порядок класів у таблицях

n_total = len(df)                                           # 14 спостережень
n_yes = int((df[TARGET] == "Yes").sum())                    # кількість Yes = 9
n_no = int((df[TARGET] == "No").sum())                      # кількість No  = 5
p_yes = n_yes / n_total                                     # апріорна P(Yes)
p_no = n_no / n_total                                       # апріорна P(No)

# ---------------------------------------------------------------------------
# Крок 1. Частотні таблиці (frequency table) — ПРОГРАМНО, по кожному атрибуту.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("КРОК 1. ЧАСТОТНІ ТАБЛИЦІ (frequency tables)")
print("=" * 78)

freq_tables = {}                                            # сюди складемо всі частотні таблиці
for col in FEATURES:                                        # по кожному атрибуту окремо
    tab = pd.crosstab(df[col], df[TARGET])                  # перехресна таблиця «значення × клас»
    tab = tab.reindex(columns=CLASSES, fill_value=0)        # фіксуємо порядок стовпців Yes, No
    tab["Разом"] = tab.sum(axis=1)                          # сума по рядку
    tab.loc["Разом"] = tab.sum(axis=0)                      # підсумковий рядок
    freq_tables[col] = tab                                  # зберігаємо
    print(f"\nFrequency Table: {col}")
    print(tab.to_string())

# ---------------------------------------------------------------------------
# Крок 2. Таблиці правдоподібності (likelihood table).
# P(значення | клас) = частота(значення, клас) / кількість об'єктів класу.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("КРОК 2. ТАБЛИЦІ ПРАВДОПОДІБНОСТІ (likelihood tables)")
print("=" * 78)

like_tables = {}                                            # словник таблиць правдоподібності
for col in FEATURES:                                        # знову по кожному атрибуту
    tab = freq_tables[col].drop(index="Разом").drop(columns="Разом")  # «чисті» частоти
    lik = pd.DataFrame(index=tab.index)                     # порожня таблиця під ймовірності
    lik["P(x|Yes)"] = tab["Yes"] / n_yes                    # умовна ймовірність для класу Yes
    lik["P(x|No)"] = tab["No"] / n_no                       # умовна ймовірність для класу No
    lik["P(x)"] = (tab["Yes"] + tab["No"]) / n_total        # безумовна (маргінальна) ймовірність
    like_tables[col] = lik                                  # зберігаємо
    print(f"\nLikelihood Table: {col}   (знаменники: Yes={n_yes}, No={n_no}, Разом={n_total})")
    print(lik.round(4).to_string())

print(f"\nАпріорні ймовірності: P(Yes) = {n_yes}/{n_total} = {p_yes:.4f} ; "
      f"P(No) = {n_no}/{n_total} = {p_no:.4f}")

# ---------------------------------------------------------------------------
# ЗАВДАННЯ 3. Прогноз для ВАРІАНТА 3: Sunny + High + Weak.
# Спосіб (а): ручний розрахунок за формулою наївного Байєса.
# ---------------------------------------------------------------------------
VARIANT = {"Outlook": "Sunny", "Humidity": "High", "Wind": "Weak"}   # умова варіанта 3
print("\n" + "=" * 78)
print("ЗАВДАННЯ 3 (а). РУЧНИЙ РОЗРАХУНОК ЗА ФОРМУЛОЮ БАЙЄСА. ВАРІАНТ 3")
print("=" * 78)
print("Умова: " + ", ".join(f"{k} = {v}" for k, v in VARIANT.items()))


def manual_posterior(condition, alpha=0.0, verbose=True):
    """Ненормовані добутки P(X|c)*P(c) для класів Yes/No.

    alpha — параметр згладжування Лапласа (0 = без згладжування). Згладжування
    застосовується лише до умовних ймовірностей P(x|c) — так само, як це робить
    CategoricalNB у sklearn; апріорні P(c) залишаються емпіричними.
    Повертає словник {клас: ненормований добуток}.
    """
    scores = {}                                             # результат по класах
    for cls, n_cls in (("Yes", n_yes), ("No", n_no)):       # перебираємо обидва класи
        prior = n_cls / n_total                             # апріорна ймовірність класу
        parts = [f"P({cls}) = {prior:.4f}"]                 # текстовий слід обчислень
        prod = prior                                        # починаємо добуток з апріорної
        for col, val in condition.items():                  # по кожній ознаці умови
            tab = freq_tables[col].drop(index="Разом").drop(columns="Разом")  # частоти
            cnt = int(tab.loc[val, cls])                    # частота «значення & клас»
            n_vals = tab.shape[0]                           # кількість різних значень ознаки
            cond = (cnt + alpha) / (n_cls + alpha * n_vals)  # P(значення|клас) зі згладжуванням
            prod *= cond                                    # накопичуємо добуток
            parts.append(f"P({col}={val}|{cls}) = {cnt}/{n_cls} = {cond:.4f}")  # слід
        scores[cls] = prod                                  # зберігаємо ненормований добуток
        if verbose:                                         # друкуємо всі проміжні множники
            print(f"\n  Клас «{cls}»:")
            for p in parts:
                print("    " + p)
            print(f"    Добуток = {prod:.6f}")
    return scores


scores = manual_posterior(VARIANT)                          # рахуємо без згладжування
denom = scores["Yes"] + scores["No"]                        # нормувальний знаменник
post_yes = scores["Yes"] / denom                            # нормована P(Yes|X)
post_no = scores["No"] / denom                              # нормована P(No|X)

print("\n  Нормування:")
print(f"    P(Yes|X) = {scores['Yes']:.6f} / ({scores['Yes']:.6f} + {scores['No']:.6f}) "
      f"= {post_yes:.4f}  ({post_yes * 100:.2f} %)")
print(f"    P(No|X)  = {scores['No']:.6f} / ({scores['Yes']:.6f} + {scores['No']:.6f}) "
      f"= {post_no:.4f}  ({post_no * 100:.2f} %)")
answer_manual = "Yes" if post_yes > post_no else "No"       # клас з більшою апостеріорною
print(f"\n  ВІДПОВІДЬ (ручний розрахунок): Play = {answer_manual} — "
      + ("матч ВІДБУДЕТЬСЯ" if answer_manual == "Yes" else "матч НЕ відбудеться"))

# ---------------------------------------------------------------------------
# Згладжування Лапласа (alpha = 1) — демонстрація і пояснення.
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("Згладжування Лапласа (alpha = 1)")
print("-" * 78)
zero_cells = []                                             # шукаємо нульові умовні ймовірності
for col in FEATURES:                                        # по всіх ознаках
    tab = freq_tables[col].drop(index="Разом").drop(columns="Разом")
    for val in tab.index:                                   # по всіх значеннях ознаки
        for cls in CLASSES:                                 # по обох класах
            if tab.loc[val, cls] == 0:                      # нульова частота
                zero_cells.append(f"P({col}={val}|{cls}) = 0")
print("Нульові умовні ймовірності у наборі: "
      + (", ".join(zero_cells) if zero_cells else "відсутні"))
print("Для умови варіанта 3 нулів немає, тож згладжування не змінює рішення,")
print("але воно потрібне, щоб один нульовий множник не обнуляв увесь добуток.")

scores_lap = manual_posterior(VARIANT, alpha=1.0, verbose=False)   # той самий розрахунок з alpha=1
den_lap = scores_lap["Yes"] + scores_lap["No"]                      # знаменник
print(f"\nЗ alpha = 1:  P(Yes|X) = {scores_lap['Yes'] / den_lap:.4f} "
      f"({scores_lap['Yes'] / den_lap * 100:.2f} %), "
      f"P(No|X) = {scores_lap['No'] / den_lap:.4f} "
      f"({scores_lap['No'] / den_lap * 100:.2f} %)")

# ---------------------------------------------------------------------------
# Спосіб (б): той самий розрахунок засобами sklearn (CategoricalNB).
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("ЗАВДАННЯ 3 (б). РОЗРАХУНОК ЗАСОБАМИ SKLEARN (CategoricalNB)")
print("=" * 78)

SK_FEATURES = list(VARIANT.keys())                          # беремо лише ознаки з умови варіанта
enc = OrdinalEncoder()                                      # кодувальник категорій у числа 0,1,2...
X = enc.fit_transform(df[SK_FEATURES])                      # матриця ознак у числовому вигляді
y = (df[TARGET] == "Yes").astype(int).to_numpy()            # цільовий вектор: Yes=1, No=0
for name, cats in zip(SK_FEATURES, enc.categories_):        # показуємо, як закодовано категорії
    print(f"  {name}: " + ", ".join(f"{c}->{i}" for i, c in enumerate(cats)))

x_new = enc.transform(pd.DataFrame([VARIANT])[SK_FEATURES])  # кодуємо умову варіанта 3

# alpha майже 0 => формула збігається з ручним розрахунком «один-в-один»
nb_raw = CategoricalNB(alpha=1e-10, force_alpha=True)       # модель без згладжування
nb_raw.fit(X, y)                                            # навчання на 14 рядках
proba_raw = nb_raw.predict_proba(x_new)[0]                  # апостеріорні ймовірності [No, Yes]
pred_raw = nb_raw.predict(x_new)[0]                         # прогнозований клас

# alpha = 1 => класичне згладжування Лапласа
nb_lap = CategoricalNB(alpha=1.0, force_alpha=True)         # модель зі згладжуванням
nb_lap.fit(X, y)                                            # навчання
proba_lap = nb_lap.predict_proba(x_new)[0]                  # апостеріорні ймовірності

print(f"\nCategoricalNB(alpha=1e-10): P(No|X) = {proba_raw[0]:.4f} ({proba_raw[0] * 100:.2f} %), "
      f"P(Yes|X) = {proba_raw[1]:.4f} ({proba_raw[1] * 100:.2f} %)")
print(f"CategoricalNB(alpha=1.0)  : P(No|X) = {proba_lap[0]:.4f} ({proba_lap[0] * 100:.2f} %), "
      f"P(Yes|X) = {proba_lap[1]:.4f} ({proba_lap[1] * 100:.2f} %)")
print(f"Прогноз sklearn: Play = {'Yes' if pred_raw == 1 else 'No'}")

# Контроль збігу двох способів
diff = abs(proba_raw[1] - post_yes)                         # різниця між ручним і sklearn
print(f"\nПеревірка збігу: |ручний P(Yes|X) - sklearn P(Yes|X)| = {diff:.2e}")
assert diff < 1e-6, "Ручний розрахунок і sklearn НЕ збіглися!"   # жорстка перевірка
print("Результати двох способів ЗБІГАЮТЬСЯ.")

# Для порівняння — приклад самої методички (Rain, High, Weak)
EX = {"Outlook": "Rain", "Humidity": "High", "Wind": "Weak"}       # умова прикладу з методички
sc_ex = manual_posterior(EX, verbose=False)                        # чесний перерахунок
den_ex = sc_ex["Yes"] + sc_ex["No"]                                # знаменник
print(f"\nДовідково, приклад методички (Rain, High, Weak), чесний перерахунок за 14 рядками:")
print(f"  Yes = {sc_ex['Yes']:.6f}, No = {sc_ex['No']:.6f}  =>  "
      f"P(Yes|X) = {sc_ex['Yes'] / den_ex * 100:.2f} %, P(No|X) = {sc_ex['No'] / den_ex * 100:.2f} %")
print("  (у методичці наведено 0,0199 / 0,0166 і 55 % / 45 % — ці числа не відтворюються)")

# ---------------------------------------------------------------------------
# Рисунок 1: таблиці правдоподібності + апостеріорні ймовірності варіанта 3.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))           # 4 панелі в один ряд
for ax, col in zip(axes[:3], SK_FEATURES):                  # перші три панелі — правдоподібності
    lik = like_tables[col]                                  # таблиця для цієї ознаки
    idx = np.arange(len(lik))                               # позиції груп стовпчиків
    ax.bar(idx - 0.2, lik["P(x|Yes)"], 0.4, label="P(x|Yes)", color="#2a9d8f")   # Yes
    ax.bar(idx + 0.2, lik["P(x|No)"], 0.4, label="P(x|No)", color="#e76f51")     # No
    ax.set_xticks(idx)                                      # підписи категорій
    ax.set_xticklabels(lik.index)
    ax.set_ylim(0, 1.0)                                     # ймовірність не виходить за 0..1
    ax.set_title(f"Правдоподібності: {col}", pad=26)        # відступ під смугу легенди
    ax.set_ylabel("ймовірність")                            # підпис осі
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=2, frameon=False, fontsize=9)            # легенда ПОЗА полем даних
    ax.grid(axis="y", alpha=0.3)                            # сітка

ax = axes[3]                                                # четверта панель — результат
bars = ax.bar(["Yes", "No"], [post_yes * 100, post_no * 100],
              color=["#2a9d8f", "#e76f51"])                 # апостеріорні у відсотках
for b, v in zip(bars, [post_yes * 100, post_no * 100]):     # підписуємо значення над стовпчиками
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f} %", ha="center", fontweight="bold")
ax.set_ylim(0, 100)                                         # шкала 0..100 %
ax.set_ylabel("P(клас | X), %")                             # підпис осі
ax.set_title("Варіант 3: Sunny, High, Weak")                # заголовок
ax.grid(axis="y", alpha=0.3)                                # сітка
fig.suptitle("ЛР6, завдання 3. Наївний Байєс на наборі Play Tennis", fontweight="bold")
fig.tight_layout()                                          # прибираємо накладання
fig.savefig(BASE / "fig1_play_tennis.png", dpi=150)          # зберігаємо рисунок
plt.close(fig)                                              # звільняємо пам'ять
print("\nЗбережено рисунок: fig1_play_tennis.png")
