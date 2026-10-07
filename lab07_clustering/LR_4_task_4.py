"""
Лабораторна робота №7 (у методичці — "ЛР №4")
Завдання 2.4. Знаходження підгруп на фондовому ринку
              з використанням моделі поширення подібності (Affinity Propagation).

Студент: Камінський Олексій Дмитрович, група ВТ-23-2, варіант 3.

Керуюча ознака — варіація котирувань між відкриттям і закриттям біржі
(close - open) за кожен торговий день.

ВАЖЛИВО: методичка пропонує matplotlib.finance.quotes_historical_yahoo_ochl.
Модуль matplotlib.finance видалено з matplotlib ще у версії 2.2 (2018 р.),
а сервіс Yahoo, на який він спирався, більше не працює.
Тому котирування завантажуються через бібліотеку yfinance.
Завантажені дані кешуються у quotes_cache.csv, щоб повторний запуск
не залежав від мережі.
"""

import matplotlib                                      # пакет візуалізації
matplotlib.use("Agg")                                  # бекенд без вікна

import os                                              # перевірка наявності файла кешу
import json                                            # читання прив'язок символів до назв
import datetime                                        # задання періоду котирувань
import numpy as np                                     # числові масиви
import pandas as pd                                    # таблиці котирувань
import matplotlib.pyplot as plt                        # графіки
from sklearn import covariance, cluster                # модель графа та кластеризація
from sklearn.manifold import TSNE                      # зниження розмірності для візуалізації

# ===========================================================================
# 1. ЗАВАНТАЖЕННЯ ПРИВ'ЯЗОК СИМВОЛІВ КОМПАНІЙ ДО ЇХ ПОВНИХ НАЗВ
# ===========================================================================
input_file = "company_symbol_mapping.json"             # вхідний файл із символічними позначеннями компаній

with open(input_file, "r") as f:                       # відкриваємо файл на читання
    company_symbols_map = json.loads(f.read())         # розбираємо JSON у словник {символ: назва}

symbols, names = np.array(list(company_symbols_map.items())).T   # розділяємо на два масиви
print("Усього компаній у файлі прив'язок:", len(symbols))

# ===========================================================================
# 2. ЗАВАНТАЖЕННЯ АРХІВНИХ ДАНИХ КОТИРУВАНЬ
# ===========================================================================
# Період спостереження взято ТОЧНО таким, як задано у методичці (стор. 11):
# start_date = datetime.datetime(2003, 7, 3); end_date = datetime.datetime(2007, 5, 4)
start_date = datetime.datetime(2003, 7, 3)             # початок періоду спостереження (за методичкою)
end_date = datetime.datetime(2007, 5, 4)               # кінець періоду спостереження (за методичкою)
cache_file = "quotes_cache.csv"                        # локальний кеш завантажених котирувань

if os.path.exists(cache_file):                         # якщо кеш уже є — беремо дані з нього
    print("Читаємо котирування з кешу:", cache_file)
    raw = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
else:                                                  # інакше йдемо в мережу
    import yfinance as yf                              # завантажувач котирувань (заміна matplotlib.finance)
    print("Завантаження котирувань через yfinance за період {} .. {}"
          .format(start_date.date(), end_date.date()))
    raw = yf.download(list(symbols),                   # список тікерів
                      start=start_date,                # від якої дати
                      end=end_date,                    # до якої дати
                      auto_adjust=False,               # беремо «сирі» Open/Close без коригування
                      progress=False,                  # без індикатора прогресу
                      threads=False)                   # послідовне завантаження — стабільніше
    # у кеш зберігаємо лише потрібні для роботи стовпці Open і Close:
    # High/Low/Adj Close/Volume не використовуються, а файл стає меншим у 3 рази
    raw[["Open", "Close"]].to_csv(cache_file, float_format="%.6f")   # кеш для відтворюваності

# ---------------------------------------------------------------------------
# Вилучення котирувань, що відповідають відкриттю та закриттю біржі
# ---------------------------------------------------------------------------
open_df = raw["Open"]                                  # таблиця цін відкриття (дати x тікери)
close_df = raw["Close"]                                # таблиця цін закриття

# Відсіюємо тікери, для яких дані не завантажились (компанія поглинена, тікер змінено тощо)
valid = [s for s in symbols                            # перебираємо всі символи
         if s in open_df.columns                       # колонка взагалі існує
         and open_df[s].notna().sum() > 0.9 * len(open_df)   # і заповнена щонайменше на 90 %
         and close_df[s].notna().sum() > 0.9 * len(close_df)]

missing = [s for s in symbols if s not in valid]       # перелік недоступних тікерів
print("\nДоступні тікери: {} з {}".format(len(valid), len(symbols)))
print("Недоступні тікери ({}): {}".format(len(missing), ", ".join(missing)))

# Залишаємо лише валідні тікери та дні, де є повні дані по всіх компаніях
open_df = open_df[valid].dropna()                      # прибираємо рядки з пропусками
close_df = close_df[valid].reindex(open_df.index)      # вирівнюємо закриття за тим самим індексом дат
common = open_df.index.intersection(close_df.dropna().index)   # спільні торгові дні
open_df, close_df = open_df.loc[common], close_df.loc[common]

symbols = np.array(valid)                              # оновлений масив символів
names = np.array([company_symbols_map[s] for s in valid])      # відповідні назви компаній
print("Торгових днів у вибірці:", len(common))

opening_quotes = open_df.to_numpy().T.astype(float)    # матриця (компанії x дні) — ціни відкриття
closing_quotes = close_df.to_numpy().T.astype(float)   # матриця (компанії x дні) — ціни закриття

# ---------------------------------------------------------------------------
# Обчислення різниці між двома видами котирувань
# ---------------------------------------------------------------------------
quotes_diff = closing_quotes - opening_quotes          # денна варіація: скільки акція додала/втратила за сесію

# ===========================================================================
# 3. НОРМАЛІЗАЦІЯ ДАНИХ
# ===========================================================================
# Методичка задає рівно одну операцію нормалізації — ділення на стандартне
# відхилення кожного стовпця. Жодного додаткового центрування не додаємо:
# GraphicalLassoCV з assume_centered=False центрує дані самостійно.
X = quotes_diff.copy().T                               # транспонуємо: рядки = дні, стовпці = компанії
X /= X.std(axis=0)                                     # ділимо на стандартне відхилення кожної компанії
print("Форма нормалізованої матриці X (дні x компанії):", X.shape)

# ===========================================================================
# 4. СТВОРЕННЯ ТА НАВЧАННЯ МОДЕЛІ ГРАФА
# GraphLassoCV перейменовано на GraphicalLassoCV у scikit-learn 0.20.
# Модель оцінює розріджену матрицю коваріацій — «граф» зв'язків між акціями.
# ===========================================================================
alphas = np.logspace(-1.5, 0.5, 12)                    # сітка коефіцієнтів регуляризації для крос-валідації
edge_model = covariance.GraphicalLassoCV(alphas=alphas,   # перебір alpha за крос-валідацією
                                         cv=5,            # 5 блоків крос-валідації
                                         max_iter=500,    # підвищений ліміт ітерацій (інакше не сходиться)
                                         tol=1e-3,        # послаблений поріг збіжності
                                         enet_tol=1e-3,   # поріг для внутрішнього elastic-net
                                         assume_centered=False)

print("\nНавчання GraphicalLassoCV...")
# Під час крос-валідації для завеликих alpha матриця стає виродженою і numpy
# друкує попередження у slogdet. На кінцевий результат це не впливає —
# такі alpha просто отримують погану оцінку і відкидаються. Глушимо їх.
import warnings                                        # керування попередженнями
with np.errstate(all="ignore"), warnings.catch_warnings():
    warnings.simplefilter("ignore")                    # ховаємо RuntimeWarning/ConvergenceWarning
    edge_model.fit(X)                                  # навчання моделі графа

print("Обраний alpha_: {:.5f}".format(edge_model.alpha_))

# ===========================================================================
# 5. КЛАСТЕРИЗАЦІЯ НА ОСНОВІ ПОШИРЕННЯ ПОДІБНОСТІ
# ===========================================================================
_, labels = cluster.affinity_propagation(edge_model.covariance_,   # матриця подібності = коваріації
                                         random_state=3)           # варіант 3 — для відтворюваності
num_labels = labels.max()                              # найбільший номер мітки (нумерація з нуля)

print("\n" + "=" * 70)
print("ЗНАЙДЕНІ ПІДГРУПИ НА ФОНДОВОМУ РИНКУ: {} кластерів".format(num_labels + 1))
print("=" * 70)
for i in range(num_labels + 1):                        # перебираємо всі кластери
    print("Cluster", i + 1, "==>", ', '.join(names[labels == i]))   # друкуємо склад кластера

# ---------------------------------------------------------------------------
# Зведена таблиця розмірів кластерів
# ---------------------------------------------------------------------------
print("\nРозміри кластерів:")
for i in range(num_labels + 1):
    print("  Кластер {:<3d} — {:2d} компаній".format(i + 1, int(np.sum(labels == i))))

# ===========================================================================
# 6. РИСУНОК 9. Візуалізація підгруп на площині
# Розташування компаній отримуємо методом t-SNE із матриці часткових кореляцій.
# ===========================================================================
prec = edge_model.precision_.copy()                    # матриця точності (обернена до коваріаційної)
d = 1 / np.sqrt(np.diag(prec))                         # нормувальні множники
partial_corr = -prec * d[:, None] * d[None, :]         # часткові кореляції між акціями
np.fill_diagonal(partial_corr, 0)                      # діагональ обнуляємо

dist = 1 - np.abs(partial_corr)                        # перетворюємо подібність на відстань
np.fill_diagonal(dist, 0)                              # нульова відстань до самого себе

embedding = TSNE(n_components=2,                       # проєктуємо у 2D
                 metric="precomputed",                 # працюємо з готовою матрицею відстаней
                 init="random",                        # випадкова ініціалізація
                 perplexity=15,                        # підібрано за рівномірністю заповнення полотна
                 early_exaggeration=12.0,              # менше «розкидає» одиночні викиди
                 random_state=3).fit_transform(dist)   # варіант 3

fig, ax = plt.subplots(figsize=(14, 9.5))              # полотно рисунка
palette = plt.cm.tab20(np.linspace(0, 1, num_labels + 1))   # різні кольори для кластерів
for i in range(num_labels + 1):                        # малюємо кластер за кластером
    mask = labels == i                                 # маска компаній цього кластера
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               s=160, color=palette[i], edgecolors="black",
               label="Кластер {}".format(i + 1), zorder=3)

ax.set_title("Підгрупи учасників фондового ринку (Affinity Propagation, {} кластерів)"
             .format(num_labels + 1))
ax.legend(loc="upper left", fontsize=8, ncol=2)        # легенда з номерами кластерів
ax.grid(alpha=0.3)                                     # легка сітка
ax.set_xticks(())                                      # осі t-SNE не мають фізичного змісту
ax.set_yticks(())

# ---------------------------------------------------------------------------
# Розміщення підписів без накладань.
# Для кожної назви перебираємо кілька позицій-кандидатів навколо точки і
# беремо першу, яка не перетинається ні з уже поставленими підписами,
# ні з маркерами інших компаній. Усі розрахунки — у пікселях полотна.
# ---------------------------------------------------------------------------
ax.margins(0.09)                                       # трохи вільного місця під підписи з краю
fig.canvas.draw()                                      # малюємо, щоб отримати коректні трансформації
px_per_pt = fig.dpi / 72.0                             # перевід типографських пунктів у пікселі
pts_px = ax.transData.transform(embedding)             # координати точок у пікселях
FS = 8                                                 # розмір шрифта підпису, пт
marker_r = 7.0 * px_per_pt                             # радіус маркера (s=160) у пікселях

# набір зміщень-кандидатів (у пунктах) — від найближчих до найдальших
candidates = [(8, 6), (8, -6), (-8, 6), (-8, -6), (0, 13), (0, -15),
              (16, 0), (-16, 0), (0, 24), (0, -26), (22, 12), (-22, 12),
              (22, -12), (-22, -12), (0, 35), (0, -37)]


def boxes_overlap(a, b):                               # чи перетинаються два прямокутники
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def hits_marker(box, skip):                            # чи накриває прямокутник чужий маркер
    for j, (mx, my) in enumerate(pts_px):              # перебираємо всі точки
        if j == skip:                                  # власну точку ігноруємо
            continue
        if boxes_overlap(box, (mx - marker_r, my - marker_r, mx + marker_r, my + marker_r)):
            return True                                # знайдено накладання на маркер
    return False


placed = []                                            # уже зайняті прямокутники
for k in np.argsort(-embedding[:, 1]):                 # підписуємо згори вниз — стабільніший результат
    name = names[k]                                    # назва компанії
    w = (len(name) * 0.52 * FS + 3) * px_per_pt        # оцінка ширини тексту у пікселях
    h = 1.35 * FS * px_per_pt                          # оцінка висоти рядка у пікселях
    best = None                                        # обраний варіант розміщення
    for dx, dy in candidates:                          # перебираємо позиції-кандидати
        ha = "right" if dx < 0 else ("center" if dx == 0 else "left")   # прив'язка тексту
        shift = w if ha == "right" else (w / 2 if ha == "center" else 0)
        x0 = pts_px[k, 0] + dx * px_per_pt - shift     # лівий край прямокутника
        y0 = pts_px[k, 1] + dy * px_per_pt - h / 2     # нижній край прямокутника
        box = (x0, y0, x0 + w, y0 + h)                 # прямокутник підпису
        if any(boxes_overlap(box, b) for b in placed): # накладається на інший підпис?
            continue                                   # пробуємо наступного кандидата
        if hits_marker(box, k):                        # накриває чужий маркер?
            continue                                   # пробуємо наступного кандидата
        best = (dx, dy, ha, box)                       # місце вільне — беремо його
        break                                          # далі не шукаємо
    if best is None:                                   # усі кандидати зайняті — ставимо найдальший
        dx, dy = candidates[-1]
        ha = "center"
        x0 = pts_px[k, 0] + dx * px_per_pt - w / 2
        y0 = pts_px[k, 1] + dy * px_per_pt - h / 2
        best = (dx, dy, ha, (x0, y0, x0 + w, y0 + h))
    dx, dy, ha, box = best                             # розпаковуємо обране розміщення
    placed.append(box)                                 # фіксуємо зайнятий прямокутник
    ax.annotate(name, (embedding[k, 0], embedding[k, 1]),   # сам підпис
                fontsize=FS, xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))

plt.savefig("fig9_stock_clusters.png", dpi=150, bbox_inches="tight")
plt.close()

# ===========================================================================
# 7. РИСУНОК 10. Матриця коваріацій, впорядкована за кластерами
# ===========================================================================
order = np.argsort(labels)                             # сортуємо компанії за номером кластера
cov_sorted = edge_model.covariance_[np.ix_(order, order)]   # переставляємо рядки й стовпці

plt.figure(figsize=(11, 9))
vmax = np.percentile(np.abs(cov_sorted), 98)           # обрізаємо викиди для кращого контрасту
plt.imshow(cov_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
plt.colorbar(label="Коваріація")
plt.xticks(range(len(order)), names[order], rotation=90, fontsize=7)
plt.yticks(range(len(order)), names[order], fontsize=7)

boundaries = np.cumsum([np.sum(labels == i) for i in range(num_labels + 1)])[:-1]
for b in boundaries:                                   # малюємо межі між кластерами
    plt.axhline(b - 0.5, color="black", linewidth=1.2)
    plt.axvline(b - 0.5, color="black", linewidth=1.2)

plt.title("Матриця коваріацій, впорядкована за кластерами")
plt.tight_layout()
plt.savefig("fig10_stock_covariance.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nЗбережено рисунки: fig9_stock_clusters.png, fig10_stock_covariance.png")
