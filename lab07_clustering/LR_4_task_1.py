"""
Лабораторна робота №7 (у методичці — "ЛР №4")
Завдання 2.1. Кластеризація даних за допомогою методу k-середніх.

Студент: Камінський Олексій Дмитрович, група ВТ-23-2, варіант 3.
Вхідні дані: data_clustering.txt (350 точок, 2 ознаки, роздільник — кома).
"""

import matplotlib                      # базовий пакет візуалізації
matplotlib.use("Agg")                  # неінтерактивний бекенд: малюємо у файл, без вікна

import numpy as np                     # робота з масивами та сіткою точок
import matplotlib.pyplot as plt        # побудова графіків
from sklearn.cluster import KMeans     # алгоритм k-середніх
from sklearn import metrics            # метрики оцінки якості кластеризації

# ---------------------------------------------------------------------------
# Завантаження вхідних даних
# ---------------------------------------------------------------------------
X = np.loadtxt("data_clustering.txt", delimiter=",")   # матриця 350x2
print("Розмір вхідних даних:", X.shape)

# Щоб застосувати k-середніх, необхідно заздалегідь задати кількість кластерів.
# Візуально на графіку вхідних даних чітко видно 5 згущень.
num_clusters = 5

# ---------------------------------------------------------------------------
# Рисунок 1. Візуалізація вхідних даних (без міток)
# ---------------------------------------------------------------------------
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker="o", facecolors="none",
            edgecolors="black", s=80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1   # межі по осі X із запасом
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1   # межі по осі Y із запасом
plt.title("Вхідні дані")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())                         # приховуємо поділки — вони тут не інформативні
plt.yticks(())
plt.savefig("fig1_input_data.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Створення та навчання моделі KMeans
# init='k-means++' — "розумна" ініціалізація центроїдів (далеко один від одного),
# що гарантує швидку збіжність; n_init=10 — 10 незалежних запусків, беремо найкращий.
# ---------------------------------------------------------------------------
kmeans = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10, random_state=3)
kmeans.fit(X)                          # навчання: пошук оптимальних положень центроїдів

# ---------------------------------------------------------------------------
# Побудова сітки точок для візуалізації меж кластерів
# ---------------------------------------------------------------------------
step_size = 0.01                       # крок сітки: що менший — то гладкіші межі
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))

# Передбачення вихідних міток для всіх вузлів сітки
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)  # повертаємо форму сітки для imshow

# ---------------------------------------------------------------------------
# Рисунок 2. Межі кластерів: області, вхідні точки, центри
# ---------------------------------------------------------------------------
plt.figure()
plt.clf()
plt.imshow(output, interpolation="nearest",
           extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired,         # кожній мітці — свій колір області
           aspect="auto",
           origin="lower")             # origin='lower' — щоб вісь Y зростала вгору

# Відображення вхідних точок поверх зафарбованих областей
plt.scatter(X[:, 0], X[:, 1], marker="o", facecolors="none",
            edgecolors="black", s=80)

# Відображення центрів кластерів, знайдених методом k-середніх
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker="o", s=210, linewidths=4, color="black",
            zorder=12, facecolors="black")
plt.title("Межі кластерів (k-середніх, k=5)")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig("fig2_kmeans_boundaries.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nЦентри кластерів (k=5):")
for i, c in enumerate(cluster_centers, start=1):
    print("  Кластер {}: ({:7.4f}, {:7.4f})".format(i, c[0], c[1]))

# ---------------------------------------------------------------------------
# Оцінка якості кластеризації
# silhouette_score ∈ [-1, 1]: що ближче до 1 — то щільніші й краще розділені кластери.
# ---------------------------------------------------------------------------
labels = kmeans.labels_
sil = metrics.silhouette_score(X, labels, metric="euclidean")
db = metrics.davies_bouldin_score(X, labels)       # менше — краще
ch = metrics.calinski_harabasz_score(X, labels)    # більше — краще

print("\nОцінка якості кластеризації при k = 5:")
print("  Silhouette score        = {:.4f}".format(sil))
print("  Davies-Bouldin index    = {:.4f}".format(db))
print("  Calinski-Harabasz index = {:.4f}".format(ch))
print("  Inertia (сума квадратів відстаней) = {:.4f}".format(kmeans.inertia_))

# ---------------------------------------------------------------------------
# Додатково: підбір оптимального k за силуетним коефіцієнтом та "ліктем"
# ---------------------------------------------------------------------------
print("\nПошук оптимальної кількості кластерів:")
print("  k   silhouette    inertia")
scores, inertias, ks = [], [], range(2, 11)
for k in ks:
    km = KMeans(init="k-means++", n_clusters=k, n_init=10, random_state=3).fit(X)
    s = metrics.silhouette_score(X, km.labels_, metric="euclidean")
    scores.append(s)
    inertias.append(km.inertia_)
    print("  {:<3d} {:10.4f} {:10.2f}".format(k, s, km.inertia_))

best_k = list(ks)[int(np.argmax(scores))]
print("\nНайкраще значення silhouette досягається при k =", best_k)

# Рисунок 3. Графіки залежності метрик від k
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), scores, "o-", color="tab:blue")
ax[0].axvline(best_k, color="red", linestyle="--", label="k = {}".format(best_k))
ax[0].set_xlabel("Кількість кластерів k")
ax[0].set_ylabel("Silhouette score")
ax[0].set_title("Силуетний коефіцієнт")
ax[0].grid(alpha=0.3)
ax[0].legend()
ax[1].plot(list(ks), inertias, "s-", color="tab:orange")
ax[1].axvline(5, color="red", linestyle="--", label="k = 5 ('лікоть')")
ax[1].set_xlabel("Кількість кластерів k")
ax[1].set_ylabel("Inertia")
ax[1].set_title("Метод 'ліктя'")
ax[1].grid(alpha=0.3)
ax[1].legend()
plt.tight_layout()
plt.savefig("fig3_kmeans_quality.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nЗбережено рисунки: fig1_input_data.png, fig2_kmeans_boundaries.png, "
      "fig3_kmeans_quality.png")
