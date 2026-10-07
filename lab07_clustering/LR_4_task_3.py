"""
Лабораторна робота №7 (у методичці — "ЛР №4")
Завдання 2.3. Оцінка кількості кластерів методом зсуву середнього (Mean Shift).

Студент: Камінський Олексій Дмитрович, група ВТ-23-2, варіант 3.
Вхідні дані: data_clustering.txt — той самий файл, що й у завданні 2.1.

Принципова відмінність від k-середніх: кількість кластерів НЕ задається
вручну, а визначається алгоритмом автоматично — через пошук максимумів
(піків) оцінки щільності розподілу даних.
"""

import matplotlib                                   # пакет візуалізації
matplotlib.use("Agg")                               # бекенд без вікна — зберігаємо у PNG

import numpy as np                                  # масиви та числові операції
import matplotlib.pyplot as plt                     # побудова графіків
from sklearn.cluster import MeanShift, estimate_bandwidth   # алгоритм зсуву середнього
from sklearn.metrics import silhouette_score        # оцінка якості кластеризації
from itertools import cycle                         # циклічний перебір маркерів/кольорів

# ---------------------------------------------------------------------------
# Завантаження вхідних даних
# ---------------------------------------------------------------------------
X = np.loadtxt("data_clustering.txt", delimiter=",")
print("Розмір вхідних даних:", X.shape)

# ---------------------------------------------------------------------------
# Оцінка ширини вікна (bandwidth) для X.
# Ширина вікна — параметр ядрової оцінки щільності. Замала ширина дає забагато
# кластерів, завелика — зливає сусідні кластери в один.
# Параметр quantile керує шириною вікна: більший quantile -> ширше вікно ->
# менше кластерів.
# ---------------------------------------------------------------------------
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))
print("Оцінена ширина вікна (quantile=0.1): {:.4f}".format(bandwidth_X))

# ---------------------------------------------------------------------------
# Кластеризація даних методом зсуву середнього
# bin_seeding=True прискорює роботу: стартові точки беруться з решітки,
# а не з кожної точки даних.
# ---------------------------------------------------------------------------
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)                              # навчання моделі

# Витягування центрів кластерів
cluster_centers = meanshift_model.cluster_centers_
print("\nCenters of clusters:\n", np.round(cluster_centers, 4))

# Оцінка кількості кластерів — визначається алгоритмом АВТОМАТИЧНО
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)

# Детальний вивід координат центрів і розмірів кластерів
print("\nКоординати центрів кластерів та кількість точок у кожному:")
for i, c in enumerate(cluster_centers, start=1):
    size = int(np.sum(labels == i - 1))             # скільки точок потрапило у кластер
    print("  Кластер {}: центр = ({:7.4f}, {:7.4f}), точок = {}".format(i, c[0], c[1], size))

# Оцінка якості
sil = silhouette_score(X, labels)
print("\nSilhouette score (Mean Shift): {:.4f}".format(sil))

# ---------------------------------------------------------------------------
# Рисунок 7. Відображення на графіку точок та центрів кластерів
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 6))
markers = 'o*xvs'                                   # набір маркерів для різних кластерів
for i, marker in zip(range(num_clusters), cycle(markers)):
    # Відображення на графіку точок, що належать поточному кластеру
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='black')
    # Відображення на графіку центру поточного кластера
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
             markerfacecolor='black', markeredgecolor='black', markersize=15)
plt.title("Кластери (Mean Shift), знайдено автоматично: {}".format(num_clusters))
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.grid(alpha=0.3)
plt.savefig("fig7_meanshift_clusters.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Додаткове дослідження: як quantile впливає на кількість знайдених кластерів
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("ВПЛИВ ПАРАМЕТРА quantile НА ШИРИНУ ВІКНА І КІЛЬКІСТЬ КЛАСТЕРІВ")
print("=" * 68)
print("  quantile   bandwidth   кластерів   silhouette")

quantiles = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
q_bw, q_n, q_sil = [], [], []
for q in quantiles:
    bw = estimate_bandwidth(X, quantile=q, n_samples=len(X))   # ширина вікна для цього quantile
    ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(X)      # кластеризація
    n = len(np.unique(ms.labels_))                             # скільки кластерів вийшло
    s = silhouette_score(X, ms.labels_) if n > 1 else float('nan')
    q_bw.append(bw)
    q_n.append(n)
    q_sil.append(s)
    print("  {:<10.2f} {:9.4f} {:10d} {:12.4f}".format(q, bw, n, s))

# Рисунок 8. Залежність кількості кластерів і ширини вікна від quantile
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(quantiles, q_n, "o-", color="tab:blue")
ax[0].axhline(5, color="red", linestyle="--", label="k = 5 (завдання 2.1)")
ax[0].set_xlabel("quantile")
ax[0].set_ylabel("Кількість знайдених кластерів")
ax[0].set_title("Кількість кластерів vs quantile")
ax[0].grid(alpha=0.3)
ax[0].legend()
ax[1].plot(quantiles, q_bw, "s-", color="tab:green")
ax[1].set_xlabel("quantile")
ax[1].set_ylabel("bandwidth")
ax[1].set_title("Ширина вікна vs quantile")
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig8_meanshift_quantile.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Порівняння центрів Mean Shift із центрами k-середніх із завдання 2.1
# ---------------------------------------------------------------------------
from sklearn.cluster import KMeans                  # імпорт тут, щоб підкреслити порівняльний характер
km = KMeans(init="k-means++", n_clusters=5, n_init=10, random_state=3).fit(X)
km_centers = km.cluster_centers_

print("\n" + "=" * 68)
print("ПОРІВНЯННЯ ЦЕНТРІВ: Mean Shift (авто) vs k-середніх (k=5 задано вручну)")
print("=" * 68)
# Для кожного центру Mean Shift шукаємо найближчий центр k-середніх
for i, c in enumerate(cluster_centers, start=1):
    d = np.linalg.norm(km_centers - c, axis=1)      # відстані до всіх центрів KMeans
    j = int(np.argmin(d))                           # індекс найближчого
    print("  MS центр {}: ({:7.4f}, {:7.4f})  <->  KMeans центр {}: ({:7.4f}, {:7.4f})  |  відстань = {:.4f}"
          .format(i, c[0], c[1], j + 1, km_centers[j][0], km_centers[j][1], d[j]))

print("\nSilhouette: Mean Shift = {:.4f}, KMeans(k=5) = {:.4f}"
      .format(sil, silhouette_score(X, km.labels_)))

print("\nЗбережено рисунки: fig7_meanshift_clusters.png, fig8_meanshift_quantile.png")
