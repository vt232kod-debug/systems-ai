"""ЛР-5, завдання 2.3. Пошук оптимальних навчальних параметрів
за допомогою сіткового пошуку (GridSearchCV).

Запуск:
    python LR_5_task_3.py
"""

import matplotlib                                # налаштування бекенда ДО pyplot
matplotlib.use("Agg")                            # малюємо у файли

import numpy as np                               # масиви
import matplotlib.pyplot as plt                  # графіки
from sklearn.metrics import classification_report                  # звіт якості
from sklearn.model_selection import GridSearchCV, train_test_split # сітковий пошук і поділ
from sklearn.ensemble import ExtraTreesClassifier                  # гранично випадковий ліс

from utilities import visualize_classifier       # власна функція візуалізації

# Завантаження вхідних даних
input_file = 'data_random_forests.txt'           # той самий файл, що й у завданні 2.1
data = np.loadtxt(input_file, delimiter=',')     # читаємо числа через кому
X, y = data[:, :-1], data[:, -1]                 # ознаки та мітки

# Розбиття даних на три класи на підставі міток
class_0 = np.array(X[y == 0])                    # точки класу 0
class_1 = np.array(X[y == 1])                    # точки класу 1
class_2 = np.array(X[y == 2])                    # точки класу 2

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# Визначення сітки значень параметрів.
# Перший словник: фіксуємо n_estimators=100 і варіюємо max_depth.
# Другий словник: фіксуємо max_depth=4 і варіюємо n_estimators.
parameter_grid = [
    {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
    {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]},
]

# Метричні характеристики для вибору найкращої комбінації.
# Задача БАГАТОКЛАСОВА (3 класи при 2 ознаках), тому потрібні усереднені
# варіанти метрик ('precision'/'recall' без суфікса працюють лише для бінарної).
metrics = ['precision_weighted', 'recall_weighted']

results = {}                                     # сюди складемо результати для рисунка

for metric in metrics:                           # окремий сітковий пошук для кожної метрики
    print("\n##### Searching optimal parameters for", metric)

    # GridSearchCV перебирає всі комбінації сітки з 5-кратною крос-валідацією
    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0),
        parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)             # запускаємо перебір

    # Виведення оцінки для кожної комбінації параметрів.
    # УВАГА: атрибут grid_scores_ (з методички) вилучено зі scikit-learn 0.22,
    # сучасний еквівалент — словник cv_results_.
    print("\nGrid scores for the parameter grid:")
    params_list = classifier.cv_results_['params']           # список комбінацій
    means = classifier.cv_results_['mean_test_score']        # середні оцінки
    for params, avg_score in zip(params_list, means):        # друкуємо попарно
        print(params, '-->', round(avg_score, 3))

    print("\nBest parameters:", classifier.best_params_)     # найкраща комбінація
    results[metric] = (params_list, means)                   # запам'ятовуємо для графіка

    # Виведення звіту з результатами роботи класифікатора
    y_pred = classifier.predict(X_test)          # прогноз найкращої моделі на тесті
    print("\nPerformance report:\n")
    print(classification_report(y_test, y_pred))

# Побудова рисунка: оцінки всіх комбінацій для обох метрик
fig, axes = plt.subplots(1, 2, figsize=(13, 5))  # дві панелі поруч
for ax, metric in zip(axes, metrics):            # по панелі на метрику
    params_list, means = results[metric]         # дістаємо збережені результати
    labels = [f"d={p['max_depth']}\nn={p['n_estimators']}" for p in params_list]
    ax.bar(range(len(means)), means, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(means)))             # підписи під стовпчиками
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.75, 0.90)                      # масштаб, щоб бачити різницю
    ax.set_title(metric)                         # назва метрики
    ax.set_ylabel('mean CV score')
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Сітковий пошук: оцінки всіх комбінацій параметрів')
plt.savefig('fig13_grid_search_scores.png', dpi=150, bbox_inches='tight')
print('\n[saved] fig13_grid_search_scores.png')
