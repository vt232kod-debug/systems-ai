"""ЛР-5, завдання 2.4. Обчислення відносної важливості ознак
за допомогою регресора AdaBoost поверх дерева рішень.

Запуск:
    python LR_5_task_4.py
"""

import matplotlib                                # налаштування бекенда ДО pyplot
matplotlib.use("Agg")                            # малюємо у файли

import numpy as np                               # масиви
import matplotlib.pyplot as plt                  # графіки
from sklearn.tree import DecisionTreeRegressor               # базова (слабка) модель
from sklearn.ensemble import AdaBoostRegressor               # ансамбль-бустинг
from sklearn.utils import shuffle                            # перемішування даних
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split         # поділ вибірки

# Завантаження даних із цінами на нерухомість.
# УВАГА: у методичці використано load_boston(), але цю функцію ВИЛУЧЕНО
# зі scikit-learn з версії 1.2 (етичні застереження щодо ознаки B).
# Офіційна заміна — набір California Housing (20640 зразків, 8 ознак).
from sklearn.datasets import fetch_california_housing
housing_data = fetch_california_housing()        # завантажуємо набір даних

# Перемішування даних, щоб підвищити об'єктивність аналізу
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# Розбиття даних на навчальний та тестовий набори (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# Модель на основі регресора AdaBoost:
# базова модель — дерево рішень глибини 4, таких дерев будується 400
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)                  # навчання ансамблю

# Обчислення показників ефективності регресора AdaBoost
y_pred = regressor.predict(X_test)               # прогноз на тестовій вибірці
mse = mean_squared_error(y_test, y_pred)         # середньоквадратична помилка
evs = explained_variance_score(y_test, y_pred)   # частка поясненої дисперсії
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Вилучення важливості ознак
feature_importances = regressor.feature_importances_   # внески ознак (сума = 1)
feature_names = np.array(housing_data.feature_names)   # назви ознак

# Нормалізація значень важливості ознак: найважливіша ознака = 100%
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортування та перестановка значень (за спаданням важливості)
index_sorted = np.flipud(np.argsort(feature_importances))

# Розміщення міток уздовж осі X
pos = np.arange(index_sorted.shape[0]) + 0.5

# Побудова стовпчастої діаграми
plt.figure(figsize=(9, 5))
plt.bar(pos, feature_importances[index_sorted], align='center',
        color='steelblue', edgecolor='black')
plt.xticks(pos, feature_names[index_sorted], rotation=20)    # підписи ознак
plt.ylabel('Relative Importance, %')
plt.title('Feature importance using AdaBoost regressor (California Housing)')
plt.grid(axis='y', alpha=0.3)
plt.savefig('fig14_feature_importance.png', dpi=150, bbox_inches='tight')
print('[saved] fig14_feature_importance.png')

# Друкуємо числові значення важливості — вони потрібні для висновку у звіті
print("\nRelative feature importance (normalized to 100):")
for name, value in zip(feature_names[index_sorted],
                       feature_importances[index_sorted]):
    print(f"  {name:12s} {value:6.2f}")
