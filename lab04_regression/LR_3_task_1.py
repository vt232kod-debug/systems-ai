# -*- coding: utf-8 -*-
"""
ЛР №4. Частина Б, Завдання 2.1. Створення регресора однієї змінної.
Вхідний файл: data_singlevar_regr.txt
Камінський Олексій Дмитрович, ВТ-23-2.
"""

import matplotlib                               # backend задаємо до імпорту pyplot
matplotlib.use("Agg")                           # без GUI, результат — у файл PNG
import matplotlib.pyplot as plt                 # побудова графіків
import pickle                                   # серіалізація моделі у файл
import numpy as np                              # числові масиви
from sklearn import linear_model                # лінійні моделі sklearn
import sklearn.metrics as sm                    # метрики якості регресії

# Вхідний файл, який містить дані
input_file = "data_singlevar_regr.txt"

# Завантаження даних: у файлі роздільник — кома
data = np.loadtxt(input_file, delimiter=",")
# Останній стовпець — цільова змінна y, решта — ознаки X
X, y = data[:, :-1], data[:, -1]

print("=" * 70)
print("ЗАВДАННЯ 2.1. РЕГРЕСОР ОДНІЄЇ ЗМІННОЇ (data_singlevar_regr.txt)")
print("=" * 70)
print("Розмір набору: X %s, y %s" % (X.shape, y.shape))

# Розбивка даних на навчальний та тестовий набори (80% / 20%)
num_training = int(0.8 * len(X))                # кількість навчальних зразків
num_test = len(X) - num_training                # кількість тестових зразків

# Тренувальні дані
X_train, y_train = X[:num_training], y[:num_training]
# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]
print("Навчальна вибірка: %d зразків, тестова: %d зразків" % (num_training, num_test))

# Створення об'єкта лінійного регресора
regressor = linear_model.LinearRegression()
# Навчання моделі на тренувальних даних
regressor.fit(X_train, y_train)

# Прогнозування результату для тестового набору
y_test_pred = regressor.predict(X_test)

# Знайдені параметри прямої y = k*x + b
print("\nПараметри навченої моделі:")
print("  Коефіцієнт нахилу k = %.6f" % regressor.coef_[0])
print("  Зсув            b = %.6f" % regressor.intercept_)
print("  Рівняння: y = %.4f·x %+.4f" % (regressor.coef_[0], regressor.intercept_))

# ------------------------------------------------------------ побудова графіка
fig, ax = plt.subplots(figsize=(8, 5.5))        # полотно для графіка
ax.scatter(X_train, y_train, color="lightsteelblue", edgecolors="gray",   # навчальні дані
           s=45, label="Навчальні дані")
ax.scatter(X_test, y_test, color="green", s=70, zorder=5,                 # тестові дані
           label="Тестові дані (істинні значення)")
# Апроксимуюча пряма: сортуємо X_test, щоб лінія малювалась без зламів
order = np.argsort(X_test[:, 0])                # індекси сортування за зростанням x
ax.plot(X_test[order], y_test_pred[order], color="black", linewidth=4,
        label="Прогноз лінійного регресора")
ax.set_xlabel("X")                              # підпис осі абсцис
ax.set_ylabel("y")                              # підпис осі ординат
ax.set_title("Завдання 2.1: лінійна регресія однієї змінної")
ax.grid(True, linestyle="--", alpha=0.5)        # сітка
ax.legend()                                     # легенда
fig.tight_layout()
fig.savefig("fig3_task1_singlevar.png", dpi=150)   # зберігаємо рисунок
plt.close(fig)

# ----------------------------------------------- метрики якості регресора
print("\nLinear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# ------------------------------------------- збереження моделі через pickle
# Файл для збереження моделі
output_model_file = "model_task1.pkl"

# Збереження моделі
with open(output_model_file, "wb") as f:        # відкриваємо файл для бінарного запису
    pickle.dump(regressor, f)                   # серіалізуємо навчений регресор
print("\nМодель збережено у файл: %s" % output_model_file)

# Завантаження моделі з файлу на диску та побудова прогнозу
# (у методичці крок завантаження пропущено — об'єкт regressor_model не створювався)
with open(output_model_file, "rb") as f:        # відкриваємо файл для бінарного читання
    regressor_model = pickle.load(f)            # десеріалізуємо модель
y_test_pred_new = regressor_model.predict(X_test)   # прогноз завантаженою моделлю

print("\nNew mean absolute error =",
      round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
# Контроль: прогнози вихідної та завантаженої моделі мають бути ідентичними
print("Максимальна різниця прогнозів до/після pickle: %.2e"
      % np.max(np.abs(y_test_pred - y_test_pred_new)))
print("\nРисунок збережено: fig3_task1_singlevar.png")
