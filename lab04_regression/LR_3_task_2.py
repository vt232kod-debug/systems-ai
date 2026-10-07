# -*- coding: utf-8 -*-
"""
ЛР №4. Частина Б, Завдання 2.2. Передбачення за допомогою регресії однієї змінної.
Варіант визначається за списком групи (табл. 2.1): № у списку 3 -> № варіанту 3.
Вхідний файл свого варіанту: data_regr_3.txt
Камінський Олексій Дмитрович, ВТ-23-2, № у списку 3.
"""

import matplotlib                               # backend до pyplot
matplotlib.use("Agg")                           # рендеринг у файл
import matplotlib.pyplot as plt                 # графіки
import pickle                                   # збереження моделі
import numpy as np                              # масиви
from sklearn import linear_model                # лінійна регресія
import sklearn.metrics as sm                    # метрики

# Вхідний файл свого варіанту (варіант 3)
input_file = "data_regr_3.txt"

# Завантаження даних, роздільник — кома
data = np.loadtxt(input_file, delimiter=",")
X, y = data[:, :-1], data[:, -1]                # ознаки та цільова змінна

print("=" * 70)
print("ЗАВДАННЯ 2.2. РЕГРЕСІЯ ОДНІЄЇ ЗМІННОЇ, ВАРІАНТ 3 (data_regr_3.txt)")
print("=" * 70)
print("Розмір набору: X %s, y %s" % (X.shape, y.shape))
print("Діапазон X: [%.2f; %.2f], діапазон y: [%.2f; %.2f]"
      % (X.min(), X.max(), y.min(), y.max()))

# Розбивка даних на навчальний та тестовий набори (80% / 20%)
num_training = int(0.8 * len(X))                # розмір навчальної вибірки
num_test = len(X) - num_training                # розмір тестової вибірки

X_train, y_train = X[:num_training], y[:num_training]   # тренувальні дані
X_test, y_test = X[num_training:], y[num_training:]     # тестові дані
print("Навчальна вибірка: %d зразків, тестова: %d зразків" % (num_training, num_test))

# Створення та навчання лінійного регресора
regressor = linear_model.LinearRegression()     # об'єкт моделі
regressor.fit(X_train, y_train)                 # підгонка за методом найменших квадратів

# Прогнозування результату на тестовому наборі
y_test_pred = regressor.predict(X_test)

print("\nПараметри навченої моделі:")
print("  Коефіцієнт нахилу k = %.6f" % regressor.coef_[0])
print("  Зсув            b = %.6f" % regressor.intercept_)
print("  Рівняння: y = %.4f·x %+.4f" % (regressor.coef_[0], regressor.intercept_))

# ------------------------------------------------------------ побудова графіка
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(X_train, y_train, color="lightsteelblue", edgecolors="gray",
           s=45, label="Навчальні дані")
ax.scatter(X_test, y_test, color="green", s=70, zorder=5,
           label="Тестові дані (істинні значення)")
order = np.argsort(X_test[:, 0])                # сортуємо для гладкої лінії
ax.plot(X_test[order], y_test_pred[order], color="black", linewidth=4,
        label="Прогноз лінійного регресора")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title("Завдання 2.2: лінійна регресія, варіант 3")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig("fig4_task2_regr3.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------- метрики якості
print("\nLinear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# --------------------------------------------- збереження/завантаження моделі
output_model_file = "model_task2.pkl"           # ім'я файлу моделі
with open(output_model_file, "wb") as f:        # бінарний запис
    pickle.dump(regressor, f)                   # серіалізація моделі
with open(output_model_file, "rb") as f:        # бінарне читання
    regressor_model = pickle.load(f)            # десеріалізація моделі
y_test_pred_new = regressor_model.predict(X_test)   # прогноз відновленою моделлю

print("\nМодель збережено/завантажено: %s" % output_model_file)
print("New mean absolute error =",
      round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))

# ------------------- діагностика: чому R2 виявився близьким до нуля/від'ємним
# Перевіряємо, чи взагалі у цьому наборі є лінійна залежність між X та y.
r = np.corrcoef(X[:, 0], y)[0, 1]               # вибірковий коефіцієнт кореляції Пірсона
print("\nДІАГНОСТИКА НАБОРУ ДАНИХ ВАРІАНТА 3:")
print("  Коефіцієнт кореляції Пірсона r(X, y) = %.4f" % r)
print("  r² = %.4f  -> лінійною моделлю можна пояснити лише %.1f%% дисперсії y"
      % (r ** 2, 100 * r ** 2))

# Модель, навчена на ВСЬОМУ наборі (найкраща можлива пряма для цих даних)
full = linear_model.LinearRegression().fit(X, y)    # підгонка по всіх 60 зразках
print("  Найкраща пряма по всьому набору: y = %.4f·x %+.4f, R² = %.4f"
      % (full.coef_[0], full.intercept_, full.score(X, y)))

# Контроль: чи не є від'ємний R² артефактом послідовного розбиття 80/20?
from sklearn.model_selection import train_test_split, cross_val_score
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, test_size=0.2, random_state=0)
rnd = linear_model.LinearRegression().fit(Xtr2, ytr2)   # модель на випадковому розбитті
print("  Випадкове розбиття (random_state=0): R² на тесті = %.4f"
      % sm.r2_score(yte2, rnd.predict(Xte2)))
cv = cross_val_score(linear_model.LinearRegression(), X, y, cv=5, scoring="r2")
print("  5-блокова перехресна перевірка R²: %s, середнє = %.4f"
      % (np.round(cv, 3), cv.mean()))
print("  ВИСНОВОК: у файлі data_regr_3.txt лінійний зв'язок практично відсутній,")
print("  тому низький/від'ємний R² — це властивість самих даних, а не помилка коду.")

print("\nРисунок збережено: fig4_task2_regr3.png")
