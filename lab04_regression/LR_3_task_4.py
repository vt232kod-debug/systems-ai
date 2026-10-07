# -*- coding: utf-8 -*-
"""
ЛР №4. Частина Б, Завдання 2.4. Регресія багатьох змінних на наборі diabetes.
Набір sklearn.datasets.load_diabetes(): 442 екземпляри, 10 ознак.
Камінський Олексій Дмитрович, ВТ-23-2.
"""

import matplotlib                               # backend до pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # графіки
import numpy as np                              # масиви
from sklearn import datasets, linear_model      # вбудовані набори та лінійні моделі
from sklearn.metrics import mean_squared_error, r2_score     # метрики MSE та R²
from sklearn.metrics import mean_absolute_error              # метрика MAE
from sklearn.model_selection import train_test_split         # розбиття вибірки

# Завантаження вбудованого набору даних про діабет
diabetes = datasets.load_diabetes()
X = diabetes.data                               # матриця ознак 442x10
y = diabetes.target                             # цільова змінна — прогресування хвороби

print("=" * 70)
print("ЗАВДАННЯ 2.4. ЛІНІЙНА РЕГРЕСІЯ НА НАБОРІ DIABETES")
print("=" * 70)
print("Розмір набору: X %s, y %s" % (X.shape, y.shape))
print("Назви ознак: %s" % ", ".join(diabetes.feature_names))
print("Цільова змінна y: min = %.1f, max = %.1f, середнє = %.2f"
      % (y.min(), y.max(), y.mean()))

# Поділ на навчальну та тестову вибірки з параметрами з методички
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)
print("Навчальна вибірка: %d зразків, тестова: %d зразків" % (len(Xtrain), len(Xtest)))

# Створення та навчання моделі лінійної регресії
regr = linear_model.LinearRegression()          # об'єкт лінійного регресора
regr.fit(Xtrain, ytrain)                        # навчання методом найменших квадратів

# Прогноз по тестовій вибірці
ypred = regr.predict(Xtest)

# ------------------------------------------- коефіцієнти регресії з підписами
print("\nКоефіцієнти регресії (regr.coef_) по ознаках:")
for name, c in zip(diabetes.feature_names, regr.coef_):     # кожна ознака і її вага
    print("   %-6s : %12.4f" % (name, c))
print("Вільний член (regr.intercept_) = %.4f" % regr.intercept_)

# ------------------------------------------------------------- метрики якості
r2 = r2_score(ytest, ypred)                     # коефіцієнт детермінації
mae = mean_absolute_error(ytest, ypred)         # середня абсолютна помилка
mse = mean_squared_error(ytest, ypred)          # середньоквадратична помилка

print("\nПоказники якості на тестовій вибірці:")
print("   R2 score (коефіцієнт кореляції R²) = %.4f" % r2)
print("   Mean absolute error (MAE)          = %.4f" % mae)
print("   Mean squared error (MSE)           = %.4f" % mse)
print("   RMSE = sqrt(MSE)                   = %.4f" % np.sqrt(mse))

# -------------------------------------------- графік "Виміряно / Передбачено"
# Це НЕ залежність від однієї ознаки: по осі x — істинні значення y,
# по осі y — передбачені. Пряма y = x показує ідеальний прогноз.
fig, ax = plt.subplots(figsize=(7.5, 7))        # квадратне полотно
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), alpha=0.75, s=45)   # точки прогнозів
ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4,       # лінія y = x
        label="Ідеальний прогноз (y = x)")
ax.set_xlabel("Виміряно")                       # підпис осі абсцис (за методичкою)
ax.set_ylabel("Передбачено")                    # підпис осі ординат (за методичкою)
ax.set_title("Завдання 2.4: diabetes, виміряно проти передбаченого\n"
             "R² = %.4f, MAE = %.2f, MSE = %.2f" % (r2, mae, mse))
ax.grid(True, linestyle="--", alpha=0.5)        # сітка
ax.legend()                                     # легенда
fig.tight_layout()
fig.savefig("fig6_task4_diabetes.png", dpi=150)    # збереження рисунка
plt.close(fig)

print("\nРисунок збережено: fig6_task4_diabetes.png")
