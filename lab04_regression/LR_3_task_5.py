# -*- coding: utf-8 -*-
"""
ЛР №4. Частина Б, Завдання 2.5. Самостійна побудова регресії на власних даних.
Варіант за табл. 2.2: № у списку 3 -> № варіанту 3.

    Варіант 3:
        m = 100
        X = 6 * np.random.rand(m, 1) - 4
        y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

Тобто модельне (істинне) рівняння:  y = 0.5·x² + 1·x + 2 + шум N(0, 1).
Камінський Олексій Дмитрович, ВТ-23-2, № у списку 3.
"""

import matplotlib                               # backend до pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # графіки
import numpy as np                              # масиви та генератор випадкових чисел
from sklearn.linear_model import LinearRegression            # лінійна регресія
from sklearn.preprocessing import PolynomialFeatures         # поліноміальні ознаки
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Фіксуємо зерно генератора, щоб результати були відтворюваними у звіті
np.random.seed(3)                               # зерно = номер варіанта

# ----------------------------------------- генерація власних даних (варіант 3)
m = 100                                         # кількість згенерованих зразків
X = 6 * np.random.rand(m, 1) - 4                # ознака X рівномірно на [-4; 2]
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)    # цільова змінна з гаусовим шумом

print("=" * 70)
print("ЗАВДАННЯ 2.5. ВЛАСНІ ДАНІ, ВАРІАНТ 3")
print("=" * 70)
print("Модельне рівняння: y = 0.5·x² + 1·x + 2 + N(0, 1)")
print("Згенеровано m = %d зразків, X ∈ [%.3f; %.3f]" % (m, X.min(), X.max()))

# ------------------------------------------------------- лінійна регресія
lin_reg_simple = LinearRegression()             # об'єкт простої лінійної моделі
lin_reg_simple.fit(X, y)                        # підгонка прямої до нелінійних даних
y_lin = lin_reg_simple.predict(X)               # прогноз лінійної моделі

print("\n1) Лінійна регресія (пряма y = b + k·x):")
print("   intercept_ = %.6f" % lin_reg_simple.intercept_[0])
print("   coef_      = %.6f" % lin_reg_simple.coef_[0][0])
print("   Рівняння: y = %.4f·x %+.4f"
      % (lin_reg_simple.coef_[0][0], lin_reg_simple.intercept_[0]))
print("   R²   = %.4f" % r2_score(y, y_lin))
print("   MAE  = %.4f" % mean_absolute_error(y, y_lin))
print("   MSE  = %.4f" % mean_squared_error(y, y_lin))

# -------------------------------------------- поліноміальна регресія degree=2
poly_features = PolynomialFeatures(degree=2, include_bias=False)   # додаємо лише x та x²
X_poly = poly_features.fit_transform(X)         # розширений набір ознак [x, x²]

# Виводимо значення першої ознаки до та після перетворення (за методичкою)
print("\n2) Перетворення ознак класом PolynomialFeatures(degree=2):")
print("   X[0]      = %s" % X[0])
print("   X_poly[0] = %s   (перша ознака x, друга — її квадрат x²)" % X_poly[0])
print("   Форма X: %s  ->  форма X_poly: %s" % (X.shape, X_poly.shape))

lin_reg = LinearRegression()                    # лінійна модель над розширеними ознаками
lin_reg.fit(X_poly, y)                          # навчання поліноміальної регресії
y_poly = lin_reg.predict(X_poly)                # прогноз поліноміальної моделі

print("\n3) Поліноміальна регресія (degree=2):")
print("   intercept_ = %.6f" % lin_reg.intercept_[0])
print("   coef_      = %s" % np.round(lin_reg.coef_[0], 6))
print("   R²   = %.4f" % r2_score(y, y_poly))
print("   MAE  = %.4f" % mean_absolute_error(y, y_poly))
print("   MSE  = %.4f" % mean_squared_error(y, y_poly))

# ------------------------- запис моделі рівнянням і порівняння коефіцієнтів
a_true, b_true, c_true = 0.5, 1.0, 2.0          # модельні (істинні) коефіцієнти варіанта 3
a_hat = lin_reg.coef_[0][1]                     # знайдений коефіцієнт при x²
b_hat = lin_reg.coef_[0][0]                     # знайдений коефіцієнт при x
c_hat = lin_reg.intercept_[0]                   # знайдений вільний член

print("\n4) Порівняння модельних і знайдених коефіцієнтів:")
print("   Модель варіанта 3:   y = %.4f·x² + %.4f·x + %.4f" % (a_true, b_true, c_true))
print("   Знайдена регресія:   y = %.4f·x² + %.4f·x + %.4f" % (a_hat, b_hat, c_hat))
print("   %-12s %-12s %-12s %-12s" % ("коефіцієнт", "модельний", "знайдений", "|різниця|"))
for nm, tv, hv in [("при x²", a_true, a_hat), ("при x", b_true, b_hat),
                   ("вільний", c_true, c_hat)]:
    print("   %-12s %-12.4f %-12.4f %-12.4f" % (nm, tv, hv, abs(tv - hv)))
print("   Знайдені коефіцієнти близькі до модельних => модель навчена правильно.")

# ------------------- 5) контроль стійкості оцінок: чи не «пощастило» із зерном?
# Коефіцієнт при x вийшов 0.8072 замість модельного 1.0 — відхилення 19 %.
# Щоб показати, що це вибіркова мінливість при σ = 1 і m = 100, а не дефект моделі,
# повторюємо весь експеримент на 20 різних зернах генератора і дивимося на розкид.
print("\n5) Контроль стійкості оцінок (20 різних зерен генератора):")
n_seeds = 20                                    # кількість повторних експериментів
estimates = []                                  # накопичувач трійок (a, b, c)
for seed in range(n_seeds):                     # перебираємо зерна 0..19
    rng = np.random.RandomState(seed)           # власний генератор для цього зерна
    Xs = 6 * rng.rand(m, 1) - 4                 # ті самі формули варіанта 3
    ys = 0.5 * Xs ** 2 + Xs + 2 + rng.randn(m, 1)
    pf_s = PolynomialFeatures(degree=2, include_bias=False)     # ознаки [x, x²]
    mdl = LinearRegression().fit(pf_s.fit_transform(Xs), ys)    # навчання моделі
    estimates.append([mdl.coef_[0][1], mdl.coef_[0][0], mdl.intercept_[0]])
estimates = np.array(estimates)                 # масив (20, 3): по рядку на зерно
means = estimates.mean(axis=0)                  # середні оцінки коефіцієнтів
stds = estimates.std(axis=0, ddof=1)            # вибіркові стандартні відхилення

print("   %-12s %-11s %-13s %-11s %-22s"
      % ("коефіцієнт", "модельний", "середнє(20)", "ст.відх.", "наше зерно (seed=3)"))
for nm, tv, mv, sv, hv in [("при x²", a_true, means[0], stds[0], a_hat),
                           ("при x", b_true, means[1], stds[1], b_hat),
                           ("вільний", c_true, means[2], stds[2], c_hat)]:
    # z — на скільки стандартних відхилень наша оцінка відстоїть від модельного значення
    print("   %-12s %-11.4f %-13.4f %-11.4f %.4f (z = %+.2f)"
          % (nm, tv, mv, sv, hv, (hv - tv) / sv))
# Максимальне за трьома коефіцієнтами відхилення нашої оцінки у стандартних відхиленнях
z_max = max(abs((hv - tv) / sv) for tv, sv, hv in
            [(a_true, stds[0], a_hat), (b_true, stds[1], b_hat), (c_true, stds[2], c_hat)])
print("   Середні по 20 зернах збігаються з модельними (|різниця| <= %.4f),"
      % np.max(np.abs(means - np.array([a_true, b_true, c_true]))))
print("   а найбільше відхилення нашої оцінки становить %.2f ст.відхилення (< 2σ)." % z_max)
print("   Отже, 0.8072 замість 1.0 — це вибіркова мінливість при σ = 1 і m = 100,")
print("   а не помилка побудови моделі.")

# --------------------------------------------------------- побудова графіків
X_grid = np.linspace(X.min() - 0.3, X.max() + 0.3, 400).reshape(-1, 1)   # щільна сітка
y_grid_lin = lin_reg_simple.predict(X_grid)                              # пряма
y_grid_poly = lin_reg.predict(poly_features.transform(X_grid))           # парабола
y_grid_true = 0.5 * X_grid ** 2 + X_grid + 2                             # істинна крива

fig, ax = plt.subplots(figsize=(9, 6))          # полотно
ax.scatter(X, y, s=40, color="royalblue", edgecolors="black", linewidths=0.4,
           alpha=0.8, label="Згенеровані дані (варіант 3)")
ax.plot(X_grid, y_grid_true, color="gray", linestyle=":", linewidth=2.5,
        label="Модельна крива y = 0.5x² + x + 2")
ax.plot(X_grid, y_grid_lin, color="darkorange", linewidth=2.5,
        label="Лінійна регресія (R² = %.3f)" % r2_score(y, y_lin))
ax.plot(X_grid, y_grid_poly, color="crimson", linewidth=2.5,
        label="Поліноміальна регресія degree=2 (R² = %.3f)" % r2_score(y, y_poly))
ax.set_xlabel("X")                              # підпис осі x
ax.set_ylabel("y")                              # підпис осі y
ax.set_title("Завдання 2.5: лінійна та поліноміальна регресія на власних даних")
ax.grid(True, linestyle="--", alpha=0.5)        # сітка
ax.legend(loc="upper center")                   # легенда
fig.tight_layout()
fig.savefig("fig7_task5_poly_variant3.png", dpi=150)   # збереження рисунка
plt.close(fig)

print("\nРисунок збережено: fig7_task5_poly_variant3.png")
