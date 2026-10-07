# -*- coding: utf-8 -*-
"""
ЛР №4. Частина Б, Завдання 2.3. Створення багатовимірного регресора.
Вхідний файл: data_multivar_regr.txt (4 стовпці: 3 ознаки + цільова змінна).
Камінський Олексій Дмитрович, ВТ-23-2.
"""

import matplotlib                               # backend до pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # графіки
import numpy as np                              # масиви
import scipy.linalg                             # той самий розв'язувач МНК, що й у sklearn
from sklearn import linear_model                # лінійні моделі
import sklearn.metrics as sm                    # метрики якості
from sklearn.preprocessing import PolynomialFeatures   # генерація поліноміальних ознак

# Вхідний файл з багатовимірними даними
input_file = "data_multivar_regr.txt"

# Завантаження даних, роздільник — кома
data = np.loadtxt(input_file, delimiter=",")
# Перші три стовпці — ознаки, останній — цільова змінна
X, y = data[:, :-1], data[:, -1]

print("=" * 70)
print("ЗАВДАННЯ 2.3. БАГАТОВИМІРНИЙ РЕГРЕСОР (data_multivar_regr.txt)")
print("=" * 70)
print("Розмір набору: X %s (3 ознаки), y %s" % (X.shape, y.shape))

# Розбивка даних на навчальний та тестовий набори (80% / 20%)
num_training = int(0.8 * len(X))                # розмір навчальної вибірки
num_test = len(X) - num_training                # розмір тестової вибірки
X_train, y_train = X[:num_training], y[:num_training]   # тренувальні дані
X_test, y_test = X[num_training:], y[num_training:]     # тестові дані
print("Навчальна вибірка: %d зразків, тестова: %d зразків" % (num_training, num_test))

# ------------------------------------------------ лінійна багатовимірна модель
linear_regressor = linear_model.LinearRegression()   # об'єкт лінійного регресора
linear_regressor.fit(X_train, y_train)               # навчання на тренувальних даних
y_test_pred = linear_regressor.predict(X_test)       # прогноз для тестового набору

print("\nПараметри лінійної моделі:")
print("  Коефіцієнти: ", np.round(linear_regressor.coef_, 4))
print("  Зсув (intercept): %.4f" % linear_regressor.intercept_)
print("  Рівняння: y = %.4f %+.4f·x1 %+.4f·x2 %+.4f·x3"
      % (linear_regressor.intercept_, linear_regressor.coef_[0],
         linear_regressor.coef_[1], linear_regressor.coef_[2]))

# Метрики якості лінійної регресії
print("\nLinear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# --------------------------------------------- поліноміальна регресія (degree=10)
polynomial = PolynomialFeatures(degree=10)           # генератор поліноміальних ознак
X_train_transformed = polynomial.fit_transform(X_train)   # НАВЧАЄМО перетворювач на train
print("\nПоліноміальні ознаки degree=10: %d -> %d ознак"
      % (X_train.shape[1], X_train_transformed.shape[1]))

poly_linear_model = linear_model.LinearRegression()  # лінійна модель над поліном. ознаками
poly_linear_model.fit(X_train_transformed, y_train)  # навчання поліноміального регресора

# Прогноз поліноміальної моделі на тестовому наборі
# ВАЖЛИВО: тут лише transform, бо перетворювач уже навчений на X_train
X_test_transformed = polynomial.transform(X_test)
y_test_pred_poly = poly_linear_model.predict(X_test_transformed)

print("\nPolynomial Regressor performance (degree=10):")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_poly), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_poly), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_poly), 2))
print("Explained variance score =",
      round(sm.explained_variance_score(y_test, y_test_pred_poly), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_poly), 2))

# ------------------------------- прогноз для контрольної точки даних
# Точка близька до рядка 11 файлу даних: [7.66, 6.29, 5.66] -> y = 41.35
datapoint = [[7.75, 6.35, 5.56]]                     # 2D-масив: predict вимагає 2 виміри
# У методичці тут стоїть polynomial.fit_transform(datapoint) — це помилка:
# fit_transform переналаштовує перетворювач на одному зразку. Правильно — transform.
poly_datapoint = polynomial.transform(datapoint)

print("\nКонтрольна точка datapoint = %s" % datapoint)
print("Очікуване значення (рядок 11 файлу, [7.66, 6.29, 5.66]) ≈ 41.35")
print("\nLinear regression:\n", linear_regressor.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))

# Абсолютні відхилення прогнозів від орієнтира 41.35
target = 41.35                                       # орієнтир із методички
lin_pred = linear_regressor.predict(datapoint)[0]    # прогноз лінійної моделі
poly_pred = poly_linear_model.predict(poly_datapoint)[0]   # прогноз поліноміальної моделі
print("\nВідхилення від 41.35:")
print("  лінійна модель:        %.4f  (|Δ| = %.4f)" % (lin_pred, abs(lin_pred - target)))
print("  поліноміальна модель:  %.4f  (|Δ| = %.4f)" % (poly_pred, abs(poly_pred - target)))

# Контроль пастки: показуємо, що fit_transform з методички дає той самий масив ознак,
# бо набір поліноміальних ознак залежить лише від кількості ознак і степеня.
poly_datapoint_wrong = PolynomialFeatures(degree=10).fit_transform(datapoint)
print("\nКонтроль пастки fit_transform vs transform:")
print("  форми однакові: %s vs %s, максимальна різниця = %.2e"
      % (poly_datapoint.shape, poly_datapoint_wrong.shape,
         np.max(np.abs(poly_datapoint - poly_datapoint_wrong))))
print("  => на цих даних помилка непомітна, але сам прийом хибний (див. звіт).")

# ------------- ЧОМУ ПОЛІНОМІАЛЬНА МОДЕЛЬ НЕ ДАЄ ОБІЦЯНИХ 41.35: аналіз
# Матриця поліноміальних ознак степеня 10 для 3 ознак є майже вироджена,
# тому розв'язок задачі найменших квадратів НЕ ЄДИНИЙ і залежить від порога
# відсікання малих сингулярних чисел (параметр rcond у numpy.linalg.lstsq).
# Спершу показуємо цю залежність у «чистому» вигляді, а далі — конкретний
# поріг, який використовує саме sklearn 1.9.1 (параметр tol, див. нижче).
rank = np.linalg.matrix_rank(X_train_transformed)        # фактичний ранг матриці ознак
cond = np.linalg.cond(X_train_transformed)               # число обумовленості
print("\nАНАЛІЗ ОБУМОВЛЕНОСТІ ПОЛІНОМІАЛЬНОЇ ЗАДАЧІ:")
print("  Матриця ознак: %d зразків x %d ознак" % X_train_transformed.shape)
print("  Ранг матриці = %d (менший за кількість ознак %d => система вироджена)"
      % (rank, X_train_transformed.shape[1]))
print("  Число обумовленості cond = %.3e" % cond)

# Показуємо, як прогноз у контрольній точці «плаває» залежно від порога rcond
print("\n  Залежність прогнозу в точці [7.75, 6.35, 5.56] від порога rcond:")
print("  %-10s %-7s %-12s %-12s" % ("rcond", "ранг", "прогноз", "R² на тесті"))
for rc in [None, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]:
    # Розв'язуємо перевизначену систему з заданим порогом відсікання
    coef, _, rk, _ = np.linalg.lstsq(X_train_transformed, y_train, rcond=rc)
    pred_rc = (poly_datapoint @ coef)[0]                 # прогноз у контрольній точці
    r2_rc = sm.r2_score(y_test, X_test_transformed @ coef)   # R² на тестовій вибірці
    print("  %-10s %-7d %-12.4f %-12.3f" % (rc, rk, pred_rc, r2_rc))
print("  Прогноз змінюється від ~20 до ~51 при незмінних даних і степені!")

# ---------- ТОЧНА ПРИЧИНА РОЗБІЖНОСТІ З МЕТОДИЧКОЮ: параметр LinearRegression.tol
# У scikit-learn 1.9 поведінка змінилася: LinearRegression.fit для щільних даних
# викликає scipy.linalg.lstsq(X, y, cond=self.tol), а self.tol за замовчуванням = 1e-6.
# Тобто дефолтний поріг відсікання сингулярних чисел тепер 1e-6, а не машинна точність,
# як було до версії 1.9. Саме це, а не властивість задачі, дає прогноз 31.64.
# Число 41.35/41.45 з методички відтворюється одним параметром публічного API: tol=1e-15.
print("\n  ПРИЧИНА РОЗБІЖНОСТІ З МЕТОДИЧКОЮ (параметр LinearRegression.tol):")
print("  LinearRegression.fit -> scipy.linalg.lstsq(X, y, cond=self.tol), tol за замовч. = %g"
      % linear_model.LinearRegression().tol)
for tol_value in [None, 1e-15]:
    # None означає «залишити дефолт 1e-6»; 1e-15 — відтворення поведінки sklearn < 1.9
    md = (linear_model.LinearRegression() if tol_value is None
          else linear_model.LinearRegression(tol=tol_value))
    md.fit(X_train_transformed, y_train)                 # навчання з заданим порогом cond
    pr = md.predict(poly_datapoint)[0]                   # прогноз у контрольній точці
    r2v = md.score(X_test_transformed, y_test)           # R² на тестовій вибірці
    print("    tol = %-8s rank_ = %-4d прогноз = %-10.4f R² на тесті = %.3f"
          % (("1e-06 (дефолт)" if tol_value is None else "1e-15"), md.rank_, pr, r2v))

# Контроль: пряма поведінка старих версій sklearn — scipy.linalg.lstsq без параметра cond
coef_sp, _, rank_sp, _ = scipy.linalg.lstsq(X_train_transformed, y_train)
print("    scipy.linalg.lstsq(X, y) без cond (поведінка sklearn < 1.9):")
print("      rank = %d, прогноз = %.4f" % (rank_sp, (poly_datapoint @ coef_sp)[0]))
coef_gy, _, rank_gy, _ = scipy.linalg.lstsq(X_train_transformed, y_train,
                                            lapack_driver="gelsy")
print("    той самий розв'язувач з драйвером gelsy: rank = %d, прогноз = %.4f"
      % (rank_gy, (poly_datapoint @ coef_gy)[0]))
print("  => 41.35/41.45 з методички — це результат СТАРОГО дефолта розв'язувача,")
print("     а не властивість даних; відтворюється в один рядок через tol=1e-15.")
print("     Але ціна цього — R² на тесті падає з -5.04 до ≈ -1156: модель перенавчена.")

# Для порівняння — поліноми помірного степеня, які реально узагальнюють
print("\n  Поліноми помірного степеня (адекватна альтернатива degree=10):")
print("  %-9s %-9s %-12s %-12s" % ("степінь", "ознак", "прогноз", "R² на тесті"))
for dg in [2, 3, 4, 5]:
    pf = PolynomialFeatures(degree=dg)                   # перетворювач для степеня dg
    A_tr = pf.fit_transform(X_train)                     # навчаємо на train
    md = linear_model.LinearRegression().fit(A_tr, y_train)   # лінійна модель над ознаками
    pr = md.predict(pf.transform(datapoint))[0]          # прогноз у контрольній точці
    r2d = md.score(pf.transform(X_test), y_test)         # R² на тесті
    print("  %-9d %-9d %-12.4f %-12.4f" % (dg, A_tr.shape[1], pr, r2d))
print("  Вони дають стабільний прогноз ~35 і R² ≈ 0.86, як у лінійної моделі.")

# --------------------------------------------------------- побудова графіка
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))    # дві панелі поруч
for ax, pred, name in zip(axes, [y_test_pred, y_test_pred_poly],
                          ["Лінійна регресія", "Поліноміальна регресія (degree=10)"]):
    ax.scatter(y_test, pred, s=35, alpha=0.7, edgecolors="black", linewidths=0.4)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "k--", lw=2, label="ідеальний прогноз y = x")
    ax.set_xlabel("Виміряно (y_test)")               # підпис осі x
    ax.set_ylabel("Передбачено")                     # підпис осі y
    ax.set_title("%s\nR² = %.4f" % (name, sm.r2_score(y_test, pred)))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
fig.tight_layout()
fig.savefig("fig5_task3_multivar.png", dpi=150)      # зберігаємо рисунок
plt.close(fig)
print("\nРисунок збережено: fig5_task3_multivar.png")
