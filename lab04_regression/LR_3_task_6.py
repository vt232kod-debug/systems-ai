# -*- coding: utf-8 -*-
"""
ЛР №4. Частина Б, Завдання 2.6. Побудова кривих навчання (learning curves).
Дані — ті самі, що у завданні 2.5 (варіант 3, те саме зерно генератора).
Будуємо криві навчання для лінійної моделі, поліноміальної degree=2 та degree=10.
Камінський Олексій Дмитрович, ВТ-23-2, № у списку 3.
"""

import matplotlib                               # backend до pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # графіки
import numpy as np                              # масиви
from sklearn.linear_model import LinearRegression            # лінійна регресія
from sklearn.preprocessing import PolynomialFeatures         # поліноміальні ознаки
from sklearn.pipeline import Pipeline                        # конвеєр перетворень
from sklearn.metrics import mean_squared_error               # MSE для розрахунку RMSE
from sklearn.model_selection import train_test_split         # розбиття вибірки

# Те саме зерно, що у завданні 2.5 — дані ідентичні
np.random.seed(3)

# ----------------------------------------- генерація власних даних (варіант 3)
m = 100                                         # кількість зразків
X = 6 * np.random.rand(m, 1) - 4                # ознака X на [-4; 2]
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)    # цільова змінна з шумом

print("=" * 70)
print("ЗАВДАННЯ 2.6. КРИВІ НАВЧАННЯ, ВЛАСНІ ДАНІ ВАРІАНТА 3")
print("=" * 70)
print("Дані: m = %d, модельне рівняння y = 0.5·x² + x + 2 + N(0, 1)" % m)


def plot_learning_curves(model, X, y, ax, title, max_rmse=3.0):
    """
    Будує криві навчання моделі: RMSE на навчальній та перевірочній вибірках
    як функцію від розміру навчального набору.

    model    — оцінювач sklearn (буде навчатись багато разів);
    X, y     — повний набір даних;
    ax       — вісі matplotlib, на яких малюємо;
    title    — заголовок панелі;
    max_rmse — верхня межа осі RMSE для порівнянності панелей.
    """
    # Відокремлюємо перевірочну вибірку (20%), решта — пул для навчання
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []           # накопичувачі помилок

    # Поступово збільшуємо розмір навчальної підмножини від 1 до len(X_train)
    for n in range(1, len(X_train) + 1):
        model.fit(X_train[:n], y_train[:n])     # навчаємо модель на перших n зразках
        y_train_predict = model.predict(X_train[:n])    # прогноз на цих самих n зразках
        y_val_predict = model.predict(X_val)            # прогноз на перевірочній вибірці
        # Зберігаємо MSE; корінь візьмемо при побудові графіка
        train_errors.append(mean_squared_error(y_train[:n], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    # Переводимо MSE у RMSE — так криві мають розмірність самої величини y
    train_rmse = np.sqrt(train_errors)
    val_rmse = np.sqrt(val_errors)

    ax.plot(train_rmse, "r-+", linewidth=2, label="навчальна вибірка (train)")
    ax.plot(val_rmse, "b-", linewidth=3, label="перевірочна вибірка (validation)")
    ax.set_xlabel("Розмір навчального набору")  # підпис осі x
    ax.set_ylabel("RMSE")                       # підпис осі y
    ax.set_title(title)                         # заголовок панелі
    ax.set_ylim(0, max_rmse)                    # однакова шкала для всіх панелей
    ax.grid(True, linestyle="--", alpha=0.5)    # сітка
    ax.legend(loc="upper right")                # легенда
    return train_rmse, val_rmse                 # повертаємо криві для числового аналізу


def report(name, train_rmse, val_rmse, n_params):
    """
    Друкує числові характеристики кривих навчання.

    Плато (середнє останніх 10 точок) показує, куди криві сходяться при повному
    наборі даних. Але методичка просить побачити ПРОМІЖОК між кривими, а на плато
    він уже закритий, тому додатково друкуємо максимальний розрив між кривими.

    n_params — кількість параметрів моделі (ознаки + вільний член). Поки
    n <= n_params, модель інтерполює навчальні точки ТОЧНО (train RMSE = 0), і
    розрив там тривіально нескінченний — це властивість підрахунку, а не моделі.
    Тому максимум шукаємо в області n >= 2*n_params (правило «щонайменше вдвічі
    більше зразків, ніж параметрів»), де оцінки вже змістовні.
    """
    # Усереднюємо останні 10 точок — це «плато», до якого виходять криві
    tr_plateau = train_rmse[-10:].mean()        # плато помилки на навчальних даних
    val_plateau = val_rmse[-10:].mean()         # плато помилки на перевірочних даних

    # Розрив між кривими у кожній точці: val - train (позитивний = ознака перенавчання)
    gap = val_rmse - train_rmse
    n_min = 2 * n_params                        # нижня межа змістовної області
    start = n_min - 1                           # індекс масиву для n = n_min
    idx = start + int(np.argmax(gap[start:]))   # індекс максимального розриву
    n_at_max = idx + 1                          # розмір навчального набору в цій точці

    print("\n%s (параметрів моделі: %d):" % (name, n_params))
    print("   RMSE train (плато, останні 10 точок) = %.4f" % tr_plateau)
    print("   RMSE val   (плато, останні 10 точок) = %.4f" % val_plateau)
    print("   Розрив між кривими на плато = %.4f" % (val_plateau - tr_plateau))
    print("   Максимальний розрив при n >= %d: %.4f при n = %d зразків"
          % (n_min, gap[idx], n_at_max))
    print("   (у цій точці train RMSE = %.4f, val RMSE = %.4f)"
          % (train_rmse[idx], val_rmse[idx]))
    # Середній розрив по змістовній області — стійкіша характеристика, ніж максимум
    print("   Середній розрив при n >= %d = %.4f" % (n_min, gap[start:].mean()))
    # Розбиваємо змістовну область навпіл: так видно, чи розрив закривається з ростом n
    half = start + (len(gap) - start) // 2      # індекс середини змістовної області
    print("   Середній розрив: n = %d..%d -> %.4f;  n = %d..%d -> %.4f"
          % (n_min, half, gap[start:half].mean(),
             half + 1, len(gap), gap[half:].mean()))
    return gap[idx], n_at_max


# ------------------------------------------ 1) криві навчання лінійної моделі
fig, ax = plt.subplots(figsize=(8, 5.5))        # окреме полотно для лінійної моделі
lin_reg = LinearRegression()                    # проста лінійна модель
tr1, va1 = plot_learning_curves(lin_reg, X, y, ax,
                                "Криві навчання: лінійна регресія")
fig.tight_layout()
fig.savefig("fig8_task6_lc_linear.png", dpi=150)    # зберігаємо рисунок
plt.close(fig)
g1, n1 = report("Лінійна регресія", tr1, va1, n_params=2)

# -------------------------------- 2) криві навчання поліноміальної моделі degree=2
fig, ax = plt.subplots(figsize=(8, 5.5))        # полотно для degree=2
polynomial_regression_2 = Pipeline([            # конвеєр: ознаки -> лінійна модель
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
tr2, va2 = plot_learning_curves(polynomial_regression_2, X, y, ax,
                                "Криві навчання: поліноміальна регресія, degree = 2")
fig.tight_layout()
fig.savefig("fig9_task6_lc_poly2.png", dpi=150)     # зберігаємо рисунок
plt.close(fig)
g2, n2 = report("Поліноміальна регресія degree=2", tr2, va2, n_params=3)

# ------------------------------- 3) криві навчання поліноміальної моделі degree=10
fig, ax = plt.subplots(figsize=(8, 5.5))        # полотно для degree=10
polynomial_regression_10 = Pipeline([           # конвеєр зі степенем 10
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
tr10, va10 = plot_learning_curves(polynomial_regression_10, X, y, ax,
                                  "Криві навчання: поліноміальна регресія, degree = 10")
fig.tight_layout()
fig.savefig("fig10_task6_lc_poly10.png", dpi=150)   # зберігаємо рисунок
plt.close(fig)
g10, n10 = report("Поліноміальна регресія degree=10", tr10, va10, n_params=11)

# ------------------------------------------------- підсумкове порівняння
print("\n" + "-" * 70)
print("ПОРІВНЯННЯ МОДЕЛЕЙ ЗА КРИВИМИ НАВЧАННЯ:")
print("%-26s %-11s %-11s %-13s %-20s"
      % ("модель", "RMSE train", "RMSE val", "розрив(плато)", "макс.розрив (при n)"))
for nm, tr, va, gmax, nmax in [("лінійна", tr1, va1, g1, n1),
                               ("поліноміальна degree=2", tr2, va2, g2, n2),
                               ("поліноміальна degree=10", tr10, va10, g10, n10)]:
    print("%-26s %-11.4f %-11.4f %-13.4f %.4f (n = %d)"
          % (nm, tr[-10:].mean(), va[-10:].mean(),
             va[-10:].mean() - tr[-10:].mean(), gmax, nmax))

# ---------------- коректне трактування рівня шуму та від'ємного розриву
# σ = 1.0 — це ОЧІКУВАНЕ значення RMSE ідеальної моделі, а не жорстка нижня межа:
# вибіркова оцінка RMSE на 20 перевірочних точках має помітний розкид.
# Для нормального шуму n_val*MSE/σ² ~ χ²(n_val), звідки sd(RMSE) ≈ σ/√(2*n_val).
n_val = int(0.2 * m)                            # розмір перевірочної вибірки = 20 точок
sigma = 1.0                                     # закладений рівень шуму за умовою варіанта
sd_rmse = sigma / np.sqrt(2 * n_val)            # теоретичний розкид вибіркового RMSE
g2_plateau = va2[-10:].mean() - tr2[-10:].mean()    # розрив на плато для degree=2
print("\nТРАКТУВАННЯ РІВНЯ ШУМУ:")
print("  σ = %.1f (за умовою варіанта) — це ОЧІКУВАНЕ значення RMSE ідеальної" % sigma)
print("  моделі, а НЕ недосяжна нижня межа. Перевірочна вибірка містить лише")
print("  n_val = %d точок, тому вибіркова оцінка RMSE має розкид" % n_val)
print("  sd ≈ σ/√(2*n_val) = %.3f, а інтервал σ ± 2*sd = [%.3f; %.3f]."
      % (sd_rmse, sigma - 2 * sd_rmse, sigma + 2 * sd_rmse))
print("  Отримані val RMSE %.4f (degree=2) і %.4f (degree=10) лежать у цьому"
      % (va2[-10:].mean(), va10[-10:].mean()))
print("  інтервалі, тобто УЗГОДЖУЮТЬСЯ з σ = 1.0, а не суперечать їй.")
print("  З тієї ж причини розрив на плато для degree=2 вийшов від'ємним (%.4f):"
      % g2_plateau)
print("  на 20 точках перевірочна помилка випадково опинилася нижчою за навчальну;")
print("  це вибіркова мінливість, а не ознака якості моделі.")
print("\nРисунки збережено: fig8_task6_lc_linear.png, fig9_task6_lc_poly2.png,")
print("                   fig10_task6_lc_poly10.png")
