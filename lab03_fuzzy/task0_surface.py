# -*- coding: utf-8 -*-
"""
Лабораторна робота №3. Приклад-розминка (Завдання 1 «приклад» з методички).

Апроксимація поверхні y = (x1^2 - 8) * cos(x2) на області x1, x2 in [0, 4]
системою нечіткого виводу (СНВ) типу Мамдані.

Структура СНВ (точно за кроками 7-16 методички):
  вхід  x1 : діапазон [0, 4],  3 терми (L, A, H)            — трикутні ФН;
  вхід  x2 : діапазон [0, 4],  5 термів (L, LA, A, HA, H)    — гаусові ФН;
  вихід y  : діапазон [-10, 10], 5 термів (L, LA, A, HA, H)  — трикутні ФН.

Запуск:
    /Users/alex/.venvs/systems-ai/bin/python task0_surface.py

Автор: Камінський Олексій Дмитрович, ВТ-23-2, № у списку 3.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # обов'язково до імпорту pyplot: рендеримо тільки у файли
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from fuzzy_utils import (matlab_gauss_params, matlab_tri_params, plot_mfs,
                         plot_surface, safe_defuzz, savefig, surface,
                         write_fis_txt)

HERE = os.path.dirname(os.path.abspath(__file__))

# Сітка 15x15 — рівно як у М-файлі методички (n = 15).
N = 15
X1_LOW, X1_HIGH = 0.0, 4.0
X2_LOW, X2_HIGH = 0.0, 4.0
Y_LOW, Y_HIGH = -10.0, 10.0


def target_function(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Точна залежність y = (x1^2 - 8) * cos(x2), повернута як матриця [x2, x1]."""
    y = np.zeros((len(x2), len(x1)))
    for j in range(len(x2)):
        y[j, :] = (x1 ** 2 - 8.0) * np.cos(x2[j])
    return y


def build_fis():
    """Зібрати систему нечіткого виводу Мамдані за кроками 7-16 методички."""
    # --- універсуми (крок дискретизації достатній для гладкої дефазифікації)
    u_x1 = np.arange(X1_LOW, X1_HIGH + 0.01, 0.01)
    u_x2 = np.arange(X2_LOW, X2_HIGH + 0.01, 0.01)
    u_y = np.arange(Y_LOW, Y_HIGH + 0.02, 0.02)

    x1 = ctrl.Antecedent(u_x1, "x1")
    x2 = ctrl.Antecedent(u_x2, "x2")
    y = ctrl.Consequent(u_y, "y")
    # Мамдані: усічення (min) + об'єднання (max) + центроїд — як у MATLAB
    y.defuzzify_method = "centroid"

    # --- крок 9-10: x1, 3 трикутні терми L / A / H
    tri_x1 = matlab_tri_params(X1_LOW, X1_HIGH, 3)
    for label, prm in zip(("L", "A", "H"), tri_x1):
        x1[label] = fuzz.trimf(u_x1, prm)

    # --- крок 11-12: x2, 5 гаусових термів L / LA / A / HA / H
    gauss_x2 = matlab_gauss_params(X2_LOW, X2_HIGH, 5)
    for label, (center, sigma) in zip(("L", "LA", "A", "HA", "H"), gauss_x2):
        x2[label] = fuzz.gaussmf(u_x2, center, sigma)

    # --- крок 13-14: y, 5 трикутних термів L / LA / A / HA / H
    tri_y = matlab_tri_params(Y_LOW, Y_HIGH, 5)
    for label, prm in zip(("L", "LA", "A", "HA", "H"), tri_y):
        y[label] = fuzz.trimf(u_y, prm)

    # --- крок 16: база знань. У методичці сказано «десять правил»,
    # але фактично перелічено дев'ять (див. розділ «Зауваження до методички»).
    rules = [
        ctrl.Rule(x1["L"] & x2["L"], y["L"], label="R1"),
        ctrl.Rule(x1["L"] & x2["H"], y["A"], label="R2"),
        ctrl.Rule(x1["L"] & x2["HA"], y["H"], label="R3"),
        ctrl.Rule(x1["H"] & x2["L"], y["HA"], label="R4"),
        ctrl.Rule(x1["H"] & x2["H"], y["L"], label="R5"),
        ctrl.Rule(x1["A"] & x2["A"], y["A"], label="R6"),
        ctrl.Rule(x1["A"] & x2["HA"], y["HA"], label="R7"),
        ctrl.Rule(x1["L"] & x2["LA"], y["LA"], label="R8"),
        # Правило 9 має той самий антецедент, що й правило 7, але інший
        # консеквент — методичка суперечить сама собі, залишаємо як у тексті.
        ctrl.Rule(x1["A"] & x2["HA"], y["A"], label="R9"),
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system, flush_after_run=N * N + 10)
    return x1, x2, y, sim, len(rules)


def main() -> None:
    print("=" * 70)
    print("ЗАВДАННЯ 0 (приклад). Апроксимація поверхні (x1^2-8)*cos(x2)")
    print("=" * 70)

    x1_grid = np.linspace(X1_LOW, X1_HIGH, N)
    x2_grid = np.linspace(X2_LOW, X2_HIGH, N)

    # ----------------------------------------------------------- точна поверхня
    y_exact = target_function(x1_grid, x2_grid)
    print(f"\nТочна функція на сітці {N}x{N}:")
    print(f"  min(y) = {y_exact.min():.4f}, max(y) = {y_exact.max():.4f}")

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    plot_surface(ax, x1_grid, x2_grid, y_exact, "x1", "x2", "y",
                 "Target: y = (x1² − 8)·cos(x2)")
    ax.view_init(elev=28, azim=-135)
    savefig(fig, os.path.join(HERE, "fig1_target_surface.png"))

    # ------------------------------------------------------------- побудова СНВ
    x1, x2, y, sim, n_rules = build_fis()
    print(f"\nСистему Мамдані зібрано: 2 входи, 1 вихід, правил — {n_rules}")
    print("  x1: 3 трикутні терми L/A/H на [0, 4]")
    print("  x2: 5 гаусових термів L/LA/A/HA/H на [0, 4]")
    print("  y : 5 трикутних термів L/LA/A/HA/H на [-10, 10]")

    # ----------------------------------------------------- функції належності
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    plot_mfs(axes[0], x1, "Вхід x1: 3 трикутні терми", "x1")
    plot_mfs(axes[1], x2, "Вхід x2: 5 гаусових термів", "x2")
    plot_mfs(axes[2], y, "Вихід y: 5 трикутних термів", "y")
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "fig2_task0_mfs.png"))

    # ------------------------------------------------- поверхня нечіткої системи
    fallback = {"y": 0.0}  # середина діапазону виходу, як у MATLAB
    y_fuzzy = surface(sim, "x1", x1_grid, "x2", x2_grid, "y", fallback)
    print(f"\nПоверхня СНВ на сітці {N}x{N}:")
    print(f"  min(y) = {y_fuzzy.min():.4f}, max(y) = {y_fuzzy.max():.4f}")

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    plot_surface(ax, x1_grid, x2_grid, y_fuzzy, "x1", "x2", "y",
                 "Поверхня «входи-вихід» СНВ (Мамдані, 9 правил)",
                 cmap="plasma")
    ax.view_init(elev=28, azim=-135)
    savefig(fig, os.path.join(HERE, "fig3_task0_fis_surface.png"))

    # ------------------------------------------------------------- порівняння
    err = y_fuzzy - y_exact
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    max_abs = float(np.abs(err).max())
    span = float(y_exact.max() - y_exact.min())
    print("\nПорівняння СНВ з точною поверхнею:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  max|похибка| = {max_abs:.4f}")
    print(f"  розмах точної функції = {span:.4f}")
    print(f"  відносна RMSE = {100.0 * rmse / span:.2f} % від розмаху")

    fig = plt.figure(figsize=(16, 4.6))
    ax = fig.add_subplot(131, projection="3d")
    plot_surface(ax, x1_grid, x2_grid, y_exact, "x1", "x2", "y", "Точна функція")
    ax.view_init(elev=28, azim=-135)
    ax.set_zlim(-12, 12)
    ax = fig.add_subplot(132, projection="3d")
    plot_surface(ax, x1_grid, x2_grid, y_fuzzy, "x1", "x2", "y",
                 "Нечітка система", cmap="plasma")
    ax.view_init(elev=28, azim=-135)
    ax.set_zlim(-12, 12)
    ax = fig.add_subplot(133, projection="3d")
    plot_surface(ax, x1_grid, x2_grid, err, "x1", "x2", "Δy",
                 f"Похибка (RMSE = {rmse:.2f})", cmap="coolwarm")
    ax.view_init(elev=28, azim=-135)
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "fig4_task0_compare.png"))

    # ------------------------------------------- контрольні точки (View rules)
    print("\nПеревірка у контрольних точках (аналог вікна View rules...):")
    print(f"  {'x1':>5} {'x2':>5} | {'y (СНВ)':>10} {'y (точне)':>10} {'Δ':>8}")
    probes = [(0.0, 0.0), (1.0, 0.5), (2.0, 2.0), (2.0, 3.0),
              (4.0, 0.0), (4.0, 4.0), (3.0, 1.0)]
    for px1, px2 in probes:
        got = safe_defuzz(sim, {"x1": px1, "x2": px2}, ["y"], fallback)["y"]
        exact = (px1 ** 2 - 8.0) * np.cos(px2)
        print(f"  {px1:5.2f} {px2:5.2f} | {got:10.4f} {exact:10.4f} "
              f"{got - exact:8.4f}")

    # ----------------------------------------------------- експорт конфігурації
    tri_x1 = matlab_tri_params(X1_LOW, X1_HIGH, 3)
    gauss_x2 = matlab_gauss_params(X2_LOW, X2_HIGH, 5)
    tri_y = matlab_tri_params(Y_LOW, Y_HIGH, 5)
    write_fis_txt(
        os.path.join(HERE, "task0_surface.fis.txt"),
        name="first",
        description=(
            "Апроксимація поверхні y = (x1^2 - 8)*cos(x2) на x1,x2 in [0,4].\n"
            "Відтворює СНВ з кроків 2-17 методички (у MATLAB система\n"
            "називається first.fis). Параметри ФН згенеровані за тим самим\n"
            "правилом, що й у Fuzzy Logic Designer при Edit -> Add MFs..."
        ),
        inputs=[
            {"name": "x1", "range": (X1_LOW, X1_HIGH),
             "comment": "3 трикутні терми: L (низький), A (середній), H (високий)",
             "mfs": [{"label": lb, "type": "trimf", "params": prm}
                     for lb, prm in zip(("L", "A", "H"), tri_x1)]},
            {"name": "x2", "range": (X2_LOW, X2_HIGH),
             "comment": "5 гаусових термів; sigma підібрана так, щоб сусідні ФН "
                        "перетиналися на рівні 0.5",
             "mfs": [{"label": lb, "type": "gaussmf", "params": (sg, c)}
                     for lb, (c, sg) in zip(("L", "LA", "A", "HA", "H"), gauss_x2)]},
        ],
        outputs=[
            {"name": "y", "range": (Y_LOW, Y_HIGH),
             "comment": "5 трикутних термів L/LA/A/HA/H",
             "mfs": [{"label": lb, "type": "trimf", "params": prm}
                     for lb, prm in zip(("L", "LA", "A", "HA", "H"), tri_y)]},
        ],
        rules=[
            "x1=L  and x2=L  => y=L   (1)",
            "x1=L  and x2=H  => y=A   (1)",
            "x1=L  and x2=HA => y=H   (1)",
            "x1=H  and x2=L  => y=HA  (1)",
            "x1=H  and x2=H  => y=L   (1)",
            "x1=A  and x2=A  => y=A   (1)",
            "x1=A  and x2=HA => y=HA  (1)",
            "x1=L  and x2=LA => y=LA  (1)",
            "x1=A  and x2=HA => y=A   (1)   # той самий антецедент, що й у правилі 7",
        ],
        notes=[
            "Методичка пише «сформуємо наступні десять правил», але перелічує 9; "
            "підпис до рис. 5 теж говорить про «усі 9 правил».",
            "Правила 7 і 9 мають однаковий антецедент (x1=A and x2=HA) та різні "
            "консеквенти (HA і A) — у Мамдані це дає усереднений результат.",
            "База правил покриває лише 8 із 15 можливих комбінацій термів, тому "
            "поверхня СНВ лише грубо повторює форму точної функції.",
        ],
    )

    print("\nГотово: завдання 0 виконано.")


if __name__ == "__main__":
    main()
