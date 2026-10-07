# -*- coding: utf-8 -*-
"""
Лабораторна робота №3. Завдання 1.
Нечітка модель системи керування кранами гарячої і холодної води змішувача.

Вхідні лінгвістичні змінні:
    t  — температура води на виході змішувача, [0, 100] °C,
         5 гаусових термів: cold / cool / warm / notveryhot / hot;
    p  — напір (витрата) води, [0, 100] % від максимуму,
         3 трикутні терми: weak / notverystrong / strong.

Вихідні лінгвістичні змінні (кут повороту крана, [-90, 90]°, вправо = «+»):
    hot_angle  — кут повороту крана гарячої води;
    cold_angle — кут повороту крана холодної води.
    По 7 трикутних термів: BL / ML / SL / Z / SR / MR / BR.

Запуск:
    /Users/alex/.venvs/systems-ai/bin/python task1_water_mixer.py

Автор: Камінський Олексій Дмитрович, ВТ-23-2, № у списку 3.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from fuzzy_utils import (matlab_gauss_params, matlab_tri_params, plot_mfs,
                         plot_surface, safe_defuzz, savefig, write_fis_txt)

HERE = os.path.dirname(os.path.abspath(__file__))

T_LOW, T_HIGH = 0.0, 100.0      # температура води, °C
P_LOW, P_HIGH = 0.0, 100.0      # напір, % від максимальної витрати
A_LOW, A_HIGH = -90.0, 90.0     # кут повороту крана, градуси (з умови задачі)

T_LABELS = ("cold", "cool", "warm", "notveryhot", "hot")
T_UKR = {"cold": "холодна", "cool": "прохолодна", "warm": "тепла",
         "notveryhot": "не дуже гаряча", "hot": "гаряча"}
P_LABELS = ("weak", "notverystrong", "strong")
P_UKR = {"weak": "слабий", "notverystrong": "не дуже сильний", "strong": "сильний"}
A_LABELS = ("BL", "ML", "SL", "Z", "SR", "MR", "BR")
A_UKR = {"BL": "великий кут вліво", "ML": "середній кут вліво",
         "SL": "невеликий кут вліво", "Z": "залишити без змін",
         "SR": "невеликий кут вправо", "MR": "середній кут вправо",
         "BR": "великий кут вправо"}

FALLBACK = {"hot_angle": 0.0, "cold_angle": 0.0}


def build_fis():
    """Зібрати СНВ Мамдані з 11 евристичними правилами методички."""
    u_t = np.arange(T_LOW, T_HIGH + 0.5, 0.5)
    u_p = np.arange(P_LOW, P_HIGH + 0.5, 0.5)
    u_a = np.arange(A_LOW, A_HIGH + 0.5, 0.5)

    t = ctrl.Antecedent(u_t, "t")
    p = ctrl.Antecedent(u_p, "p")
    hot = ctrl.Consequent(u_a, "hot_angle")
    cold = ctrl.Consequent(u_a, "cold_angle")
    hot.defuzzify_method = "centroid"
    cold.defuzzify_method = "centroid"

    # Температура — гаусові ФН: температура змінюється плавно, різких меж між
    # «прохолодною» і «теплою» водою не існує, тому гладкий терм адекватніший.
    for label, (center, sigma) in zip(T_LABELS,
                                      matlab_gauss_params(T_LOW, T_HIGH, 5)):
        t[label] = fuzz.gaussmf(u_t, center, sigma)

    # Напір — трикутні ФН: 3 терми, як для x1 у прикладі методички.
    for label, prm in zip(P_LABELS, matlab_tri_params(P_LOW, P_HIGH, 3)):
        p[label] = fuzz.trimf(u_p, prm)

    # Кути кранів — 7 симетричних трикутних термів з центрами
    # -90 / -60 / -30 / 0 / +30 / +60 / +90, тобто «великий», «середній»
    # і «невеликий» кут у кожну сторону плюс терм «залишити без змін».
    tri_a = matlab_tri_params(A_LOW, A_HIGH, 7)
    for label, prm in zip(A_LABELS, tri_a):
        hot[label] = fuzz.trimf(u_a, prm)
        cold[label] = fuzz.trimf(u_a, prm)

    # ------------------------------------------------------------- база правил
    # Там, де методичка згадує лише один кран, другий трактуємо як Z
    # («залишити в своєму положенні») — див. «Зауваження до методички».
    rules = [
        ctrl.Rule(t["hot"] & p["strong"],
                  [hot["ML"], cold["MR"]], label="R1"),
        ctrl.Rule(t["hot"] & p["notverystrong"],
                  [hot["Z"], cold["MR"]], label="R2"),
        ctrl.Rule(t["notveryhot"] & p["strong"],
                  [hot["SL"], cold["Z"]], label="R3"),
        ctrl.Rule(t["notveryhot"] & p["weak"],
                  [hot["SR"], cold["SR"]], label="R4"),
        ctrl.Rule(t["warm"] & p["notverystrong"],
                  [hot["Z"], cold["Z"]], label="R5"),
        ctrl.Rule(t["cool"] & p["strong"],
                  [hot["MR"], cold["ML"]], label="R6"),
        ctrl.Rule(t["cool"] & p["notverystrong"],
                  [hot["MR"], cold["SL"]], label="R7"),
        ctrl.Rule(t["cold"] & p["weak"],
                  [hot["BR"], cold["Z"]], label="R8"),
        # Правило 9 реалізоване буквально за текстом методички, хоч воно і
        # суперечить фізиці процесу (холодна вода -> зменшити гарячу).
        ctrl.Rule(t["cold"] & p["strong"],
                  [hot["ML"], cold["MR"]], label="R9"),
        ctrl.Rule(t["warm"] & p["strong"],
                  [hot["SL"], cold["SL"]], label="R10"),
        ctrl.Rule(t["warm"] & p["weak"],
                  [hot["SR"], cold["SR"]], label="R11"),
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system, flush_after_run=4000)
    return t, p, hot, cold, sim, len(rules)


def two_output_surface(sim, t_vals, p_vals):
    """Поверхні «входи-виходи» для обох кранів одночасно."""
    z_hot = np.zeros((len(p_vals), len(t_vals)))
    z_cold = np.zeros((len(p_vals), len(t_vals)))
    for j, pv in enumerate(p_vals):
        for i, tv in enumerate(t_vals):
            res = safe_defuzz(sim, {"t": tv, "p": pv},
                              ["hot_angle", "cold_angle"], FALLBACK)
            z_hot[j, i] = res["hot_angle"]
            z_cold[j, i] = res["cold_angle"]
    return z_hot, z_cold


def describe(angle: float) -> str:
    """Словесна інтерпретація дефазифікованого кута."""
    a = abs(angle)
    side = "вправо (більше потоку)" if angle > 0 else "вліво (менше потоку)"
    if a < 7.5:
        return "практично без змін"
    if a < 22.5:
        return f"трохи {side}"
    if a < 45.0:
        return f"невеликий кут {side}"
    if a < 70.0:
        return f"середній кут {side}"
    return f"великий кут {side}"


def main() -> None:
    print("=" * 74)
    print("ЗАВДАННЯ 1. Нечітке керування кранами гарячої/холодної води")
    print("=" * 74)

    t, p, hot, cold, sim, n_rules = build_fis()
    print(f"\nСистема Мамдані: 2 входи, 2 виходи, правил — {n_rules}")
    sigma = matlab_gauss_params(T_LOW, T_HIGH, 5)[0][1]
    print(f"  t: 5 гаусових термів, центри 0/25/50/75/100 °C, sigma = {sigma:.3f}")
    print("  p: 3 трикутні терми, центри 0/50/100 %")
    print("  кути: 7 трикутних термів, центри -90/-60/-30/0/+30/+60/+90°")

    # ----------------------------------------------------- функції належності
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    plot_mfs(axes[0], t, "Вхід «температура води» (5 гаусових термів)", "t, °C")
    plot_mfs(axes[1], p, "Вхід «напір» (3 трикутні терми)", "p, %")
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "fig5_task1_inputs.png"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    plot_mfs(axes[0], hot, "Вихід «кут крана гарячої води»", "кут, град")
    plot_mfs(axes[1], cold, "Вихід «кут крана холодної води»", "кут, град")
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "fig6_task1_outputs.png"))

    # ------------------------------------------------- поверхні «входи-виходи»
    nt, npr = 21, 21
    t_grid = np.linspace(T_LOW, T_HIGH, nt)
    p_grid = np.linspace(P_LOW, P_HIGH, npr)
    z_hot, z_cold = two_output_surface(sim, t_grid, p_grid)
    print(f"\nПоверхні виводу на сітці {nt}x{npr}:")
    print(f"  кран гарячої води : від {z_hot.min():7.2f}° до {z_hot.max():7.2f}°")
    print(f"  кран холодної води: від {z_cold.min():7.2f}° до {z_cold.max():7.2f}°")

    fig = plt.figure(figsize=(14, 5.0))
    ax = fig.add_subplot(121, projection="3d")
    plot_surface(ax, t_grid, p_grid, z_hot, "t, °C", "p, %", "кут, град",
                 "Кран ГАРЯЧОЇ води", cmap="inferno")
    ax.view_init(elev=26, azim=-130)
    ax = fig.add_subplot(122, projection="3d")
    plot_surface(ax, t_grid, p_grid, z_cold, "t, °C", "p, %", "кут, град",
                 "Кран ХОЛОДНОЇ води", cmap="winter")
    ax.view_init(elev=26, azim=-130)
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "fig7_task1_surfaces.png"))

    # ------------------------------------------------------------ перерізи
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)
    t_fine = np.linspace(T_LOW, T_HIGH, 51)
    for ax, pv, cap in zip(axes, (0.0, 50.0, 100.0),
                           ("слабий напір (p = 0 %)",
                            "не дуже сильний напір (p = 50 %)",
                            "сильний напір (p = 100 %)")):
        hv, cv = [], []
        for tv in t_fine:
            res = safe_defuzz(sim, {"t": tv, "p": pv},
                              ["hot_angle", "cold_angle"], FALLBACK)
            hv.append(res["hot_angle"])
            cv.append(res["cold_angle"])
        ax.plot(t_fine, hv, "r-", linewidth=2, label="кран гарячої")
        ax.plot(t_fine, cv, "b--", linewidth=2, label="кран холодної")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(cap, fontsize=10)
        ax.set_xlabel("t, °C")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("кут повороту, град")
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "fig8_task1_sections.png"))

    # --------------------------------------------------- перевірка на входах
    print("\nПеревірка роботи СНВ на конкретних наборах входів:")
    header = (f"  {'t,°C':>6} {'p,%':>6} | {'кран гарячої':>13} "
              f"{'кран холодної':>14} | інтерпретація")
    print(header)
    print("  " + "-" * (len(header) + 18))
    probes = [
        (95.0, 95.0, "дуже гаряча вода, сильний напір"),
        (90.0, 50.0, "гаряча вода, помірний напір"),
        (70.0, 100.0, "не дуже гаряча, сильний напір"),
        (70.0, 5.0,  "не дуже гаряча, слабий напір"),
        (50.0, 50.0, "тепла вода, помірний напір (комфорт)"),
        (50.0, 100.0, "тепла вода, сильний напір"),
        (50.0, 0.0,  "тепла вода, слабий напір"),
        (25.0, 100.0, "прохолодна, сильний напір"),
        (25.0, 50.0, "прохолодна, помірний напір"),
        (5.0, 0.0,   "холодна, слабий напір"),
        (5.0, 100.0, "холодна, сильний напір (суперечливе правило 9)"),
    ]
    for tv, pv, cap in probes:
        res = safe_defuzz(sim, {"t": tv, "p": pv},
                          ["hot_angle", "cold_angle"], FALLBACK)
        print(f"  {tv:6.1f} {pv:6.1f} | {res['hot_angle']:13.2f} "
              f"{res['cold_angle']:14.2f} | {cap}")
        print(f"  {'':6} {'':6} |   гаряча: {describe(res['hot_angle'])}; "
              f"холодна: {describe(res['cold_angle'])}")

    # ----------------------------------------------------- експорт конфігурації
    gauss_t = matlab_gauss_params(T_LOW, T_HIGH, 5)
    tri_p = matlab_tri_params(P_LOW, P_HIGH, 3)
    tri_a = matlab_tri_params(A_LOW, A_HIGH, 7)
    rules_txt = [
        "t=hot        and p=strong        => hot_angle=ML, cold_angle=MR  (1)",
        "t=hot        and p=notverystrong => hot_angle=Z,  cold_angle=MR  (1)",
        "t=notveryhot and p=strong        => hot_angle=SL, cold_angle=Z   (1)",
        "t=notveryhot and p=weak          => hot_angle=SR, cold_angle=SR  (1)",
        "t=warm       and p=notverystrong => hot_angle=Z,  cold_angle=Z   (1)",
        "t=cool       and p=strong        => hot_angle=MR, cold_angle=ML  (1)",
        "t=cool       and p=notverystrong => hot_angle=MR, cold_angle=SL  (1)",
        "t=cold       and p=weak          => hot_angle=BR, cold_angle=Z   (1)",
        "t=cold       and p=strong        => hot_angle=ML, cold_angle=MR  (1)"
        "   # буквально за методичкою, суперечить фізиці",
        "t=warm       and p=strong        => hot_angle=SL, cold_angle=SL  (1)",
        "t=warm       and p=weak          => hot_angle=SR, cold_angle=SR  (1)",
    ]
    write_fis_txt(
        os.path.join(HERE, "task1_water_mixer.fis.txt"),
        name="water_mixer",
        description=(
            "Нечітке керування кранами гарячої і холодної води змішувача.\n"
            "Кут повороту крана: [-90; 90] градусів, вправо = збільшити потік\n"
            "(діапазон задано умовою задачі). Діапазони входів і типи ФН\n"
            "методичка не задає — обрані самостійно, обґрунтування у звіті."
        ),
        inputs=[
            {"name": "t", "range": (T_LOW, T_HIGH),
             "comment": "температура води, °C; терми: "
                        + ", ".join(f"{k}={T_UKR[k]}" for k in T_LABELS),
             "mfs": [{"label": lb, "type": "gaussmf", "params": (sg, c),
                      "comment": T_UKR[lb]}
                     for lb, (c, sg) in zip(T_LABELS, gauss_t)]},
            {"name": "p", "range": (P_LOW, P_HIGH),
             "comment": "напір (витрата), % від максимуму; терми: "
                        + ", ".join(f"{k}={P_UKR[k]}" for k in P_LABELS),
             "mfs": [{"label": lb, "type": "trimf", "params": prm,
                      "comment": P_UKR[lb]}
                     for lb, prm in zip(P_LABELS, tri_p)]},
        ],
        outputs=[
            {"name": "hot_angle", "range": (A_LOW, A_HIGH),
             "comment": "кут повороту крана гарячої води, град",
             "mfs": [{"label": lb, "type": "trimf", "params": prm,
                      "comment": A_UKR[lb]}
                     for lb, prm in zip(A_LABELS, tri_a)]},
            {"name": "cold_angle", "range": (A_LOW, A_HIGH),
             "comment": "кут повороту крана холодної води, град",
             "mfs": [{"label": lb, "type": "trimf", "params": prm,
                      "comment": A_UKR[lb]}
                     for lb, prm in zip(A_LABELS, tri_a)]},
        ],
        rules=rules_txt,
        notes=[
            "Правила 2, 3, 8 методички згадують лише один кран. Другий кран у "
            "цих правилах отримує терм Z («залишити в своєму положенні») — "
            "інакше scikit-fuzzy/MATLAB не змогли б дефазифікувати цей вихід.",
            "Правило 9 («вода холодна і напір сильний -> гарячу вліво, "
            "холодну вправо») збігається з правилом 1 для гарячої води і "
            "суперечить здоровому глузду; реалізовано буквально, розбіжність "
            "описана у звіті.",
            "База правил покриває 11 із 15 комбінацій термів; не покриті "
            "(cold, notverystrong), (cool, weak), (notveryhot, notverystrong), "
            "(hot, weak).",
        ],
    )

    print("\nГотово: завдання 1 виконано.")


if __name__ == "__main__":
    main()
