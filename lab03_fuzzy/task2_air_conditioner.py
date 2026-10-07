# -*- coding: utf-8 -*-
"""
Лабораторна робота №3. Завдання 2.
Нечітка модель керування кондиціонером повітря в приміщенні.

Особливість задачі: через теплову інертність приміщення одного значення
температури недостатньо — другим входом подається швидкість її зміни
(похідна dT/dt), яка може бути від'ємною, тому обов'язковий терм «нуль».

Вхідні лінгвістичні змінні:
    t  — температура повітря, [10, 34] °C, комфортна = 22 °C,
         5 гаусових термів: VC / C / N / W / VW;
    dt — швидкість зміни температури, [-1, 1] °C/хв,
         3 трикутні терми: NEG / ZERO / POS.

Вихідна лінгвістична змінна:
    angle — кут повороту регулятора кондиціонера, [-90, 90]°,
            вліво («−») = режим «холод», вправо («+») = режим «тепло»,
            5 трикутних термів: BL / SL / OFF / SR / BR.

Запуск:
    /Users/alex/.venvs/systems-ai/bin/python task2_air_conditioner.py

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
                         plot_surface, safe_defuzz, savefig, surface,
                         write_fis_txt)

HERE = os.path.dirname(os.path.abspath(__file__))

T_LOW, T_HIGH = 10.0, 34.0      # температура повітря, °C
T_COMFORT = 22.0                # комфортна температура — центр терму N
D_LOW, D_HIGH = -1.0, 1.0       # швидкість зміни температури, °C/хв
A_LOW, A_HIGH = -90.0, 90.0     # кут регулятора, град

T_LABELS = ("VC", "C", "N", "W", "VW")
T_UKR = {"VC": "дуже холодна", "C": "холодна", "N": "в нормі",
         "W": "тепла", "VW": "дуже тепла"}
D_LABELS = ("NEG", "ZERO", "POS")
D_UKR = {"NEG": "від'ємна (температура падає)",
         "ZERO": "нуль (температура стабільна)",
         "POS": "додатня (температура зростає)"}
A_LABELS = ("BL", "SL", "OFF", "SR", "BR")
A_UKR = {"BL": "великий кут вліво — «холод» на максимум",
         "SL": "невеликий кут вліво — «холод» помірно",
         "OFF": "кондиціонер вимкнено",
         "SR": "невеликий кут вправо — «тепло» помірно",
         "BR": "великий кут вправо — «тепло» на максимум"}

FALLBACK = {"angle": 0.0}


def build_fis():
    """Зібрати СНВ Мамдані з 15 правилами керування кондиціонером."""
    u_t = np.arange(T_LOW, T_HIGH + 0.1, 0.1)
    u_d = np.arange(D_LOW, D_HIGH + 0.01, 0.01)
    u_a = np.arange(A_LOW, A_HIGH + 0.5, 0.5)

    t = ctrl.Antecedent(u_t, "t")
    d = ctrl.Antecedent(u_d, "dt")
    a = ctrl.Consequent(u_a, "angle")
    a.defuzzify_method = "centroid"

    # Температура — гаусові ФН з центрами 10/16/22/28/34 °C: центральний терм N
    # припадає точно на комфортні 22 °C, межі термів розмиті (плавний перехід).
    for label, (center, sigma) in zip(T_LABELS,
                                      matlab_gauss_params(T_LOW, T_HIGH, 5)):
        t[label] = fuzz.gaussmf(u_t, center, sigma)

    # Похідна — трикутні ФН з центрами -1 / 0 / +1: терм ZERO дає µ = 1 рівно
    # при dT/dt = 0, що потрібно для правил 9-12 і 15.
    for label, prm in zip(D_LABELS, matlab_tri_params(D_LOW, D_HIGH, 3)):
        d[label] = fuzz.trimf(u_d, prm)

    # Кут регулятора — 5 трикутних термів з центрами -90/-45/0/+45/+90.
    for label, prm in zip(A_LABELS, matlab_tri_params(A_LOW, A_HIGH, 5)):
        a[label] = fuzz.trimf(u_a, prm)

    # ------------------------------------------------------------- база правил
    # 15 правил методички; виправлено правило 7 (у тексті «режим тепло,
    # але кут вліво» — суперечність) та дописано обірване правило 4.
    rules = [
        ctrl.Rule(t["VW"] & d["POS"], a["BL"], label="R1"),
        ctrl.Rule(t["VW"] & d["NEG"], a["SL"], label="R2"),
        ctrl.Rule(t["W"] & d["POS"], a["BL"], label="R3"),
        # Правило 4: «тепло, температура вже падає» -> кондиціонер вимкнути.
        ctrl.Rule(t["W"] & d["NEG"], a["OFF"], label="R4"),
        ctrl.Rule(t["VC"] & d["NEG"], a["BR"], label="R5"),
        ctrl.Rule(t["VC"] & d["POS"], a["SR"], label="R6"),
        # Правило 7 ВИПРАВЛЕНО: режим «тепло» = поворот ВПРАВО (BR).
        ctrl.Rule(t["C"] & d["NEG"], a["BR"], label="R7"),
        ctrl.Rule(t["C"] & d["POS"], a["OFF"], label="R8"),
        ctrl.Rule(t["VW"] & d["ZERO"], a["BL"], label="R9"),
        ctrl.Rule(t["W"] & d["ZERO"], a["SL"], label="R10"),
        ctrl.Rule(t["VC"] & d["ZERO"], a["BR"], label="R11"),
        ctrl.Rule(t["C"] & d["ZERO"], a["SR"], label="R12"),
        ctrl.Rule(t["N"] & d["POS"], a["SL"], label="R13"),
        ctrl.Rule(t["N"] & d["NEG"], a["SR"], label="R14"),
        ctrl.Rule(t["N"] & d["ZERO"], a["OFF"], label="R15"),
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system, flush_after_run=4000)
    return t, d, a, sim, len(rules)


def describe(angle: float) -> str:
    """Словесна інтерпретація кута регулятора."""
    if abs(angle) < 10.0:
        return "кондиціонер практично вимкнено"
    mode = "«холод»" if angle < 0 else "«тепло»"
    side = "вліво" if angle < 0 else "вправо"
    mag = abs(angle)
    if mag < 30.0:
        power = "слабо"
    elif mag < 60.0:
        power = "помірно"
    else:
        power = "на максимум"
    return f"режим {mode} {power} (кут {side})"


def closed_loop(sim, t0: float, t_out: float, minutes: float = 180.0,
                step: float = 0.5, tau: float = 3.0,
                gain: float = 0.5, leak: float = 0.02):
    """Замкнутий контур: приміщення + інертний кондиціонер + нечіткий регулятор.

    Модель приміщення:  dT/dt = leak*(t_out - T) + gain*q,
    де q — нормований тепловий потік кондиціонера, що наздоганяє положення
    регулятора з інерцією tau (саме та інертність, про яку пише методичка).
    """
    n = int(minutes / step)
    temp = np.zeros(n + 1)
    deriv = np.zeros(n + 1)
    angles = np.zeros(n + 1)
    time = np.arange(n + 1) * step
    temp[0] = t0
    q = 0.0
    for k in range(n + 1):
        t_in = float(np.clip(temp[k], T_LOW, T_HIGH))
        d_in = float(np.clip(deriv[k], D_LOW, D_HIGH))
        angles[k] = safe_defuzz(sim, {"t": t_in, "dt": d_in},
                                ["angle"], FALLBACK)["angle"]
        if k == n:
            break
        q += step / tau * (angles[k] / 90.0 - q)       # інерція кондиціонера
        rate = leak * (t_out - temp[k]) + gain * q     # швидкість зміни, °C/хв
        temp[k + 1] = temp[k] + step * rate
        deriv[k + 1] = rate
    return time, temp, deriv, angles


def main() -> None:
    print("=" * 74)
    print("ЗАВДАННЯ 2. Нечітке керування кондиціонером повітря")
    print("=" * 74)

    t, d, a, sim, n_rules = build_fis()
    sigma = matlab_gauss_params(T_LOW, T_HIGH, 5)[0][1]
    print(f"\nСистема Мамдані: 2 входи, 1 вихід, правил — {n_rules}")
    print(f"  t : 5 гаусових термів, центри 10/16/22/28/34 °C, sigma = {sigma:.3f}")
    print(f"      комфортна температура {T_COMFORT:.0f} °C = центр терму N")
    print("  dt: 3 трикутні терми, центри -1 / 0 / +1 °C/хв")
    print("  angle: 5 трикутних термів, центри -90/-45/0/+45/+90°")
    print("  (вліво = режим «холод», вправо = режим «тепло»)")

    # ----------------------------------------------------- функції належності
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    plot_mfs(axes[0], t, "Вхід «температура повітря» (5 гаусових термів)", "t, °C")
    axes[0].axvline(T_COMFORT, color="g", linestyle=":", linewidth=1.5)
    plot_mfs(axes[1], d, "Вхід «швидкість зміни температури»", "dT/dt, °C/хв")
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "fig9_task2_inputs.png"))

    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    plot_mfs(ax, a, "Вихід «кут повороту регулятора кондиціонера»", "кут, град")
    ax.axvline(0, color="k", linewidth=0.8)
    ax.text(-70, 1.10, "режим «холод»", ha="center", fontsize=9, color="navy")
    ax.text(70, 1.10, "режим «тепло»", ha="center", fontsize=9, color="darkred")
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "fig10_task2_output.png"))

    # ------------------------------------------------- поверхня «входи-вихід»
    nt, nd = 25, 25
    t_grid = np.linspace(T_LOW, T_HIGH, nt)
    d_grid = np.linspace(D_LOW, D_HIGH, nd)
    z = surface(sim, "t", t_grid, "dt", d_grid, "angle", FALLBACK)
    print(f"\nПоверхня виводу на сітці {nt}x{nd}:")
    print(f"  кут регулятора: від {z.min():7.2f}° до {z.max():7.2f}°")

    fig = plt.figure(figsize=(13, 4.8))
    ax = fig.add_subplot(121, projection="3d")
    plot_surface(ax, t_grid, d_grid, z, "t, °C", "dT/dt, °C/хв", "кут, град",
                 "Поверхня «входи-вихід» СНВ кондиціонера", cmap="coolwarm_r")
    ax.view_init(elev=26, azim=-130)
    ax2 = fig.add_subplot(122)
    cs = ax2.contourf(*np.meshgrid(t_grid, d_grid), z, levels=20,
                      cmap="coolwarm_r")
    ax2.contour(*np.meshgrid(t_grid, d_grid), z, levels=[0], colors="k",
                linewidths=1.6)
    ax2.set_xlabel("t, °C")
    ax2.set_ylabel("dT/dt, °C/хв")
    ax2.set_title("Той самий вивід у вигляді ізоліній\n(чорна лінія — кут 0°)",
                  fontsize=10)
    fig.colorbar(cs, ax=ax2, label="кут регулятора, град")
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "fig11_task2_surface.png"))

    # --------------------------------------------------- перевірка на входах
    print("\nПеревірка роботи СНВ на конкретних наборах входів:")
    header = (f"  {'t,°C':>6} {'dT/dt':>7} | {'кут, град':>10} | "
              f"{'правила':<10} інтерпретація")
    print(header)
    print("  " + "-" * (len(header) + 20))
    probes = [
        (34.0, 0.8, "R1", "дуже тепло і далі теплішає"),
        (34.0, -0.8, "R2", "дуже тепло, але вже холодніє"),
        (34.0, 0.0, "R9", "дуже тепло, температура стабільна"),
        (28.0, 0.8, "R3", "тепло і теплішає"),
        (28.0, -0.8, "R4", "тепло, але вже холодніє"),
        (28.0, 0.0, "R10", "тепло, стабільно"),
        (22.0, 0.0, "R15", "норма, стабільно — комфорт"),
        (22.0, 0.8, "R13", "норма, але теплішає"),
        (22.0, -0.8, "R14", "норма, але холодніє"),
        (16.0, -0.8, "R7", "холодно і далі холодніє (виправлене правило 7)"),
        (16.0, 0.8, "R8", "холодно, але вже теплішає"),
        (16.0, 0.0, "R12", "холодно, стабільно"),
        (10.0, -0.8, "R5", "дуже холодно і далі холодніє"),
        (10.0, 0.8, "R6", "дуже холодно, але теплішає"),
        (10.0, 0.0, "R11", "дуже холодно, стабільно"),
    ]
    for tv, dv, rule, cap in probes:
        res = safe_defuzz(sim, {"t": tv, "dt": dv}, ["angle"], FALLBACK)["angle"]
        print(f"  {tv:6.1f} {dv:7.2f} | {res:10.2f} | {rule:<10} {cap}")
        print(f"  {'':6} {'':7} | {'':10} | {'':10} -> {describe(res)}")

    # --------------------------------------------- замкнутий контур керування
    minutes = 180.0
    print(f"\nЗамкнутий контур: приміщення 30 °C, зовні 32 °C, "
          f"{minutes:.0f} хвилин:")
    time, temp, deriv, angles = closed_loop(sim, t0=30.0, t_out=32.0,
                                            minutes=minutes)
    for mark in (0.0, 20.0, 60.0, 120.0, 180.0):
        k = int(mark / 0.5)
        print(f"  t = {mark:5.0f} хв: температура {temp[k]:6.2f} °C, "
              f"dT/dt = {deriv[k]:+6.3f} °C/хв, кут = {angles[k]:+7.2f}°")
    print(f"  мінімум температури за весь час : {temp.min():.2f} °C "
          f"(перерегулювання відсутнє)")
    print(f"  усталене відхилення від комфорту: {temp[-1] - T_COMFORT:+.2f} °C "
          f"(статична похибка П-подібного нечіткого регулятора)")

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    axes[0].plot(time, temp, "r-", linewidth=2)
    axes[0].axhline(T_COMFORT, color="g", linestyle="--",
                    label=f"комфорт {T_COMFORT:.0f} °C")
    axes[0].set_ylabel("t, °C")
    axes[0].set_title("Замкнутий контур: нечіткий регулятор + інертний "
                      "кондиціонер", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time, deriv, "b-", linewidth=2)
    axes[1].axhline(0, color="k", linewidth=0.8)
    axes[1].set_ylabel("dT/dt, °C/хв")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(time, angles, "m-", linewidth=2)
    axes[2].axhline(0, color="k", linewidth=0.8)
    axes[2].set_ylabel("кут, град")
    axes[2].set_xlabel("час, хв")
    axes[2].set_ylim(A_LOW - 5, A_HIGH + 5)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "fig12_task2_dynamics.png"))

    # ----------------------------------------------------- експорт конфігурації
    gauss_t = matlab_gauss_params(T_LOW, T_HIGH, 5)
    tri_d = matlab_tri_params(D_LOW, D_HIGH, 3)
    tri_a = matlab_tri_params(A_LOW, A_HIGH, 5)
    rules_txt = [
        "t=VW and dt=POS  => angle=BL   (1)",
        "t=VW and dt=NEG  => angle=SL   (1)",
        "t=W  and dt=POS  => angle=BL   (1)",
        "t=W  and dt=NEG  => angle=OFF  (1)   # правило 4 у методичці обірване",
        "t=VC and dt=NEG  => angle=BR   (1)",
        "t=VC and dt=POS  => angle=SR   (1)",
        "t=C  and dt=NEG  => angle=BR   (1)   # ВИПРАВЛЕНО: у методичці «вліво»",
        "t=C  and dt=POS  => angle=OFF  (1)",
        "t=VW and dt=ZERO => angle=BL   (1)",
        "t=W  and dt=ZERO => angle=SL   (1)",
        "t=VC and dt=ZERO => angle=BR   (1)",
        "t=C  and dt=ZERO => angle=SR   (1)",
        "t=N  and dt=POS  => angle=SL   (1)",
        "t=N  and dt=NEG  => angle=SR   (1)",
        "t=N  and dt=ZERO => angle=OFF  (1)",
    ]
    write_fis_txt(
        os.path.join(HERE, "task2_air_conditioner.fis.txt"),
        name="air_conditioner",
        description=(
            "Нечітке керування кондиціонером приміщення. Другий вхід —\n"
            "швидкість зміни температури (може бути від'ємною), що враховує\n"
            "теплову інертність процесу нагріву/охолодження.\n"
            "Кут регулятора: вліво («−») = «холод», вправо («+») = «тепло»."
        ),
        inputs=[
            {"name": "t", "range": (T_LOW, T_HIGH),
             "comment": "температура повітря, °C; комфорт 22 °C; терми: "
                        + ", ".join(f"{k}={T_UKR[k]}" for k in T_LABELS),
             "mfs": [{"label": lb, "type": "gaussmf", "params": (sg, c),
                      "comment": T_UKR[lb]}
                     for lb, (c, sg) in zip(T_LABELS, gauss_t)]},
            {"name": "dt", "range": (D_LOW, D_HIGH),
             "comment": "швидкість зміни температури, °C/хв; терми: "
                        + ", ".join(f"{k}={D_UKR[k]}" for k in D_LABELS),
             "mfs": [{"label": lb, "type": "trimf", "params": prm,
                      "comment": D_UKR[lb]}
                     for lb, prm in zip(D_LABELS, tri_d)]},
        ],
        outputs=[
            {"name": "angle", "range": (A_LOW, A_HIGH),
             "comment": "кут повороту регулятора кондиціонера, град",
             "mfs": [{"label": lb, "type": "trimf", "params": prm,
                      "comment": A_UKR[lb]}
                     for lb, prm in zip(A_LABELS, tri_a)]},
        ],
        rules=rules_txt,
        notes=[
            "Правило 7 методички суперечливе: сказано увімкнути режим «тепло», "
            "але повернути регулятор «на великий кут вліво», хоч вліво = "
            "«холод». Реалізовано виправлено: angle=BR (великий кут вправо).",
            "Правило 4 методички обірване: «...включити режим «холод», "
            "повернувши регулятор кондиціонеру слід вимкнути». Трактуємо як "
            "angle=OFF: температура тепла, але вже падає сама — охолоджувати "
            "не потрібно.",
            "База правил покриває всі 15 комбінацій 5 термів температури на "
            "3 терми похідної, тому розріджених зон у поверхні виводу немає.",
        ],
    )

    print("\nГотово: завдання 2 виконано.")


if __name__ == "__main__":
    main()
