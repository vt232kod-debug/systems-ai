# -*- coding: utf-8 -*-
"""
Спільні допоміжні функції для лабораторної роботи №3 (нечітка логіка).

Містить:
  * matlab_tri_params / matlab_gauss_sigma — відтворення параметрів функцій
    належності, які MATLAB Fuzzy Logic Toolbox генерує автоматично командою
    Edit -> Add MFs...;
  * safe_defuzz — обчислення виходу системи Мамдані зі «страхуванням» від
    розріджених баз правил (MATLAB у такому разі повертає середину діапазону);
  * plot_mfs — малювання функцій належності лінгвістичної змінної;
  * write_fis_txt — текстовий експорт конфігурації системи (аналог .fis-файлу).

Автор: Камінський Олексій Дмитрович, ВТ-23-2, № у списку 3.
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")  # працюємо без графічного дисплея, тільки у файли
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Відтворення автоматичних параметрів MF з MATLAB
# ---------------------------------------------------------------------------

def matlab_tri_params(low: float, high: float, n: int) -> list[list[float]]:
    """Параметри n трикутних ФН, рівномірно розставлених на [low, high].

    MATLAB для n термів ставить вершини у точках low + k*(high-low)/(n-1),
    а напівширина трикутника дорівнює кроку між вершинами. Крайні трикутники
    при цьому «виходять» за межі діапазону — так само, як у fis-редакторі.
    """
    step = (high - low) / (n - 1)
    centers = [low + k * step for k in range(n)]
    return [[c - step, c, c + step] for c in centers]


def matlab_gauss_sigma(low: float, high: float, n: int) -> float:
    """Сигма n гаусових ФН, рівномірно розставлених на [low, high].

    MATLAB підбирає сигму так, щоб сусідні ФН перетиналися на рівні 0.5:
    exp(-(step/2)^2 / (2*sigma^2)) = 0.5  =>  sigma = step / (2*sqrt(2*ln2)).
    """
    step = (high - low) / (n - 1)
    return step / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def matlab_gauss_params(low: float, high: float, n: int) -> list[tuple[float, float]]:
    """Список (центр, сигма) для n гаусових ФН на [low, high]."""
    step = (high - low) / (n - 1)
    sigma = matlab_gauss_sigma(low, high, n)
    return [(low + k * step, sigma) for k in range(n)]


# ---------------------------------------------------------------------------
# Дефазифікація зі страхуванням
# ---------------------------------------------------------------------------

def safe_defuzz(sim, inputs: dict, outputs: list[str], fallback: dict) -> dict:
    """Виконати нечіткий вивід і повернути дефазифіковані значення.

    Якщо база правил «розріджена» (жодне правило не активувалося з вагою > 0),
    scikit-fuzzy кидає помилку. MATLAB у такій ситуації повертає середину
    діапазону вихідної змінної — відтворюємо цю поведінку через fallback.
    """
    for name, value in inputs.items():
        sim.input[name] = value
    try:
        sim.compute()
        return {name: float(sim.output[name]) for name in outputs}
    except Exception:
        return {name: float(fallback[name]) for name in outputs}


def surface(sim, x_name: str, x_vals, y_name: str, y_vals,
            out_name: str, fallback: dict) -> np.ndarray:
    """Поверхня «входи-вихід»: Z[j, i] = f(x_vals[i], y_vals[j])."""
    z = np.zeros((len(y_vals), len(x_vals)))
    for j, yv in enumerate(y_vals):
        for i, xv in enumerate(x_vals):
            res = safe_defuzz(sim, {x_name: xv, y_name: yv},
                              [out_name], fallback)
            z[j, i] = res[out_name]
    return z


# ---------------------------------------------------------------------------
# Графіка
# ---------------------------------------------------------------------------

def plot_mfs(ax, var, title: str, xlabel: str) -> None:
    """Намалювати всі терми змінної scikit-fuzzy Antecedent/Consequent."""
    for label in var.terms:
        ax.plot(var.universe, var[label].mf, linewidth=2, label=label)
    n_terms = len(var.terms)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("µ")
    # Додатковий запас зверху, щоб легенда не перекривала графіки ФН.
    ax.set_ylim(-0.05, 1.42)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=min(n_terms, 7), loc="upper center",
              framealpha=0.9, columnspacing=1.0, handlelength=1.4)


def plot_surface(ax, x_vals, y_vals, z, xlabel: str, ylabel: str,
                 zlabel: str, title: str, cmap: str = "viridis"):
    """3D-поверхня (аналог surf у MATLAB)."""
    xx, yy = np.meshgrid(np.asarray(x_vals), np.asarray(y_vals))
    srf = ax.plot_surface(xx, yy, z, cmap=cmap, edgecolor="k",
                          linewidth=0.2, antialiased=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title, fontsize=11)
    return srf


# ---------------------------------------------------------------------------
# Текстовий експорт конфігурації (аналог File -> Export -> To disk)
# ---------------------------------------------------------------------------

def write_fis_txt(path: str, name: str, description: str,
                  inputs: list[dict], outputs: list[dict],
                  rules: list[str], notes: list[str] | None = None,
                  and_method: str = "min", or_method: str = "max",
                  imp_method: str = "min", agg_method: str = "max",
                  defuzz_method: str = "centroid") -> None:
    """Зберегти опис системи нечіткого виводу у читабельному текстовому файлі.

    Структура повторює логіку MATLAB .fis-файлу, щоб студент міг швидко
    ввести ту саму систему у Fuzzy Logic Designer, якщо тулбокс знайдеться.
    """
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"FIS (текстовий експорт): {name}")
    lines.append("=" * 78)
    lines.append(description.strip())
    lines.append("")
    lines.append("[System]")
    lines.append(f"Name        = {name}")
    lines.append("Type        = mamdani")
    lines.append(f"NumInputs   = {len(inputs)}")
    lines.append(f"NumOutputs  = {len(outputs)}")
    lines.append(f"NumRules    = {len(rules)}")
    lines.append(f"AndMethod   = {and_method}")
    lines.append(f"OrMethod    = {or_method}")
    lines.append(f"ImpMethod   = {imp_method}")
    lines.append(f"AggMethod   = {agg_method}")
    lines.append(f"DefuzzMethod= {defuzz_method}")
    lines.append("")

    def dump_var(kind: str, idx: int, var: dict) -> None:
        lines.append(f"[{kind}{idx}]")
        lines.append(f"Name  = {var['name']}")
        lines.append(f"Range = [{var['range'][0]} {var['range'][1]}]")
        lines.append(f"NumMFs= {len(var['mfs'])}")
        if var.get("comment"):
            lines.append(f"# {var['comment']}")
        for k, mf in enumerate(var["mfs"], start=1):
            params = " ".join(f"{p:g}" for p in mf["params"])
            tail = f"   # {mf['comment']}" if mf.get("comment") else ""
            lines.append(f"MF{k} = '{mf['label']}':'{mf['type']}',[{params}]{tail}")
        lines.append("")

    for i, var in enumerate(inputs, start=1):
        dump_var("Input", i, var)
    for i, var in enumerate(outputs, start=1):
        dump_var("Output", i, var)

    lines.append("[Rules]")
    lines.append("# формат: антецедент => консеквент (вага)")
    for k, rule in enumerate(rules, start=1):
        lines.append(f"{k:2d}. {rule}")
    lines.append("")

    if notes:
        lines.append("[Notes]")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[fis] збережено {path}")


def savefig(fig, path: str) -> None:
    """Збереження рисунка у PNG з однаковими налаштуваннями."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[png] збережено {path}")
