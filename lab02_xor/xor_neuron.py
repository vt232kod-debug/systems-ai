"""
Лабораторна робота 2. Нейронна реалізація логічних функцій AND, OR, XOR.
Студент: Камінський Олексій Дмитрович, група ВТ-23-2.

Модель нейрона:
    суматор      g(x) = W1*x1 + W2*x2 + W0
    активація    f(g) = 1, якщо g >= 0;  0, якщо g < 0
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def activation(g: float) -> int:
    """Порогова (одинична) функція активації."""
    return 1 if g >= 0 else 0


def neuron(x1: int, x2: int, w1: float, w2: float, w0: float) -> int:
    """Один формальний нейрон: суматор + порогова активація."""
    return activation(w1 * x1 + w2 * x2 + w0)


# --- Ваги підібрані так, щоб розділяюча пряма проходила між класами ---
# OR:  x1 + x2 - 0.5 = 0  ->  пряма x1 + x2 = 1/2
def or_gate(x1, x2):
    return neuron(x1, x2, w1=1.0, w2=1.0, w0=-0.5)


# AND: x1 + x2 - 1.5 = 0  ->  пряма x1 + x2 = 3/2
def and_gate(x1, x2):
    return neuron(x1, x2, w1=1.0, w2=1.0, w0=-1.5)


def xor_gate(x1, x2):
    """
    XOR через композицію OR і AND (двошаровий персептрон).

    Перший шар:   y1 = or(x1, x2),  y2 = and(x1, x2)
    Другий шар:   xor = y1 AND (NOT y2) = f(1*y1 - 1*y2 - 0.5)
    Розділяюча пряма другого шару в просторі (y1, y2):  y1 - y2 = 1/2
    """
    y1 = or_gate(x1, x2)
    y2 = and_gate(x1, x2)
    return neuron(y1, y2, w1=1.0, w2=-1.0, w0=-0.5)


def truth_table(fn, name):
    print(f"\n{name}")
    print("  x1  x2 | y")
    print("  ---------+---")
    rows = []
    for x1, x2 in INPUTS:
        y = fn(x1, x2)
        rows.append((x1, x2, y))
        print(f"   {x1}   {x2}  | {y}")
    return rows


def main():
    truth_table(or_gate, "OR  (пряма x1 + x2 = 1/2)")
    truth_table(and_gate, "AND (пряма x1 + x2 = 3/2)")
    xor_rows = truth_table(xor_gate, "XOR (через OR і AND)")

    expected = [0, 1, 1, 0]
    got = [r[2] for r in xor_rows]
    print("\nОчікувано XOR:", expected)
    print("Отримано  XOR:", got)
    print("Перевірка:", "ВІРНО" if got == expected else "ПОМИЛКА")

    print("\nПроміжні значення першого шару:")
    print("  x1  x2 | y1=OR  y2=AND | XOR")
    for x1, x2 in INPUTS:
        y1, y2 = or_gate(x1, x2), and_gate(x1, x2)
        print(f"   {x1}   {x2}  |   {y1}      {y2}    |  {xor_gate(x1, x2)}")

    make_plots()


def make_plots():
    # --- Рисунок 1: простір входів (x1, x2) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    line = np.linspace(-0.4, 1.4, 100)

    for ax, (fn, title, w0) in zip(axes[:2], [
        (or_gate, "OR:  $x_1+x_2=1/2$", 0.5),
        (and_gate, "AND:  $x_1+x_2=3/2$", 1.5),
    ]):
        for x1, x2 in INPUTS:
            y = fn(x1, x2)
            ax.scatter(x1, x2, s=260, zorder=3,
                       color="#2E7D32" if y else "#C62828",
                       marker="o" if y else "s", edgecolors="black")
            ax.annotate(f"{y}", (x1, x2), color="white", ha="center", va="center",
                        zorder=4, fontweight="bold")
        ax.plot(line, w0 - line, "b--", lw=2, label=f"$x_1+x_2={w0}$")
        ax.set_title(title); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4)
        ax.grid(alpha=0.3); ax.legend(loc="upper right")

    ax = axes[2]
    for x1, x2 in INPUTS:
        y = xor_gate(x1, x2)
        ax.scatter(x1, x2, s=260, zorder=3,
                   color="#2E7D32" if y else "#C62828",
                   marker="o" if y else "s", edgecolors="black")
        ax.annotate(f"{y}", (x1, x2), color="white", ha="center", va="center",
                    zorder=4, fontweight="bold")
    ax.plot(line, 0.5 - line, "b--", lw=2, alpha=0.5)
    ax.plot(line, 1.5 - line, "b--", lw=2, alpha=0.5)
    ax.set_title("XOR: однією прямою не розділити\n(потрібні дві)")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig1_input_space.png", dpi=150)
    print("\nЗбережено fig1_input_space.png")

    # --- Рисунок 2: простір виходів першого шару (y1, y2) ---
    fig, ax = plt.subplots(figsize=(5.5, 5))
    seen = {}
    for x1, x2 in INPUTS:
        y1, y2 = or_gate(x1, x2), and_gate(x1, x2)
        out = xor_gate(x1, x2)
        seen.setdefault((y1, y2), []).append(f"({x1},{x2})")
        ax.scatter(y1, y2, s=320, zorder=3,
                   color="#2E7D32" if out else "#C62828",
                   marker="o" if out else "s", edgecolors="black")
    for (y1, y2), src in seen.items():
        ax.annotate(" ".join(src), (y1, y2), textcoords="offset points",
                    xytext=(12, 10), fontsize=9)
    t = np.linspace(-0.4, 1.4, 100)
    ax.plot(t, t - 0.5, "b--", lw=2, label="$y_1 - y_2 = 1/2$")
    ax.set_xlabel("$y_1 = OR(x_1,x_2)$"); ax.set_ylabel("$y_2 = AND(x_1,x_2)$")
    ax.set_title("Простір виходів першого шару:\nкласи стали лінійно роздільними")
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4)
    ax.grid(alpha=0.3); ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("fig2_hidden_space.png", dpi=150)
    print("Збережено fig2_hidden_space.png")


if __name__ == "__main__":
    main()
