"""Рисунок 3: схема двошарового персептрона для XOR."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9, 5.2))
pos = {"x1": (0, 2.2), "x2": (0, 0.6), "y1": (3, 2.6), "y2": (3, 0.2), "out": (6, 1.4)}
labels = {"x1": "$x_1$", "x2": "$x_2$", "y1": "$y_1$\nOR", "y2": "$y_2$\nAND", "out": "XOR"}
colors = {"x1": "#ECEFF1", "x2": "#ECEFF1", "y1": "#BBDEFB", "y2": "#BBDEFB", "out": "#C8E6C9"}

for k, (x, y) in pos.items():
    ax.add_patch(Circle((x, y), 0.45, facecolor=colors[k], edgecolor="black", lw=1.6, zorder=3))
    ax.text(x, y, labels[k], ha="center", va="center", fontsize=11, zorder=4)

# t — положення підпису вздовж стрілки, щоб підписи на перехресті не накладались
edges = [
    ("x1", "y1", "+1", 0.30), ("x2", "y1", "+1", 0.72),
    ("x1", "y2", "+1", 0.72), ("x2", "y2", "+1", 0.30),
    ("y1", "out", "+1", 0.5), ("y2", "out", "−1", 0.5),
]
for a, b, w, t in edges:
    (xa, ya), (xb, yb) = pos[a], pos[b]
    ax.add_patch(FancyArrowPatch((xa + 0.45, ya), (xb - 0.45, yb),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#455A64", zorder=2))
    lx, ly = xa + (xb - xa) * t, ya + (yb - ya) * t
    ax.text(lx, ly + 0.16, w, fontsize=11, color="#B71C1C", ha="center",
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

ax.text(3, 3.35, "$W_0 = -0.5$", ha="center", fontsize=10, color="#1565C0")
ax.text(3, -0.55, "$W_0 = -1.5$", ha="center", fontsize=10, color="#1565C0")
ax.text(6, 2.15, "$W_0 = -0.5$", ha="center", fontsize=10, color="#1565C0")
ax.text(0, 3.35, "вхідний\nшар", ha="center", fontsize=10, style="italic")
ax.text(3, -1.25, "перший (прихований) шар", ha="center", fontsize=10, style="italic")
ax.text(6, 2.75, "другий шар", ha="center", fontsize=10, style="italic")

ax.set_xlim(-1.2, 7.4); ax.set_ylim(-1.6, 3.8); ax.axis("off")
ax.set_title("Двошаровий персептрон для функції XOR", fontsize=13)
plt.tight_layout(); plt.savefig("fig3_perceptron.png", dpi=150)
print("Збережено fig3_perceptron.png")
