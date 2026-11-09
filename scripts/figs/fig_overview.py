"""Fig. 1 (overview, full width): (a) the problem - estimator memory loss hides the rotor/PM state; (b) offline, simulation-trained
construction of the restart prior and its event-matched calibration; (c) online re-initialisation and recovery on a real
Paderborn validation reset (profile 48, winding sensor only).  Schematic drawn programmatically."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, FancyArrowPatch, Rectangle
sys.path.insert(0, str(Path(__file__).parent)); from figstyle import *
ROOT = Path(__file__).resolve().parents[2]; OUT = FIGDIR
Z = np.load(OUT / "fig1_example_reset_traces.npz")
setup()
fig = plt.figure(figsize=(W2, 156 * MM))
FW, FH = 177.5, 156
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, FW); bg.set_ylim(0, FH); bg.axis("off")
BLUE, ORNG, GRN, INK, MUT = C["CBP"], C["naive_KF"], C["oracle"], C["ink"], C["muted"]
PALE = {"a": "#F6F6F4", "b": "#EEF4FA", "c": "#F1F8F4"}
FS = 8.0                                                                     # smallest text: 8 pt at the printed size

def box(x, y, w, h, txt, fc="white", ec="#9A9A9A", fs=FS, bold=False, color=INK, lw=0.7, r=1.2):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=3))
    bg.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=color, zorder=4)

def arrow(x0, y0, x1, y1, color=INK, lw=0.9, style="-|>", ls="-"):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls, zorder=2,
                                 shrinkA=0, shrinkB=0))

def ax_at(x, y, w, h):                                                       # axes placed in figure millimetres
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

# ---------- panel frames: (a) and (c) on top, (b) full width below
for x, y, w, h, k, t in [(1, 80, 69, 75, "a", "(a) Estimator memory loss"), (72, 80, 104.5, 75, "c", "(c) Online: re-initialize and recover"),
                         (1, 1, 175.5, 77, "b", "(b) Offline: simulation-trained restart prior")]:
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2", fc=PALE[k], ec="#D8D8D4", lw=0.6, zorder=0))
    bg.text(x + 2, y + h - 1.5, t, fontsize=9, fontweight="bold", va="top", color=INK)

# ---------- (a) motor cross-section with sensors
cx, cy = 15, 129
bg.add_patch(Circle((cx, cy), 10.5, fc="#D9D9D6", ec="#7A7A7A", lw=0.6, zorder=2))          # stator yoke
bg.add_patch(Circle((cx, cy), 7.4, fc=PALE["a"], ec="#7A7A7A", lw=0.5, zorder=2))
for k in range(12):                                                                           # winding slots
    a = 2 * np.pi * k / 12
    bg.add_patch(Circle((cx + 8.6 * np.cos(a), cy + 8.6 * np.sin(a)), 1.05, fc="#E8A33D", ec="#9C6A1E", lw=0.4, zorder=3))
bg.add_patch(Circle((cx, cy), 6.0, fc="#BFBFBC", ec="#7A7A7A", lw=0.5, zorder=3))           # rotor
for k in range(8):                                                                            # magnets
    bg.add_patch(Wedge((cx, cy), 5.6, 45 * k + 6, 45 * k + 39, width=1.4, fc="#C0392B" if k % 2 else "#2E5E9E", ec="none", zorder=4))
bg.add_patch(Circle((cx, cy), 1.2, fc="#7A7A7A", ec="none", zorder=5))
bg.text(29, 141.0, "winding sensor\n(measured)", fontsize=FS, color=INK, va="center")
arrow(28.6, 140.0, cx + 8.6 * np.cos(np.pi / 6) + 0.6, cy + 8.6 * np.sin(np.pi / 6) + 0.6, color=INK, lw=0.6)
bg.text(29, 129.0, "magnets\n(hidden state)", fontsize=FS, color="#C0392B", va="center", fontweight="bold")
arrow(28.6, 129.0, cx + 5.0, cy - 1.0, color="#C0392B", lw=0.6)
bg.text(29, 117.5, "coolant, ambient,\ncurrent, voltage, speed", fontsize=FS, color=MUT, va="center")

# (a) timeline inset: history erased at reset
ax = ax_at(12, 88, 55, 21)
ax.axvspan(Z["pre_t"][0], 0, color="#DADAD6", alpha=0.7, lw=0)
ax.plot(Z["pre_t"], Z["pre_w"], color="#8E8E8A", lw=1.0); ax.plot(Z["pre_t"], Z["pre_pm"], color="#8E8E8A", lw=1.4, ls=(0, (4, 1.5)))
ax.plot(Z["tm"], Z["wind"], color=INK, lw=1.1); ax.plot(Z["tm"], Z["pm"], color="#C0392B", lw=1.6, ls=(0, (4, 1.5)))
ax.axvline(0, color=ORNG, lw=1.4)
ax.text(-15, 166, "history erased", ha="center", va="top", fontsize=FS, color=MUT)
ax.text(1.5, 166, "reset", ha="left", va="top", fontsize=FS, color=ORNG, fontweight="bold")
ax.text(29.5, 150, "winding (measured)", ha="right", va="top", fontsize=FS, color=INK)
ax.annotate("PM = ?", (1.5, Z["pm"][180]), xytext=(11, 44), fontsize=FS, color="#C0392B", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#C0392B"))
ax.set_ylim(40, 182); ax.set_xlim(-30, 30); ax.set_xticks([-30, 0, 30]); ax.set_yticks([40, 80, 120])
ax.set_xlabel("Time [min]", labelpad=1); ax.set_ylabel("T [°C]", labelpad=1)
ax.set_facecolor("white"); ax.grid(alpha=0.6)

# ---------- (b) offline pipeline
box(5, 54, 38, 14, "Training input\nsequences\n(i, ω, u, coolant)")
box(53, 54, 38, 14, "", fc="white")
bg.text(72, 63.6, "4-node thermal\nnetwork (LPTN)", ha="center", va="center", fontsize=FS, zorder=4)
for k, xx in enumerate([63.5, 69.5, 75.5, 81.5]):                                           # tiny RC chain
    bg.add_patch(Circle((xx, 57.3), 1.0, fc=["#C0392B", "#7A7A7A", "#7A7A7A", "#E8A33D"][k], ec="none", zorder=5))
    if k < 3: bg.plot([xx + 1.0, xx + 5.0], [57.3, 57.3], color="#7A7A7A", lw=0.8, zorder=4)
box(101, 54, 40, 14, "Simulated\nhistories +\nrandom standstill\n(no PM labels)", ec=BLUE, lw=0.9)
arrow(43, 61, 53, 61); arrow(91, 61, 101, 61)
box(101, 33, 40, 14, "Gradient-boosted prior\nμ(z) = E[hidden state |\nmeasured, context]", fc="#DCEAF6", ec=BLUE, bold=False, lw=1.0)
arrow(121, 54, 121, 47)
bg.text(123, 50.5, "train on\nsnapshots", fontsize=FS, color=MUT, va="center")
box(53, 33, 38, 14, "Novelty ν(z)\nkNN distance to\nsimulated states")
arrow(101, 40, 91, 40, style="<|-")
box(5, 11, 136, 17, "", fc="white", ec=BLUE, lw=0.9)
bg.text(73, 25.3, "Event-matched split conformal (90%)", ha="center", va="center", fontsize=FS, fontweight="bold", color=BLUE, zorder=4)
bg.text(73, 19.3, "score s = |PM − μ(z)| / (1 + ν(z)) on real training\nevents of the same type (starts → starts,\nresets → mid-run points)",
        ha="center", va="center", fontsize=FS, zorder=4, linespacing=1.1)
bg.text(73, 13.4, "half-width  h(z) = $q_{0.9}$ · (1 + ν(z))", ha="center", va="center", fontsize=FS, fontweight="bold", zorder=4)
arrow(72, 33, 72, 28); arrow(121, 33, 121, 28)
box(146, 11, 27, 36, "Split process\nnoise Q\n\nq rule: smallest q\nwith train coverage\n≥ 0.88")
bg.text(73, 5.5, "Measured PM temperatures identify the network and calibrate the band;\nthey are not used as regression targets of the prior",
        ha="center", va="center", fontsize=FS, color=MUT, style="italic")

# ---------- (c) online
box(75, 135, 98.5, 9, "At the event: z = [stator temps, coolant, ambient, i², ω, u²]", fc="white")
box(75, 122, 44, 10, "$\\hat{x}_0$ = μ(z)\n$P_0$ from h(z)", fc="#DCEAF6", ec=BLUE, lw=1.0)
box(129.5, 122, 44, 10, "Kalman filter\non LPTN, split Q", fc="white", ec=BLUE, lw=1.0)
arrow(97, 135, 97, 132); arrow(119, 127, 129.5, 127)
ax = ax_at(89, 88, 83, 22)
t = Z["tm"]
ax.fill_between(t, Z["cbp"] - 1.645 * Z["sd"], Z["cbp"] + 1.645 * Z["sd"], color=BLUE, alpha=0.18, lw=0, label="CBP ±1.645σ band")
ax.plot(t, Z["pm"], color=INK, lw=1.8, label="Measured PM (eval. only)")
ax.plot(t, Z["naive"], color=ORNG, lw=1.4, ls="--", label="Naive KF")
ax.plot(t, Z["cbp"], color=BLUE, lw=1.6, label="CBP")
ax.plot(t, Z["oracle"], color=GRN, lw=1.3, ls=":", label="Oracle")
ax.set_xlim(0, 30); ax.set_xticks([0, 10, 20, 30]); ax.set_xlabel("Time after reset [min]", labelpad=1)
ax.set_ylabel("PM temperature [°C]", labelpad=1); ax.set_facecolor("white")
ax.legend(fontsize=FS, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.8, labelspacing=0.2, columnspacing=1.0,
          borderaxespad=0.2)
e0n, e0c = abs(Z["naive"][0] - Z["pm"][0]), abs(Z["cbp"][0] - Z["pm"][0])
ax.annotate(f"naive error {e0n:.0f} K", (0.15, min(Z["naive"][0], 104)), xytext=(3.0, 98), color=ORNG, fontsize=FS, fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=ORNG))
ax.annotate(f"CBP error {e0c:.0f} K", (0.15, Z["cbp"][0]), xytext=(3.0, 38), color=BLUE, fontsize=FS, fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=BLUE))
ax.set_ylim(30, 106)
save(fig, OUT / "fig_overview")
print("ok", e0n, e0c)
