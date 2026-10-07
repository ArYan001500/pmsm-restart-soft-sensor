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
fig = plt.figure(figsize=(W2, 84 * MM))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 190); bg.set_ylim(0, 84); bg.axis("off")
BLUE, ORNG, GRN, INK, MUT = C["CBP"], C["naive_KF"], C["oracle"], C["ink"], C["muted"]
PALE = {"a": "#F6F6F4", "b": "#EEF4FA", "c": "#F1F8F4"}

def box(x, y, w, h, txt, fc="white", ec="#9A9A9A", fs=6.4, bold=False, color=INK, lw=0.7, r=1.2):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=3))
    bg.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=color, zorder=4)

def arrow(x0, y0, x1, y1, color=INK, lw=0.9, style="-|>", ls="-"):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, color=color, lw=lw, ls=ls, zorder=2,
                                 shrinkA=0, shrinkB=0))

# ---------- panel frames
for x, w, k, t in [(1, 52, "a", "(a) Estimator memory loss"), (55, 74, "b", "(b) Offline: simulation-trained restart prior"),
                   (131, 58, "c", "(c) Online: re-initialise and recover")]:
    bg.add_patch(FancyBboxPatch((x, 1), w, 82, boxstyle="round,pad=0,rounding_size=2", fc=PALE[k], ec="#D8D8D4", lw=0.6, zorder=0))
    bg.text(x + 2, 80.6, t, fontsize=7.6, fontweight="bold", va="top", color=INK)

# ---------- (a) motor cross-section with sensors
cx, cy = 14, 60
bg.add_patch(Circle((cx, cy), 10.5, fc="#D9D9D6", ec="#7A7A7A", lw=0.6, zorder=2))          # stator yoke
bg.add_patch(Circle((cx, cy), 7.4, fc=PALE["a"], ec="#7A7A7A", lw=0.5, zorder=2))
for k in range(12):                                                                           # winding slots
    a = 2 * np.pi * k / 12
    bg.add_patch(Circle((cx + 8.6 * np.cos(a), cy + 8.6 * np.sin(a)), 1.05, fc="#E8A33D", ec="#9C6A1E", lw=0.4, zorder=3))
bg.add_patch(Circle((cx, cy), 6.0, fc="#BFBFBC", ec="#7A7A7A", lw=0.5, zorder=3))           # rotor
for k in range(8):                                                                            # magnets
    bg.add_patch(Wedge((cx, cy), 5.6, 45 * k + 6, 45 * k + 39, width=1.4, fc="#C0392B" if k % 2 else "#2E5E9E", ec="none", zorder=4))
bg.add_patch(Circle((cx, cy), 1.2, fc="#7A7A7A", ec="none", zorder=5))
bg.text(cx + 12.5, 70.5, "winding sensor\n(measured)", fontsize=5.9, color=INK, va="center")
arrow(cx + 12.2, 70.0, cx + 8.6 * np.cos(np.pi / 6) + 0.6, cy + 8.6 * np.sin(np.pi / 6) + 0.6, color=INK, lw=0.6)
bg.text(cx + 12.5, 57.0, "magnets\n(hidden state)", fontsize=5.9, color="#C0392B", va="center", fontweight="bold")
arrow(cx + 12.2, 57.0, cx + 5.0, cy - 1.0, color="#C0392B", lw=0.6)
bg.text(cx + 12.5, 49.5, "coolant, ambient,\ncurrent, voltage, speed", fontsize=5.6, color=MUT, va="center")

# (a) timeline inset: history erased at reset
ax = fig.add_axes([0.045, 0.085, 0.235, 0.36])
ax.axvspan(Z["pre_t"][0], 0, color="#DADAD6", alpha=0.7, lw=0)
ax.plot(Z["pre_t"], Z["pre_w"], color="#8E8E8A", lw=1.0); ax.plot(Z["pre_t"], Z["pre_pm"], color="#8E8E8A", lw=1.4, ls=(0, (4, 1.5)))
ax.plot(Z["tm"], Z["wind"], color=INK, lw=1.1); ax.plot(Z["tm"], Z["pm"], color="#C0392B", lw=1.6, ls=(0, (4, 1.5)))
ax.axvline(0, color=ORNG, lw=1.4)
ax.text(-15, 136, "history erased", ha="center", va="top", fontsize=5.9, color=MUT)
ax.text(1.5, 136, "reset", ha="left", va="top", fontsize=6.2, color=ORNG, fontweight="bold")
ax.text(29, 149, "winding (measured)", ha="right", va="top", fontsize=5.6, color=INK)
ax.annotate("PM = ?", (1.5, Z["pm"][180]), xytext=(9, 50), fontsize=6.4, color="#C0392B", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#C0392B"))
ax.set_ylim(40, 152); ax.set_xlim(-30, 30); ax.set_xticks([-30, 0, 30]); ax.set_yticks([40, 80, 120])
ax.set_xlabel("Time [min]", labelpad=1); ax.set_ylabel("T [°C]", labelpad=1); ax.tick_params(labelsize=6)
ax.set_facecolor("white"); ax.grid(alpha=0.6)

# ---------- (b) offline pipeline
box(58, 63, 20, 11, "Training input\nsequences\n(i, ω, u, coolant)", fs=5.9)
box(83, 63, 21, 11, "", fc="white")
bg.text(93.5, 71.6, "4-node thermal\nnetwork (LPTN)", ha="center", va="center", fontsize=5.9, zorder=4)
for k, xx in enumerate([86, 91, 96, 101]):                                                     # tiny RC chain
    bg.add_patch(Circle((xx, 66.0), 0.9, fc=["#C0392B", "#7A7A7A", "#7A7A7A", "#E8A33D"][k], ec="none", zorder=5))
    if k < 3: bg.plot([xx + 0.9, xx + 4.1], [66.0, 66.0], color="#7A7A7A", lw=0.8, zorder=4)
box(109, 63, 18, 11, "Simulated\nhistories +\nrandom standstill\n(no PM labels)", fs=5.6, ec=BLUE, lw=0.9)
arrow(78, 68.5, 83, 68.5); arrow(104, 68.5, 109, 68.5)
box(80, 44, 26, 12, "Gradient-boosted prior\nμ(x) = E[hidden state |\nmeasured, context]", fs=5.8, fc="#DCEAF6", ec=BLUE, bold=False, lw=1.0)
arrow(118, 63, 118, 50.3); arrow(118, 50.3, 106, 50.3)
bg.text(119, 57.5, "train on\nsnapshots", fontsize=5.4, color=MUT, va="center")
box(58, 45, 18, 10.5, "Novelty ν(x)\nkNN distance to\nsimulated states", fs=5.6)
arrow(80, 50.3, 76, 50.3, style="<|-")
box(58, 19, 48, 17, "", fc="white", ec=BLUE, lw=0.9)
bg.text(82, 32.8, "Event-matched split conformal (90 %)", ha="center", fontsize=6.0, fontweight="bold", color=BLUE, zorder=4)
bg.text(82, 27.4, "score s = |PM − μ(x)| / (1 + ν(x)) on real training\nevents of the same type (starts → starts,\nresets → mid-run points)",
        ha="center", va="center", fontsize=5.5, zorder=4)
bg.text(82, 21.6, "half-width  h(x) = $q_{0.9}$ · (1 + ν(x))", ha="center", va="center", fontsize=5.9, fontweight="bold", zorder=4)
arrow(67, 45, 67, 36); arrow(93, 45, 93, 36)
box(109, 19, 18, 17, "Split process\nnoise Q\n\nq rule: smallest q\nwith train coverage\n≥ 0.88", fs=5.4)
bg.text(92, 9.5, "Measured PM temperatures identify the network and calibrate the band;\nthey are never used to train the prior", ha="center", fontsize=5.7, color=MUT, style="italic")
bg.add_patch(Rectangle((60, 8.2), 64, 0.01, color="none"))

# ---------- (c) online
box(134, 66, 52, 8.5, "At the event: x = [stator temps, coolant, ambient, i², ω, u²]", fs=5.3, fc="white")
box(134, 54.5, 24.5, 8.5, "$\\hat{x}_0$ = μ(x)\n$P_0$ from h(x)", fs=5.9, fc="#DCEAF6", ec=BLUE, lw=1.0)
box(161.5, 54.5, 24.5, 8.5, "Kalman filter\non LPTN, split Q", fs=5.9, fc="white", ec=BLUE, lw=1.0)
arrow(146, 66, 146, 63); arrow(158.5, 58.75, 161.5, 58.75)
ax = fig.add_axes([0.725, 0.085, 0.255, 0.50])
t = Z["tm"]
ax.fill_between(t, Z["cbp"] - 1.645 * Z["sd"], Z["cbp"] + 1.645 * Z["sd"], color=BLUE, alpha=0.18, lw=0, label="CBP 90 % band")
ax.plot(t, Z["pm"], color=INK, lw=1.8, label="Measured PM (eval. only)")
ax.plot(t, Z["naive"], color=ORNG, lw=1.4, ls="--", label="Naive KF")
ax.plot(t, Z["cbp"], color=BLUE, lw=1.6, label="CBP")
ax.plot(t, Z["oracle"], color=GRN, lw=1.3, ls=":", label="Oracle")
ax.set_xlim(0, 30); ax.set_xticks([0, 10, 20, 30]); ax.set_xlabel("Time after reset [min]", labelpad=1)
ax.set_ylabel("PM temperature [°C]", labelpad=1); ax.tick_params(labelsize=6); ax.set_facecolor("white")
ax.legend(fontsize=5.4, loc="lower right", handlelength=1.8, labelspacing=0.25)
e0n, e0c = abs(Z["naive"][0] - Z["pm"][0]), abs(Z["cbp"][0] - Z["pm"][0])
ax.annotate(f"naive error {e0n:.0f} K", (0.15, min(Z["naive"][0], 104)), xytext=(3.0, 100), color=ORNG, fontsize=5.8, fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=ORNG))
ax.annotate(f"CBP error {e0c:.0f} K", (0.15, Z["cbp"][0]), xytext=(3.0, 40), color=BLUE, fontsize=5.8, fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=BLUE))
ax.set_ylim(30, 106)
save(fig, OUT / "fig_overview")
print("ok", e0n, e0c)
