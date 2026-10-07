"""Graphical abstract (>= 531 x 1328 px h x w, readable at 5 x 13 cm). Numbers = locked single test results
(Paderborn G resets, winding sensor only; induction-motor test starts, winding sensor only)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
sys.path.insert(0, str(Path(__file__).parent)); from figstyle import *
ROOT = Path(__file__).resolve().parents[2]; OUT = FIGDIR
Z = np.load(OUT / "fig1_example_reset_traces.npz")
setup()
fig = plt.figure(figsize=(130 * MM, 52 * MM))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 130); bg.set_ylim(-1, 51); bg.axis("off")
BLUE, ORNG, INK, MUT, RED = C["CBP"], C["naive_KF"], C["ink"], C["muted"], "#C0392B"
def card(x, w, fc): bg.add_patch(FancyBboxPatch((x, 2), w, 46, boxstyle="round,pad=0,rounding_size=2", fc=fc, ec="none", zorder=0))
card(1, 36, "#F6F6F4"); card(40, 44, "#EEF4FA"); card(87, 42, "#F1F8F4")
# 1 problem
bg.text(19, 45, "Estimator reset", ha="center", fontsize=8, fontweight="bold", color=INK)
bg.text(19, 40.8, "memory erased (emulated)", ha="center", fontsize=6.3, color=MUT)
cx, cy = 19, 24
bg.add_patch(Circle((cx, cy), 11, fc="#D9D9D6", ec="#7A7A7A", lw=0.6))
bg.add_patch(Circle((cx, cy), 7.8, fc="#F6F6F4", ec="#7A7A7A", lw=0.5))
for k in range(12):
    a = 2 * np.pi * k / 12; bg.add_patch(Circle((cx + 9.1 * np.cos(a), cy + 9.1 * np.sin(a)), 1.1, fc="#E8A33D", ec="none"))
bg.add_patch(Circle((cx, cy), 6.3, fc="#BFBFBC", ec="#7A7A7A", lw=0.5))
for k in range(8): bg.add_patch(Wedge((cx, cy), 5.9, 45 * k + 6, 45 * k + 39, width=1.5, fc=RED if k % 2 else "#2E5E9E", ec="none"))
bg.text(cx, cy, "?", ha="center", va="center", fontsize=15, fontweight="bold", color=RED)
bg.text(19, 7.5, "magnet temperature\nunknown, no history", ha="center", fontsize=6.4, color=RED, fontweight="bold")
# 2 method
bg.text(62, 45, "Simulation-trained restart prior", ha="center", fontsize=8, fontweight="bold", color=BLUE)
for i, (t, y) in enumerate([("simulations of a thermal network\nidentified on labelled data", 34.5), ("gradient-boosted prior\n+ novelty-scaled conformal band", 23.5),
                            ("Kalman filter seeded from stator,\nboundary and operating point", 12.5)]):
    bg.add_patch(FancyBboxPatch((45, y - 4), 34, 8, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=BLUE, lw=0.9))
    bg.text(62, y, t, ha="center", va="center", fontsize=6.2, color=INK)
    if i < 2: bg.add_patch(FancyArrowPatch((62, y - 4), (62, y - 7), arrowstyle="-|>", mutation_scale=7, color=BLUE, lw=0.9))
bg.text(62, 4.6, "2 machines, one locked test each", ha="center", fontsize=6.0, color=MUT, style="italic")
for x0 in (37.3, 84.3): bg.add_patch(FancyArrowPatch((x0, 25), (x0 + 2.4, 25), arrowstyle="-|>", mutation_scale=9, color=INK, lw=1.2))
# 3 result
bg.text(108, 45, "Faster recovery", ha="center", fontsize=8, fontweight="bold", color=C["oracle"])
bg.text(108, 40.8, "initial band calibrated offline", ha="center", fontsize=6.3, color=MUT)
ax = fig.add_axes([0.705, 0.36, 0.27, 0.43])
ax.fill_between(Z["tm"], Z["cbp"] - 1.645 * Z["sd"], Z["cbp"] + 1.645 * Z["sd"], color=BLUE, alpha=0.18, lw=0)
ax.plot(Z["tm"], Z["pm"], color=INK, lw=1.6); ax.plot(Z["tm"], Z["naive"], color=ORNG, lw=1.2, ls="--"); ax.plot(Z["tm"], Z["cbp"], color=BLUE, lw=1.4)
ax.set_xlim(0, 30); ax.set_ylim(35, 105); ax.set_xticks([0, 15, 30]); ax.set_yticks([40, 70, 100]); ax.tick_params(labelsize=5.5, pad=1)
ax.set_xlabel("min after reset", fontsize=5.8, labelpad=0); ax.set_facecolor("white")
ax.text(3, 100, "hardest validation reset", color=MUT, fontsize=5.6, ha="left", va="top", style="italic")
ax.text(29, 100, "naive", color=ORNG, fontsize=5.8, ha="right", va="top", fontweight="bold")
ax.text(29, 44, "proposed + band", color=BLUE, fontsize=5.8, ha="right", fontweight="bold")
ax.text(16, 81, "measured", color=INK, fontsize=5.8, ha="left", fontweight="bold")
bg.text(108, 9.6, "median PMSM error at reset 11.9 → 5.1 K", ha="center", fontsize=5.8, color=INK, fontweight="bold")
bg.text(108, 3.6, "locked PMSM test: 26/32 covered (3 profiles);\nstrict split: 91 % of CV resets covered", ha="center", fontsize=5.7, color=INK)
fig.savefig(OUT / "graphical_abstract.pdf"); fig.savefig(OUT / "graphical_abstract.png", dpi=300); fig.savefig(OUT / "graphical_abstract.tiff", dpi=300)
print("ok")
