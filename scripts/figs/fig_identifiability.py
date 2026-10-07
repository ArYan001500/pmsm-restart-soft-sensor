"""Fig. identifiability: (a) real twin snapshots - same measured stator/boundary/operating values, PM 50 K apart (Paderborn,
profiles 69 and 75); (b) distribution of the irreducible ambiguity (per-snapshot worst |dPM| over all twins) vs the CBP band width
and the naive band (CV resets, S1)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent)); from figstyle import *
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from pmsm_softsense import realdata
OUT = FIGDIR; MP = latest_run("*_matched_pairs_*")
J = json.loads((MP / "summary.json").read_text())["S1"]["example_max_pair"]; Z = np.load(MP / "pairs.npz")
setup()
fig, axs = plt.subplots(1, 2, figsize=(W2, 62 * MM), gridspec_kw={"wspace": 0.28, "width_ratios": [1.25, 1]})
ax = axs[0]; D = realdata.load([int(J["A"]["pid"]), int(J["B"]["pid"])])
for key, col, name in [("A", "#0072B2", "Profile %d" % J["A"]["pid"]), ("B", "#D55E00", "Profile %d" % J["B"]["pid"])]:
    o, t = D[int(J[key]["pid"])]; r = int(J[key]["r"]); lo, hi = max(0, r - 7 * 120), min(len(o), r + 15 * 120)
    tm = (np.arange(lo, hi) - r) * 0.5 / 60
    ax.plot(tm, o.stator_winding.values[lo:hi], color=col, lw=1.2, ls="-", alpha=0.9)
    ax.plot(tm, t.pm.values[lo:hi], color=col, lw=2.0, ls=(0, (5, 1.5)))
    ax.plot([0], [t.pm.values[r]], marker="o", ms=5, color=col, mec="white", mew=0.7, zorder=5)
    ax.text(0.6, t.pm.values[r] + (3.5 if key == "B" else -6.5), f"{name}: PM {t.pm.values[r]:.0f} °C", fontsize=6.6, color=col, fontweight="bold")
ax.axvline(0, color=C["ink"], lw=0.8, ls=(0, (2, 2)))
yA = (J["A"]["Tw"] + J["B"]["Tw"]) / 2
ax.text(0.2, 97, "matched instant: winding, coolant, ambient,\ncurrent, voltage, speed all equal", fontsize=6.0, color=C["ink"], va="top")
ax.annotate("", xy=(-0.6, J["B"]["pm"]), xytext=(-0.6, J["A"]["pm"]), arrowprops=dict(arrowstyle="<->", lw=1.0, color=C["ink"]))
ax.text(-0.9, (J["A"]["pm"] + J["B"]["pm"]) / 2, f"{J['B']['pm'] - J['A']['pm']:.0f} K\nhidden\ndifference", ha="right", va="center", fontsize=6.4, fontweight="bold")
ax.set_xlabel("Time relative to the matched instant [min]"); ax.set_ylabel("Temperature [°C]"); ax.set_xlim(-7, 15); ax.set_ylim(18, 98)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], color=C["ink"], lw=1.2), Line2D([], [], color=C["ink"], lw=2.0, ls=(0, (5, 1.5)))],
          ["Measured winding", "Permanent magnet (hidden)"], loc="lower right", fontsize=6.2, handlelength=1.8, ncol=2, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
panel(ax, "(a)", x=0.0, y=1.02)
ax = axs[1]
a, b, d = Z["S1_a"], Z["S1_b"], Z["S1_dpm"]
E = pd.read_csv(latest_run("*_cv5_ablation_v2_paderborn_*") / "episodes.csv")
hw = E[(E.event == "reset") & (E.scenario == "S1") & (E.method == "CBP")].hw_0.values
bins = np.arange(0, 52, 2)
ax.hist(d, bins=bins, color=C["ink"], alpha=0.85, rwidth=0.88)
ax.set_yscale("log"); ax.set_xlabel("|ΔPM| between twin snapshots [K]"); ax.set_ylabel("Number of twin pairs")
for thr, txt in [(5, f"{np.mean(d > 5) * 100:.0f} % > 5 K"), (10, f"{np.mean(d > 10) * 100:.1f} % > 10 K")]:
    ax.axvline(thr, color=C["naive_KF"], lw=1.0, ls="--"); ax.text(thr + 0.6, ax.get_ylim()[1] * (0.35 if thr == 5 else 0.12), txt, fontsize=6.4, color=C["naive_KF"])
ax.text(49.5, 12, "max 50 K", ha="right", fontsize=6.4, color=C["ink"])
ax.text(0.98, 0.98, f"{len(d):,} twin pairs\n(winding sensor only)", transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=C["muted"])
panel(ax, "(b)", x=0.0, y=1.02)
save(fig, OUT / "fig_identifiability")
print(len(d), np.mean(d > 5), np.mean(d > 10), d.max())
