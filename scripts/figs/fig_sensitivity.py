"""Fig. sensitivity: one-factor changes of the frozen recipe (5-fold CV, winding sensor only S1, CBP only).
Groups: standstill-gap maximum (60/120/240 min), novelty neighbours k (10/20/40), miscoverage alpha (0.20/0.10/0.05); the frozen
setting (gap 120, k 20, alpha 0.10) is repeated in every group and highlighted.  Columns: median |error| 10 min after the event (whole pipeline: prior, band and filter; the error AT the event is
unchanged by k and alpha by construction, since they only set the band); coverage at
the event drawn as a lollipop from the nominal level 1-alpha; median band half-width at the event."""
import sys, json, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, str(Path(__file__).parent)); from figstyle import *
ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "outputs/runs"; OUT = FIGDIR
BASE = {"paderborn": latest_run("*_cv5_ablation_v2_paderborn_*"), "induction": latest_run("*_cv5_ablation_v2_induction_*")}
EV = {"paderborn": "reset", "induction": "start"}
GROUPS = [("Standstill gap in simulated histories", [("gap60", "≤ 60 min", 0.90), ("base", "≤ 120 min", 0.90), ("gap240", "≤ 240 min", 0.90)]),
          ("Novelty neighbours", [("k10", "k = 10", 0.90), ("base", "k = 20", 0.90), ("k40", "k = 40", 0.90)]),
          ("Conformal miscoverage", [("a20", "α = 0.20", 0.80), ("base", "α = 0.10", 0.90), ("a05", "α = 0.05", 0.95)])]
DS_C = {"paderborn": "#0072B2", "induction": "#D55E00"}; DS_M = {"paderborn": "o", "induction": "s"}
DS_L = {"paderborn": "PMSM, mid-run resets (64 profiles)", "induction": "Induction motor, starts (223 sessions)"}
def load(ds, tag, t="t0"):
    d = BASE[ds] if tag == "base" else Path(sorted(glob.glob(str(R / f"*_cv5_sens_{tag}_{ds}_*")))[-1])
    return json.loads((d / "summary.json").read_text())["summary"][f"{EV[ds]}|S1|CBP"][t]
rows, y, ylab, gtitle = [], 0.0, [], []
for g, (title, lev) in enumerate(GROUPS):
    gtitle.append((title, y - 0.75))
    for tag, lab, nom in lev: rows.append((y, tag, lab, nom, g)); y += 1
    y += 0.9
YMAX = y - 0.9
setup()
fig, axs = plt.subplots(1, 3, figsize=(W2, 80 * MM), sharey=True, gridspec_kw={"wspace": 0.07, "width_ratios": [1, 1.1, 1]})
off = {"paderborn": -0.13, "induction": 0.13}
for ax in axs:
    ax.grid(axis="y", visible=False); ax.set_ylim(YMAX - 0.35, -1.25)
    for yy, tag, lab, nom, g in rows:
        if tag == "base": ax.axhspan(yy - 0.45, yy + 0.45, color="#DCEAF6", lw=0, zorder=0)
for ds in BASE:
    for g in range(3):
        R_ = [(yy, load(ds, tag), nom) for yy, tag, lab, nom, gg in rows if gg == g]
        ys = [r[0] + off[ds] for r in R_]
        R10 = [load(ds, tag, "t10")["err_med"] for yy, tag, lab, nom, gg in rows if gg == g]
        axs[0].plot(R10, ys, color=DS_C[ds], lw=0.9, alpha=0.55, zorder=2)
        axs[0].plot(R10, ys, DS_M[ds], color=DS_C[ds], ms=5, mec="white", mew=0.6, zorder=3)
        for ax, key in [(axs[2], "hw_med")]:
            ax.plot([r[1][key] for r in R_], ys, color=DS_C[ds], lw=0.9, alpha=0.55, zorder=2)
            ax.plot([r[1][key] for r in R_], ys, DS_M[ds], color=DS_C[ds], ms=5, mec="white", mew=0.6, zorder=3)
        for (yy, d, nom), yo in zip(R_, ys):
            axs[1].plot([nom, d["cov"]], [yo, yo], color=DS_C[ds], lw=1.4, alpha=0.75, zorder=2)
            axs[1].plot(d["cov"], yo, DS_M[ds], color=DS_C[ds], ms=5, mec="white", mew=0.6, zorder=3)
for yy, tag, lab, nom, g in rows:
    axs[1].plot([nom, nom], [yy - 0.38, yy + 0.38], color=C["ink"], lw=1.3, zorder=4, solid_capstyle="butt")
# range annotations (spread of the median error within each group)
for g in range(3):
    for ds, dx in [("paderborn", 0), ("induction", 1)]:
        v = [load(ds, tag, "t10")["err_med"] for yy, tag, lab, nom, gg in rows if gg == g]
        y0 = [yy for yy, tag, lab, nom, gg in rows if gg == g]
        axs[0].text(3.95, y0[0] + 0.45 + dx * 0.9, f"Δ {max(v) - min(v):.2f} K", ha="right", va="center", fontsize=5.8, color=DS_C[ds], fontweight="bold")
axs[0].set_yticks([r[0] for r in rows]); axs[0].set_yticklabels([r[2] for r in rows])
for t in axs[0].get_yticklabels():
    if t.get_text() in ("≤ 120 min", "k = 20", "α = 0.10"): t.set_fontweight("bold"); t.set_color(C["CBP"])
for title, yt in gtitle: axs[0].text(-0.02, yt, title, transform=axs[0].get_yaxis_transform(), ha="right", va="center", fontsize=7, fontweight="bold", color=C["ink"])
axs[0].set_xlim(0, 4.0); axs[0].set_xlabel("Median |error| 10 min after the event [K]")
axs[1].set_xlim(0.76, 1.0); axs[1].set_xticks([0.8, 0.85, 0.9, 0.95, 1.0]); axs[1].set_xticklabels(["0.80", "0.85", "0.90", "0.95", "1"]); axs[1].set_xlabel("Coverage at the event\n(black tick: nominal 1 − α)")
axs[2].set_xlim(0, 36); axs[2].set_xticks([10, 20, 30]); axs[2].set_xlabel("Band half-width at the event [K]")
for ax, s in zip(axs, "abc"): panel(ax, f"({s})", x=0.0, y=1.01); ax.yaxis.set_tick_params(length=0)
axs[2].text(35, rows[1][0], "frozen\nrecipe", ha="right", va="center", fontsize=5.8, color=C["CBP"], fontweight="bold")
from matplotlib.lines import Line2D
h = [Line2D([], [], ls="-", lw=0.9, marker=DS_M[d], color=DS_C[d], mec="white", ms=5) for d in BASE]
fig.legend(h, [DS_L[d] for d in BASE], loc="upper center", ncol=2, bbox_to_anchor=(0.56, 1.045))
save(fig, OUT / "fig_sensitivity")
print("ok")
