"""Fig. results: |PM/rotor error| and 90 % band behaviour vs time after the event, both machines, S1 and S3.
Lines/bands = 5-fold CV (median, IQR); hollow markers = locked test medians.  Rows: error, coverage, half-width."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent)); from figstyle import *
ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "outputs/runs"; OUT = FIGDIR
CV = {"paderborn": latest_run("*_cv5_ablation_v2_paderborn_*").name, "induction": latest_run("*_cv5_ablation_v2_induction_*").name}
TE = {"paderborn": latest_run("*_LOCKED_TEST_paderborn_*").name, "induction": latest_run("*_LOCKED_TEST_induction_*").name}
T = [0, 2, 4, 5, 10, 20, 30]
COLS = [("paderborn", "reset", "S1", "(a) PMSM, mid-run resets\nwinding sensor only"), ("paderborn", "reset", "S3", "(b) PMSM, mid-run resets\nthree stator sensors"),
        ("induction", "start", "S1", "(c) Induction motor, starts\nwinding sensor only"), ("induction", "start", "S3", "(d) Induction motor, starts\nthree stator sensors")]
METH = ["naive_KF", "A1_labelprior", "CBP", "oracle"]
setup()
fig, axs = plt.subplots(3, 4, figsize=(W2, 122 * MM), sharex=True, gridspec_kw={"hspace": 0.16, "wspace": 0.28, "height_ratios": [1.35, 1, 1]})
for j, (ds, ev, sc, title) in enumerate(COLS):
    E = pd.read_csv(R / CV[ds] / "episodes.csv"); E = E[(E.event == ev) & (E.scenario == sc)]
    X = pd.read_csv(R / TE[ds] / "episodes.csv"); X = X[(X.event == ev) & (X.scenario == sc)]
    for m in METH:
        g = E[E.method == m]
        med = [g[f"err_{t}"].median() for t in T]; q1 = [g[f"err_{t}"].quantile(.25) for t in T]; q3 = [g[f"err_{t}"].quantile(.75) for t in T]
        z = {"CBP": 5, "naive_KF": 4, "oracle": 3, "A1_labelprior": 2}[m]
        lw = 1.1 if m == "A1_labelprior" else 1.7
        axs[0, j].plot(T, med, color=C[m], ls=LS[m], marker=MK[m], ms=3.6 if m != "A1_labelprior" else 2.8, lw=lw, label=LABEL[m], zorder=z,
                       markeredgecolor="white", markeredgewidth=0.5)
        if m in ("CBP", "naive_KF"):
            if m == "CBP": axs[0, j].fill_between(T, q1, q3, color=C[m], alpha=0.16, lw=0, zorder=1)
            gt = X[X.method == m]
            axs[0, j].plot(T, [gt[f"err_{t}"].median() for t in T], ls="none", marker=MK[m], ms=5, mfc="none", mec=C[m], mew=0.9, zorder=6)
        if m != "A1_labelprior":
            axs[1, j].plot(T, [g[f"cov_{t}"].mean() for t in T], color=C[m], ls=LS[m], marker=MK[m], ms=3.2, lw=1.3, markeredgecolor="white", markeredgewidth=0.4)
            axs[2, j].plot(T, [g[f"hw_{t}"].median() for t in T], color=C[m], ls=LS[m], marker=MK[m], ms=3.2, lw=1.3, markeredgecolor="white", markeredgewidth=0.4)
    axs[0, j].set_title(title, fontsize=7.3, color=C["ink"], pad=5, loc="left", fontweight="bold")
    axs[1, j].axhline(0.90, color=C["muted"], lw=0.8, ls=(0, (4, 2))); axs[1, j].set_ylim(0.78, 1.02)
    axs[1, j].text(30.5, 0.893, "target 0.90", fontsize=6, color=C["muted"], ha="right", va="top")
    axs[2, j].set_xlabel("Time after event [min]"); axs[2, j].set_xticks([0, 5, 10, 20, 30])
    axs[0, j].set_ylim(bottom=0); axs[2, j].set_ylim(bottom=0)
axs[0, 0].set_ylabel("Median |error| [K]\n(CBP IQR shaded)"); axs[1, 0].set_ylabel("Coverage of\n±1.645σ band"); axs[2, 0].set_ylabel("Band half-width [K]")
h, l = axs[0, 0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h += [Line2D([], [], ls="none", marker="o", mfc="none", mec=C["ink"], ms=5, mew=0.9)]; l += ["Locked test median"]
fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.6)
fig.subplots_adjust(top=0.86)
save(fig, OUT / "fig_results_time")
print("ok")
