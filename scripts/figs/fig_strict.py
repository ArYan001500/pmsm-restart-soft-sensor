"""Fig. strict split and data-driven estimators.
(a) coverage of the nominal 90 % band at the event with and without novelty scaling under the strict split (fit half and
    calibration half disjoint), in cross-validation and on withheld fast/slow units; 95 % cluster-bootstrap intervals;
(b) the price of the novelty scaling: median band half-width at the event;
(c) median PMSM magnet-temperature error after mid-run resets (cross-validation, winding sensor only): CBP of the frozen
    recipe against the naive filter and the warm-started moving-average estimators on identical events.
Numbers from outputs/analysis_strict_split/results.json and the run folders."""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent)); from figstyle import *
ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "outputs/runs"; OUT = FIGDIR
S = json.loads((ROOT / "outputs/analysis_strict_split/results.json").read_text())["strict"]
def latest(p, exclude=None):
    c = [x for x in sorted(glob.glob(str(R / p))) if not (exclude and exclude in Path(x).name)]
    return Path(c[-1])
ROWS = [("cv|paderborn|S1", "PMSM resets", "cross-validation"), ("cv|induction|S1", "IM starts", "cross-validation"),
        ("shift|paderborn|high|S1", "PMSM resets", "fastest units withheld"), ("shift|paderborn|low|S1", "PMSM resets", "slowest units withheld"),
        ("shift|induction|high|S1", "IM starts", "fastest units withheld"), ("shift|induction|low|S1", "IM starts", "slowest units withheld")]
CB, A3 = C["CBP"], C["A3_plainconf"]
setup()
fig = plt.figure(figsize=(W2, 72 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.8, 1.15], wspace=0.08)
axa = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1], sharey=axa); axc = fig.add_subplot(gs[2])
y = {k: len(ROWS) - 1 - i + (0.0 if i < 2 else -0.6) for i, (k, _, _) in enumerate(ROWS)}
for ax in (axa, axb):
    for i, (k, _, _) in enumerate(ROWS):
        if i % 2 == 0: ax.axhspan(y[k] - 0.45, y[k] + 0.45, color="#F4F4F2", lw=0, zorder=0)
    ax.grid(axis="y", visible=False)
# (a) coverage dumbbells
axa.axvspan(0.5, 0.9, color="#FCF0EB", lw=0, zorder=0); axa.axvline(0.9, color=C["ink"], lw=0.8, ls="--", zorder=1)
axa.text(0.9, y[ROWS[0][0]] + 0.72, "nominal 0.90", ha="center", va="bottom", fontsize=6.3, color=C["ink"])
axa.text(0.515, y[ROWS[-1][0]] - 0.7, "under-coverage", ha="left", va="bottom", fontsize=6.3, color="#B0461E", style="italic")
for k, ds, cond in ROWS:
    m = S[k]["methods"]; c3, cc = m["A3_plainconf"], m["CBP"]
    axa.plot([c3["cov0"], cc["cov0"]], [y[k]] * 2, color="#9AA5B1", lw=1.4, zorder=2, solid_capstyle="round")
    for r_, col, mk, dy in [(c3, A3, MK["A3_plainconf"], -0.13), (cc, CB, "o", 0.13)]:
        axa.errorbar(r_["cov0"], y[k] + dy, xerr=[[r_["cov0"] - r_["cov0_ci"][0]], [r_["cov0_ci"][1] - r_["cov0"]]], fmt=mk, ms=4.6,
                     color=col, mec="white", mew=0.5, elinewidth=0.9, capsize=0, zorder=4)
    dc = S[k]["cov0_CBP_minus_A3"]
    if dc > 0.03:
        axa.annotate(f"+{dc:.2f}", xy=(cc["cov0"], y[k]), xytext=(cc["cov0"] + 0.035, y[k]), fontsize=6.4, color=CB, va="center", fontweight="bold")
axa.set_xlim(0.5, 1.07); axa.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axa.set_xlabel("Coverage of the 90 % band at the event")
axa.set_yticks([y[k] for k, _, _ in ROWS]); axa.set_yticklabels([f"{ds}\n{cond}" for _, ds, cond in ROWS], fontsize=6.6, linespacing=1.05)
axa.tick_params(axis="y", length=0)
axa.plot([0, 0], [0, 0], "o", color=CB, label="CBP (novelty-scaled)"); axa.plot([0, 0], [0, 0], MK["A3_plainconf"], color=A3, label="A3 plain conformal")
axa.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0)
panel(axa, "(a)", x=-0.02, y=1.10)
# (b) half-width dumbbells
for k, _, _ in ROWS:
    m = S[k]["methods"]; h3, hc = m["A3_plainconf"]["hw0_med"], m["CBP"]["hw0_med"]
    axb.plot([h3, hc], [y[k]] * 2, color="#9AA5B1", lw=1.4, zorder=2)
    axb.plot(h3, y[k], MK["A3_plainconf"], color=A3, ms=4.6, mec="white", mew=0.5, zorder=4); axb.plot(hc, y[k], "o", color=CB, ms=4.6, mec="white", mew=0.5, zorder=4)
axb.axvline(49.35, color=C["naive_KF"], lw=0.9, ls=":", zorder=1); axb.text(48.5, y[ROWS[-1][0]] - 0.55, "naive\nfilter", ha="right", va="bottom", fontsize=6.2, color=C["naive_KF"])
axb.set_xlim(0, 52); axb.set_xticks([0, 10, 20, 30, 40, 50]); axb.set_xlabel("Band half-width at the event [K]")
plt.setp(axb.get_yticklabels(), visible=False); axb.tick_params(axis="y", length=0)
panel(axb, "(b)", x=-0.02, y=1.10)
# (c) error after resets: CBP vs warm-started moving-average estimators (CV, PMSM, S1), identical events
E = pd.read_csv(latest("*_cv5_ablation_v2_paderborn_*") / "episodes.csv"); E = E[(E.event == "reset") & (E.scenario == "S1")]
W = pd.read_csv(latest("*_ewma_baseline_cv_paderborn_warm_*") / "episodes.csv"); W = W[(W.event == "reset") & (W.scenario == "S1")]
W0 = pd.read_csv(latest("*_ewma_baseline_cv_paderborn_*", exclude="warm") / "episodes.csv"); W0 = W0[(W0.event == "reset") & (W0.scenario == "S1")]
T = [0, 2, 4, 5, 10, 20, 30]; key = ["unit", "r"]
base = E[E.method == "CBP"][key]
series = [("Naive KF", E[E.method == "naive_KF"], C["naive_KF"], "s", "--"),
          ("EWMA-OLS, no reset", W0[W0.method == "EWMA_OLS_no_reset"], "#8C8C8C", None, ":"),
          ("EWMA-GBT, warm start", W[W.method == "EWMA_GBT_reset"], "#E69F00", "v", "-."),
          ("EWMA-OLS, warm start", W[W.method == "EWMA_OLS_reset"], "#CC79A7", "D", "-."),
          ("CBP (proposed)", E[E.method == "CBP"], CB, "o", "-")]
for lab, G, col, mk, ls in series:
    G = base.merge(G, on=key, how="inner")
    med = [np.nanmedian(G[f"err_{t}"]) for t in T]
    axc.plot(T, med, color=col, ls=ls, marker=mk, ms=3.6 if mk else 0, mec="white", mew=0.4, lw=1.5 if lab.startswith("CBP") else 1.2, label=lab, zorder=5 if lab.startswith("CBP") else 3)
axc.set_xlim(-0.8, 31); axc.set_xticks([0, 5, 10, 20, 30]); axc.set_ylim(0, 12.5)
axc.set_xlabel("Time after the reset [min]"); axc.set_ylabel("Median |error| of magnet temperature [K]")
axc.yaxis.set_label_position("right"); axc.yaxis.tick_right(); axc.spines["right"].set_visible(True); axc.spines["left"].set_visible(False)
axc.legend(loc="upper right", fontsize=6.4, handlelength=2.2, labelspacing=0.35)
axc.text(0.96, 0.50, "PMSM resets, cross-validation,\nwinding sensor only, identical events", transform=axc.transAxes, fontsize=6.2, color=C["muted"], va="bottom", ha="right")
panel(axc, "(c)", x=-0.02, y=1.10)
save(fig, OUT / "fig_strict")
print("ok")
