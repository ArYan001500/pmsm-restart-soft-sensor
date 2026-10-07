"""Fig. derating: closed-loop thermal derating in the first 10 min after a reset (Paderborn validation resets, 61 events;
plant = equation-error LPTN, estimator = output-error LPTN).  Dumbbell: naive KF -> CBP -> oracle mean derating factor
(fraction of requested current admitted), 95 % cluster-bootstrap intervals over profiles; label = share of the
naive-to-oracle gap recovered by CBP.  Thermal-limit violations were identical for all three estimators (heat soak)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent)); from figstyle import *
ROOT = Path(__file__).resolve().parents[2]; OUT = FIGDIR
E = pd.read_csv(latest_run("*_derating_sim_*") / "episodes.csv"); E = E[E.event == "reset"]
rng = np.random.default_rng(20261006); M = {"naive_KF": "naive_KF", "CBP_cal": "CBP", "oracle": "oracle"}
cases = [(70.0, "S1"), (70.0, "S3"), (85.0, "S1"), (85.0, "S3")]
setup()
fig, ax = plt.subplots(figsize=(W1, 60 * MM))
for i, (tl, sc) in enumerate(cases):
    g = E[(E.T_lim == tl) & (E.scenario == sc)]; y = len(cases) - 1 - i
    units = g.pid.unique(); P = g.pivot_table(index=["pid", "r"], columns="method", values="torque_frac_10min")
    vals = {m: P[m].mean() for m in M}
    rows = {u: np.where(P.index.get_level_values(0) == u)[0] for u in units}; d = (P["CBP_cal"] - P["naive_KF"]).values
    bs = [d[np.concatenate([rows[u] for u in rng.choice(units, len(units))])].mean() for _ in range(2000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    ax.plot([vals["naive_KF"], vals["oracle"]], [y, y], color="#D5D5D1", lw=3.4, solid_capstyle="round", zorder=1)
    ax.annotate("", xy=(vals["CBP_cal"] - 0.012, y), xytext=(vals["naive_KF"] + 0.012, y),
                arrowprops=dict(arrowstyle="-|>", color=C["CBP"], lw=1.2, mutation_scale=7), zorder=2)
    for m, key in M.items():
        ax.plot(vals[m], y, MK[key], color=C[key], ms=6 if key == "CBP" else 5, mec="white", mew=0.6, zorder=3)
    rec = (vals["CBP_cal"] - vals["naive_KF"]) / (vals["oracle"] - vals["naive_KF"])
    ax.text(vals["CBP_cal"], y + 0.22, f"{rec * 100:.0f} % of gap", ha="center", fontsize=6.2, color=C["CBP"], fontweight="bold")
    ax.text(vals["CBP_cal"], y - 0.36, f"+{d.mean():.2f} [{lo:.2f}, {hi:.2f}]", ha="center", fontsize=5.6, color=C["muted"])
ax.set_yticks(range(len(cases))); ax.set_yticklabels([f"{int(tl)} °C limit\n{'winding only' if sc == 'S1' else 'three sensors'}" for tl, sc in cases][::-1])
ax.set_xlim(0.35, 1.03); ax.set_ylim(-0.6, len(cases) - 0.35); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean admitted current fraction, first 10 min after reset")
from matplotlib.lines import Line2D
h = [Line2D([], [], ls="none", marker=MK[k], color=C[k], mec="white", ms=5) for k in ["naive_KF", "CBP", "oracle"]]
ax.legend(h, [LABEL[k] for k in ["naive_KF", "CBP", "oracle"]], loc="upper center", bbox_to_anchor=(0.45, 1.14), ncol=3, fontsize=6.4, handletextpad=0.2)
save(fig, OUT / "fig_derating")
print("ok")
