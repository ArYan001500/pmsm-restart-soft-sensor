"""Fig. derating: simulated thermal derating in the first 10 min after a reset (Paderborn validation resets, 61 events;
plant = equation-error LPTN, estimator = output-error LPTN).  Dumbbell: naive KF with a conformally calibrated band ->
CBP -> oracle mean derating factor (fraction of requested current admitted), 95 % cluster-bootstrap intervals over
profiles; label = share of the calibrated-naive-to-oracle gap recovered by CBP.  The naive KF with its uncalibrated
49.4 K band is shown as an open marker.  Thermal-limit violations were identical for all estimators (heat soak)."""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).parent)); from figstyle import *
ROOT = Path(__file__).resolve().parents[2]; OUT = FIGDIR
RUN = Path(sorted(glob.glob(str(ROOT / "outputs/runs/*_derating_sim_*")))[-1])
E = pd.read_csv(RUN / "episodes.csv"); E = E[E.event == "reset"]
assert "naive_conf" in set(E.method), "derating run without the calibrated naive comparator"
rng = np.random.default_rng(20261006)
cases = [(70.0, "S1"), (70.0, "S3"), (85.0, "S1"), (85.0, "S3")]
stats = {}
setup()
fig, ax = plt.subplots(figsize=(W1, 62 * MM))
for i, (tl, sc) in enumerate(cases):
    g = E[(E.T_lim == tl) & (E.scenario == sc)]; y = len(cases) - 1 - i
    units = g.pid.unique(); P = g.pivot_table(index=["pid", "r"], columns="method", values="torque_frac_10min")
    vals = {m: P[m].mean() for m in ["naive_KF", "naive_conf", "CBP_cal", "oracle"]}
    rows = {u: np.where(P.index.get_level_values(0) == u)[0] for u in units}
    idx = [np.concatenate([rows[u] for u in rng.choice(units, len(units))]) for _ in range(2000)]
    rec = {}
    for ref in ["naive_conf", "naive_KF"]:
        d = (P["CBP_cal"] - P[ref]).values; bs = [d[b].mean() for b in idx]
        rec[ref] = {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "gap_recovered": float((vals["CBP_cal"] - vals[ref]) / (vals["oracle"] - vals[ref]))}
    stats[f"{int(tl)}|{sc}"] = {"means": {k: float(v) for k, v in vals.items()}, "vs": rec}
    ax.plot([vals["naive_conf"], vals["oracle"]], [y, y], color="#D5D5D1", lw=3.4, solid_capstyle="round", zorder=1)
    ax.annotate("", xy=(vals["CBP_cal"] - 0.012, y), xytext=(vals["naive_conf"] + 0.012, y),
                arrowprops=dict(arrowstyle="-|>", color=C["CBP"], lw=1.2, mutation_scale=7), zorder=2)
    ax.plot(vals["naive_KF"], y, MK["naive_KF"], mfc="white", mec=C["naive_KF"], mew=0.9, ms=4.6, zorder=3)
    ax.plot(vals["naive_conf"], y, MK["naive_KF"], color=C["naive_KF"], mec="white", mew=0.6, ms=5, zorder=3)
    ax.plot(vals["CBP_cal"], y, MK["CBP"], color=C["CBP"], mec="white", mew=0.6, ms=6, zorder=3)
    ax.plot(vals["oracle"], y, MK["oracle"], color=C["oracle"], mec="white", mew=0.6, ms=5, zorder=3)
    r_ = rec["naive_conf"]
    ax.text(vals["CBP_cal"], y + 0.22, f"{r_['gap_recovered'] * 100:.0f} % of gap", ha="center", fontsize=6.2, color=C["CBP"], fontweight="bold")
    ax.text(vals["CBP_cal"], y - 0.36, f"+{r_['diff']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]", ha="center", fontsize=5.6, color=C["muted"])
ax.set_yticks(range(len(cases))); ax.set_yticklabels([f"{int(tl)} °C limit\n{'winding only' if sc == 'S1' else 'three sensors'}" for tl, sc in cases][::-1])
ax.set_xlim(0.35, 1.03); ax.set_ylim(-0.6, len(cases) - 0.35); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean admitted current fraction, first 10 min after reset")
h = [Line2D([], [], ls="none", marker=MK["naive_KF"], mfc="white", mec=C["naive_KF"], mew=0.9, ms=4.6),
     Line2D([], [], ls="none", marker=MK["naive_KF"], color=C["naive_KF"], mec="white", ms=5),
     Line2D([], [], ls="none", marker=MK["CBP"], color=C["CBP"], mec="white", ms=5),
     Line2D([], [], ls="none", marker=MK["oracle"], color=C["oracle"], mec="white", ms=5)]
ax.legend(h, ["Naive KF", "Naive, calibrated band", "CBP", "Oracle"], loc="upper center", bbox_to_anchor=(0.45, 1.2), ncol=2,
          fontsize=6.2, handletextpad=0.2, columnspacing=0.8)
save(fig, OUT / "fig_derating")
(OUT / "fig_derating_stats.json").write_text(json.dumps({"run": RUN.name, "stats": stats}, indent=1))
print(json.dumps(stats, indent=1))
