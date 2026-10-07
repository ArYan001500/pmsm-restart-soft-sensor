"""Fig. ablation: one-factor ablations of CBP on both machines (5-fold CV, winding sensor only S1).
(a) difference in median |error| at the event (variant - CBP), (b) coverage of the 90 % band at the event, (c) coverage at 30 min.
95 % intervals: cluster bootstrap over profiles / sessions (2000 replicates)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent)); from figstyle import *
ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "outputs/runs"; OUT = FIGDIR
CV = {"paderborn": (latest_run("*_cv5_ablation_v2_paderborn_*").name, "reset", "PMSM resets (64 profiles)"),
      "induction": (latest_run("*_cv5_ablation_v2_induction_*").name, "start", "Induction motor starts (223 sessions)")}
ROWS = ["naive_KF", "A1_labelprior", "A2_linear", "A3_plainconf", "A4_uncal", "A5_stdQ", "A7_no_event_match", "A8_simstart", "CBP"]
DS_C = {"paderborn": "#0072B2", "induction": "#D55E00"}; DS_M = {"paderborn": "o", "induction": "s"}
rng = np.random.default_rng(20261006); B = 2000; res = []
for ds, (run, ev, _) in CV.items():
    E = pd.read_csv(R / run / "episodes.csv"); E = E[(E.event == ev) & (E.scenario == "S1")]
    units = np.array(sorted(E.unit.unique()))
    piv = {c: E.pivot_table(index=["unit", "r"], columns="method", values=c) for c in ["err_0", "cov_0", "cov_30", "hw_0"]}
    piv = {c: v.reindex(piv["err_0"].index) for c, v in piv.items()}
    uix = piv["err_0"].index.get_level_values(0); rows_of = {u: np.where(uix == u)[0] for u in units}
    boots = [np.concatenate([rows_of[u] for u in rng.choice(units, len(units))]) for _ in range(B)]
    for m in ROWS:
        rec = {"ds": ds, "m": m}
        if m != "CBP":
            d = piv["err_0"][m].values; c = piv["err_0"]["CBP"].values
            rec["derr"] = np.nanmedian(d) - np.nanmedian(c); bs = [np.nanmedian(d[b]) - np.nanmedian(c[b]) for b in boots]
            rec["derr_lo"], rec["derr_hi"] = np.percentile(bs, [2.5, 97.5])
        for k in ["cov_0", "cov_30", "hw_0"]:
            v = piv[k][m].values; ok = ~np.isnan(v); f = np.nanmedian if k == "hw_0" else np.nanmean; rec[k] = f(v)
            bs = [f(v[b]) for b in boots]; rec[k + "_lo"], rec[k + "_hi"] = np.percentile(bs, [2.5, 97.5])
        res.append(rec)
D = pd.DataFrame(res); D.to_csv(OUT / "fig_ablation_data.csv", index=False)
setup()
fig, axs = plt.subplots(1, 4, figsize=(W2, 70 * MM), sharey=True, gridspec_kw={"wspace": 0.10, "width_ratios": [1.15, 1, 1, 1]})
y = {m: len(ROWS) - 1 - i for i, m in enumerate(ROWS)}; off = {"paderborn": 0.17, "induction": -0.17}
for ax in axs:
    for i, m in enumerate(ROWS):
        if i % 2 == 0: ax.axhspan(y[m] - 0.5, y[m] + 0.5, color="#F4F4F2", lw=0, zorder=0)
    ax.grid(axis="y", visible=False)
for ds in CV:
    g = D[D.ds == ds]
    for _, r in g.iterrows():
        yy = y[r.m] + off[ds]
        if r.m != "CBP" and r.derr_hi - r.derr_lo < 1e-9:
            axs[0].text(0.15, yy, "same prior", fontsize=5.6, color=C["muted"], va="center", ha="left") if ds == "paderborn" else None
        elif r.m != "CBP":
            axs[0].errorbar(r.derr, yy, xerr=[[r.derr - r.derr_lo], [r.derr_hi - r.derr]], fmt=DS_M[ds], color=DS_C[ds], ms=4, lw=1.1, capsize=0,
                            mec="white", mew=0.5, zorder=3)
        for ax, k in [(axs[1], "cov_0"), (axs[2], "cov_30"), (axs[3], "hw_0")]:
            ax.errorbar(r[k], yy, xerr=[[r[k] - r[k + "_lo"]], [r[k + "_hi"] - r[k]]], fmt=DS_M[ds], color=DS_C[ds], ms=4, lw=1.1, capsize=0,
                        mec="white", mew=0.5, zorder=3)
axs[0].axvline(0, color=C["ink"], lw=0.8); axs[0].set_xlim(-2.6, 8.2); axs[0].set_xticks([-2, 0, 2, 4, 6, 8])
axs[0].set_xlabel("Δ median |error| vs CBP\nat the event [K]  (>0: worse)")
for ax, t in [(axs[1], "at the event"), (axs[2], "after 30 min")]:
    ax.axvline(0.90, color=C["muted"], lw=0.8, ls=(0, (4, 2))); ax.set_xlim(0.2, 1.03); ax.set_xlabel(f"Coverage of 90 % band\n{t}"); ax.set_xticks([0.25, 0.5, 0.75, 0.9, 1.0]); ax.set_xticklabels(["0.25", "0.50", "0.75", "0.90", "1"])
axs[0].set_yticks([y[m] for m in ROWS]); axs[0].set_yticklabels([LABEL[m] for m in ROWS]); axs[0].set_ylim(-0.6, len(ROWS) - 0.4)
axs[0].get_yticklabels()[-1].set_fontweight("bold")
axs[3].set_xlabel("Band half-width\nat the event [K]"); axs[3].set_xlim(0, 52)
for ax, s in zip(axs, "abcd"): panel(ax, f"({s})", x=0.0, y=1.02); ax.yaxis.set_tick_params(length=0)
from matplotlib.lines import Line2D
h = [Line2D([], [], ls="-", marker=DS_M[d], color=DS_C[d], mec="white", ms=5) for d in CV]
fig.legend(h, [CV[d][2] for d in CV], loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.06))
save(fig, OUT / "fig_ablation_forest")
print(D.round(3).to_string())
