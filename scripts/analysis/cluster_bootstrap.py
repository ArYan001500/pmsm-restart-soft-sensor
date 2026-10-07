"""Cluster (unit-level) paired bootstrap CIs on v2 CV episodes, both machines.  Pre-committed: all comparisons below are
reported whatever the result.  Statistic: difference of medians and of p90 of |PM error| (method - CBP), and coverage.
Resampling unit = profile (Paderborn) / session (induction); 4000 replicates; 95 % percentile intervals."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/figs")); from figstyle import latest_run
RUNS = {"paderborn": latest_run("*_cv5_ablation_v2_paderborn_*").name, "induction": latest_run("*_cv5_ablation_v2_induction_*").name}
COMP = ["naive_KF", "A1_labelprior", "A2_linear", "A3_plainconf", "A4_uncal", "A5_stdQ", "A7_no_event_match", "A8_simstart"]
TIMES = [0, 2, 10, 30]; B = 4000; rng = np.random.default_rng(20261006); out = []
for ds, run in RUNS.items():
    E = pd.read_csv(ROOT / "outputs/runs" / run / "episodes.csv")
    for ev in sorted(E.event.unique()):
        for sc in ["S1", "S3"]:
            G = E[(E.event == ev) & (E.scenario == sc)]
            units = np.array(sorted(G.unit.unique())); idx = {u: G.index[G.unit == u].values for u in units}
            boots = [np.concatenate([idx[u] for u in rng.choice(units, len(units))]) for _ in range(B)]
            for tm in TIMES:
                piv = G.pivot_table(index=["unit", "r"], columns="method", values=f"err_{tm}")
                cov = G.pivot_table(index=["unit", "r"], columns="method", values=f"cov_{tm}")
                if "CBP" not in piv: continue
                uidx = {u: np.where(piv.index.get_level_values(0) == u)[0] for u in units}
                bi = [np.concatenate([uidx[u] for u in rng.choice(units, len(units))]) for _ in range(B)]
                c = piv["CBP"].values
                for mth in COMP + ["CBP"]:
                    if mth not in piv: continue
                    x = piv[mth].values; ok = ~np.isnan(x) & ~np.isnan(c)
                    rec = {"dataset": ds, "event": ev, "scenario": sc, "t_min": tm, "method": mth, "n_events": int(ok.sum()), "n_units": len(units)}
                    if mth == "CBP":
                        cv = cov["CBP"].values; bs = [np.nanmean(cv[b]) for b in bi]
                        rec.update(cov=float(np.nanmean(cv)), cov_lo=float(np.percentile(bs, 2.5)), cov_hi=float(np.percentile(bs, 97.5)))
                    else:
                        for stat, f in [("dmed", np.nanmedian), ("dp90", lambda a: np.nanpercentile(a, 90))]:
                            d0 = f(x[ok]) - f(c[ok]); bs = []
                            for b in bi:
                                bb = b[ok[b]]; bs.append(f(x[bb]) - f(c[bb]))
                            lo, hi = np.percentile(bs, [2.5, 97.5])
                            rec.update({stat: float(d0), stat + "_lo": float(lo), stat + "_hi": float(hi), stat + "_sig": bool(lo > 0 or hi < 0)})
                        if mth in cov:
                            cm = cov[mth].values; rec["cov_method"] = float(np.nanmean(cm))
                    out.append(rec)
R = pd.DataFrame(out); here = ROOT / "outputs/analysis"; here.mkdir(parents=True, exist_ok=True)
R.to_csv(here / "cluster_bootstrap_ci.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
cols = ["dataset", "event", "scenario", "t_min", "method", "n_units", "dmed", "dmed_lo", "dmed_hi", "dmed_sig", "dp90", "dp90_lo", "dp90_hi", "dp90_sig", "cov", "cov_lo", "cov_hi"]
print(R[cols].round(2).to_string(index=False))
