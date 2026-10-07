"""Robustness summaries (2026-10-07, after the locked test; existing runs only, no new fitting):
(1) profile-balanced medians (median of per-profile medians) for the cross-validated PMSM resets and induction-motor starts;
(2) a common cohort: only events followed for the full 30 min, at 0, 10 and 30 min;
(3) numerical stability of the forward-Euler thermal network: spectral radius of I + Ts*A(omega) over the speed range,
    and whether any filter trajectory of the cross-validation became non-finite.
Writes outputs/analysis_robustness/results.json."""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from pmsm_softsense.lptn_real import RateLPTN, TS
from pmsm_softsense.reset_tools import LPTNSim
OUT = ROOT / "outputs/analysis_robustness"; OUT.mkdir(parents=True, exist_ok=True)
R = ROOT / "outputs/runs"
def latest(p): return Path(sorted(glob.glob(str(R / p)))[-1])
res = {}
for ds, ev in [("paderborn", "reset"), ("induction", "start")]:
    E = pd.read_csv(latest(f"*_cv5_ablation_v2_{ds}_*") / "episodes.csv")
    for sc in ["S1", "S3"]:
        G = E[(E.event == ev) & (E.scenario == sc)]
        d = {}
        for m in ["naive_KF", "CBP", "oracle"]:
            g = G[G.method == m]
            pb = {f"t{t}": float(g.groupby("unit")[f"err_{t}"].median().median()) for t in [0, 10, 30]}
            full = g[g["err_30"].notna()]
            co = {f"t{t}": float(full[f"err_{t}"].median()) for t in [0, 10, 30]}
            d[m] = {"pooled": {f"t{t}": float(g[f"err_{t}"].median()) for t in [0, 10, 30]}, "profile_balanced": pb,
                    "common_cohort": co, "common_cohort_n": int(len(full)),
                    "cov0_profile_balanced": float(g.groupby("unit").cov_0.mean().mean()),
                    "nonfinite_errors": int((~np.isfinite(g[[c for c in g.columns if c.startswith("err_")]].to_numpy(dtype=float)) & g[[c for c in g.columns if c.startswith("err_")]].notna().to_numpy()).sum())}
        res[f"{ds}|{ev}|{sc}"] = d
pars = json.loads((latest("*_lptn_real_baseline_*") / "lptn_params.json").read_text())
stab = {}
for k in ["equation_error", "output_error"]:
    sim = LPTNSim(RateLPTN.from_dict(pars[k]))
    rho = [float(np.max(np.abs(np.linalg.eigvals(np.eye(4) + TS * sim.mats(w)[0])))) for w in np.linspace(0, 1.0, 41)]
    stab[k] = {"max_spectral_radius_I_plus_TsA": max(rho), "min": min(rho)}
res["euler_stability_main_network"] = stab
(OUT / "results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
