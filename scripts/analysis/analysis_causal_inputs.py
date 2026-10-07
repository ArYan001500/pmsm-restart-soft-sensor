"""Additional analysis (2026-10-07, after the locked test): induction-motor cross-validation with causal inputs
(scripts/run_cv5_causal_im.py) against the frozen-recipe cross-validation (scripts/run_cv5_ablation_v2.py --dataset induction)
on identical session starts.  Paired cluster bootstrap over sessions for the CBP error and coverage at the event.
Writes outputs/analysis_causal_inputs/results.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "outputs/runs"
OUT = ROOT / "outputs/analysis_causal_inputs"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261006); B = 4000
def latest(pat): return sorted(R.glob(pat))[-1]
CA, V2 = latest("*_cv5_causal_induction_*"), latest("*_cv5_ablation_v2_induction_*")
A, Bv = pd.read_csv(CA / "episodes.csv"), pd.read_csv(V2 / "episodes.csv")
def ci(v): return [float(x) for x in np.percentile(v, [2.5, 97.5])]
res = {"causal_run": CA.name, "frozen_run": V2.name}
for sc in ["S1", "S3"]:
    a = A[(A.event == "start") & (A.scenario == sc)]; b = Bv[(Bv.event == "start") & (Bv.scenario == sc)]
    d = {}
    for m in ["CBP", "naive_KF", "A1_labelprior", "A3_plainconf", "oracle"]:
        x = a[a.method == m].set_index("unit"); y = b[b.method == m].set_index("unit")
        idx = x.index.intersection(y.index); x, y = x.loc[idx], y.loc[idx]
        st = lambda e: {"e0_med": float(e.err_0.median()), "cov0": float(e.cov_0.mean()), "hw0_med": float(e.hw_0.median()),
                        "e10_med": float(e.err_10.median()), "cov30": float(e.cov_30.mean())}
        d[m] = {"n": int(len(idx)), "causal": st(x), "frozen": st(y)}
        if m == "CBP":
            bs = [rng.choice(len(idx), len(idx)) for _ in range(B)]; xe, ye = x.err_0.values, y.err_0.values
            d[m]["e0_diff"] = float(np.median(xe) - np.median(ye)); d[m]["e0_diff_ci"] = ci([np.median(xe[k]) - np.median(ye[k]) for k in bs])
            d[m]["cov0_diff"] = float(x.cov_0.mean() - y.cov_0.mean())
    res[sc] = d
(OUT / "results.json").write_text(json.dumps(res, indent=1))
for sc in ["S1", "S3"]:
    c = res[sc]["CBP"]; print(sc, "CBP e0 frozen", round(c["frozen"]["e0_med"], 2), "causal", round(c["causal"]["e0_med"], 2),
                              "diff", round(c["e0_diff"], 2), [round(v, 2) for v in c["e0_diff_ci"]],
                              "cov0", round(c["frozen"]["cov0"], 3), "->", round(c["causal"]["cov0"], 3))
