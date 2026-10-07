"""Per-session summary of the induction-motor dataset (Stender, Wallscheid, Boecker, version 4).

Writes outputs/diagnostics/20261006_induction_audit/per_operating_point.csv, which the induction-motor experiments read
(mean speed and torque of each session, which holds one operating point), and a summary.json.
Identical to the audit script used for the paper, apart from the file locations.
"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DD = ROOT / "data/Second Dataset"
OUT = ROOT / "outputs/diagnostics/20261006_induction_audit"; OUT.mkdir(parents=True, exist_ok=True)
S = pd.read_csv(DD / "Second_Part.csv"); T3 = pd.read_csv(DD / "Third_Part.csv")
st = [c for c in S.columns if c.startswith("theta_{S")]; ro = [c for c in S.columns if c.startswith("theta_{R")]
sp, tq = [], []
for ch in pd.read_csv(DD / "First_Part.csv", usecols=["l", "n_{m}", "T_{TS,m}"], chunksize=5_000_000):
    sp.append(ch.groupby("l")["n_{m}"].agg(["mean", "std", "count"]))
    tq.append(ch.groupby("l")["T_{TS,m}"].agg(["sum", "count"]))
SP = pd.concat(sp).groupby(level=0).apply(lambda g: pd.Series({"speed_mean": np.average(g["mean"], weights=g["count"]), "speed_std_max": g["std"].max()}))
TQ = pd.concat(tq).groupby(level=0).sum(); TQ["torque_mean"] = TQ["sum"] / TQ["count"]   # mean torque-sensor signal per session
rows = []
for l, g in S.groupby("l"):
    R0 = g[ro].iloc[0].mean(); S0 = g[st].iloc[0].mean(); R1 = g[ro].iloc[-1].mean(); S1 = g[st].iloc[-1].mean()
    t3 = T3[T3.l == l]
    rows.append({"l": int(l), "dur_min": float(g.t.iloc[-1] / 60), "n2Hz": len(g), "rotor_start": R0, "stator_start": S0,
                 "rotor_minus_stator_start": R0 - S0, "rotor_end": R1, "stator_end": S1,
                 "stator_spread_start": float(g[st].iloc[0].max() - g[st].iloc[0].min()),
                 "rotor_spread_start": float(g[ro].iloc[0].max() - g[ro].iloc[0].min()),
                 "i_rms_mean": float(t3[["i_{a,r,m}", "i_{b,r,m}", "i_{c,r,m}"]].mean(axis=1).mean()) if len(t3) else np.nan,
                 "P_mech_mean": float(t3["P_{mech,m}"].mean()) if len(t3) else np.nan,
                 "speed_mean": float(SP.loc[l, "speed_mean"]) if l in SP.index else np.nan,
                 "torque_mean": float(TQ.loc[l, "torque_mean"]) if l in TQ.index else np.nan})
A = pd.DataFrame(rows); A.to_csv(OUT / "per_operating_point.csv", index=False)
summ = {"n_operating_points": int(len(A)), "total_hours": float(A.dur_min.sum() / 60), "dur_min_median": float(A.dur_min.median()),
        "stator_sensors": len(st), "rotor_sensors": len(ro),
        "rotor_minus_stator_start_quantiles": A.rotor_minus_stator_start.quantile([0, .1, .5, .9, 1]).round(2).to_dict(),
        "rotor_start_range": [float(A.rotor_start.min()), float(A.rotor_start.max())],
        "speed_range_rpm": [float(A.speed_mean.min()), float(A.speed_mean.max())],
        "i_rms_range": [float(A.i_rms_mean.min()), float(A.i_rms_mean.max())],
        "files_bytes": {p.name: p.stat().st_size for p in DD.glob("*.csv")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT / "summary.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
