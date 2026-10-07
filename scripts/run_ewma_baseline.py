"""Data-driven baseline after estimator resets (2026-10-07, added after the locked test; CBP recipe unchanged).

Feature engineering of the public deep-pmsm code that accompanies the Paderborn benchmark (Kirchgaessner et al., 2021):
raw inputs plus exponentially weighted moving means and standard deviations with spans 840, 1320, 3360 and 6360 samples,
computed with the code's start-up convention (the history before the first sample is padded with zeros for electrical
quantities and with the first value for temperatures).  Two regressors predict the rotor/PM temperature directly:
ordinary least squares on standardised features (OLS) and a gradient-boosted tree ensemble (GBT).
Inputs: PMSM - ambient, coolant, u_d, u_q, speed, i_d, i_q, |i|, |u|, |i||u| plus the measured stator temperatures of the
scenario (S1: winding; S3: yoke, tooth, winding); induction motor - housing, ambient, RMS current and voltage, speed,
P_in - P_mech, torque plus the stator temperatures.  The models are trained on whole training units (from their start).
After an event the estimator's memory is gone, so the features are recomputed from the event onwards with the same
start-up convention ("reset" methods); for mid-run resets the same model with uninterrupted features is also reported
("no reset", not available after a real reset).
--dataset paderborn|induction  --mode cv|test   (test: trained on all development units, evaluated on the locked units)
--pad benchmark|current   benchmark: start-up convention above; current: the history before the first sample is padded
                          with the first measured value of every input (warm start from the current measurement).
"""
from __future__ import annotations
import sys, json, time, platform, uuid, argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=["paderborn", "induction"])
ap.add_argument("--mode", required=True, choices=["cv", "test"])
ap.add_argument("--pad", default="benchmark", choices=["benchmark", "current"]); args = ap.parse_args()
TS = 0.5; SPANS = [840, 1320, 3360, 6360]; TIMES = [0, 2, 4, 5, 10, 20, 30]; HOR = int(31 * 60 / TS)
CFG = {"dataset": args.dataset, "mode": args.mode, "pad": args.pad, "spans": SPANS, "seed": 20261006, "train_every": 4,
       "gbt": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 40}}

if args.dataset == "paderborn":
    from pmsm_softsense import realdata
    M = realdata.manifest(); FOLDS = M["dev_cv5_over_non_G"]; DEV = sorted(p for f in FOLDS for p in f)
    TEST = sorted(M["primary_protocol"]["test_locked_G"])
    UNITS = DEV + (TEST if args.mode == "test" else [])
    D = realdata.load(UNITS, final_evaluation=(args.mode == "test"))
    ELEC = ["u_d", "u_q", "motor_speed", "i_d", "i_q", "i_s", "u_s", "P_el"]; TEMP = ["ambient", "coolant"]
    STAT = ["stator_yoke", "stator_tooth", "stator_winding"]
    def unit(p):
        o, t = D[p]; df = o.copy()
        df["i_s"] = np.hypot(df.i_d, df.i_q); df["u_s"] = np.hypot(df.u_d, df.u_q); df["P_el"] = df.i_s * df.u_s
        return df[ELEC + TEMP + STAT].reset_index(drop=True), t.pm.values
    def events(n): return list(range(0, n - int(5 * 60 / TS), int(15 * 60 / TS)))
else:
    DD = ROOT / "data/Second Dataset"
    S2 = pd.read_csv(DD / "Second_Part.csv"); T3 = pd.read_csv(DD / "Third_Part.csv")
    audit = pd.read_csv(ROOT / "outputs/diagnostics/20261006_induction_audit/per_operating_point.csv").set_index("l")
    ALLL = [int(l) for l in sorted(S2.l.unique())]; DEV = [l for l in ALLL if l % 5 != 0]; TEST = [l for l in ALLL if l % 5 == 0]
    FOLDS = [[l for l in DEV if (l // 5) % 5 == k] for k in range(5)]
    UNITS = DEV + (TEST if args.mode == "test" else [])
    ELEC = ["i_rms", "u_rms", "speed", "p_loss", "torque"]; TEMP = ["housing", "ambient"]; STAT = ["stator_yoke", "stator_tooth", "stator_winding"]
    def unit(l):
        g = S2[S2.l == l]; t = g.t.values; p = T3[T3.l == l].sort_values("t")
        ip = lambda col: np.interp(t, p.t.values, col) if len(p) else np.zeros(len(t))
        df = pd.DataFrame({
            "i_rms": ip(p[["i_{a,r,m}", "i_{b,r,m}", "i_{c,r,m}"]].mean(axis=1).values) if len(p) else np.zeros(len(t)),
            "u_rms": ip(p[["u_{1,r,m}", "u_{2,r,m}", "u_{3,r,m}"]].mean(axis=1).values) if len(p) else np.zeros(len(t)),
            "speed": np.full(len(t), abs(audit.loc[l, "speed_mean"])), "p_loss": np.maximum(ip((p["P_{in,mot,m}"] - p["P_{mech,m}"]).values), 0) if len(p) else np.zeros(len(t)),
            "torque": np.full(len(t), float(audit.loc[l, "torque_mean"])),
            "housing": g[["theta_{S,14}", "theta_{S,15}", "theta_{S,18}"]].mean(axis=1).values,
            "ambient": g[["theta_{S,16}", "theta_{S,17}"]].mean(axis=1).values,
            "stator_yoke": g["theta_{S,5}"].values, "stator_tooth": g["theta_{S,7}"].values,
            "stator_winding": g[["theta_{S,10}", "theta_{S,11}"]].mean(axis=1).values})
        return df, g[["theta_{R,1}", "theta_{R,4}", "theta_{R,5}"]].mean(axis=1).values
    def events(n): return [0]

SCEN = {"S1": ["stator_winding"], "S3": STAT}

def featurize(df):
    """deep-pmsm style features with its start-up convention (history padded before the first sample)."""
    pad = max(SPANS); first = df.iloc[0]
    padded = pd.DataFrame({c: np.r_[np.full(pad, first[c] if (c not in ELEC or args.pad == "current") else 0.0), df[c].values] for c in df.columns})
    parts = [df.reset_index(drop=True)]
    for sp in SPANS:
        ew = padded.ewm(span=sp)
        parts.append(ew.mean().iloc[pad:].reset_index(drop=True).add_suffix(f"_m{sp}"))
        parts.append(ew.std().iloc[pad:].reset_index(drop=True).add_suffix(f"_s{sp}"))
    return pd.concat(parts, axis=1)

def cols_for(sc, all_cols):
    drop = [s for s in STAT if s not in SCEN[sc]]
    return [c for c in all_cols if not any(c == d or c.startswith(d + "_") for d in drop)]

t0 = time.time()
data = {u: unit(u) for u in UNITS}
full = {u: featurize(data[u][0]) for u in UNITS}                       # uninterrupted features from the unit start
print(f"features ready {time.time() - t0:.0f}s", flush=True)
splits = [(i, [u for u in DEV if u not in f], f) for i, f in enumerate(FOLDS)] if args.mode == "cv" else [(0, DEV, TEST)]
rows = []
for fi, train, held in splits:
    allc = list(full[train[0]].columns)
    X = pd.concat([full[u].iloc[::CFG["train_every"]] for u in train]); y = np.concatenate([data[u][1][::CFG["train_every"]] for u in train])
    models = {}
    for sc in SCEN:
        c = cols_for(sc, allc); mu, sd = X[c].mean(), X[c].std() + 1e-9; Z = ((X[c] - mu) / sd).values
        ols = LinearRegression().fit(Z, y)
        gbt = HistGradientBoostingRegressor(random_state=CFG["seed"], **CFG["gbt"]).fit(Z[::2], y[::2])
        models[sc] = (c, mu, sd, {"EWMA_OLS": ols, "EWMA_GBT": gbt})
    for u in held:
        df, pm = data[u]
        for r in events(len(df)):
            n = min(len(df), r + HOR); ks = [int(tm * 60 / TS) for tm in TIMES if r + int(tm * 60 / TS) < n]
            reset_feats = featurize(df.iloc[r:n].reset_index(drop=True))
            base = {"fold": fi, "unit": u, "r": r, "event": "start" if r == 0 else "reset"}
            for sc, (c, mu, sd, mods) in models.items():
                Zr = ((reset_feats[c].iloc[ks] - mu) / sd).values
                Zc = ((full[u][c].iloc[[r + k for k in ks]] - mu) / sd).values
                for name, m in mods.items():
                    for tag, Zx in [("reset", Zr), ("no_reset", Zc)]:
                        if tag == "no_reset" and r == 0: continue
                        pred = m.predict(Zx); rec = {**base, "scenario": sc, "method": f"{name}_{tag}"}
                        for k, pv, tm in zip(ks, pred, TIMES): rec[f"err_{tm}"] = float(abs(pv - pm[r + k]))
                        rows.append(rec)
    print(f"fold {fi} done {time.time() - t0:.0f}s", flush=True)

R = pd.DataFrame(rows)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN = ROOT / "outputs/runs" / (f"{stamp}_ewma_baseline_{args.mode}_{args.dataset}_" + ("warm_" if args.pad == "current" else "") + f"{uuid.uuid4().hex[:8]}")
RUN.mkdir(parents=True); (RUN / "source_snapshot").mkdir(); (RUN / "source_snapshot" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
(RUN / "resolved_config.json").write_text(json.dumps(CFG, indent=1)); R.to_csv(RUN / "episodes.csv", index=False)
S = {}
for (ev, sc, mth), g in R.groupby(["event", "scenario", "method"]):
    d = {"n": int(len(g))}
    for tm in TIMES:
        if f"err_{tm}" in g and g[f"err_{tm}"].notna().any():
            d[f"t{tm}"] = {"err_med": round(float(g[f"err_{tm}"].median()), 2), "err_p90": round(float(g[f"err_{tm}"].quantile(.9)), 2)}
    S[f"{ev}|{sc}|{mth}"] = d
(RUN / "summary.json").write_text(json.dumps({"meta": {"runtime_s": time.time() - t0, "python": sys.version, "platform": platform.platform()}, "summary": S}, indent=1))
(RUN / "completion.json").write_text(json.dumps({"status": "COMPLETED"}))
print(RUN)
for k, v in S.items(): print(k, v.get("t0"), v.get("t10"), v.get("t30"))
