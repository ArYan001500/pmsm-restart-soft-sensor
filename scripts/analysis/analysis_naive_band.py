"""Additional analysis (2026-10-07, after the locked test): a naive initialiser with a conformally calibrated band (no new fitting).
The point estimate is the measured winding temperature at the event, as in the naive Kalman filter; its band half-width is
the split-conformal quantile of |winding - rotor| on the same calibration events that CBP uses in the frozen recipe
(PMSM resets: points every 5 min of training half B; induction-motor starts: the first sample of every training session).
Evaluated on the held-out events of the five-fold cross-validation (winding sensor only), with cluster-bootstrap
intervals.  Also the group-wise variant (one calibration event per profile, as in run_cv5_groupconf.py) for the naive band
and the CBP results of that run.  Writes outputs/analysis_naive_band/results.json."""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/analysis_naive_band"; OUT.mkdir(parents=True, exist_ok=True)
TS = 0.5; ALPHA = 0.10; rng = np.random.default_rng(20261006); B = 4000
def cq(s, a):
    s = np.sort(np.asarray(s)); n = len(s); k = int(np.ceil((n + 1) * (1 - a))); return float(s[min(k, n) - 1])
def mid_points(n): return list(range(0, n - 600, int(5 * 60 / TS)))

def evaluate(Y, FOLDS, ALL, events, calib):
    rows = []
    for fi, held in enumerate(FOLDS):
        train = [u for u in ALL if u not in held]; q = cq(calib(train), ALPHA)
        for u in held:
            for r in events(u):
                e = abs(Y[u][r, 3] - Y[u][r, 0]); rows.append({"fold": fi, "unit": u, "r": r, "err": e, "cov": float(e <= q), "hw": q})
    R = pd.DataFrame(rows); units = np.array(sorted(R.unit.unique())); g = {u: np.where(R.unit.values == u)[0] for u in units}
    bs = [np.concatenate([g[u] for u in rng.choice(units, len(units))]) for _ in range(B)]
    c = R["cov"].values
    return {"n_events": int(len(R)), "n_units": int(len(units)), "err_med": float(R.err.median()), "cov": float(c.mean()),
            "cov_ci": [float(np.percentile([c[b].mean() for b in bs], p)) for p in (2.5, 97.5)], "hw_med": float(R.hw.median()),
            "hw_per_fold": [round(float(x), 2) for x in R.groupby("fold").hw.first()]}

res = {}
from pmsm_softsense import realdata
M = realdata.manifest(); FOLDS = M["dev_cv5_over_non_G"]; ALL = sorted(p for f in FOLDS for p in f); D = realdata.load(ALL)
Y = {p: np.column_stack([D[p][1].pm.values, D[p][0].stator_yoke.values, D[p][0].stator_tooth.values, D[p][0].stator_winding.values]) for p in ALL}
reset_events = lambda u: [r for r in range(0, len(Y[u]) - int(5 * 60 / TS), int(15 * 60 / TS)) if r > 0]
calib_mid = lambda train: [abs(Y[p][r, 3] - Y[p][r, 0]) for p in sorted(train)[1::2] for r in mid_points(len(Y[p]))]
res["paderborn_reset_naive_conformal"] = evaluate(Y, FOLDS, ALL, reset_events, calib_mid)
# group-wise calibration: one event per calibration profile, drawn as in run_cv5_groupconf.py
pick = lambda fi, p: int(np.random.default_rng([20261006, fi, int(p)]).choice(mid_points(len(Y[p]))))
def evaluate_group(Y, FOLDS, ALL, events):
    rows = []
    for fi, held in enumerate(FOLDS):
        train = [u for u in ALL if u not in held]
        q = cq([abs(Y[p][pick(fi, p), 3] - Y[p][pick(fi, p), 0]) for p in sorted(train)[1::2]], ALPHA)
        for u in held:
            for r in events(u):
                e = abs(Y[u][r, 3] - Y[u][r, 0]); rows.append({"unit": u, "cov": float(e <= q), "hw": q})
    R = pd.DataFrame(rows); units = np.array(sorted(R.unit.unique())); g = {u: np.where(R.unit.values == u)[0] for u in units}
    c = R["cov"].values; bs = [c[np.concatenate([g[u] for u in rng.choice(units, len(units))])].mean() for _ in range(B)]
    return {"n_events": int(len(R)), "cov": float(c.mean()), "cov_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "hw_med": float(R.hw.median())}
res["paderborn_reset_naive_groupconformal"] = evaluate_group(Y, FOLDS, ALL, reset_events)
# CBP and A3 of the group-conformal run (run_cv5_groupconf.py), cluster-bootstrap intervals
G = pd.read_csv(sorted(glob.glob(str(ROOT / "outputs/runs/*_cv5_groupconf_paderborn_*")))[-1] + "/episodes.csv")
for sc in ["S1", "S3"]:
    for m in ["CBP", "A3_plainconf", "A1_labelprior"]:
        g_ = G[(G.event == "reset") & (G.scenario == sc) & (G.method == m)]
        units = np.array(sorted(g_.unit.unique())); ix = {u: np.where(g_.unit.values == u)[0] for u in units}
        c = g_.cov_0.values; bs = [c[np.concatenate([ix[u] for u in rng.choice(units, len(units))])].mean() for _ in range(B)]
        res[f"groupconf|paderborn|{sc}|{m}"] = {"e0_med": float(g_.err_0.median()), "cov0": float(c.mean()), "cov0_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                               "hw0_med": float(g_.hw_0.median()), "cov10": float(g_.cov_10.mean()), "cov30": float(g_.cov_30.mean())}

DD = ROOT / "data/Second Dataset"; S2 = pd.read_csv(DD / "Second_Part.csv")
ALLI = [int(l) for l in sorted(S2.l.unique()) if int(l) % 5 != 0]; FOLDSI = [[l for l in ALLI if (l // 5) % 5 == k] for k in range(5)]
YI = {}
for l in ALLI:
    g = S2[S2.l == l]
    YI[l] = np.column_stack([g[["theta_{R,1}", "theta_{R,4}", "theta_{R,5}"]].mean(axis=1).values, g["theta_{S,5}"].values,
                             g["theta_{S,7}"].values, g[["theta_{S,10}", "theta_{S,11}"]].mean(axis=1).values])
calib_start = lambda train: [abs(YI[p][0, 3] - YI[p][0, 0]) for p in train]
res["induction_start_naive_conformal"] = evaluate(YI, FOLDSI, ALLI, lambda u: [0], calib_start)
(OUT / "results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
