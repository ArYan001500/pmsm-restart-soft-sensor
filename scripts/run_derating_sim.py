"""Closed-loop thermal derating after estimator memory loss (2026-10-06).

Plant  : equation-error LPTN (identified separately from the estimator's output-error LPTN), driven by real V1/V2
         input profiles; plant state at reset = real measured temperatures incl. PM (realistic hidden state).
Control: every 10 s choose the largest torque factor f in {1, .75, .5, .25, 0} (current scaled by f, speed unchanged)
         such that the estimator's predicted PM upper bound (mean + 1.645 sigma) over the next 60 s stays <= T_lim.
Estimators (same OE-LPTN Kalman filter, winding-only S1 and 3-sensor S3):
         naive_KF  (PM = winding, P0 = 30^2), CBP_cal (label-free prior + novelty-scaled conformal band, q_pm x2),
         naive_conf (PM = winding, rotor P0 from a split-conformal quantile of |winding - PM| on the same calibration
                    events as CBP: training half B, every 5 min; an uncertainty-matched comparator),
         oracle    (true plant PM at reset, tiny P0; not available online).
Research threshold T_lim is a study choice (not an OEM value): two levels.
Metrics over 30 min after each reset: delivered torque fraction (mean f), violation degree-seconds (plant PM above T_lim).
"""
from __future__ import annotations
import sys, json, glob, time, hashlib, platform, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pmsm_softsense import realdata
from pmsm_softsense.lptn_real import RateLPTN, features, TS
from pmsm_softsense.reset_tools import LPTNSim, simulate_histories

CFG = {"seed": 20261006, "n_orders": 3, "gap_min_max": 120, "record_every": 60, "calib_every_min": 5,
       "eval_reset_every_min": 15, "alpha": 0.10, "knn_k": 20, "horizon_min": 30, "decision_every_s": 10,
       "lookahead_s": 60, "factors": [1.0, 0.75, 0.5, 0.25, 0.0], "T_lim_C": [70.0, 85.0], "sigma_meas": 0.3,
       "q_pm": 2.0, "q_stator": 8.0, "plant_meas_noise": 0.1,
       "gbr": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 40}}
rng = np.random.default_rng(CFG["seed"])
LPTN_RUN = Path(sorted(glob.glob(str(ROOT / "outputs/runs/*_lptn_real_baseline_*")))[-1])
pars = json.loads((LPTN_RUN / "lptn_params.json").read_text())
est = LPTNSim(RateLPTN.from_dict(pars["output_error"]))
plant = LPTNSim(RateLPTN.from_dict(pars["equation_error"]))
TRAIN = realdata.protocol_ids("train"); VAL = realdata.protocol_ids("val")
D = realdata.load(TRAIN + VAL)
SCEN = {"S3": {"m": [1, 2, 3], "h": [0]}, "S1": {"m": [3], "h": [0, 1, 2]}}
def prof(pid):
    o, t = D[pid]; F = features(o)
    return o, F, np.column_stack([t.pm.values, o.stator_yoke.values, o.stator_tooth.values, o.stator_winding.values])
cache = {p: prof(p) for p in TRAIN + VAL}
res = []
for pid in TRAIN[::3]:
    o, F, Y = cache[pid]
    for k in range(0, len(o) - 1, 7):
        Tn, _ = est.step(Y[k], k, F); res.append(Y[k + 1] - Tn)
v = np.var(np.array(res), 0); QP = np.diag([v[0] * CFG["q_pm"]] + list(v[1:] * CFG["q_stator"]))
def cq(s, a):
    s = np.sort(np.asarray(s)); n = len(s); k = int(np.ceil((n + 1) * (1 - a))); return float(s[min(k, n) - 1])
def ctx(F, r): return [F["bnd"][r, 0], F["bnd"][r, 1], F["is2"][r], F["wn"][r], F["us2"][r]]

# ---- CBP priors + calibration (identical recipe to run_cbp_qpm_tuning.py, group 'all')
Hh = simulate_histories(est, cache, TRAIN, CFG["n_orders"], CFG["gap_min_max"], CFG["record_every"], rng)
B_ids = set(sorted(TRAIN)[1::2]); PRI = {}
for sc, sp in SCEN.items():
    m, h = sp["m"], sp["h"]; X = np.column_stack([Hh[:, m], Hh[:, 4:9]]); gbs = []
    for j in h:
        kw = dict(CFG["gbr"]); kw["random_state"] = CFG["seed"]; gbs.append(HistGradientBoostingRegressor(**kw).fit(X, Hh[:, j]))
    Csim = np.cov(np.column_stack([Hh[:, j] - gbs[i].predict(X) for i, j in enumerate(h)]).T).reshape(len(h), len(h))
    a, b = X.mean(0), X.std(0) + 1e-9; nn = NearestNeighbors(n_neighbors=CFG["knn_k"]).fit((X - a) / b)
    dref = float(np.median(nn.kneighbors((X[::7] - a) / b)[0].mean(1)))
    def pri(x, gbs=gbs, nn=nn, a=a, b=b, dref=dref):
        mu = np.array([gg.predict(x[None])[0] for gg in gbs])
        return mu, max(float(nn.kneighbors(((x - a) / b)[None])[0].mean() / dref) - 1, 0.0)
    scB = []
    for pid in [p for p in TRAIN if p in B_ids]:
        o, F, Y = cache[pid]
        for r in range(0, len(o) - 600, int(CFG["calib_every_min"] * 60 / TS)):
            mu, nov = pri(np.concatenate([Y[r, m], ctx(F, r)])); scB.append(abs(mu[0] - Y[r, 0]) / (1 + nov))
    PRI[sc] = (pri, Csim, cq(scB, CFG["alpha"]))
Q_NAIVE = cq([abs(cache[p][2][r, 3] - cache[p][2][r, 0]) for p in TRAIN if p in B_ids
              for r in range(0, len(cache[p][0]) - 600, int(CFG["calib_every_min"] * 60 / TS))], CFG["alpha"])
print("naive conformal half-width", round(Q_NAIVE, 2), flush=True)

def scaled(F, k, f):
    return {"wn": F["wn"], "us2": F["us2"], "bnd": F["bnd"], "is2": F["is2"] * f * f}

def episode(pid, r, sc, method, T_lim):
    o, F, Y = cache[pid]; m, h = SCEN[sc]["m"], SCEN[sc]["h"]
    n = min(len(o), r + int(CFG["horizon_min"] * 60 / TS))
    xp = Y[r].copy()                                         # plant true state at reset
    x = Y[r].copy(); P = np.zeros((4, 4))
    for i in m: P[i, i] = CFG["sigma_meas"] ** 2
    if method == "naive_KF":
        x[h] = Y[r, 3]
        for i in h: P[i, i] = (30.0 if i == 0 else 15.0) ** 2
    elif method == "naive_conf":
        x[h] = Y[r, 3]
        for i in h: P[i, i] = (Q_NAIVE / 1.645) ** 2 if i == 0 else 15.0 ** 2
    elif method == "CBP_cal":
        pri, Csim, qn = PRI[sc]
        mu, nov = pri(np.concatenate([Y[r, m], ctx(F, r)])); x[h] = mu
        hw = qn * (1 + nov); P[np.ix_(h, h)] = Csim * ((hw / 1.645) ** 2 / max(Csim[0, 0], 1e-9))
    else:
        for i in h: P[i, i] = 0.25
    H = np.zeros((len(m), 4))
    for i, j in enumerate(m): H[i, j] = 1
    Rm = np.eye(len(m)) * CFG["sigma_meas"] ** 2
    f = 1.0; fs = []; viol = 0.0; dk = int(CFG["decision_every_s"] / TS); la = int(CFG["lookahead_s"] / TS)
    for k in range(r, n - 1):
        if (k - r) % dk == 0:
            f = 0.0
            for cand in CFG["factors"]:
                Fs = scaled(F, k, cand); xx = x.copy(); PP = P.copy(); ok = True
                for j in range(k, min(n - 1, k + la)):
                    xx, J = est.step(xx, j, Fs); PP = J @ PP @ J.T + QP
                    if xx[0] + 1.645 * np.sqrt(max(PP[0, 0], 0)) > T_lim: ok = False; break
                if ok: f = cand; break
        Fs = scaled(F, k, f)
        xp, _ = plant.step(xp, k, Fs)                       # plant under derated current
        x, J = est.step(x, k, Fs); P = J @ P @ J.T + QP
        y = xp[m] + rng.normal(0, CFG["plant_meas_noise"], len(m))
        S = H @ P @ H.T + Rm; K = P @ H.T @ np.linalg.inv(S); x = x + K @ (y - H @ x); P = (np.eye(4) - K @ H) @ P
        fs.append(f); viol += max(xp[0] - T_lim, 0.0) * TS
    fs = np.array(fs); k10 = int(600 / TS)
    return {"torque_frac_30min": float(fs.mean()), "torque_frac_10min": float(fs[:k10].mean()), "viol_degsec": viol,
            "plant_pm_max": None}

t0 = time.time(); rows = []
for T_lim in CFG["T_lim_C"]:
    for pid in VAL:
        o, F, Y = cache[pid]
        for r in range(0, len(o) - int(5 * 60 / TS), int(CFG["eval_reset_every_min"] * 60 / TS)):
            for sc in SCEN:
                for meth in ["naive_KF", "naive_conf", "CBP_cal", "oracle"]:
                    rec = episode(pid, r, sc, meth, T_lim)
                    rec.update({"T_lim": T_lim, "pid": pid, "r": r, "event": "start" if r == 0 else "reset", "scenario": sc, "method": meth})
                    rows.append(rec)
        print(f"T_lim {T_lim} profile {pid} done {time.time()-t0:.0f}s", flush=True)
R = pd.DataFrame(rows)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN = ROOT / "outputs/runs" / f"{stamp}_derating_sim_{uuid.uuid4().hex[:8]}"
RUN.mkdir(parents=True); (RUN / "source_snapshot").mkdir()
for p in [Path(__file__), ROOT / "src/pmsm_softsense/reset_tools.py"]:
    (RUN / "source_snapshot" / p.name).write_bytes(p.read_bytes())
(RUN / "resolved_config.json").write_text(json.dumps(CFG, indent=1)); R.to_csv(RUN / "episodes.csv", index=False)
S = {}
for (tl, sc, mth), g in R[R.event == "reset"].groupby(["T_lim", "scenario", "method"]):
    S[f"Tlim={tl}|{sc}|{mth}"] = {"n": int(len(g)), "torque_frac_10min_mean": round(float(g.torque_frac_10min.mean()), 3),
                                  "torque_frac_30min_mean": round(float(g.torque_frac_30min.mean()), 3),
                                  "episodes_with_violation": int((g.viol_degsec > 0).sum()),
                                  "viol_degsec_total": round(float(g.viol_degsec.sum()), 1)}
meta = {"naive_conformal_halfwidth_K": Q_NAIVE, "runtime_s": time.time() - t0, "lptn_run": str(LPTN_RUN), "python": sys.version, "platform": platform.platform(),
        "manifest_sha256": hashlib.sha256(realdata.MANIFEST.read_bytes()).hexdigest(), "locked_G_loaded": False}
(RUN / "summary.json").write_text(json.dumps({"meta": meta, "summary": S}, indent=1))
(RUN / "completion.json").write_text(json.dumps({"status": "COMPLETED"}))
print(RUN)
for k, val in S.items(): print(k, val)
