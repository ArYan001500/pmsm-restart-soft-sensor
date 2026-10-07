"""Liang et al. 2025 (TTE 11(1):2204-2218) faithful-structure re-implementation vs CBP (2026-10-06).

Liang: one winding sensor; the measured winding temperature replaces the model's winding state (integration model);
the deviation between measured and one-interval-ahead predicted winding over N sampling intervals (tp = 10 s,
N = 25 -> ~4 min in the paper) is linear in the unknown initial temperatures of the other nodes, solved by
optimisation (PSO in the paper; exact linear least squares here, same objective, Eq. 16 structure).
Our model: same identified 4-node LPTN for all methods (fairness).  S1 scenario (winding only measured).
Compared at windows 1, 2, 4, 8 min: PM error at reset and at the moment the estimate becomes available.
CBP: zero-history prior at t=0 (label-free), and CBP + split-Q KF at the same times.  V1/V2 resets every 15 min.
"""
from __future__ import annotations
import sys, json, glob, time, hashlib, platform, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pmsm_softsense import realdata
from pmsm_softsense.lptn_real import RateLPTN, features, TS
from pmsm_softsense.reset_tools import LPTNSim, kalman, simulate_histories

CFG = {"seed": 20261006, "tp_s": 10.0, "windows_min": [1, 2, 4, 8], "eval_reset_every_min": 15, "n_orders": 3,
       "gap_min_max": 120, "record_every": 60, "sigma_meas": 0.3, "q_split_pm": 0.5, "q_split_stator": 8.0,
       "gbr": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 40}}
rng = np.random.default_rng(CFG["seed"])
LPTN_RUN = Path(sorted(glob.glob(str(ROOT / "outputs/runs/*_lptn_real_baseline_*")))[-1])
sim = LPTNSim(RateLPTN.from_dict(json.loads((LPTN_RUN / "lptn_params.json").read_text())["output_error"]))
TRAIN = realdata.protocol_ids("train"); VAL = realdata.protocol_ids("val")
D = realdata.load(TRAIN + VAL)
H = [0, 1, 2]; WIND = 3
def prof(pid):
    o, t = D[pid]; F = features(o)
    return o, F, np.column_stack([t.pm.values, o.stator_yoke.values, o.stator_tooth.values, o.stator_winding.values])
cache = {p: prof(p) for p in TRAIN + VAL}
t0 = time.time()
HIST = simulate_histories(sim, cache, TRAIN, CFG["n_orders"], CFG["gap_min_max"], CFG["record_every"], rng)
X = np.column_stack([HIST[:, [3]], HIST[:, 4:9]])
GB = []
for j in H:
    kw = dict(CFG["gbr"]); kw["random_state"] = CFG["seed"]; GB.append(HistGradientBoostingRegressor(**kw).fit(X, HIST[:, j]))
C = np.cov(np.column_stack([HIST[:, j] - GB[i].predict(X) for i, j in enumerate(H)]).T)
res = []
for pid in TRAIN[::3]:
    o, F, Y = cache[pid]
    for k in range(0, len(o) - 1, 7):
        Tn, _ = sim.step(Y[k], k, F); res.append(Y[k + 1] - Tn)
v = np.var(np.array(res), 0); Q_SPLIT = np.diag([v[0] * CFG["q_split_pm"]] + list(v[1:] * CFG["q_split_stator"]))
m_tp = int(CFG["tp_s"] / TS)

def liang_affine(Y, F, r, W):
    """Residuals of the integration model as an affine function of hidden initial temperatures."""
    cols = []
    for basis in [None, 0, 1, 2]:
        x = Y[r].copy(); x[H] = 0.0
        if basis is not None: x[H[basis]] = 1.0
        if basis is not None:          # remove the affine part (inputs/boundaries): superpose on zero-input copy
            pass
        resid = []; wpred = Y[r, WIND]
        for k in range(r, r + W):
            if (k - r) % m_tp == 0: wpred = Y[k, WIND]                # interval start: measured winding
            xs = x.copy(); xs[WIND] = Y[k, WIND]                       # integration model: winding forced to measurement
            xn, _ = sim.step(xs, k, F)
            xw = xs.copy(); xw[WIND] = wpred; wpred = sim.step(xw, k, F)[0][WIND]
            x = xn
            if (k + 1 - r) % m_tp == 0: resid.append(Y[k + 1, WIND] - wpred)
        cols.append(np.array(resid))
    base = cols[0]; A = np.column_stack([cols[i + 1] - base for i in range(3)])
    return A, base

def liang_estimate(Y, F, r, W):
    A, base = liang_affine(Y, F, r, W)   # resid(x_h) = base + A x_h ; minimise ||resid||^2
    xh = np.linalg.lstsq(A, -base, rcond=None)[0]
    return xh

def propagate(x0, r, k1, F, Y):
    """Liang's compensated model after estimation (integration model: winding forced to measurement)."""
    x = x0.copy()
    for k in range(r, k1):
        x[WIND] = Y[k, WIND]; x, _ = sim.step(x, k, F)
    return x

rows = []
for pid in VAL:
    o, F, Y = cache[pid]; pm = Y[:, 0]
    for r in range(0, len(o) - int(10 * 60 / TS), int(CFG["eval_reset_every_min"] * 60 / TS)):
        xv = np.concatenate([[Y[r, WIND]], [F["bnd"][r, 0], F["bnd"][r, 1], F["is2"][r], F["wn"][r], F["us2"][r]]])[None]
        mu = np.array([g.predict(xv)[0] for g in GB])
        x0 = Y[r].copy(); x0[H] = mu
        P0 = np.zeros((4, 4)); P0[np.ix_(H, H)] = C; P0[3, 3] = CFG["sigma_meas"] ** 2
        nmax = min(len(o), r + int(max(CFG["windows_min"]) * 60 / TS) + 1)
        kf_pm, _, _ = kalman(sim, F, Y, [3], x0, P0, Q_SPLIT, r, nmax, CFG["sigma_meas"])
        rec0 = {"profile": pid, "reset_idx": r, "event": "start" if r == 0 else "reset", "cbp_t0_err": abs(mu[0] - pm[r])}
        for wmin in CFG["windows_min"]:
            W = int(wmin * 60 / TS)
            if r + W >= len(o): continue
            xh = liang_estimate(Y, F, r, W)
            xL0 = Y[r].copy(); xL0[H] = xh
            xLW = propagate(xL0, r, r + W, F, Y)
            # CBP open-loop propagated to the same moment, and CBP-KF at that moment
            xC = propagate(x0, r, r + W, F, Y)
            rows.append({**rec0, "W_min": wmin, "liang_pm0_err": abs(xh[0] - pm[r]), "liang_pmW_err": abs(xLW[0] - pm[r + W]),
                         "cbp_ol_pmW_err": abs(xC[0] - pm[r + W]), "cbp_kf_pmW_err": abs(kf_pm[W] - pm[r + W])})
    print(f"profile {pid} done {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rows)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN = ROOT / "outputs/runs" / f"{stamp}_liang_comparison_{uuid.uuid4().hex[:8]}"
RUN.mkdir(parents=True); (RUN / "source_snapshot").mkdir()
for p in [Path(__file__), ROOT / "src/pmsm_softsense/reset_tools.py"]:
    (RUN / "source_snapshot" / p.name).write_bytes(p.read_bytes())
(RUN / "resolved_config.json").write_text(json.dumps(CFG, indent=1)); R.to_csv(RUN / "metrics.csv", index=False)
S = {}
for (ev, w), g in R.groupby(["event", "W_min"]):
    S[f"{ev}|W={w}min"] = {"n": int(len(g)), **{f"{c}_med": float(g[c].median()) for c in
                            ["cbp_t0_err", "liang_pm0_err", "liang_pmW_err", "cbp_ol_pmW_err", "cbp_kf_pmW_err"]},
                           **{f"{c}_p90": float(g[c].quantile(.9)) for c in ["liang_pmW_err", "cbp_kf_pmW_err"]}}
meta = {"runtime_s": time.time() - t0, "lptn_run": str(LPTN_RUN), "python": sys.version, "platform": platform.platform(),
        "manifest_sha256": hashlib.sha256(realdata.MANIFEST.read_bytes()).hexdigest(), "locked_G_loaded": False}
(RUN / "summary.json").write_text(json.dumps({"meta": meta, "summary": S}, indent=1))
(RUN / "completion.json").write_text(json.dumps({"status": "COMPLETED"}))
print(RUN)
for k, val in S.items(): print(k, {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in val.items()})
