"""CAUSAL-INPUT INDUCTION-MOTOR CV (2026-10-07, added after the locked test; development sessions only).
Same code path as run_cv5_ablation_v2.py --dataset induction, but every input at thermal sample time t uses only data
recorded up to t: speed and torque are the means of the 10 kHz signals n_m and T_TS,m over the 0.5 s before t (instead of
the session means), and the 0.1 Hz electrical quantities are held from the last sample at or before t (instead of linear
interpolation).  Before the first sample of a signal its first available value is used (about 0.7 s for speed/torque,
about 1 s for the electrical quantities after the first thermal sample).
--- original v2 docstring ---
v2 (2026-10-06, after v1 CV): event-matched calibration made generic.  Every calibrated variant uses the SAME
sim-trained prior for all events; its conformal quantile is computed on events of the same type: session starts ->
all train session starts, mid-run resets -> train-half-B points every 5 min.  The v1 start prior (simulated
zero-input cooling) is kept only as ablation A8_simstart (it gave very wide bands / worse 2-10 min error in v1 CV on
both machines).  A7_no_event_match now = mid-run calibration used at every event.
5-fold CV with contribution ablations, both machines (2026-10-06).

--dataset paderborn : 66 dev profiles (manifest dev_cv5 folds); events = mid-run resets every 15 min + natural starts;
                      calibration events = real train-half-B points every 5 min (event-matched to resets).
--dataset induction : 223 non-test sessions (l%5 != 0), 5 folds by l%25 bucket; events = session starts (warm restarts);
                      calibration events = train-half-B session starts.  Test sessions (l%5==0) never built.
Per fold the whole recipe is re-run: LPTN (EE NNLS + OE refine), simulated histories, priors, calibration, q rule.
Methods (all share the fold's LPTN and filter):
  naive_KF      PM/hidden = winding, P0 = (30 K)^2 (baseline)            oracle   true hidden state at the event
  CBP           full method: sim-trained GBR prior + UI-009 novelty conformal + q rule (split Q)
  A1_labelprior prior trained on REAL train labels (label-free cost)     A2_linear linear prior on sim histories
  A3_plainconf  conformal without novelty inflation                       A4_uncal  simulated residual variance, no conformal
  A5_stdQ       uniform process-noise inflation x4 (no split / no q rule)
  A7_no_event_match  reset-type prior/calibration used also at session starts (event-matching ablation)
  CBP at session starts uses a start prior (simulated histories + zero-input cooling gaps) calibrated on train starts.
  A6_liang      (S1 only) Liang-2025-style integration-model inversion with a 4-min window (estimate available at 4 min)
Deterministic initializers (init error only): winding, ambient/coolant mean (Aguilar), coolant.
"""
from __future__ import annotations
import sys, json, time, hashlib, platform, uuid, argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pmsm_softsense.lptn_real import TS
from pmsm_softsense.lptn_fit import ee_fit, oe_refine
from pmsm_softsense.reset_tools import LPTNSim, kalman, simulate_histories

ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=["paderborn", "induction"])
ap.add_argument("--oe-epochs", type=int, default=15); args = ap.parse_args()
CFG = {"dataset": args.dataset, "seed": 20261006, "oe_epochs": args.oe_epochs, "n_orders": 3, "gap_min_max": 120,
       "record_every": 60, "alpha": 0.10, "knn_k": 20, "eval_every_min": 15, "horizon_min": 31, "sigma_meas": 0.3,
       "q_stator": 8.0, "q_grid": [0.5, 2, 8, 32, 128], "q_cov_target": 0.88, "times_min": [0, 2, 4, 5, 10, 20, 30],
       "liang_tp_s": 10.0, "liang_window_min": 4,
       "gbr": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 40}}
rng = np.random.default_rng(CFG["seed"])

# ------------------------------------------------------------------ data adapters
if args.dataset == "paderborn":
    from pmsm_softsense import realdata
    from pmsm_softsense.lptn_real import features
    M = realdata.manifest(); FOLDS = M["dev_cv5_over_non_G"]; ALL = sorted(p for f in FOLDS for p in f)
    D = realdata.load(ALL)
    def build(pid):
        o, t = D[pid]; F = features(o)
        return np.zeros(len(o)), F, np.column_stack([t.pm.values, o.stator_yoke.values, o.stator_tooth.values, o.stator_winding.values])
    cache = {p: build(p) for p in ALL}
    def eval_points(Y): return list(range(0, len(Y) - int(5 * 60 / TS), int(CFG["eval_every_min"] * 60 / TS)))
    def qsel_points(Y): return list(range(0, len(Y) - int(31 * 60 / TS), int(30 * 60 / TS)))
    MANIFEST_SHA = hashlib.sha256(realdata.MANIFEST.read_bytes()).hexdigest()
else:
    DD = ROOT / "data/Second Dataset"
    S2 = pd.read_csv(DD / "Second_Part.csv"); T3 = pd.read_csv(DD / "Third_Part.csv")
    FP = pd.read_csv(DD / "First_Part.csv", usecols=["l", "t", "n_{m}", "T_{TS,m}"]).sort_values(["l", "t"])
    audit = pd.read_csv(ROOT / "outputs/diagnostics/20261006_induction_audit/per_operating_point.csv").set_index("l")
    ALL = [int(l) for l in sorted(S2.l.unique()) if int(l) % 5 != 0]
    FOLDS = [[l for l in ALL if (l // 5) % 5 == k] for k in range(5)]
    def build(l):
        g = S2[S2.l == l]; t = g.t.values; p = T3[T3.l == l].sort_values("t")
        Y = np.column_stack([g[["theta_{R,1}", "theta_{R,4}", "theta_{R,5}"]].mean(axis=1).values, g["theta_{S,5}"].values,
                             g["theta_{S,7}"].values, g[["theta_{S,10}", "theta_{S,11}"]].mean(axis=1).values])
        def ip(col):
            if not len(p): return np.zeros(len(t))
            j = np.clip(np.searchsorted(p.t.values, t, side="right") - 1, 0, len(p) - 1); return np.asarray(col)[j]
        fs = FP[FP.l == l]; tf = fs.t.values
        def causal_mean(v):
            c = np.r_[0.0, np.cumsum(v)]; hi = np.searchsorted(tf, t, side="right")
            hi = np.where(hi == 0, max(np.searchsorted(tf, tf[0] + 0.5, side="right"), 1), hi)   # before first sample: first 0.5 s
            lo = np.searchsorted(tf, tf[hi - 1] - 0.5, side="right")                               # 0.5 s ending at the last available sample
            return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
        spd = np.abs(causal_mean(fs["n_{m}"].values)); trq = causal_mean(fs["T_{TS,m}"].values)
        irms = ip(p[["i_{a,r,m}", "i_{b,r,m}", "i_{c,r,m}"]].mean(axis=1).values) if len(p) else np.zeros(len(t))
        urms = ip(p[["u_{1,r,m}", "u_{2,r,m}", "u_{3,r,m}"]].mean(axis=1).values) if len(p) else np.zeros(len(t))
        ploss = ip((p["P_{in,mot,m}"] - p["P_{mech,m}"]).values) if len(p) else np.zeros(len(t))
        F = {"is2": (irms / 2.0) ** 2, "us2": (urms / 230.0) ** 2, "wn": spd / 3000.0,
             "extra": np.column_stack([np.maximum(ploss, 0) / 100.0, (trq / 5.0) ** 2]),
             "bnd": np.column_stack([g[["theta_{S,14}", "theta_{S,15}", "theta_{S,18}"]].mean(axis=1).values,
                                     g[["theta_{S,16}", "theta_{S,17}"]].mean(axis=1).values])}
        return np.zeros(len(Y)), F, Y
    cache = {l: build(l) for l in ALL}
    def eval_points(Y): return [0]
    def qsel_points(Y): return [0]
    MANIFEST_SHA = "induction: split rule l%5==0 locked test; folds (l//5)%5"

SCEN = {"S3": {"m": [1, 2, 3], "h": [0]}, "S1": {"m": [3], "h": [0, 1, 2]}}
def mid_points(Y): return list(range(0, len(Y) - 600, int(5 * 60 / TS)))
def ctx(F, r): return [F["bnd"][r, 0], F["bnd"][r, 1], F["is2"][r], F["wn"][r], F["us2"][r]]
def cq(s, a):
    s = np.sort(np.asarray(s)); n = len(s); k = int(np.ceil((n + 1) * (1 - a))); return float(s[min(k, n) - 1])

def liang_estimate(sim, Y, F, r, W):
    """Liang 2025 integration-model inversion (winding measured), exact LS (verified exact on synthetic data)."""
    H = [0, 1, 2]; m_tp = int(CFG["liang_tp_s"] / TS); cols = []
    for basis in [None, 0, 1, 2]:
        x = Y[r].copy(); x[H] = 0.0
        if basis is not None: x[H[basis]] = 1.0
        resid = []; wpred = Y[r, 3]
        for k in range(r, r + W):
            if (k - r) % m_tp == 0: wpred = Y[k, 3]
            xs = x.copy(); xs[3] = Y[k, 3]; xn, _ = sim.step(xs, k, F)
            xw = xs.copy(); xw[3] = wpred; wpred = sim.step(xw, k, F)[0][3]; x = xn
            if (k + 1 - r) % m_tp == 0: resid.append(Y[k + 1, 3] - wpred)
        cols.append(np.array(resid))
    A = np.column_stack([cols[i + 1] - cols[0] for i in range(3)])
    xh = np.linalg.lstsq(A, -cols[0], rcond=None)[0]
    x = Y[r].copy(); x[H] = xh
    for k in range(r, r + W):
        x[3] = Y[k, 3]; x, _ = sim.step(x, k, F)
    return x[0]

t0 = time.time(); rows = []; fold_meta = []
for fi, held in enumerate(FOLDS):
    train = [p for p in ALL if p not in held]
    lp = ee_fit(cache, train)
    if CFG["oe_epochs"] > 0: lp = oe_refine(lp, cache, train, epochs=CFG["oe_epochs"], seed=CFG["seed"])
    sim = LPTNSim(lp)
    res = []
    for p in train[::3]:
        _, F, Y = cache[p]
        for k in range(0, len(Y) - 1, 7):
            Tn, _ = sim.step(Y[k], k, F); res.append(Y[k + 1] - Tn)
    v = np.var(np.array(res), 0)
    QS = {q: np.diag([v[0] * q] + list(v[1:] * CFG["q_stator"])) for q in CFG["q_grid"]}
    Q_STD = np.diag(v * 4.0)
    Hh = simulate_histories(sim, cache, train, CFG["n_orders"], CFG["gap_min_max"], CFG["record_every"], rng)
    def sim_starts(n_samp=5000):
        idx = rng.choice(len(Hh), size=min(n_samp, len(Hh)), replace=False); out = []
        starts0 = [(cache[p][1], cache[p][2]) for p in train]
        Am0, B0 = sim.mats(0.0); Ad = np.eye(4) + TS * Am0
        for i in idx:
            Fn, Yn = starts0[rng.integers(len(starts0))]; bnd0 = Fn["bnd"][0]
            gap = int(rng.uniform(0, CFG["gap_min_max"]) * 60 / TS)
            Teq = np.linalg.solve(-Am0, B0 @ bnd0)
            T = Teq + np.linalg.matrix_power(Ad, gap) @ (Hh[i, :4] - Teq)
            out.append(np.concatenate([T, [bnd0[0], bnd0[1], Fn["is2"][0], Fn["wn"][0], Fn["us2"][0]]]))
        return np.array(out)
    Hs = sim_starts()
    Lab = np.array([np.concatenate([cache[p][2][r], ctx(cache[p][1], r)]) for p in train for r in range(0, len(cache[p][2]), 60)])
    B_ids = set(sorted(train)[1::2]); A_ids = [p for p in train if p not in B_ids]
    def build_scenario(sc, sp):
        m, h = sp["m"], sp["h"]
        def fit_prior(rows_, kind):
            X = np.column_stack([rows_[:, m], rows_[:, 4:9]]); preds = []
            if kind == "gbr":
                for j in h:
                    kw = dict(CFG["gbr"]); kw["random_state"] = CFG["seed"]; preds.append(HistGradientBoostingRegressor(**kw).fit(X, rows_[:, j]))
                f = lambda x, P=preds: np.array([g.predict(x[None])[0] for g in P])
            else:
                Xa = np.column_stack([np.ones(len(X)), X]); Bc = np.linalg.lstsq(Xa, rows_[:, h], rcond=None)[0]
                f = lambda x, Bc=Bc: np.concatenate([[1.0], x]) @ Bc
            R_ = np.column_stack([rows_[:, j] for j in h]) - np.array([f(x) for x in X[::5]]).repeat(5, 0)[:len(X)] if False else None
            Rs = np.array([rows_[i, h] - f(X[i]) for i in range(0, len(X), 7)])
            return f, np.cov(Rs.T).reshape(len(h), len(h))
        Xs = np.column_stack([Hh[:, m], Hh[:, 4:9]]); a, b = Xs.mean(0), Xs.std(0) + 1e-9
        nn = NearestNeighbors(n_neighbors=CFG["knn_k"]).fit((Xs - a) / b); dref = float(np.median(nn.kneighbors((Xs[::7] - a) / b)[0].mean(1)))
        novf = lambda x, nn=nn, a=a, b=b, dref=dref: max(float(nn.kneighbors(((x - a) / b)[None])[0].mean() / dref) - 1, 0.0)
        priors = {"gbr_sim": fit_prior(Hh, "gbr"), "lin_sim": fit_prior(Hh, "lin"), "gbr_lab": fit_prior(Lab, "gbr"),
                  "gbr_simstart": fit_prior(Hs, "gbr")}
        Xst = np.column_stack([Hs[:, m], Hs[:, 4:9]]); a2, b2 = Xst.mean(0), Xst.std(0) + 1e-9
        nn2 = NearestNeighbors(n_neighbors=CFG["knn_k"]).fit((Xst - a2) / b2); dref2 = float(np.median(nn2.kneighbors((Xst[::7] - a2) / b2)[0].mean(1)))
        novf_start = lambda x, nn=nn2, a=a2, b=b2, dref=dref2: max(float(nn.kneighbors(((x - a) / b)[None])[0].mean() / dref) - 1, 0.0)
        def calib(fname, use_nov, kind):
            f, _ = priors[fname]; sc_ = []
            pts = [(p, r) for p in B_ids for r in mid_points(cache[p][2])] if kind == "mid" else [(p, 0) for p in train]
            for p, r in pts:
                _, F, Y = cache[p]
                x = np.concatenate([Y[r, m], ctx(F, r)]); e = abs(f(x)[0] - Y[r, 0])
                sc_.append(e / (1 + novf(x)) if use_nov else e)
            return cq(sc_, CFG["alpha"])
        Q_cal = {(fn, nv, kd): calib(fn, nv, kd) for fn, nv in [("gbr_sim", True), ("gbr_sim", False), ("lin_sim", True), ("gbr_lab", True)]
                 for kd in ("mid", "start")}
        def calib_start():
            f, _ = priors["gbr_simstart"]; sc_ = []
            for p in train:
                _, F, Y = cache[p]; x = np.concatenate([Y[0, m], ctx(F, 0)]); sc_.append(abs(f(x)[0] - Y[0, 0]) / (1 + novf_start(x)))
            return cq(sc_, CFG["alpha"])
        Q_start = calib_start()
        def seed_start(Y, F, r):
            f, C = priors["gbr_simstart"]; x = np.concatenate([Y[r, m], ctx(F, r)]); mu = f(x); x0 = Y[r].copy(); x0[h] = mu
            P0 = np.zeros((4, 4)); hw = Q_start * (1 + novf_start(x)); P0[np.ix_(h, h)] = C * ((hw / 1.645) ** 2 / max(C[0, 0], 1e-9))
            for i in m: P0[i, i] = CFG["sigma_meas"] ** 2
            return x0, P0
        def make_seed(fname, mode, force_kind=None):
            f, C = priors[fname]
            def seed(Y, F, r):
                kind = force_kind or ("start" if r == 0 else "mid")
                x = np.concatenate([Y[r, m], ctx(F, r)]); mu = f(x); x0 = Y[r].copy(); x0[h] = mu; P0 = np.zeros((4, 4))
                if mode == "uncal": P0[np.ix_(h, h)] = C
                else:
                    use_nov = mode == "nov"; hw = Q_cal[(fname, use_nov, kind)] * ((1 + novf(x)) if use_nov else 1.0)
                    P0[np.ix_(h, h)] = C * ((hw / 1.645) ** 2 / max(C[0, 0], 1e-9))
                for i in m: P0[i, i] = CFG["sigma_meas"] ** 2
                return x0, P0
            return seed
        cbp_seed = make_seed("gbr_sim", "nov")
        def simstart(Y, F, r, rs=cbp_seed, ss=seed_start): return ss(Y, F, r) if r == 0 else rs(Y, F, r)
        seeds = {"CBP": cbp_seed, "A7_no_event_match": make_seed("gbr_sim", "nov", "mid"), "A8_simstart": simstart, "A1_labelprior": make_seed("gbr_lab", "nov"), "A2_linear": make_seed("lin_sim", "nov"),
                 "A3_plainconf": make_seed("gbr_sim", "plain"), "A4_uncal": make_seed("gbr_sim", "uncal")}
        covs = {q: [] for q in CFG["q_grid"]}
        for p in A_ids:
            _, F, Y = cache[p]
            for r in qsel_points(Y):
                if r + int(31 * 60 / TS) > len(Y): continue
                x0, P0 = seeds["CBP"](Y, F, r); n = r + int(31 * 60 / TS)
                for q, QQ in QS.items():
                    trj, sd, _ = kalman(sim, F, Y, m, x0, P0, QQ, r, n, CFG["sigma_meas"])
                    covs[q].append([abs(trj[k] - Y[r + k, 0]) <= 1.645 * sd[k] for k in (int(1200 / TS), int(1800 / TS))])
        ok = [q for q in CFG["q_grid"] if len(covs[q]) and np.mean(np.array(covs[q]), 0).min() >= CFG["q_cov_target"]]
        return {"seeds": seeds, "qcal": {f"{k[0]}|{k[1]}|{k[2]}": round(v_, 3) for k, v_ in Q_cal.items()} | {"simstart": round(Q_start, 3)}, "q": ok[0] if ok else CFG["q_grid"][-1]}
    VARIANTS = {sc: build_scenario(sc, sp) for sc, sp in SCEN.items()}
    fold_meta.append({"fold": fi, "n_held": len(held), "q_selected": {sc: VARIANTS[sc]["q"] for sc in SCEN}, "qcal": {sc: VARIANTS[sc]["qcal"] for sc in SCEN}})
    for p in held:
        _, F, Y = cache[p]; pm = Y[:, 0]
        for r in eval_points(Y):
            n = min(len(Y), r + int(CFG["horizon_min"] * 60 / TS))
            base = {"fold": fi, "unit": p, "r": r, "event": "start" if r == 0 else "reset"}
            Ta, Tc = F["bnd"][r, 1], F["bnd"][r, 0]
            for nm, val in {"init_winding": Y[r, 3], "init_mean_Ta_Tc": 0.5 * (Ta + Tc), "init_coolant_or_housing": Tc}.items():
                rows.append({**base, "scenario": "-", "method": nm, "err_0": abs(val - pm[r])})
            for sc, sp in SCEN.items():
                m, h = sp["m"], sp["h"]; QQ = QS[VARIANTS[sc]["q"]]
                xn = Y[r].copy(); xn[h] = Y[r, 3]
                Pn = np.diag([30.0 ** 2 if i == 0 else (15.0 ** 2 if i in h else CFG["sigma_meas"] ** 2) for i in range(4)])
                xo = Y[r].copy(); Po = np.diag([0.25] + [CFG["sigma_meas"] ** 2] * 3)
                runs = {"naive_KF": (xn, Pn, QQ), "oracle": (xo, Po, QQ)}
                for nm, sd_ in VARIANTS[sc]["seeds"].items():
                    x0, P0 = sd_(Y, F, r); runs[nm] = (x0, P0, QQ)
                runs["A5_stdQ"] = (*VARIANTS[sc]["seeds"]["CBP"](Y, F, r), Q_STD)
                for nm, (xx, PP, QX) in runs.items():
                    trj, sd, _ = kalman(sim, F, Y, m, xx, PP, QX, r, n, CFG["sigma_meas"])
                    rec = {**base, "scenario": sc, "method": nm}
                    for tm in CFG["times_min"]:
                        k = int(tm * 60 / TS)
                        if k < len(trj):
                            e = abs(trj[k] - pm[r + k]); rec[f"err_{tm}"] = float(e); rec[f"cov_{tm}"] = float(e <= 1.645 * sd[k]); rec[f"hw_{tm}"] = float(1.645 * sd[k])
                    rows.append(rec)
                if sc == "S1":
                    W = int(CFG["liang_window_min"] * 60 / TS)
                    if r + W < len(Y):
                        rows.append({**base, "scenario": "S1", "method": "A6_liang", "err_4": float(abs(liang_estimate(sim, Y, F, r, W) - pm[r + W]))})
    print(f"fold {fi} done q={fold_meta[-1]['q_selected']} {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rows)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN = ROOT / "outputs/runs" / f"{stamp}_cv5_causal_{args.dataset}_{uuid.uuid4().hex[:8]}"
RUN.mkdir(parents=True); (RUN / "source_snapshot").mkdir()
for pth in [Path(__file__), ROOT / "src/pmsm_softsense/reset_tools.py", ROOT / "src/pmsm_softsense/lptn_fit.py", ROOT / "src/pmsm_softsense/lptn_real.py"]:
    (RUN / "source_snapshot" / pth.name).write_bytes(pth.read_bytes())
(RUN / "resolved_config.json").write_text(json.dumps(CFG, indent=1)); R.to_csv(RUN / "episodes.csv", index=False)
S = {}
for (ev, sc, mth), g in R.groupby(["event", "scenario", "method"]):
    d = {"n": int(len(g))}
    for tm in CFG["times_min"]:
        c = f"err_{tm}"
        if c in g and g[c].notna().any():
            gg = g[g[c].notna()]; pf = gg.groupby("fold")[c].quantile(.9)
            d[f"t{tm}"] = {"err_med": round(float(gg[c].median()), 2), "err_p90": round(float(gg[c].quantile(.9)), 2),
                           "p90_fold_min_max": [round(float(pf.min()), 2), round(float(pf.max()), 2)]}
            if f"cov_{tm}" in gg and gg[f"cov_{tm}"].notna().any():
                d[f"t{tm}"]["cov"] = round(float(gg[f"cov_{tm}"].mean()), 3); d[f"t{tm}"]["hw_med"] = round(float(gg[f"hw_{tm}"].median()), 2)
    S[f"{ev}|{sc}|{mth}"] = d
meta = {"folds": fold_meta, "runtime_s": time.time() - t0, "python": sys.version, "platform": platform.platform(), "split_provenance": MANIFEST_SHA}
(RUN / "summary.json").write_text(json.dumps({"meta": meta, "summary": S}, indent=1))
(RUN / "completion.json").write_text(json.dumps({"status": "COMPLETED"}))
print(RUN)
