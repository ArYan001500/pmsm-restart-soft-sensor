"""Identify the 4-node rate-form LPTN on TNN-protocol TRAIN profiles; evaluate on V1/V2.

Stage 1: equation-error nonnegative least squares (Savitzky-Golay derivatives).
Stage 2: output-error refinement (torch, TBPTT, true initial state at profile start = prototype calibration).
Evaluation on V1 and V2 (validation folds) with three initializations of the PM node:
  true (literature setting), naive PM=winding, coolant/ambient mean (Aguilar-Zamorate rule).
Locked G profiles are never loaded.  Writes a fresh run directory.
"""
from __future__ import annotations
import sys, json, time, hashlib, platform, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pmsm_softsense import realdata
from pmsm_softsense.lptn_real import RateLPTN, features, phi_matrix, ALPHA_CU, TS, STATE_NAMES

CFG = {"ee_subsample": 4, "sg_window": 31, "sg_poly": 2, "oe_epochs": 40, "oe_chunk": 1024,
       "oe_lr": 3e-3, "seed": 20261006, "torch_threads": 4, "dtype": "float64"}
if "--smoke" in sys.argv:
    CFG["oe_epochs"] = 1
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
torch.set_num_threads(CFG["torch_threads"])
DT = torch.float64

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN = ROOT / "outputs/runs" / f"{stamp}_lptn_real_baseline_{uuid.uuid4().hex[:8]}"
RUN.mkdir(parents=True)
(RUN / "resolved_config.json").write_text(json.dumps(CFG, indent=1))
(RUN / "source_snapshot").mkdir()
for p in [Path(__file__), ROOT / "src/pmsm_softsense/lptn_real.py", ROOT / "src/pmsm_softsense/realdata.py"]:
    (RUN / "source_snapshot" / p.name).write_bytes(p.read_bytes())

TRAIN = realdata.protocol_ids("train"); V1 = realdata.protocol_ids("V1"); V2 = realdata.protocol_ids("V2")
data = realdata.load(TRAIN + V1 + V2)


def prof_arrays(pid):
    obs, tgt = data[pid]
    f = features(obs)
    T = np.column_stack([tgt["pm"].to_numpy(), obs["stator_yoke"].to_numpy(),
                         obs["stator_tooth"].to_numpy(), obs["stator_winding"].to_numpy()])
    return f, T


# ---------------- Stage 1: equation error ----------------
t0 = time.time()
rows = {i: [] for i in range(4)}; ys = {i: [] for i in range(4)}
for pid in TRAIN:
    f, T = prof_arrays(pid)
    if len(T) < CFG["sg_window"] + 2:
        continue
    dT = savgol_filter(T, CFG["sg_window"], CFG["sg_poly"], deriv=1, delta=TS, axis=0)
    ph = phi_matrix(f, T[:, 3])
    sl = slice(CFG["sg_window"], len(T) - CFG["sg_window"], CFG["ee_subsample"])
    for i in range(4):
        cols = []
        for j in range(4):
            if j == i: continue
            d = T[:, j] - T[:, i]; cols += [d, f["wn"] * d]
        for b in range(2):
            d = f["bnd"][:, b] - T[:, i]; cols += [d, f["wn"] * d]
        X = np.column_stack(cols + [ph])
        rows[i].append(X[sl]); ys[i].append(dT[sl, i])
Ka = np.zeros((4, 4)); Kc = np.zeros((4, 4)); Ba = np.zeros((4, 2)); Bc = np.zeros((4, 2)); Q = np.zeros((4, 6))
ee_r2 = {}
for i in range(4):
    X = np.vstack(rows[i]); y = np.concatenate(ys[i])
    sc = np.maximum(np.abs(X).max(0), 1e-9)
    res = lsq_linear(X / sc, y, bounds=(0, np.inf), lsmr_tol="auto", max_iter=5000)
    c = res.x / sc
    ee_r2[STATE_NAMES[i]] = float(1 - np.sum((y - X @ c) ** 2) / np.sum((y - y.mean()) ** 2))
    k = 0
    for j in range(4):
        if j == i: continue
        Ka[i, j], Kc[i, j] = c[k], c[k + 1]; k += 2
    for b in range(2):
        Ba[i, b], Bc[i, b] = c[k], c[k + 1]; k += 2
    Q[i] = c[k:k + 6]
ee_model = RateLPTN(Ka, Kc, Ba, Bc, Q)
t_ee = time.time() - t0


# ---------------- torch simulator ----------------
def batch(pids):
    fs = [prof_arrays(p) for p in pids]
    L = max(len(T) for _, T in fs); B = len(fs)
    wn = np.zeros((B, L)); is2 = np.zeros((B, L)); us2 = np.zeros((B, L)); bnd = np.zeros((B, L, 2))
    Y = np.zeros((B, L, 4)); M = np.zeros((B, L))
    for b, (f, T) in enumerate(fs):
        n = len(T); wn[b, :n] = f["wn"]; is2[b, :n] = f["is2"]; us2[b, :n] = f["us2"]
        bnd[b, :n] = f["bnd"]; bnd[b, n:] = f["bnd"][-1]; Y[b, :n] = T; M[b, :n] = 1
    t = lambda a: torch.tensor(a, dtype=DT)
    return t(wn), t(is2), t(us2), t(bnd), t(Y), t(M)


def inv_softplus(x):
    x = np.maximum(x, 1e-7)
    return np.log(np.expm1(x))


class TorchLPTN(torch.nn.Module):
    def __init__(self, m: RateLPTN):
        super().__init__()
        P = lambda a: torch.nn.Parameter(torch.tensor(inv_softplus(a), dtype=DT))
        self.Ka, self.Kc, self.Ba, self.Bc, self.Q = P(m.Ka), P(m.Kc), P(m.Ba), P(m.Bc), P(m.Q)
        self.register_buffer("offdiag", 1 - torch.eye(4, dtype=DT))

    def pos(self):
        sp = torch.nn.functional.softplus
        return sp(self.Ka) * self.offdiag, sp(self.Kc) * self.offdiag, sp(self.Ba), sp(self.Bc), sp(self.Q)

    def step(self, T, wn, is2, us2, bnd, P):
        Ka, Kc, Ba, Bc, Q = P
        R = 1 + ALPHA_CU * (T[:, 3] - 20.0)
        phi = torch.stack([is2 * R, wn, wn ** 2, is2 * wn, is2 * wn ** 2, us2], -1)
        K = Ka + Kc * wn[:, None, None]; Bm = Ba + Bc * wn[:, None, None]
        dT = (K * (T[:, None, :] - T[:, :, None])).sum(-1) + (Bm * (bnd[:, None, :] - T[:, :, None])).sum(-1)
        dT = dT + (phi[:, None, :] * Q).sum(-1)
        return T + TS * dT

    def rollout(self, T0, wn, is2, us2, bnd, k0, k1):
        P = self.pos(); T = T0; out = []
        for k in range(k0, k1):
            out.append(T)
            T = self.step(T, wn[:, k], is2[:, k], us2[:, k], bnd[:, k], P)
        return torch.stack(out, 1), T

    def to_rate(self):
        with torch.no_grad():
            return RateLPTN(*(p.numpy() for p in self.pos()))


# ---------------- Stage 2: output error ----------------
t0 = time.time()
wn, is2, us2, bnd, Y, M = batch(TRAIN)
net = TorchLPTN(ee_model)
opt = torch.optim.Adam(net.parameters(), lr=CFG["oe_lr"])
L = Y.shape[1]; hist = []
for ep in range(CFG["oe_epochs"]):
    T = Y[:, 0].clone(); tot = 0.0; cnt = 0.0
    for k0 in range(0, L - 1, CFG["oe_chunk"]):
        k1 = min(k0 + CFG["oe_chunk"], L)
        pred, Tn = net.rollout(T, wn, is2, us2, bnd, k0, k1)
        m = M[:, k0:k1]
        loss = (((pred - Y[:, k0:k1]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        tot += float(loss) * float(m.sum()); cnt += float(m.sum())
        T = Tn.detach()
        # re-anchor finished profiles to their last valid targets (padding keeps them frozen)
    hist.append(tot / cnt)
    print(f"epoch {ep} train MSE(avg 4 targets) {hist[-1]:.3f}", flush=True)
oe_model = net.to_rate(); t_oe = time.time() - t0


# ---------------- evaluation ----------------
def evaluate(model_rate: RateLPTN, pids, init: str):
    tn = TorchLPTN(model_rate)
    res = []
    for pid in pids:
        wn_, is2_, us2_, bnd_, Y_, M_ = batch([pid])
        T0 = Y_[:, 0].clone()
        if init == "naive_pm_eq_winding":
            T0[:, 0] = T0[:, 3]
        elif init == "coolant_ambient_mean":
            T0[:, 0] = 0.5 * (bnd_[:, 0, 0] + bnd_[:, 0, 1])
        with torch.no_grad():
            pred, _ = tn.rollout(T0, wn_, is2_, us2_, bnd_, 0, Y_.shape[1])
        e = (pred - Y_)[0].numpy(); n = len(e)
        t_min = np.arange(n) * TS / 60
        r = {"profile": pid, "init": init, "n": n, "pm_init_error_C": float(e[0, 0])}
        for i, nm in enumerate(STATE_NAMES):
            r[f"mse_{nm}"] = float(np.mean(e[:, i] ** 2))
        r["pm_mae"] = float(np.mean(np.abs(e[:, 0]))); r["pm_maxabs"] = float(np.max(np.abs(e[:, 0])))
        for w in [5, 10, 30]:
            sel = t_min < w
            r[f"pm_maxabs_first{w}min"] = float(np.max(np.abs(e[sel, 0])))
        res.append(r)
    return res


ev = []
for name, mdl in [("equation_error", ee_model), ("output_error", oe_model)]:
    for init in ["true", "naive_pm_eq_winding", "coolant_ambient_mean"]:
        for fold, pids in [("V1", V1), ("V2", V2)]:
            for r in evaluate(mdl, pids, init):
                r["model"] = name; r["fold"] = fold; ev.append(r)
ev = pd.DataFrame(ev)
ev.to_csv(RUN / "validation_metrics_per_profile.csv", index=False)


def pooled(g):
    w = g["n"]
    out = {f"mse_{nm}": float((g[f"mse_{nm}"] * w).sum() / w.sum()) for nm in STATE_NAMES}
    out["mse_avg4"] = float(np.mean([out[f"mse_{nm}"] for nm in STATE_NAMES]))
    out["pm_mae"] = float((g["pm_mae"] * w).sum() / w.sum()); out["pm_maxabs"] = float(g["pm_maxabs"].max())
    out["pm_maxabs_first10min_median"] = float(g["pm_maxabs_first10min"].median())
    return out


summary = {f"{m}|{i}|{fd}": pooled(g) for (m, i, fd), g in ev.groupby(["model", "init", "fold"])}
(RUN / "lptn_params.json").write_text(json.dumps({"equation_error": ee_model.to_dict(),
                                                    "output_error": oe_model.to_dict()}, indent=1))
metrics = {"equation_error_r2_derivative": ee_r2, "oe_train_mse_history": hist, "summary": summary,
           "time_s": {"equation_error": t_ee, "output_error": t_oe}}
(RUN / "metrics.json").write_text(json.dumps(metrics, indent=1))
prov = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "torch": torch.__version__, "manifest_sha256": hashlib.sha256(realdata.MANIFEST.read_bytes()).hexdigest(),
        "csv_bytes": realdata.CSV.stat().st_size, "train_profiles": TRAIN, "eval_profiles": {"V1": V1, "V2": V2},
        "locked_G_loaded": False, "source_commit": "UNVERSIONED"}
(RUN / "provenance.json").write_text(json.dumps(prov, indent=1))
(RUN / "completion.json").write_text(json.dumps({"status": "COMPLETED", "utc": datetime.now(timezone.utc).isoformat()}))
print(json.dumps({"run": str(RUN), "ee_r2": ee_r2}, indent=1))
for k, v in summary.items():
    print(k, {kk: round(vv, 2) for kk, vv in v.items()})
