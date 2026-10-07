"""Fig. 1 (draft): one real mid-run reset on a Paderborn validation profile (winding-only S1).
Truth PM vs naive KF, CBP (calibrated band) and oracle, 30 min after the reset.  Uses the main OE LPTN and the frozen
recipe (sim prior, novelty-scaled conformal calibration on train half B, q=2).  Validation data only; G never loaded."""
import sys, json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from pmsm_softsense import realdata
from pmsm_softsense.lptn_real import RateLPTN, features, TS
from pmsm_softsense.reset_tools import LPTNSim, kalman, simulate_histories
C = {"cbp": "#2a78d6", "naive": "#eb6834", "oracle": "#1baf7a", "truth": "#0b0b0b"}
rng = np.random.default_rng(20261006)
sim = LPTNSim(RateLPTN.from_dict(json.loads((Path(sorted(glob.glob(str(ROOT / "outputs/runs/*_lptn_real_baseline_*")))[-1]) / "lptn_params.json").read_text())["output_error"]))
TR = realdata.protocol_ids("train"); VA = realdata.protocol_ids("val"); D = realdata.load(TR + VA)
cache = {}
for p in TR + VA:
    o, t = D[p]; F = features(o); cache[p] = (o, F, np.column_stack([t.pm.values, o.stator_yoke.values, o.stator_tooth.values, o.stator_winding.values]))
H = simulate_histories(sim, cache, TR, 3, 120, 60, rng)
m, h = [3], [0, 1, 2]
X = np.column_stack([H[:, m], H[:, 4:9]])
gbs = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, random_state=20261006).fit(X, H[:, j]) for j in h]
Cs = np.cov(np.column_stack([H[:, j] - gbs[i].predict(X) for i, j in enumerate(h)]).T)
a, b = X.mean(0), X.std(0) + 1e-9; nn = NearestNeighbors(n_neighbors=20).fit((X - a) / b); dref = float(np.median(nn.kneighbors((X[::7] - a) / b)[0].mean(1)))
def ctx(F, r): return [F["bnd"][r, 0], F["bnd"][r, 1], F["is2"][r], F["wn"][r], F["us2"][r]]
def pri(x): return np.array([g.predict(x[None])[0] for g in gbs]), max(float(nn.kneighbors(((x - a) / b)[None])[0].mean() / dref) - 1, 0)
sc = []
for p in sorted(TR)[1::2]:
    o, F, Y = cache[p]
    for r in range(0, len(Y) - 600, 600):
        mu, nv = pri(np.concatenate([Y[r, m], ctx(F, r)])); sc.append(abs(mu[0] - Y[r, 0]) / (1 + nv))
sc = np.sort(sc); qn = sc[int(np.ceil((len(sc) + 1) * 0.9)) - 1]
res = []
for p in TR[::3]:
    o, F, Y = cache[p]
    for k in range(0, len(Y) - 1, 7):
        Tn, _ = sim.step(Y[k], k, F); res.append(Y[k + 1] - Tn)
v = np.var(np.array(res), 0); Q = np.diag([v[0] * 2] + list(v[1:] * 8))
# pick the validation reset with the largest naive initial error (illustrative, stated in caption)
best = None
for p in VA:
    o, F, Y = cache[p]
    for r in range(1800, len(Y) - 3600, 1800):
        e = abs(Y[r, 3] - Y[r, 0])
        if best is None or e > best[0]: best = (e, p, r)
_, p, r = best; o, F, Y = cache[p]; n = r + 3600
mu, nv = pri(np.concatenate([Y[r, m], ctx(F, r)])); hw = qn * (1 + nv)
x0 = Y[r].copy(); x0[h] = mu; P0 = np.zeros((4, 4)); P0[np.ix_(h, h)] = Cs * ((hw / 1.645) ** 2 / Cs[0, 0]); P0[3, 3] = 0.09
cbp, sd, _ = kalman(sim, F, Y, m, x0, P0, Q, r, n)
xn = Y[r].copy(); xn[h] = Y[r, 3]; Pn = np.diag([900, 225, 225, 0.09]); nai, sdn, _ = kalman(sim, F, Y, m, xn, Pn, Q, r, n)
xo = Y[r].copy(); Po = np.diag([0.25, .09, .09, .09]); ora, _, _ = kalman(sim, F, Y, m, xo, Po, Q, r, n)
tm = np.arange(n - r) * TS / 60
fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=200)
ax.fill_between(tm, cbp - 1.645 * sd, cbp + 1.645 * sd, color=C["cbp"], alpha=0.15, lw=0, label="CBP 90 % band")
ax.plot(tm, Y[r:n, 0], color=C["truth"], lw=2, label="Measured PM (evaluation only)")
ax.plot(tm, nai, color=C["naive"], lw=2, ls="--", label="Naive KF (PM = winding)")
ax.plot(tm, cbp, color=C["cbp"], lw=2, label="CBP (label-free prior)")
ax.plot(tm, ora, color=C["oracle"], lw=2, ls=":", label="Oracle (true PM at reset)")
ax.plot(tm, Y[r:n, 3], color="#8a8986", lw=1.2, label="Measured winding")
ax.set_xlabel("Time after estimator reset [min]"); ax.set_ylabel("Temperature [°C]")
ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="best")
ax.set_title(f"Paderborn validation profile {p}, reset at {r*TS/60:.0f} min (winding sensor only)", fontsize=8)
out = ROOT / "outputs/figures"; out.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(out / "fig1_example_reset.png"); fig.savefig(out / "fig1_example_reset.pdf")
np.savez_compressed(out / "fig1_example_reset_traces.npz", tm=tm, pm=Y[r:n, 0], wind=Y[r:n, 3], cbp=cbp, sd=sd, naive=nai, sdn=sdn, oracle=ora,
                    pre_t=(np.arange(max(0, r - 3600), r) - r) * TS / 60, pre_pm=Y[max(0, r - 3600):r, 0], pre_w=Y[max(0, r - 3600):r, 3])
json.dump({"profile": int(p), "reset_idx": int(r), "pm0": float(Y[r, 0]), "winding0": float(Y[r, 3]), "cbp0": float(mu[0]), "hw0": float(hw)},
          open(out / "fig1_example_reset.json", "w"), indent=1)
print(p, r, Y[r, 0], Y[r, 3], mu[0], hw)
