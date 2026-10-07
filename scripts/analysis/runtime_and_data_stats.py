"""Data statistics and online cost of the CBP recipe (2026-10-07). Dev profiles only (G never loaded).
(1) Paderborn: profiles, hours, samples. (2) Size of the simulated snapshot set for the 59 TNN-train profiles.
(3) Wall-clock cost per event of: GBR prior (S1: 3 hidden nodes), kNN novelty query, and one KF step."""
import sys, json, time, glob, platform
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from pmsm_softsense import realdata
from pmsm_softsense.lptn_real import RateLPTN, features, TS
from pmsm_softsense.reset_tools import LPTNSim, simulate_histories, kalman
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
out = {}
df = pd.read_pickle(realdata.CACHE)
g = df.groupby("profile_id").size()
out["paderborn_all_profiles"] = int(len(g)); out["paderborn_all_hours"] = round(float(g.sum() * TS / 3600), 1)
M = realdata.manifest(); G = M["primary_protocol"]["test_locked_G"]
dev = [p for p in g.index if p not in G]
out["paderborn_dev_profiles"] = len(dev); out["paderborn_dev_hours"] = round(float(g.loc[dev].sum() * TS / 3600), 1)
out["paderborn_profile_length_min_minutes"] = [round(float(g.loc[dev].min() * TS / 60), 1), round(float(g.loc[dev].max() * TS / 60), 1)]
rng = np.random.default_rng(20261006)
sim = LPTNSim(RateLPTN.from_dict(json.loads((Path(sorted(glob.glob(str(ROOT / "outputs/runs/*_lptn_real_baseline_*")))[-1]) / "lptn_params.json").read_text())["output_error"]))
TR = realdata.protocol_ids("train"); D = realdata.load(TR)
cache = {}
for p in TR:
    o, t = D[p]; F = features(o); cache[p] = (o, F, np.column_stack([t.pm.values, o.stator_yoke.values, o.stator_tooth.values, o.stator_winding.values]))
t0 = time.perf_counter(); H = simulate_histories(sim, cache, TR, 3, 120, 60, rng); out["sim_histories_seconds"] = round(time.perf_counter() - t0, 1)
out["n_simulated_snapshots"] = int(len(H))
m, h = [3], [0, 1, 2]; X = np.column_stack([H[:, m], H[:, 4:9]])
t0 = time.perf_counter()
gbs = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, random_state=20261006).fit(X, H[:, j]) for j in h]
out["gbr_fit_seconds_S1"] = round(time.perf_counter() - t0, 1)
a, b = X.mean(0), X.std(0) + 1e-9; nn = NearestNeighbors(n_neighbors=20).fit((X - a) / b)
Q = X[rng.choice(len(X), 500, replace=False)]
t0 = time.perf_counter()
for x in Q: [gg.predict(x[None]) for gg in gbs]
out["gbr_predict_ms_per_event"] = round((time.perf_counter() - t0) / len(Q) * 1e3, 3)
t0 = time.perf_counter()
for x in Q: nn.kneighbors(((x - a) / b)[None])
out["knn_query_ms_per_event"] = round((time.perf_counter() - t0) / len(Q) * 1e3, 3)
o, F, Y = cache[TR[0]]; x0 = Y[0].copy(); P0 = np.eye(4); Qm = np.eye(4) * 1e-3
n = 2000; t0 = time.perf_counter(); kalman(sim, F, Y, m, x0, P0, Qm, 0, n); out["kf_step_us"] = round((time.perf_counter() - t0) / (n - 1) * 1e6, 1)
out["platform"] = platform.platform(); out["processor"] = platform.processor(); out["python"] = sys.version.split()[0]
import sklearn, torch, scipy; out["versions"] = {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "torch": torch.__version__, "scipy": scipy.__version__}
((ROOT / "outputs/analysis").mkdir(parents=True, exist_ok=True) or ROOT / "outputs/analysis" / "runtime_and_data_stats.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
