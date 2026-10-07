"""Restart identifiability via matched pairs (2026-10-06).

Real snapshots every 30 s (TRAIN + V1/V2).  Two snapshots are 'matched' if everything a zero-history re-initializer may use
is (nearly) identical: winding (and tooth, yoke for S3) within 1 K, coolant and ambient within 1 K, speed within 200 rpm,
current magnitude within 5 A, voltage magnitude within 5 V; and they come from different profiles or are >= 30 min apart.
For matched pairs, any snapshot-only estimator returns the same value for both, so its error on the worse member is
>= |dPM|/2: the distribution of |dPM|/2 is an empirical ambiguity floor.  Compared with the CBP errors and calibrated
bands (run_cbp_qpm_tuning recipe) on V resets.  PM is used only for analysis.  G never loaded.
"""
from __future__ import annotations
import sys, json, glob, time, hashlib, platform, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pmsm_softsense import realdata

CFG = {"stride_s": 30, "tol": {"T": 1.0, "rpm": 200.0, "I": 5.0, "U": 5.0}, "min_sep_min": 30, "max_pairs_per_point": 5}
TRAIN = realdata.protocol_ids("train"); VAL = realdata.protocol_ids("val")
D = realdata.load(TRAIN + VAL)
rows = []
for pid in TRAIN + VAL:
    o, t = D[pid]
    for r in range(0, len(o), int(CFG["stride_s"] / 0.5)):
        rows.append({"pid": pid, "r": r, "split": "train" if pid in TRAIN else "val", "Tw": o.stator_winding.values[r],
                     "Tt": o.stator_tooth.values[r], "Ty": o.stator_yoke.values[r], "Tc": o.coolant.values[r],
                     "Ta": o.ambient.values[r], "rpm": abs(o.motor_speed.values[r]),
                     "I": float(np.hypot(o.i_d.values[r], o.i_q.values[r])), "U": float(np.hypot(o.u_d.values[r], o.u_q.values[r])),
                     "pm": t.pm.values[r]})
S = pd.DataFrame(rows)
tol = CFG["tol"]; out = {"n_snapshots": int(len(S))}; pairs_all = {}
for sc, cols in {"S1": ["Tw"], "S3": ["Tw", "Tt", "Ty"]}.items():
    feats = cols + ["Tc", "Ta", "rpm", "I", "U"]
    scale = np.array([tol["T"]] * len(cols) + [tol["T"], tol["T"], tol["rpm"], tol["I"], tol["U"]])
    Z = S[feats].values / scale
    tree = cKDTree(Z)
    pr = tree.query_pairs(r=1.0, p=np.inf, output_type="ndarray")      # Chebyshev: every feature within its tolerance
    a, b = pr[:, 0], pr[:, 1]
    sep_ok = (S.pid.values[a] != S.pid.values[b]) | (np.abs(S.r.values[a] - S.r.values[b]) * 0.5 / 60 >= CFG["min_sep_min"])
    a, b = a[sep_ok], b[sep_ok]
    dpm = np.abs(S.pm.values[a] - S.pm.values[b])
    pairs_all[sc] = (a, b, dpm)
    floor = dpm / 2
    out[sc] = {"matched_pairs": int(len(a)), "points_with_a_match": int(len(np.unique(np.r_[a, b]))),
               "abs_dPM_median": float(np.median(dpm)), "abs_dPM_p90": float(np.quantile(dpm, .9)),
               "frac_dPM_gt_5K": float(np.mean(dpm > 5)), "frac_dPM_gt_10K": float(np.mean(dpm > 10)),
               "ambiguity_floor_median_K": float(np.median(floor)), "ambiguity_floor_p90_K": float(np.quantile(floor, .9)),
               "abs_dPM_max": float(dpm.max())}
    # per-point worst ambiguity: for each snapshot, max |dPM| over its matches (the spread a calibrated band should cover)
    worst = pd.Series(np.r_[dpm, dpm]).groupby(np.r_[a, b]).max()
    out[sc]["per_point_worst_dPM_median"] = float(worst.median()); out[sc]["per_point_worst_dPM_p90"] = float(worst.quantile(.9))
    # example: largest-ambiguity pair, with the measured inputs
    k = int(np.argmax(dpm))
    out[sc]["example_max_pair"] = {"A": S.iloc[a[k]][["pid", "r"] + feats + ["pm"]].to_dict(),
                                   "B": S.iloc[b[k]][["pid", "r"] + feats + ["pm"]].to_dict()}

# compare with CBP on V resets (from the q_pm tuning run, q=2) where available
qrun = sorted(glob.glob(str(ROOT / "outputs/runs/*_cbp_qpm_tuning_*")))
if qrun:
    B = pd.read_csv(Path(qrun[-1]) / "bands.csv")
    for sc in ["S1", "S3"]:
        g = B[(B.group == "val") & (B.scenario == sc) & (B.method == "q2") & (B.event == "reset")]
        out[sc]["CBP_t0_abs_err_median_on_V"] = float(g.err_0.median()); out[sc]["CBP_t0_halfwidth_median_on_V"] = float(g.hw_0.median())
        out[sc]["CBP_t0_abs_err_p90_on_V"] = float(g.err_0.quantile(.9))
out["config"] = CFG
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN = ROOT / "outputs/runs" / f"{stamp}_matched_pairs_{uuid.uuid4().hex[:8]}"
RUN.mkdir(parents=True); (RUN / "script.py").write_bytes(Path(__file__).read_bytes())
(RUN / "summary.json").write_text(json.dumps(out, indent=1, default=float))
np.savez_compressed(RUN / "pairs.npz", **{f"{sc}_{n}": v for sc, t in pairs_all.items() for n, v in zip(["a", "b", "dpm"], t)},
                    pid=S.pid.values, r=S.r.values, pm=S.pm.values)   # added 2026-10-07 for the identifiability figure
print(RUN); print(json.dumps(out, indent=1, default=float))
