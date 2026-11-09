"""Additional analysis (2026-11, added during the revision, after the locked test): the thermal neural network (TNN)
baseline of scripts/run_tnn_official.py compared with CBP on identical events, as analysis_shift_ewma.py does for the
EWMA estimators. Uses the newest outputs/runs/*_tnn_official folder and the newest CBP runs
(*_cv5_ablation_v2_paderborn_*, *_LOCKED_TEST_paderborn_*).
Seed selection (fixed before results): per CV fold, and for the test model, the seed with the lowest training loss of
the last epoch. Differences of medians (TNN minus CBP) with 95% cluster-bootstrap intervals over profiles (CV only).
Also the continuous accuracy of the test model on profiles 60, 62 and 74 started from the true state (the TNN paper's
generalisation set; the paper reports a mean squared error of 3.18 K^2 for its optimised model).
Writes outputs/analysis_tnn/results.json."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "outputs/runs"; OUT = ROOT / "outputs/analysis_tnn"; OUT.mkdir(parents=True, exist_ok=True)
def latest(p): return Path(sorted(glob.glob(str(R / p)))[-1])
RUN = latest("*_tnn_official")
rng = np.random.default_rng(20261006); B = 4000
def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

metas = {Path(f).stem[5:]: json.load(open(f)) for f in glob.glob(str(RUN / "meta_*.json"))}
sel = {}
for tag, m in metas.items():
    key = tag.rsplit("_s", 1)[0]
    if key not in sel or m["final_train_mse_K2"] < metas[sel[key]]["final_train_mse_K2"]:
        sel[key] = tag
res = {"run": RUN.name, "selected": sel, "final_train_mse_K2": {t: m["final_train_mse_K2"] for t, m in metas.items()}}

for mode, src in [("cv", "*_cv5_ablation_v2_paderborn_*"), ("test", "*_LOCKED_TEST_paderborn_*")]:
    tags = [t for k, t in sel.items() if k.startswith(mode)]
    if not tags: continue
    if mode == "cv" and len(tags) < 5: print("WARNING: only", len(tags), "CV folds found")
    T = pd.concat([pd.read_csv(RUN / f"episodes_{t}.csv") for t in tags])
    C = pd.read_csv(latest(src) / "episodes.csv")
    for ev in ["reset", "start"]:
        c = C[(C.event == ev) & (C.scenario == "S1") & (C.method == "CBP")].set_index(["unit", "r"])
        d = {}
        for m in sorted(T.method.unique()):
            w = T[(T.event == ev) & (T.method == m)].set_index(["unit", "r"])
            j = c.join(w, how="inner", lsuffix="_cbp", rsuffix="_w")
            units = np.array(sorted(set(j.index.get_level_values(0)))); rows_of = {u: np.where(j.index.get_level_values(0) == u)[0] for u in units}
            bs = [np.concatenate([rows_of[u] for u in rng.choice(units, len(units))]) for _ in range(B)] if mode == "cv" else None
            rec = {"n_events": int(len(j)), "n_units": int(len(units))}
            for t in [0, 10, 30]:
                a, b_ = j[f"err_{t}_w"].values, j[f"err_{t}_cbp"].values; ok = ~np.isnan(a) & ~np.isnan(b_)
                rec[f"t{t}"] = {"med": float(np.median(a[ok])), "p90": float(np.percentile(a[ok], 90)), "cbp_med": float(np.median(b_[ok])),
                                "diff_med": float(np.median(a[ok]) - np.median(b_[ok])), "n": int(ok.sum())}
                if bs is not None:
                    aa, bb_ = np.where(ok, a, np.nan), np.where(ok, b_, np.nan)
                    rec[f"t{t}"]["diff_ci"] = ci([np.nanmedian(aa[bb]) - np.nanmedian(bb_[bb]) for bb in bs])
            d[m] = rec
        res[f"{mode}|paderborn|{ev}"] = d

for t in [t for k, t in sel.items() if k.startswith("test")]:
    cont = metas[t]["continuous"]; n = sum(v["n"] for v in cont.values())
    res["fidelity_test"] = {"per_profile": cont, "mse_all_K2_weighted": sum(v["mse_all_K2"] * v["n"] for v in cont.values()) / n,
                            "max_abs_pm_K": max(v["max_abs_pm_K"] for v in cont.values()), "paper_mse_all_K2": 3.18, "paper_max_abs_K": 5.84}
(OUT / "results.json").write_text(json.dumps(res, indent=1))
for k, d in res.items():
    if "|" in k:
        for m, r in d.items():
            print(k, m, r["n_events"], {t: (round(r[t]["med"], 2), round(r[t]["p90"], 2), round(r[t]["diff_med"], 2),
                                         [round(x, 2) for x in r[t].get("diff_ci", [])]) for t in ["t0", "t10", "t30"]})
if "fidelity_test" in res:
    f = res["fidelity_test"]; print("test profiles 60/62/74, true start: MSE", round(f["mse_all_K2_weighted"], 2), "K^2, max PM error", round(f["max_abs_pm_K"], 2), "K")
