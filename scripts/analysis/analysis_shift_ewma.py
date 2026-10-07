"""Additional analyses (2026-10-07, after the locked test), all on existing runs (no new fitting):
(1) shift test: CBP vs naive / A3 / A4 / A1 with cluster-bootstrap intervals (held-out high/low-speed groups);
(2) data-driven EWMA baseline vs CBP on the same events (paired, cluster bootstrap), CV and test;
(3) number of events and units available at each horizon (censoring at the end of a recording);
(4) coverage per profile/session and the profile-averaged (equal-weight) coverage.
Writes outputs/analysis_shift_ewma/ (results.json and LaTeX tables)."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "outputs/runs"; OUT = ROOT / "outputs/analysis_shift_ewma"; OUT.mkdir(parents=True, exist_ok=True)
def latest(p): return Path([x for x in sorted(glob.glob(str(R / p))) if "_warm_" not in Path(x).name][-1])
rng = np.random.default_rng(20261006); B = 4000; res = {}
TIMES = [0, 2, 4, 5, 10, 20, 30]

def boot_units(units, rows_of):
    return [np.concatenate([rows_of[u] for u in rng.choice(units, len(units))]) for _ in range(B)]

def ci(vals): return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

# ---------------- (1) shift test
shift = {}
for ds, ev in [("paderborn", "reset"), ("induction", "start")]:
    for grp in ["high", "low"]:
        run = latest(f"*_shift_{grp}_{ds}_*"); E = pd.read_csv(run / "episodes.csv"); cfg = json.loads((run / "resolved_config.json").read_text())
        for sc in ["S1", "S3"]:
            G = E[(E.event == ev) & (E.scenario == sc)]
            piv = {c: G.pivot_table(index=["unit", "r"], columns="method", values=c) for c in ["err_0", "cov_0", "hw_0", "err_10", "cov_10", "cov_30"]}
            idx = piv["err_0"].index; units = np.array(sorted(set(idx.get_level_values(0))))
            piv = {c: v.reindex(idx) for c, v in piv.items()}
            rows_of = {u: np.where(idx.get_level_values(0) == u)[0] for u in units}; bs = boot_units(units, rows_of)
            d = {"n_events": int(len(idx)), "n_units": int(len(units)), "held_out_speed_range": cfg["held_out_speed_range"],
                 "train_speed_range": cfg["train_speed_range"], "methods": {}}
            for m in ["naive_KF", "CBP", "A3_plainconf", "A4_uncal", "A1_labelprior"]:
                if m not in piv["err_0"]: continue
                e0 = piv["err_0"][m].values; c0 = piv["cov_0"][m].values
                rec = {"e0_med": float(np.nanmedian(e0)), "e0_p90": float(np.nanpercentile(e0, 90)), "cov0": float(np.nanmean(c0)),
                       "cov0_ci": ci([np.nanmean(c0[b]) for b in bs]), "hw0_med": float(np.nanmedian(piv["hw_0"][m].values)),
                       "e10_med": float(np.nanmedian(piv["err_10"][m].values)), "cov10": float(np.nanmean(piv["cov_10"][m].values)),
                       "cov30": float(np.nanmean(piv["cov_30"][m].values))}
                if m != "CBP":
                    c = piv["err_0"]["CBP"].values
                    rec["d_e0_vs_CBP"] = float(np.nanmedian(e0) - np.nanmedian(c))
                    rec["d_e0_vs_CBP_ci"] = ci([np.nanmedian(e0[b]) - np.nanmedian(c[b]) for b in bs])
                d["methods"][m] = rec
            shift[f"{ds}|{grp}|{sc}"] = d
res["shift"] = shift

# ---------------- (2) EWMA baseline vs CBP (paired on identical events)
ewma = {}
for mode, src in [("cv", {"paderborn": "*_cv5_ablation_v2_paderborn_*", "induction": "*_cv5_ablation_v2_induction_*"}),
                  ("test", {"paderborn": "*_LOCKED_TEST_paderborn_*", "induction": "*_LOCKED_TEST_induction_*"})]:
    for ds, pat in src.items():
        C = pd.read_csv(latest(pat) / "episodes.csv"); W = pd.read_csv(latest(f"*_ewma_baseline_{mode}_{ds}_*") / "episodes.csv")
        for ev in (["reset", "start"] if ds == "paderborn" else ["start"]):
            for sc in ["S1", "S3"]:
                c = C[(C.event == ev) & (C.scenario == sc)]
                w = W[(W.event == ev) & (W.scenario == sc)]
                key = ["unit", "r"]
                cb = c[c.method == "CBP"].set_index(key); nv = c[c.method == "naive_KF"].set_index(key)
                d = {}
                for m in sorted(w.method.unique()):
                    wm = w[w.method == m].set_index(key); j = cb.join(wm, how="inner", lsuffix="_cbp", rsuffix="_w")
                    units = np.array(sorted(set(j.index.get_level_values(0)))); rows_of = {u: np.where(j.index.get_level_values(0) == u)[0] for u in units}
                    bs = boot_units(units, rows_of) if mode == "cv" else None
                    rec = {"n_events": int(len(j)), "n_units": int(len(units))}
                    for t in [0, 10, 30]:
                        a, b_ = j[f"err_{t}_w"].values, j[f"err_{t}_cbp"].values; ok = ~np.isnan(a) & ~np.isnan(b_)
                        rec[f"t{t}"] = {"ewma_med": float(np.median(a[ok])), "ewma_p90": float(np.percentile(a[ok], 90)),
                                        "cbp_med": float(np.median(b_[ok])), "cbp_p90": float(np.percentile(b_[ok], 90)),
                                        "diff_med": float(np.median(a[ok]) - np.median(b_[ok])), "n": int(ok.sum())}
                        if bs is not None:
                            rec[f"t{t}"]["diff_ci"] = ci([np.nanmedian(np.where(ok, a, np.nan)[bb]) - np.nanmedian(np.where(ok, b_, np.nan)[bb]) for bb in bs])
                    d[m] = rec
                ewma[f"{mode}|{ds}|{ev}|{sc}"] = d
res["ewma"] = ewma

# ---------------- (3) events per horizon and (4) coverage per unit
cohort, perunit = {}, {}
for mode, src in [("cv", {"paderborn": "*_cv5_ablation_v2_paderborn_*", "induction": "*_cv5_ablation_v2_induction_*"}),
                  ("test", {"paderborn": "*_LOCKED_TEST_paderborn_*", "induction": "*_LOCKED_TEST_induction_*"})]:
    for ds, pat in src.items():
        C = pd.read_csv(latest(pat) / "episodes.csv")
        for ev in (["reset", "start"] if ds == "paderborn" else ["start"]):
            for sc in ["S1", "S3"]:
                g = C[(C.event == ev) & (C.scenario == sc) & (C.method == "CBP")]
                cohort[f"{mode}|{ds}|{ev}|{sc}"] = {f"t{t}": [int(g[f"err_{t}"].notna().sum()), int(g[g[f"err_{t}"].notna()].unit.nunique())] for t in TIMES}
                pu = g.groupby("unit").cov_0.mean()
                perunit[f"{mode}|{ds}|{ev}|{sc}"] = {"pooled": float(g.cov_0.mean()), "unit_mean": float(pu.mean()),
                                                     "unit_min": float(pu.min()), "unit_q10": float(pu.quantile(.1)),
                                                     "share_units_below_0.8": float((pu < 0.8).mean()), "n_units": int(len(pu)),
                                                     "per_unit": {str(k): round(float(v), 3) for k, v in pu.items()} if mode == "test" else None}
res["cohort_events_units"] = cohort; res["coverage_per_unit"] = perunit
(OUT / "results.json").write_text(json.dumps(res, indent=1))

# ---------------- LaTeX tables
def f(x): return f"{x:.2f}"
L = [r"\begin{table*}[t]", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\caption{Shift test (development data). The fastest or slowest 20\,\% of the units, ranked by mean speed, were held out of identification, simulation, prior, calibration and $q$ selection. Winding sensor only (S1). $e_0$: median/90th percentile of the absolute error at the event (K); $c_0$: coverage of the nominal 90\,\% band at the event with a 95\,\% cluster-bootstrap interval; $h_0$: median half-width (K).}",
     r"\label{tab:shift}", r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}llccccc@{}}", r"\toprule",
     r"Held-out group & Method & $e_0$ & $c_0$ & $h_0$ & $c_{10}$ & $c_{30}$\\", r"\midrule"]
names = {"naive_KF": "Naive KF", "CBP": r"\textbf{CBP}", "A3_plainconf": "A3 no novelty", "A4_uncal": "A4 no conformal"}
for ds, lab in [("paderborn", "PMSM resets"), ("induction", "IM starts")]:
    for grp in ["high", "low"]:
        d = shift[f"{ds}|{grp}|S1"]; sr = d["held_out_speed_range"]
        head = f"{lab}, {grp} speed ({sr[0]:.0f}--{sr[1]:.0f}\\,rpm; {d['n_events']} events, {d['n_units']} units)"
        for i, m in enumerate(["naive_KF", "CBP", "A3_plainconf", "A4_uncal"]):
            r_ = d["methods"][m]
            L.append((head if i == 0 else "") + f" & {names[m]} & {f(r_['e0_med'])}/{f(r_['e0_p90'])} & {f(r_['cov0'])} [{f(r_['cov0_ci'][0])}, {f(r_['cov0_ci'][1])}] & {r_['hw0_med']:.1f} & {f(r_['cov10'])} & {f(r_['cov30'])}\\\\")
        L.append(r"\midrule" if not (ds == "induction" and grp == "low") else "")
L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
(OUT / "tab_shift.tex").write_text("\n".join(l for l in L if l != ""), encoding="utf-8")

L = [r"\begin{table*}[t]", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\caption{Data-driven estimators of the benchmark family \citep{kirchgassner2021benchmark} (exponentially weighted moving-average features, OLS or gradient-boosted trees, trained with rotor labels) compared with CBP on the same events. After an event the features restart from the event; ``no reset'' uses uninterrupted features and is not available after a real reset. Median/90th percentile of the absolute error (K) at 0, 10 and 30\,min, winding sensor only (S1); $\Delta_0$: difference of medians at the event (estimator minus CBP) with a 95\,\% cluster-bootstrap interval (cross-validation).}",
     r"\label{tab:ewma}", r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lllcccc@{}}", r"\toprule",
     r"Data & Events & Estimator & $e_0$ & $e_{10}$ & $e_{30}$ & $\Delta_0$\\", r"\midrule"]
lab = {"EWMA_OLS_reset": "EWMA-OLS (reset)", "EWMA_GBT_reset": "EWMA-GBT (reset)", "EWMA_OLS_no_reset": "EWMA-OLS (no reset)", "EWMA_GBT_no_reset": "EWMA-GBT (no reset)"}
for mode in ["cv", "test"]:
    for ds, ev, evl in [("paderborn", "reset", "PMSM resets"), ("paderborn", "start", "PMSM starts"), ("induction", "start", "IM starts")]:
        d = ewma[f"{mode}|{ds}|{ev}|S1"]; first = True
        ms = [m for m in ["EWMA_OLS_reset", "EWMA_GBT_reset", "EWMA_OLS_no_reset", "EWMA_GBT_no_reset"] if m in d]
        r0 = d[ms[0]]
        cbp = f"{f(r0['t0']['cbp_med'])}/{f(r0['t0']['cbp_p90'])} & {f(r0['t10']['cbp_med'])}/{f(r0['t10']['cbp_p90'])} & {f(r0['t30']['cbp_med'])}/{f(r0['t30']['cbp_p90'])} & --"
        L.append(f"{'CV' if mode == 'cv' else 'Test'} & {evl} ({r0['n_events']}) & \\textbf{{CBP}} & {cbp}\\\\")
        for m in ms:
            r_ = d[m]; dl = f"{f(r_['t0']['diff_med'])}" + (f" [{f(r_['t0']['diff_ci'][0])}, {f(r_['t0']['diff_ci'][1])}]" if "diff_ci" in r_["t0"] else "")
            L.append(f" & & {lab[m]} & {f(r_['t0']['ewma_med'])}/{f(r_['t0']['ewma_p90'])} & {f(r_['t10']['ewma_med'])}/{f(r_['t10']['ewma_p90'])} & {f(r_['t30']['ewma_med'])}/{f(r_['t30']['ewma_p90'])} & {dl}\\\\")
        L.append(r"\midrule")
L = L[:-1] + [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
(OUT / "tab_ewma.tex").write_text("\n".join(L), encoding="utf-8")
print(json.dumps({k: {m: (v['cov0'], v['cov0_ci'], v['e0_med']) for m, v in d['methods'].items()} for k, d in shift.items() if k.endswith('S1')}, indent=0)[:2500])
print(json.dumps(perunit, indent=0)[:2500]); print(json.dumps({k: v for k, v in cohort.items() if 'S1' in k}, indent=0)[:2000])
