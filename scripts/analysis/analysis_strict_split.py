"""Additional analyses (2026-10-07, after the locked test), all on existing runs (no new fitting):
(1) strict-split CV (run_cv5_strictsplit.py): CBP and ablations with cluster-bootstrap intervals, and the paired change of
    the CBP error against the frozen recipe (run_cv5_ablation_v2.py) on identical events;
(2) strict-split shift test (run_shift_strict.py): coverage with and without novelty scaling;
(3) data-driven EWMA estimators with both start-up conventions (benchmark zero padding and warm start from the current
    measurement, run_ewma_baseline.py --pad current) against CBP on identical events.
Writes outputs/analysis_strict_split/ (results.json, tab_strict.tex, tab_ewma.tex)."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "outputs/runs"; OUT = ROOT / "outputs/analysis_strict_split"; OUT.mkdir(parents=True, exist_ok=True)
def latest(p, exclude=None):
    c = [x for x in sorted(glob.glob(str(R / p))) if not (exclude and exclude in Path(x).name)]
    return Path(c[-1])
rng = np.random.default_rng(20261006); B = 4000; res = {}
def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def boot(units, rows_of): return [np.concatenate([rows_of[u] for u in rng.choice(units, len(units))]) for _ in range(B)]
EV = {"paderborn": "reset", "induction": "start"}

def summarise(E, ev, sc, methods, ref=None):
    """Per-method statistics on the events of one run; ref: episodes of another run for a paired CBP comparison."""
    G = E[(E.event == ev) & (E.scenario == sc)]
    piv = {c: G.pivot_table(index=["unit", "r"], columns="method", values=c) for c in ["err_0", "cov_0", "hw_0", "err_10", "cov_10", "cov_30"]}
    idx = piv["err_0"].index; piv = {c: v.reindex(idx) for c, v in piv.items()}
    units = np.array(sorted(set(idx.get_level_values(0)))); rows_of = {u: np.where(idx.get_level_values(0) == u)[0] for u in units}
    bs = boot(units, rows_of); d = {"n_events": int(len(idx)), "n_units": int(len(units)), "methods": {}}
    for m in methods:
        e0 = piv["err_0"][m].values; c0 = piv["cov_0"][m].values
        d["methods"][m] = {"e0_med": float(np.nanmedian(e0)), "e0_p90": float(np.nanpercentile(e0, 90)), "cov0": float(np.nanmean(c0)),
                           "cov0_ci": ci([np.nanmean(c0[b]) for b in bs]), "hw0_med": float(np.nanmedian(piv["hw_0"][m].values)),
                           "e10_med": float(np.nanmedian(piv["err_10"][m].values)), "cov10": float(np.nanmean(piv["cov_10"][m].values)),
                           "cov30": float(np.nanmean(piv["cov_30"][m].values))}
    c_cbp, c_a3 = piv["cov_0"]["CBP"].values, piv["cov_0"]["A3_plainconf"].values
    d["cov0_CBP_minus_A3"] = float(np.nanmean(c_cbp) - np.nanmean(c_a3)); d["cov0_CBP_minus_A3_ci"] = ci([np.nanmean(c_cbp[b]) - np.nanmean(c_a3[b]) for b in bs])
    if ref is not None:
        Rr = ref[(ref.event == ev) & (ref.scenario == sc) & (ref.method == "CBP")].set_index(["unit", "r"]).reindex(idx)
        a, b_ = piv["err_0"]["CBP"].values, Rr["err_0"].values
        d["e0_CBP_strict_minus_frozen"] = float(np.nanmedian(a) - np.nanmedian(b_))
        d["e0_CBP_strict_minus_frozen_ci"] = ci([np.nanmedian(a[b]) - np.nanmedian(b_[b]) for b in bs])
        d["frozen_CBP"] = {"e0_med": float(np.nanmedian(b_)), "cov0": float(np.nanmean(Rr["cov_0"].values)), "hw0_med": float(np.nanmedian(Rr["hw_0"].values))}
    return d

M = ["naive_KF", "CBP", "A3_plainconf", "A4_uncal", "A1_labelprior", "oracle"]
strict = {}
for ds in ["paderborn", "induction"]:
    E = pd.read_csv(latest(f"*_cv5_strictsplit_{ds}_*") / "episodes.csv"); F = pd.read_csv(latest(f"*_cv5_ablation_v2_{ds}_*") / "episodes.csv")
    for sc in ["S1", "S3"]:
        strict[f"cv|{ds}|{sc}"] = summarise(E, EV[ds], sc, M, ref=F)
        if ds == "paderborn": strict[f"cv|{ds}|start|{sc}"] = summarise(E, "start", sc, M, ref=F)
    for grp in ["high", "low"]:
        run = latest(f"*_shift_strict_{grp}_{ds}_*"); E = pd.read_csv(run / "episodes.csv"); cfg = json.loads((run / "resolved_config.json").read_text())
        F = pd.read_csv(latest(f"*_shift_{grp}_{ds}_*") / "episodes.csv")
        for sc in ["S1", "S3"]:
            d = summarise(E, EV[ds], sc, ["naive_KF", "CBP", "A3_plainconf", "A4_uncal"], ref=F); d["held_out_speed_range"] = cfg["held_out_speed_range"]
            strict[f"shift|{ds}|{grp}|{sc}"] = d
res["strict"] = strict

ewma = {}
for mode, src in [("cv", "*_cv5_ablation_v2_{}_*"), ("test", "*_LOCKED_TEST_{}_*")]:
    for ds in ["paderborn", "induction"]:
        C = pd.read_csv(latest(src.format(ds)) / "episodes.csv")
        W0 = pd.read_csv(latest(f"*_ewma_baseline_{mode}_{ds}_*", exclude="warm") / "episodes.csv")
        W1 = pd.read_csv(latest(f"*_ewma_baseline_{mode}_{ds}_warm_*") / "episodes.csv")
        W1 = W1.assign(method=W1.method.str.replace("_reset", "_warm", regex=False)); W1 = W1[~W1.method.str.contains("no_")]
        W = pd.concat([W0, W1])
        for ev in (["reset", "start"] if ds == "paderborn" else ["start"]):
            c = C[(C.event == ev) & (C.scenario == "S1") & (C.method == "CBP")].set_index(["unit", "r"]); w = W[(W.event == ev) & (W.scenario == "S1")]
            d = {}
            for m in sorted(w.method.unique()):
                j = c.join(w[w.method == m].set_index(["unit", "r"]), how="inner", lsuffix="_cbp", rsuffix="_w")
                units = np.array(sorted(set(j.index.get_level_values(0)))); rows_of = {u: np.where(j.index.get_level_values(0) == u)[0] for u in units}
                bs = boot(units, rows_of) if mode == "cv" else None; rec = {"n_events": int(len(j))}
                for t in [0, 10, 30]:
                    a, b_ = j[f"err_{t}_w"].values, j[f"err_{t}_cbp"].values; ok = ~np.isnan(a) & ~np.isnan(b_)
                    rec[f"t{t}"] = {"ewma_med": float(np.median(a[ok])), "ewma_p90": float(np.percentile(a[ok], 90)), "cbp_med": float(np.median(b_[ok])),
                                    "cbp_p90": float(np.percentile(b_[ok], 90)), "diff_med": float(np.median(a[ok]) - np.median(b_[ok]))}
                    if bs is not None:
                        aa, bb_ = np.where(ok, a, np.nan), np.where(ok, b_, np.nan)
                        rec[f"t{t}"]["diff_ci"] = ci([np.nanmedian(aa[bb]) - np.nanmedian(bb_[bb]) for bb in bs])
                d[m] = rec
            ewma[f"{mode}|{ds}|{ev}"] = d
res["ewma"] = ewma
(OUT / "results.json").write_text(json.dumps(res, indent=1))

def f(x): return f"{x:.2f}"
def cc(r_): return f"{f(r_['cov0'])} [{f(r_['cov0_ci'][0])}, {f(r_['cov0_ci'][1])}]"
L = [r"\begin{table*}[t]", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\caption{Strict split. In every fold, or in the shift test, the training units were divided into a fit half (identification, simulations, prior, novelty index, $q$ rule) and a calibration half used only for the conformal quantile, so that no rotor label of a calibration unit enters a fitted component. Winding sensor only (S1). $e_0$: median/90th percentile of the CBP error at the event (K), with the change of the median against the frozen recipe on the same events; $c_0$: coverage of the nominal 90\,\% band at the event with a 95\,\% cluster-bootstrap interval; $h_0$: median half-width (K).}",
     r"\label{tab:strict}", r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lccccccc@{}}", r"\toprule",
     r" & \multicolumn{3}{c}{CBP} & \multicolumn{2}{c}{A3 no novelty} & A4 & \\ \cmidrule(lr){2-4}\cmidrule(lr){5-6}",
     r"Events & $e_0$ (change) & $c_0$ & $h_0$ & $c_0$ & $h_0$ & $c_0$ & $c_{30}$ CBP\\", r"\midrule"]
rows = [("cv|paderborn|S1", "CV, PMSM resets"), ("cv|induction|S1", "CV, induction-motor starts"),
        ("shift|paderborn|high|S1", "Shift, PMSM resets, fast"), ("shift|paderborn|low|S1", "Shift, PMSM resets, slow"),
        ("shift|induction|high|S1", "Shift, IM starts, fast"), ("shift|induction|low|S1", "Shift, IM starts, slow")]
for k, lab in rows:
    d = strict[k]; m = d["methods"]; ch = d["e0_CBP_strict_minus_frozen"]
    L.append(f"{lab} ({d['n_events']}) & {f(m['CBP']['e0_med'])}/{f(m['CBP']['e0_p90'])} ({ch:+.2f}) & {cc(m['CBP'])} & {m['CBP']['hw0_med']:.1f} & {cc(m['A3_plainconf'])} & {m['A3_plainconf']['hw0_med']:.1f} & {f(m['A4_uncal']['cov0'])} & {f(m['CBP']['cov30'])}\\\\")
    if k == "cv|induction|S1": L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
(OUT / "tab_strict.tex").write_text("\n".join(L).replace("+-", "$-$").replace("(-", "($-$").replace("(+", "(+"), encoding="utf-8")

L = [r"\begin{table*}[t]", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\caption{Data-driven estimators of the benchmark family \citep{kirchgassner2021benchmark} (exponentially weighted moving-average features, OLS or gradient-boosted trees, trained with rotor labels) compared with CBP on the same events. After an event the moving averages restart: with the start-up convention of the benchmark code (zero history for electrical inputs) or with a warm start, in which the history is filled with the current measurement of every input. ``No reset'' uses uninterrupted features and is not available after a real reset. Median/90th percentile of the absolute error (K) at 0, 10 and 30\,min, winding sensor only (S1); $\Delta_0$: difference of medians at the event (estimator minus CBP) with a 95\,\% cluster-bootstrap interval (cross-validation only). The three PMSM test starts are not shown.}",
     r"\label{tab:ewma}", r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lllcccc@{}}", r"\toprule",
     r"Data & Events & Estimator & $e_0$ & $e_{10}$ & $e_{30}$ & $\Delta_0$\\", r"\midrule"]
lab = {"EWMA_OLS_reset": "EWMA-OLS, benchmark start", "EWMA_GBT_reset": "EWMA-GBT, benchmark start", "EWMA_OLS_warm": "EWMA-OLS, warm start",
       "EWMA_GBT_warm": "EWMA-GBT, warm start", "EWMA_OLS_no_reset": "EWMA-OLS, no reset", "EWMA_GBT_no_reset": "EWMA-GBT, no reset"}
order = ["EWMA_OLS_reset", "EWMA_GBT_reset", "EWMA_OLS_warm", "EWMA_GBT_warm", "EWMA_OLS_no_reset", "EWMA_GBT_no_reset"]
for mode, ds, ev, evl in [("cv", "paderborn", "reset", "PMSM resets"), ("cv", "paderborn", "start", "PMSM starts"), ("cv", "induction", "start", "IM starts"),
                          ("test", "paderborn", "reset", "PMSM resets"), ("test", "induction", "start", "IM starts")]:
    d = ewma[f"{mode}|{ds}|{ev}"]; ms = [m for m in order if m in d]; r0 = d[ms[0]]
    L.append(f"{'CV' if mode == 'cv' else 'Test'} & {evl} ({r0['n_events']}) & \\textbf{{CBP}} & {f(r0['t0']['cbp_med'])}/{f(r0['t0']['cbp_p90'])} & {f(r0['t10']['cbp_med'])}/{f(r0['t10']['cbp_p90'])} & {f(r0['t30']['cbp_med'])}/{f(r0['t30']['cbp_p90'])} & --\\\\")
    for m in ms:
        r_ = d[m]; dl = f(r_["t0"]["diff_med"]) + (f" [{f(r_['t0']['diff_ci'][0])}, {f(r_['t0']['diff_ci'][1])}]" if "diff_ci" in r_["t0"] else "")
        L.append(f" & & {lab[m]} & {f(r_['t0']['ewma_med'])}/{f(r_['t0']['ewma_p90'])} & {f(r_['t10']['ewma_med'])}/{f(r_['t10']['ewma_p90'])} & {f(r_['t30']['ewma_med'])}/{f(r_['t30']['ewma_p90'])} & {dl}\\\\")
    L.append(r"\midrule")
L = L[:-1] + [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
import re
txt = re.sub(r"(?<![\w$.\[])-(\d+\.\d+)", r"$-$\1", "\n".join(L)); txt = re.sub(r"\[-(\d)", r"[$-$\1", txt); txt = re.sub(r", -(\d)", r", $-$\1", txt)
(OUT / "tab_ewma.tex").write_text(txt, encoding="utf-8")
for k, d in strict.items():
    if k.endswith("S1"):
        m = d["methods"]; print(k, "CBP e0", round(m["CBP"]["e0_med"], 2), "chg", round(d["e0_CBP_strict_minus_frozen"], 2), [round(x, 2) for x in d["e0_CBP_strict_minus_frozen_ci"]],
                                 "cov", round(m["CBP"]["cov0"], 3), [round(x, 2) for x in m["CBP"]["cov0_ci"]], "A3", round(m["A3_plainconf"]["cov0"], 3), "dcov", round(d["cov0_CBP_minus_A3"], 3), [round(x, 2) for x in d["cov0_CBP_minus_A3_ci"]])
for k, d in ewma.items():
    for m, r_ in d.items():
        if "warm" in m: print(k, m, round(r_["t0"]["ewma_med"], 2), round(r_["t0"]["cbp_med"], 2), round(r_["t0"]["diff_med"], 2), r_["t0"].get("diff_ci"))
