"""LaTeX tables for the manuscript, computed directly from the registered run episodes (no manual transcription).
Table (CV): winding-only scenario S1, PMSM mid-run resets and induction-motor starts, 5-fold CV.
Table (test): locked single test, S1 and S3, with exact Clopper-Pearson intervals for the coverage at the event."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); from figstyle import latest_run
from scipy.stats import beta
ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "outputs/runs"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/tables"; OUT.mkdir(parents=True, exist_ok=True)
CV = {"pmsm": (latest_run("*_cv5_ablation_v2_paderborn_*").name, "reset"), "im": (latest_run("*_cv5_ablation_v2_induction_*").name, "start")}
TE = {"pmsm": (latest_run("*_LOCKED_TEST_paderborn_*").name, "reset"), "im": (latest_run("*_LOCKED_TEST_induction_*").name, "start")}

def fmt(v): return "0" if v == 0 else (f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}")
def mp(x):
    x = pd.Series(x).dropna()
    return f"{fmt(x.median())}/{fmt(x.quantile(.9))}" if len(x) else "--"
def cp(k, n):
    lo = beta.ppf(.025, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(.975, k + 1, n - k) if k < n else 1.0
    return lo, hi
def cov(g, t=0, ci=False):
    c = g[f"cov_{t}"].dropna(); k, n = int(c.sum()), len(c)
    if not ci: return f"{k / n:.2f}"
    lo, hi = cp(k, n); return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}]"
def hw(g, t=0): return f"{g[f'hw_{t}'].median():.1f}"

def block(E, ev, sc, meths, ci):
    rows = {}
    for label, m in meths:
        if m.startswith("init_"):
            g = E[(E.event == ev) & (E.scenario == "-") & (E.method == m)]
            rows[label] = [mp(g.err_0), "--", "--", "--"]
        elif m == "A6_liang":
            g = E[(E.event == ev) & (E.scenario == sc) & (E.method == m)]
            rows[label] = [mp(g.err_4), "--", "--", "--"]
        else:
            g = E[(E.event == ev) & (E.scenario == sc) & (E.method == m)]
            rows[label] = [mp(g.err_0), mp(g.err_10), cov(g, 0, ci) if m != "oracle" else "--", hw(g, 0) if m != "oracle" else "--"]
    return rows

def n_events(E, ev, sc="S1"):
    g = E[(E.event == ev) & (E.scenario == sc) & (E.method == "CBP")]
    return len(g), g.unit.nunique()

# ---------------- CV table (S1)
meths = [("Rotor $=$ winding temperature", "init_winding"), ("Rotor $=(T_\\mathrm{a}+T_\\mathrm{c})/2$", "init_mean_Ta_Tc"),
         ("Rotor $=$ coolant/housing temp.", "init_coolant_or_housing"), ("Naive KF", "naive_KF"),
         ("Liang-type inversion$^{a}$", "A6_liang"), ("CBP (proposed)", "CBP"), ("Oracle", "oracle")]
E = {k: pd.read_csv(R / v[0] / "episodes.csv") for k, v in CV.items()}
B = {k: block(E[k], CV[k][1], "S1", meths, False) for k in CV}
nP, uP = n_events(E["pmsm"], "reset"); nI, uI = n_events(E["im"], "start")
lines = [r"\begin{table*}[t]", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", r"\caption{Five-fold cross-validation, winding sensor only (S1). Absolute PM/rotor temperature error"
         r" (median/90th percentile, K) at the event ($e_0$) and 10\,min later ($e_{10}$), empirical coverage of the nominal 90\,\%"
         r" band at the event ($c_0$) and its median half-width ($h_0$, K). Deterministic initialisers have no filter, so only $e_0$ is defined.}",
         r"\label{tab:cv}", r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lcccccccc@{}}", r"\toprule",
         rf" & \multicolumn{{4}}{{c}}{{PMSM, mid-run resets ({nP} events, {uP} profiles)}} & \multicolumn{{4}}{{c}}{{Induction motor, starts ({nI} events)}}\\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}", r"Method & $e_0$ & $e_{10}$ & $c_0$ & $h_0$ & $e_0$ & $e_{10}$ & $c_0$ & $h_0$\\", r"\midrule"]
for label, _ in meths:
    a, b = B["pmsm"][label], B["im"][label]
    if label.startswith("Naive"): lines.append(r"\midrule")
    lab = r"\textbf{CBP (proposed)}" if label.startswith("CBP") else label
    lines.append(f"{lab} & " + " & ".join(a) + " & " + " & ".join(b) + r"\\")
lines += [r"\bottomrule", r"\end{tabular*}", r"\par\smallskip{\footnotesize $^{a}$Error 4\,min after the event, when the estimate first becomes available.}", r"\end{table*}"]
(OUT / "tab_cv.tex").write_text("\n".join(lines), encoding="utf-8")

# ---------------- locked test table (S1 and S3)
meths_t = [("Naive KF", "naive_KF"), ("Liang-type inversion$^{a}$", "A6_liang"), ("CBP (proposed)", "CBP"), ("Oracle", "oracle")]
E = {k: pd.read_csv(R / v[0] / "episodes.csv") for k, v in TE.items()}
nP, uP = n_events(E["pmsm"], "reset"); nI, uI = n_events(E["im"], "start")
lines = [r"\begin{table*}[t]", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", r"\caption{Locked single test (run once after the recipe was frozen). Same quantities as Table~\ref{tab:cv};"
         r" $c_0$ is given with an exact 95\,\% Clopper--Pearson interval. The PMSM test set is the generalisation set of"
         r" \citet{kirchgassner2023tnn}; the induction-motor test set contains every fifth session.}", r"\label{tab:test}",
         r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}llcccccccc@{}}", r"\toprule",
         rf" & & \multicolumn{{4}}{{c}}{{PMSM, mid-run resets ({nP} events, {uP} profiles)}} & \multicolumn{{4}}{{c}}{{Induction motor, starts ({nI} events)}}\\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}", r"Sensors & Method & $e_0$ & $e_{10}$ & $c_0$ & $h_0$ & $e_0$ & $e_{10}$ & $c_0$ & $h_0$\\", r"\midrule"]
for sc, name in [("S1", "Winding"), ("S3", "Three stator")]:
    BP = block(E["pmsm"], "reset", sc, meths_t, True); BI = block(E["im"], "start", sc, meths_t, True)
    for i, (label, m) in enumerate(meths_t):
        if sc == "S3" and m == "A6_liang": continue
        lab = r"\textbf{CBP (proposed)}" if label.startswith("CBP") else label
        lines.append((name if i == 0 else "") + f" & {lab} & " + " & ".join(BP[label]) + " & " + " & ".join(BI[label]) + r"\\")
    if sc == "S1": lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular*}", r"\par\smallskip{\footnotesize $^{a}$Error 4\,min after the event. The inversion uses the winding sensor only.}", r"\end{table*}"]
(OUT / "tab_test.tex").write_text("\n".join(lines), encoding="utf-8")
print((OUT / "tab_cv.tex").read_text(encoding="utf-8")); print((OUT / "tab_test.tex").read_text(encoding="utf-8"))
