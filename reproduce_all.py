"""Reproduce every number, table and figure of the paper from the raw public data.

    python scripts/download_data.py          # once: fetch and verify the two datasets
    python reproduce_all.py                  # everything except the sensitivity study (about 4 h sequentially)
    python reproduce_all.py --parallel       # same, independent experiments in parallel (about 2 h on 8+ cores)
    python reproduce_all.py --sensitivity    # additionally the 12 sensitivity runs of Fig. 7 (adds about 2.5 h in parallel)
    python reproduce_all.py --only figures   # only re-draw figures and tables from existing runs

Each experiment writes a new time-stamped folder in outputs/runs/ with its configuration, a copy of the source code and
its results (summary.json, episodes.csv). Figures go to outputs/figures/, tables to outputs/tables/ and the bootstrap
intervals to outputs/analysis/; the shift-test and data-driven-baseline tables and statistics go to
outputs/analysis_shift_ewma/. Times are wall-clock times measured on a desktop CPU; they vary with the machine.
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SENS = [("gap60", ["--gap-max", "60"]), ("gap240", ["--gap-max", "240"]), ("k10", ["--knn-k", "10"]), ("k40", ["--knn-k", "40"]),
        ("a20", ["--alpha", "0.20"]), ("a05", ["--alpha", "0.05"])]


def run(group, parallel):
    """Run a list of commands, sequentially or as parallel subprocesses; stop on the first failure."""
    t0 = time.time()
    if parallel and len(group) > 1:
        procs = [(c, subprocess.Popen([PY] + c, cwd=ROOT)) for c in group]
        bad = [c for c, p in procs if p.wait() != 0]
    else:
        bad = []
        for c in group:
            print(">>", " ".join(c), flush=True)
            if subprocess.call([PY] + c, cwd=ROOT) != 0:
                bad.append(c); break
    if bad:
        sys.exit(f"FAILED: {bad}")
    print(f"   done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", action="store_true", help="run independent experiments as parallel processes")
    ap.add_argument("--sensitivity", action="store_true", help="also run the 12 one-factor sensitivity runs (Fig. 7)")
    ap.add_argument("--only", choices=["experiments", "figures"], help="run only one part")
    a = ap.parse_args()
    data = [ROOT / "data/raw/electric_motor_temperature/measures_v2.csv"] + [ROOT / "data/Second Dataset" / f for f in
                                                                             ["First_Part.csv", "Second_Part.csv", "Third_Part.csv"]]
    if a.only != "figures":
        missing = [str(p) for p in data if not p.exists()]
        if missing:
            sys.exit("Missing data files (run python scripts/download_data.py first):\n  " + "\n  ".join(missing))
        if not (ROOT / "outputs/diagnostics/20261006_induction_audit/per_operating_point.csv").exists():
            run([["scripts/prepare_induction_metadata.py"]], False)
        # main thermal network (Fig. 1 example, derating study, runtime statistics)
        run([["scripts/fit_lptn_real.py"]], False)
        # cross-validation with ablations and the locked single tests (Tables 5 and 7, Figs. 2 and 3)
        run([["scripts/run_cv5_ablation_v2.py", "--dataset", "paderborn"], ["scripts/run_cv5_ablation_v2.py", "--dataset", "induction"],
             ["scripts/run_locked_test.py", "--dataset", "paderborn"], ["scripts/run_locked_test.py", "--dataset", "induction"]], a.parallel)
        # identifiability (Fig. 5), derating (Fig. 6) and the example reset of Fig. 1
        run([["scripts/run_matched_pairs_identifiability.py"], ["scripts/run_derating_sim.py"], ["scripts/make_fig_example_reset.py"]], a.parallel)
        # shift test on withheld speed regimes and the data-driven (EWMA) baselines (Tables 6 and 8)
        run([["scripts/run_shift_test.py", "--dataset", ds, "--group", g] for ds in ["paderborn", "induction"] for g in ["high", "low"]]
            + [["scripts/run_ewma_baseline.py", "--dataset", ds, "--mode", m, "--pad", pad] for ds in ["paderborn", "induction"]
               for m in ["cv", "test"] for pad in ["benchmark", "current"]],
            a.parallel)
        # strict split of fitting and calibration units: cross-validation and shift test (Table 9, Fig. 4)
        run([["scripts/run_cv5_strictsplit.py", "--dataset", ds] for ds in ["paderborn", "induction"]]
            + [["scripts/run_shift_strict.py", "--dataset", ds, "--group", g] for ds in ["paderborn", "induction"] for g in ["high", "low"]]
            + [["scripts/run_cv5_groupconf.py", "--dataset", "paderborn"]],
            a.parallel)
        if a.sensitivity:
            run([["scripts/run_cv5_sensitivity.py", "--dataset", ds, "--tag", tag] + extra for ds in ["paderborn", "induction"]
                 for tag, extra in SENS], a.parallel)
        run([["scripts/analysis/cluster_bootstrap.py"], ["scripts/analysis/runtime_and_data_stats.py"],
             ["scripts/analysis/liang_sanity_synthetic.py"], ["scripts/analysis/analysis_shift_ewma.py"], ["scripts/analysis/analysis_strict_split.py"], ["scripts/analysis/analysis_naive_band.py"]], False)
    if a.only != "experiments":
        figs = ["fig_overview", "fig_results", "fig_ablation", "fig_identifiability", "fig_derating", "fig_strict", "fig_graphical_abstract", "make_tables"]
        if a.sensitivity or any((ROOT / "outputs/runs").glob("*_cv5_sens_*")):
            figs.append("fig_sensitivity")
        run([[f"scripts/figs/{f}.py"] for f in figs], False)
    print("Finished. Compare outputs/tables/*.tex and outputs/analysis/*.csv with the paper.")


if __name__ == "__main__":
    main()
