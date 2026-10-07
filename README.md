# Rotor temperature soft sensing after estimator resets

Code for the paper

> A. Ashoori, *Rotor temperature soft sensing after estimator resets using thermal-model simulations and calibrated uncertainty*, manuscript under review.

When a drive controller resets, a rotor-temperature soft sensor loses its thermal state. The method in this repository,
CBP (calibrated background prior), re-initialises the hidden rotor and stator node temperatures of a four-node
lumped-parameter thermal network (LPTN) at the moment of the reset, using only the stator temperatures, boundary
temperatures and operating point measured at that moment. A gradient-boosted prior is trained on simulations of the
identified network, its band is calibrated with novelty-scaled split conformal prediction on events of the same type,
and the result initialises a Kalman filter.

The repository contains code only. The two public datasets are downloaded by a script.

## Requirements

* Python 3.11 or newer (the paper used Python 3.14.5 on Windows 11)
* the packages in `requirements.txt` (the exact versions used for the paper)
* about 4 GB of free disk space for the data and 8 GB of RAM

```bash
python -m pip install -r requirements.txt
python -m pytest -q          # 8 fast checks, no data needed
```

## Data

```bash
python scripts/download_data.py
```

downloads, unpacks and verifies (SHA-256) both datasets:

| Dataset | Source | Files | Licence |
|---|---|---|---|
| PMSM, 69 profiles, 185 h, 2 Hz | Kirchgässner, Wallscheid, Böcker, *Electric Motor Temperature*, Kaggle, version 3, [doi:10.34740/KAGGLE/DSV/2161054](https://doi.org/10.34740/KAGGLE/DSV/2161054) | `data/raw/electric_motor_temperature/measures_v2.csv` | CC BY-SA 4.0 |
| Induction motor, 278 sessions, 262 h | Stender, Wallscheid, Böcker, *Induction Motor Data Set*, Kaggle, version 4, [kaggle.com/datasets/stender/induction-motor-data-set](https://www.kaggle.com/datasets/stender/induction-motor-data-set); description: [doi:10.13140/RG.2.2.13010.98248](https://doi.org/10.13140/RG.2.2.13010.98248) | `data/Second Dataset/First_Part.csv`, `Second_Part.csv`, `Third_Part.csv` | as stated by the dataset authors |

The script uses Kaggle's public download endpoint, which did not require a login when the paper was written. If it
does, download the two archives in a browser and run
`python scripts/download_data.py --pmsm-zip <archive.zip> --im-zip <archive.zip>`.
`python scripts/download_data.py --only pmsm` fetches only the PMSM data (0.12 GB).

## Reproducing the paper

```bash
python reproduce_all.py --parallel                 # all experiments, figures and tables except Fig. 7
python reproduce_all.py --parallel --sensitivity   # also the 12 sensitivity runs of Fig. 7
```

Wall-clock times on the desktop CPU used for the paper (Intel, 20 logical cores): main thermal network about 25 min;
five-fold cross-validation with all ablations 104 min (PMSM) and 82 min (induction motor); each locked test about
60 to 100 min; identifiability, derating and the Fig. 1 example a few minutes each; sensitivity study about 2.5 h when
the 12 runs are executed in parallel. Without `--parallel` the steps run one after another.

Every experiment writes a new time-stamped folder in `outputs/runs/` containing its configuration
(`resolved_config.json`), a copy of the source files it used, the per-event results (`episodes.csv`) and the
aggregates (`summary.json`). Figures and tables are written to `outputs/figures/` and `outputs/tables/` and always use
the newest matching run.

| Paper item | Script | Output |
|---|---|---|
| Thermal network, Fig. 1 example | `scripts/fit_lptn_real.py`, `scripts/make_fig_example_reset.py` | `outputs/runs/*_lptn_real_baseline_*` |
| Table 5, Figs. 2 and 3 (cross-validation, ablations) | `scripts/run_cv5_ablation_v2.py --dataset {paderborn,induction}` | `outputs/runs/*_cv5_ablation_v2_*` |
| Table 7 (locked test) | `scripts/run_locked_test.py --dataset {paderborn,induction}` | `outputs/runs/*_LOCKED_TEST_*` |
| Table 8 (withheld speed ranges) | `scripts/run_shift_test.py --dataset {paderborn,induction} --group {high,low}` | `outputs/runs/*_shift_*` |
| Table 6 (data-driven EWMA estimators) | `scripts/run_ewma_baseline.py --dataset {paderborn,induction} --mode {cv,test}` | `outputs/runs/*_ewma_baseline_*` |
| Tables 6 and 8, horizon counts, per-profile coverage | `scripts/analysis/analysis_shift_ewma.py` | `outputs/analysis_shift_ewma/` |
| Table 9, Fig. 4a,b (strict split of fitting and calibration units) | `scripts/run_cv5_strictsplit.py --dataset ...`, `scripts/run_shift_strict.py --dataset ... --group ...` | `outputs/runs/*_cv5_strictsplit_*`, `outputs/runs/*_shift_strict_*` |
| Table 6 warm-start rows, Fig. 4c | `scripts/run_ewma_baseline.py ... --pad current` | `outputs/runs/*_ewma_baseline_*_warm_*` |
| Table 9 and the warm-start comparison | `scripts/analysis/analysis_strict_split.py` | `outputs/analysis_strict_split/` |
| Table 9 second row (one calibration event per profile) | `scripts/run_cv5_groupconf.py --dataset paderborn` | `outputs/runs/*_cv5_groupconf_*` |
| Calibrated naive band (Sections 5.2, 5.7) | `scripts/analysis/analysis_naive_band.py` | `outputs/analysis_naive_band/` |
| Fig. 5 (twin snapshots) | `scripts/run_matched_pairs_identifiability.py` | `outputs/runs/*_matched_pairs_*` |
| Fig. 6 (derating, incl. the naive filter with a calibrated band) | `scripts/run_derating_sim.py` | `outputs/runs/*_derating_sim_*` |
| Fig. 7 (sensitivity) | `scripts/run_cv5_sensitivity.py --dataset ... --tag ...` | `outputs/runs/*_cv5_sens_*` |
| Sensitivity to the filter settings, PMSM (Section 5.10) | `scripts/run_cv5_sensitivity.py --dataset paderborn --tag ... --sigma-meas/--q-stator/--q-target ...` | `outputs/runs/*_cv5_sens_*` |
| Profile-balanced errors, common 30-min cohort, Euler stability (Sections 4.4, 5.2) | `scripts/analysis/analysis_robustness.py` | `outputs/analysis_robustness/` |
| Bootstrap intervals quoted in the text | `scripts/analysis/cluster_bootstrap.py` | `outputs/analysis/cluster_bootstrap_ci.csv` |
| Data statistics and run times | `scripts/analysis/runtime_and_data_stats.py` | `outputs/analysis/runtime_and_data_stats.json` |
| Synthetic check of the Liang-type inversion (Section 4.3) | `scripts/analysis/liang_sanity_synthetic.py` (uses `scripts/run_liang_comparison.py`) | `outputs/analysis/liang_sanity_synthetic.json` |
| Figures and tables | `scripts/figs/*.py` | `outputs/figures/`, `outputs/tables/` |

All random choices use the fixed seed 20261006. The induction-motor experiments first need a per-session summary,
which `reproduce_all.py` creates with `scripts/prepare_induction_metadata.py`.

## Protocol and test lock

* PMSM: the test profiles 60, 62 and 74 are the generalisation set of Kirchgässner et al. (2023); the 66 other
  profiles are split into the five folds listed in `data/manifests/profile_splits_literature_v2.json`.
* Induction motor: sessions whose index is divisible by five are the test set; the folds group the remaining sessions
  by `(index // 5) % 5`.
* `src/pmsm_softsense/realdata.py` refuses to load the PMSM test profiles unless the caller asks for the final
  evaluation and `data/manifests/FREEZE_FINAL_EVALUATION.json` exists. That file was written before the test data were
  first loaded; it records the SHA-256 hashes of the frozen recipe (`docs/FREEZE_RECIPE.md`), the experiment scripts and
  the split manifest, which are byte-identical to the files in this repository. The manifest lists the recipe under
  its original working path `reports/paper_prep/FREEZE_RECIPE_v1.md`; in this repository the same file (same hash) is
  `docs/FREEZE_RECIPE.md`.
  Because these files are hash-locked, their comments were not edited afterwards; for example, the docstring of
  `scripts/run_locked_test.py` still mentions the start prior of an earlier recipe version, which the final recipe
  keeps only as ablation A8.
  To keep these hashes valid, the comments of the frozen files are unchanged; in them, the internal label
  `UI-009` denotes the novelty-scaled conformal calibration described in the paper.
* `scripts/run_shift_test.py`, `scripts/run_ewma_baseline.py`, `scripts/run_cv5_strictsplit.py`, `scripts/run_shift_strict.py`, `scripts/run_cv5_groupconf.py`
  and `scripts/analysis/analysis_{shift_ewma,strict_split,naive_band}.py` were added after the locked test. The shift test reuses the code path of `run_cv5_ablation_v2.py` on development data only; the
  EWMA baseline is a separate estimator evaluated on the same events. Neither changes the frozen recipe.

## Repository layout

```
src/pmsm_softsense/   thermal network, identification, Kalman filter, simulated histories, data loader
scripts/              data download, experiments, analysis, figures and tables
docs/FREEZE_RECIPE.md the recipe as frozen before the locked test
data/manifests/       split definition and the freeze record
tests/                fast unit checks
reproduce_all.py      runs everything in the order used for the paper
```

## Licence

Code: MIT licence (see `LICENSE`). The datasets are not part of this repository and keep their own licences.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`) and the two datasets.
