# Frozen recipe v1 — post-reset hidden-state re-initialisation (CBP)

Status: LOCKED 2026-10-06 (v2) after CV runs ec2e6fab (Paderborn) and 5fd6ecea (induction), BEFORE any test profile/session
is loaded. After lock, no setting below may change because of test results. Date of draft: 2026-10-06.

## Information contract (online, after estimator memory loss)
Allowed: measured stator node temperature(s) (S1: winding only; S3: winding + tooth + yoke), boundary temperatures
(Paderborn: coolant, ambient; induction: housing, ambient air), operating context (current magnitude, voltage magnitude,
speed; induction additionally measured loss P_in−P_mech and torque^2 as loss features). Not allowed: rotor/PM temperature,
pre-reset history, flux/back-EMF temperature channel, signal injection.

## Thermal model
Rate-form 4-node LPTN [rotor/PM, yoke, tooth, winding], conductances a + c·|w_n|, nonnegative coefficients, forward Euler 0.5 s.
Identification on training units only: equation-error NNLS (Savitzky–Golay window 31, order 2, every 4th sample),
then output-error refinement (Adam lr 3e-3, TBPTT 1024, 15 epochs — identical in CV and in the locked test).

## Prior (label-free)
Simulated operating histories: LPTN driven by training input sequences in random order, 3 permutations, standstill gaps
U(0,120) min with frozen boundaries, snapshots every 60 samples.
Reset prior: HistGradientBoostingRegressor (max_iter 300, lr 0.05, 31 leaves, min leaf 40) per hidden node on
[measured nodes, boundaries, is2, wn, us2]; covariance = residual covariance on simulated rows.
The same prior is used for every event type (v2). The v1 start prior (simulated zero-input cooling) is removed: in v1
CV it widened bands (Paderborn S1 t0 half-width 61 K) and worsened 2-10 min error on both machines (kept as ablation A8).

## Calibration (UI-009, event-matched)
Novelty = kNN (k=20) mean distance in standardised feature space to the simulated rows, divided by its median, minus 1, clipped ≥0.
Score = |error| / (1 + novelty); split-conformal quantile at 90 % on real training labels unused by the prior:
reset events → training half B, points every 5 min; start events → all training session starts (same prior, v2).
Initial hidden-state covariance scaled so that 1.645 σ_PM equals the calibrated half-width.

## Filter
Kalman filter on the LPTN (exact Jacobian), measured-node noise σ = 0.3 K.
Process noise: diagonal one-step residual variance on training data; stator nodes ×8; PM node ×q with
q = smallest of {0.5, 2, 8, 32, 128} whose training-half-A coverage of ±1.645σ at 20 and 30 min is ≥ 0.88.

## Evaluation (single, after lock)
Paderborn: TNN-protocol test G = {60, 62, 74}; events = resets every 15 min + natural starts; model trained on all 66 dev profiles.
Induction: sessions l % 5 == 0 (55); events = session starts; model trained on all 223 non-test sessions.
Code: scripts/run_locked_test.py (= scripts/run_cv5_ablation_v2.py with one fold, train = all dev, held-out = test).
Deep-pmsm secondary test {65,72} is NOT used: those profiles are in the primary dev set.
Metrics: PM/rotor absolute error at 0, 2, 4, 5, 10, 20, 30 min (median, p90), coverage and half-width of the 90 % band,
baselines naive KF, deterministic initialisers, Liang-2025-style inversion (4 min), oracle.
