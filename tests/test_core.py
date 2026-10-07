"""Fast checks of the core numerical components on a small synthetic network (no dataset needed)."""
import numpy as np
from pmsm_softsense.lptn_real import RateLPTN, TS, ALPHA_CU
from pmsm_softsense.reset_tools import LPTNSim, kalman, simulate_histories


def toy_network(seed=0):
    rng = np.random.default_rng(seed)
    Ka = rng.uniform(1e-4, 2e-3, (4, 4)); np.fill_diagonal(Ka, 0)
    Kc = rng.uniform(0, 1e-3, (4, 4)); np.fill_diagonal(Kc, 0)
    return RateLPTN(Ka, Kc, rng.uniform(1e-4, 3e-3, (4, 2)), rng.uniform(0, 1e-3, (4, 2)), rng.uniform(0, 5e-3, (4, 6)))


def inputs(n, seed=1):
    rng = np.random.default_rng(seed)
    return {"is2": rng.uniform(0, 1, n), "us2": rng.uniform(0, 1, n), "wn": rng.uniform(0, 1, n),
            "bnd": np.column_stack([np.full(n, 40.0), np.full(n, 25.0)])}


def test_step_jacobian_matches_finite_differences():
    sim, F = LPTNSim(toy_network()), inputs(10)
    T = np.array([60.0, 50.0, 55.0, 70.0])
    _, J = sim.step(T, 3, F)
    eps = 1e-5
    Jn = np.column_stack([(sim.step(T + eps * e, 3, F)[0] - sim.step(T - eps * e, 3, F)[0]) / (2 * eps) for e in np.eye(4)])
    assert np.allclose(J, Jn, atol=1e-8)


def test_step_equals_rate_model_rollout():
    lp = toy_network(); sim, F = LPTNSim(lp), inputs(50)
    T0 = np.array([30.0, 35.0, 33.0, 40.0])
    assert np.allclose(sim.rollout(T0, 0, 50, F), lp.simulate(F, T0), atol=1e-10)


def test_kalman_recovers_hidden_rotor_with_perfect_model():
    sim, F = LPTNSim(toy_network()), inputs(4000)
    truth = sim.rollout(np.array([80.0, 50.0, 55.0, 60.0]), 0, 4000, F)
    x0 = truth[0].copy(); x0[0] = truth[0, 3]                       # naive start: rotor = winding
    P0 = np.diag([30.0 ** 2, 0.09, 0.09, 0.09]); Q = np.eye(4) * 1e-6
    pm, sd, _ = kalman(sim, F, truth, [1, 2, 3], x0, P0, Q, 0, 4000, sigma_meas=0.3)
    assert abs(pm[0] - truth[0, 0]) > 15 and abs(pm[-1] - truth[-1, 0]) < 1.0 and sd[-1] < sd[0]


def test_simulated_histories_shape_and_label_free():
    sim = LPTNSim(toy_network())
    cache = {p: (np.zeros(600), inputs(600, seed=p), None) for p in range(4)}   # no labels: Y is None
    H = simulate_histories(sim, cache, list(cache), n_orders=2, gap_max_min=10, record_every=60, rng=np.random.default_rng(0))
    assert H.shape[1] == 9 and len(H) == 2 * 3 * 10 and np.isfinite(H).all()
