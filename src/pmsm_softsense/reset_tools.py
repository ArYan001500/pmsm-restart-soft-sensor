"""Shared tools for post-reset PM-temperature experiments on real data (2026-10-06).

Thermal model: rate-form 4-node LPTN (see lptn_real.py), state order [pm, yoke, tooth, winding].
Kalman filter with an arbitrary (possibly correlated) Gaussian prior and an arbitrary subset of measured nodes.
"""
from __future__ import annotations
import numpy as np
from .lptn_real import RateLPTN, ALPHA_CU, TS

IDX = {"pm": 0, "stator_yoke": 1, "stator_tooth": 2, "stator_winding": 3}


class LPTNSim:
    def __init__(self, lp: RateLPTN):
        self.lp = lp

    def mats(self, wn):
        lp = self.lp
        K = lp.Ka + lp.Kc * wn; B = lp.Ba + lp.Bc * wn
        Am = K.copy(); np.fill_diagonal(Am, 0.0); Am -= np.diag(K.sum(1) - np.diag(K) + B.sum(1))
        return Am, B

    def phi(self, F, k, Tw):
        R = 1 + ALPHA_CU * (Tw - 20.0); wn = F["wn"][k]; is2 = F["is2"][k]
        base = [is2 * R, wn, wn ** 2, is2 * wn, is2 * wn ** 2, F["us2"][k]]
        if "extra" in F:
            base += list(F["extra"][k])
        return np.array(base)

    def step(self, T, k, F):
        Am, B = self.mats(F["wn"][k])
        Tn = T + TS * (Am @ T + B @ F["bnd"][k] + self.lp.Q @ self.phi(F, k, T[3]))
        J = np.eye(4) + TS * Am; J[:, 3] += TS * self.lp.Q[:, 0] * F["is2"][k] * ALPHA_CU
        return Tn, J

    def rollout(self, T0, k0, k1, F):
        T = np.array(T0, float); out = np.empty((k1 - k0, 4)); out[0] = T
        for j, k in enumerate(range(k0, k1 - 1)):
            T, _ = self.step(T, k, F); out[j + 1] = T
        return out


def kalman(sim: LPTNSim, F, Y, measured, x0, P0, Q, r, n, sigma_meas=0.3):
    """Linear(ised) KF from index r to n-1. Y: (N,4) array of measured node temperatures (NaN where unused).
    measured: list of node indices observed. Returns pm mean, pm std, full state trace."""
    H = np.zeros((len(measured), 4))
    for i, j in enumerate(measured): H[i, j] = 1.0
    Rm = np.eye(len(measured)) * sigma_meas ** 2
    x = np.array(x0, float); P = np.array(P0, float)
    pm, sd, xs = [x[0]], [np.sqrt(max(P[0, 0], 0))], [x.copy()]
    for k in range(r, n - 1):
        x, J = sim.step(x, k, F); P = J @ P @ J.T + Q
        y = Y[k + 1, measured]
        S = H @ P @ H.T + Rm; K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (y - H @ x); P = (np.eye(4) - K @ H) @ P
        pm.append(x[0]); sd.append(np.sqrt(max(P[0, 0], 0))); xs.append(x.copy())
    return np.array(pm), np.array(sd), np.array(xs)


def score_trace(trace, pm_true, r, sd=None):
    e = trace - pm_true[r:r + len(trace)]
    t = np.arange(len(e)) * TS / 60
    rec = {}
    for w in [0, 1, 2, 5, 10, 20, 29]:
        k = int(w * 60 / TS)
        rec[f"abs_err_{w}min"] = float(abs(e[k])) if k < len(e) else np.nan
    bad = np.where(np.abs(e) > 5)[0]
    rec["never_within_5K"] = bool(len(bad) and bad[-1] == len(e) - 1)
    rec["mae_first10"] = float(np.mean(np.abs(e[t < 10])))
    if sd is not None:
        rec["cov2s_first10"] = float(np.mean(np.abs(e[t < 10]) <= 2 * np.maximum(sd[t < 10], 1e-6)))
        rec["sd_at_2min"] = float(sd[min(len(sd) - 1, int(120 / TS))])
    return rec


def simulate_histories(sim: LPTNSim, cache, pids, n_orders, gap_max_min, record_every, rng):
    """Label-free operating histories: LPTN driven by real input sequences of `pids` in random order with random
    standstill gaps. cache[pid] = (obs, F, Y). Returns rows [T(4), coolant, ambient, is2, wn, us2]."""
    rows = []
    for _ in range(n_orders):
        perm = rng.permutation(pids)
        T = np.full(4, cache[perm[0]][1]["bnd"][0, 0])
        for j, pid in enumerate(perm):
            o, F, _ = cache[pid]
            gap = int(rng.uniform(0, gap_max_min) * 60 / TS)
            Fg = {"wn": np.zeros(gap + 1), "is2": np.zeros(gap + 1), "us2": np.zeros(gap + 1),
                  "bnd": np.repeat(F["bnd"][:1], gap + 1, 0)}
            if "extra" in F:
                Fg["extra"] = np.zeros((gap + 1, F["extra"].shape[1]))
            for k in range(gap):
                T, _ = sim.step(T, k, Fg)
            for k in range(len(o) - 1):
                if j > 0 and k % record_every == 0:
                    rows.append(np.concatenate([T, [F["bnd"][k, 0], F["bnd"][k, 1], F["is2"][k], F["wn"][k], F["us2"][k]]]))
                T, _ = sim.step(T, k, F)
    return np.array(rows)


def kalman_consider(sim: LPTNSim, F, Y, measured, x0, P0, Q, r, n, sigma_meas=0.3, consider=(0,)):
    """Schmidt (consider-state) KF: states in `consider` are propagated with covariance but never corrected by
    measurement innovations (their gain rows are zeroed); cross-covariances are kept consistent."""
    H = np.zeros((len(measured), 4))
    for i, j in enumerate(measured): H[i, j] = 1.0
    Rm = np.eye(len(measured)) * sigma_meas ** 2
    x = np.array(x0, float); P = np.array(P0, float)
    pm, sd = [x[0]], [np.sqrt(max(P[0, 0], 0))]
    for k in range(r, n - 1):
        x, J = sim.step(x, k, F); P = J @ P @ J.T + Q
        S = H @ P @ H.T + Rm; K = P @ H.T @ np.linalg.inv(S)
        K[list(consider), :] = 0.0
        x = x + K @ (Y[k + 1, measured] - H @ x)
        IKH = np.eye(4) - K @ H
        P = IKH @ P @ IKH.T + K @ Rm @ K.T          # Joseph form (valid for suboptimal gain)
        pm.append(x[0]); sd.append(np.sqrt(max(P[0, 0], 0)))
    return np.array(pm), np.array(sd)
