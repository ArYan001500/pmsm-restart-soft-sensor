"""Four-node lumped-parameter thermal network identified on real data (rate form).

States  x = [T_pm, T_yoke, T_tooth, T_winding]  (deg C)
Boundary b = [T_coolant, T_ambient]
dT_i/dt = sum_j k_ij(w)(T_j - T_i) + sum_b k_ib(w)(T_b - T_i) + sum_m q_im phi_m
k(w) = a + c*|w_n|,  all coefficients >= 0.  Forward Euler, Ts = 0.5 s.
phi = [is2*R(Tw), w_n, w_n^2, is2*w_n, is2*w_n^2, us2]
  is2 = (i_d^2+i_q^2)/1e4,  us2 = (u_d^2+u_q^2)/130^2,  w_n = rpm/6000,  R(Tw)=1+0.00393(Tw-20).
PM labels are used only for identification on training profiles (prototype calibration).
"""
from __future__ import annotations
import numpy as np

STATE_NAMES = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
N_STATE, N_BND, N_PHI = 4, 2, 6
ALPHA_CU = 0.00393
TS = 0.5


def features(obs):
    """obs: DataFrame with realdata.OBS_COLS. Returns dict of numpy arrays."""
    is2 = (obs["i_d"].to_numpy() ** 2 + obs["i_q"].to_numpy() ** 2) / 1e4
    us2 = (obs["u_d"].to_numpy() ** 2 + obs["u_q"].to_numpy() ** 2) / 130.0 ** 2
    wn = np.abs(obs["motor_speed"].to_numpy()) / 6000.0
    bnd = np.column_stack([obs["coolant"].to_numpy(), obs["ambient"].to_numpy()])
    return {"is2": is2, "us2": us2, "wn": wn, "bnd": bnd}


def phi_matrix(f, Tw):
    """Base 6 loss features; optional machine-specific extra loss features in f["extra"] (N x E) are appended."""
    R = 1 + ALPHA_CU * (Tw - 20.0)
    cols = [f["is2"] * R, f["wn"], f["wn"] ** 2, f["is2"] * f["wn"], f["is2"] * f["wn"] ** 2, f["us2"]]
    if "extra" in f:
        cols += [f["extra"][:, j] for j in range(f["extra"].shape[1])]
    return np.column_stack(cols)


class RateLPTN:
    """Parameters: Ka, Kc (4x4, zero diag), Ba, Bc (4x2), Q (4x6); all >= 0."""

    def __init__(self, Ka, Kc, Ba, Bc, Q):
        self.Ka, self.Kc, self.Ba, self.Bc, self.Q = map(np.asarray, (Ka, Kc, Ba, Bc, Q))

    def derivative(self, T, wn, bnd, phi):
        """T: (...,4); wn: (...,); bnd: (...,2); phi: (...,6)."""
        K = self.Ka + self.Kc * wn[..., None, None]
        B = self.Ba + self.Bc * wn[..., None, None]
        dT = (K * (T[..., None, :] - T[..., :, None])).sum(-1)
        dT += (B * (bnd[..., None, :] - T[..., :, None])).sum(-1)
        dT += (self.Q * phi[..., None, :]).sum(-1)
        return dT

    def simulate(self, f, T0, Tw_override=None):
        """Open-loop rollout. Copper resistance uses the model's own winding state."""
        n = len(f["wn"]); T = np.empty((n, N_STATE)); T[0] = T0
        for k in range(n - 1):
            Tw = T[k, 3]
            ph = phi_matrix({key: (v[k:k + 1] if v.ndim == 1 else v[k:k + 1]) for key, v in f.items()}, np.array([Tw]))[0]
            T[k + 1] = T[k] + TS * self.derivative(T[k], np.array(f["wn"][k]), f["bnd"][k], ph)
        return T

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ["Ka", "Kc", "Ba", "Bc", "Q"]}

    @classmethod
    def from_dict(cls, d):
        return cls(*(np.array(d[k]) for k in ["Ka", "Kc", "Ba", "Bc", "Q"]))
