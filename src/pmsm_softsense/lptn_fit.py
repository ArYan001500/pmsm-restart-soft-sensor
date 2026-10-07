"""Reusable LPTN identification (2026-10-06): equation-error NNLS + optional output-error refinement (torch).
cache[pid] = (obs_or_placeholder, F, Y); F may contain 'extra' loss features."""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear
from .lptn_real import RateLPTN, phi_matrix, ALPHA_CU, TS


def ee_fit(cache, ids, sg_window=31, subsample=4):
    rows = {i: [] for i in range(4)}; ys = {i: [] for i in range(4)}; nphi = None
    for pid in ids:
        _, F, Y = cache[pid]
        if len(Y) < 3 * sg_window: continue
        dY = savgol_filter(Y, sg_window, 2, deriv=1, delta=TS, axis=0); ph = phi_matrix(F, Y[:, 3]); nphi = ph.shape[1]
        sl = slice(sg_window, len(Y) - sg_window, subsample)
        for i in range(4):
            cols = []
            for j in range(4):
                if j == i: continue
                d = Y[:, j] - Y[:, i]; cols += [d, F["wn"] * d]
            for b in range(2):
                d = F["bnd"][:, b] - Y[:, i]; cols += [d, F["wn"] * d]
            X = np.column_stack(cols + [ph]); rows[i].append(X[sl]); ys[i].append(dY[sl, i])
    Ka = np.zeros((4, 4)); Kc = np.zeros((4, 4)); Ba = np.zeros((4, 2)); Bc = np.zeros((4, 2)); Q = np.zeros((4, nphi))
    for i in range(4):
        X = np.vstack(rows[i]); y = np.concatenate(ys[i]); sc = np.maximum(np.abs(X).max(0), 1e-9)
        c = lsq_linear(X / sc, y, bounds=(0, np.inf), lsmr_tol="auto", max_iter=5000).x / sc; k = 0
        for j in range(4):
            if j == i: continue
            Ka[i, j], Kc[i, j] = c[k], c[k + 1]; k += 2
        for b in range(2):
            Ba[i, b], Bc[i, b] = c[k], c[k + 1]; k += 2
        Q[i] = c[k:k + nphi]
    return RateLPTN(Ka, Kc, Ba, Bc, Q)


def oe_refine(model: RateLPTN, cache, ids, epochs=15, chunk=1024, lr=3e-3, threads=4, seed=0):
    import torch
    torch.manual_seed(seed); torch.set_num_threads(threads); DT = torch.float64
    fs = [cache[p] for p in ids]; L = max(len(Y) for _, _, Y in fs); B = len(fs); E = model.Q.shape[1] - 6
    def arr(shape): return np.zeros(shape)
    wn, is2, us2 = arr((B, L)), arr((B, L)), arr((B, L)); ex = arr((B, L, max(E, 1))); bnd = arr((B, L, 2)); Y = arr((B, L, 4)); M = arr((B, L))
    for b, (_, F, Yb) in enumerate(fs):
        n = len(Yb); wn[b, :n] = F["wn"]; is2[b, :n] = F["is2"]; us2[b, :n] = F["us2"]; bnd[b, :n] = F["bnd"]; bnd[b, n:] = F["bnd"][-1]
        if E: ex[b, :n] = F["extra"]
        Y[b, :n] = Yb; M[b, :n] = 1
    t = lambda a: torch.tensor(a, dtype=DT)
    wn, is2, us2, ex, bnd, Y, M = map(t, (wn, is2, us2, ex, bnd, Y, M))
    isp = lambda a: torch.nn.Parameter(t(np.log(np.expm1(np.maximum(a, 1e-7)))))
    P = [isp(model.Ka), isp(model.Kc), isp(model.Ba), isp(model.Bc), isp(model.Q)]
    off = 1 - torch.eye(4, dtype=DT); opt = torch.optim.Adam(P, lr=lr); sp = torch.nn.functional.softplus
    def pos(): return sp(P[0]) * off, sp(P[1]) * off, sp(P[2]), sp(P[3]), sp(P[4])
    for _ in range(epochs):
        T = Y[:, 0].clone()
        for k0 in range(0, L - 1, chunk):
            k1 = min(k0 + chunk, L); Ka, Kc, Ba, Bc, Q = pos(); Tk = T; out = []
            for k in range(k0, k1):
                out.append(Tk); w = wn[:, k]; R = 1 + ALPHA_CU * (Tk[:, 3] - 20.0)
                cols = [is2[:, k] * R, w, w ** 2, is2[:, k] * w, is2[:, k] * w ** 2, us2[:, k]] + ([ex[:, k, j] for j in range(E)] if E else [])
                phi = torch.stack(cols, -1); K = Ka + Kc * w[:, None, None]; Bm = Ba + Bc * w[:, None, None]
                dT = (K * (Tk[:, None, :] - Tk[:, :, None])).sum(-1) + (Bm * (bnd[:, k][:, None, :] - Tk[:, :, None])).sum(-1) + (phi[:, None, :] * Q).sum(-1)
                Tk = Tk + TS * dT
            pred = torch.stack(out, 1); m = M[:, k0:k1]
            loss = (((pred - Y[:, k0:k1]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(P, 1.0); opt.step(); T = Tk.detach()
    with torch.no_grad():
        return RateLPTN(*(p.numpy() for p in pos()))
