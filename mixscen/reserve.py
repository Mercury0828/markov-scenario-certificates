"""The reserve problem of Section 5.2.

Regimes r in {normal, elevated, surge} with stationary probabilities (0.70, 0.25, 0.05); the regime stays put with
probability 1 - theta and is otherwise redrawn from these probabilities. Demand of region j = 1..5:
D_{t,j} = mu[R_t] + S_t + E_{t,j} with mu = (20, 24, 30), S_t ~ Bin(20, 1/2) - 10, E_{t,j} ~ Bin(2, 1/2) - 1, all independent
given the regime path. Program: min sum_j x_j subject to x >= D_t for t = 1..N and x in [0, 60]^5, whose solution is the
coordinatewise maximum. The greedy rule keeps an index exactly when it is the last index attaining the maximum of some
coordinate whose maximum no earlier kept index attains. The risk V(x) is computed exactly.

simulate_chunk(rng, theta, N, R, gaps)   R independent trajectories of length N; also the thinned samples for each gap
"""
import math

import numpy as np
from scipy import stats

from . import certificates as C

D_DIM = 5
PI_R = np.array([0.70, 0.25, 0.05])
MU = np.array([20, 24, 30], dtype=np.int16)
S_VALS = np.arange(-10, 11)
Q_S = stats.binom.pmf(np.arange(21), 20, 0.5)
BLOCK = 200
# (theta, N) of the four rows of Table 2; the mixing times are 5 and 20
SETTINGS = [("t5_N1e4", 0.25, 10 ** 4), ("t5_N1e5", 0.25, 10 ** 5), ("t20_N1e4", 0.066, 10 ** 4),
            ("t20_N1e5", 0.066, 10 ** 5)]
BETAS = (1e-3, 0.05)
SEED0 = 20260927
CHUNKS, CHUNK_SIZE = 20, 5000


def seed(setting_index, chunk):
    return SEED0 + 1000 * setting_index + chunk


def regime_chain(theta):
    return (1 - theta) * np.eye(3) + theta * np.outer(np.ones(3), PI_R)


def tmix_formula(theta):
    """Mixing time of the regime chain (and of the latent chain): ceil(log(3.8) / log(1/(1-theta)))."""
    return math.ceil(math.log(4 * (1 - PI_R.min())) / (-math.log(1 - theta)) - 1e-12)


def risk(x):
    """V(x) = 1 - sum_r pi_R(r) sum_s P(S = s) prod_j P(E <= x_j - mu_r - s), for an integer array x of shape (R, 5)."""
    x = x.astype(np.int32)
    acc = np.zeros(x.shape[0])
    for r in range(3):
        for si, s in enumerate(S_VALS):
            u = x - int(MU[r]) - int(s)
            fe = np.where(u >= 1, 1.0, np.where(u == 0, 0.75, np.where(u == -1, 0.25, 0.0)))
            acc += PI_R[r] * Q_S[si] * fe.prod(axis=1)
    return 1.0 - acc


def greedy_k(M, L, DL):
    """Size of the greedy reconstructing set from the coordinatewise maxima M (R, d), the index L (R, d) of the last
    attainer of each maximum and the demand vector DL (R, d, d) at that index."""
    R, d = M.shape
    order = np.argsort(L, axis=1, kind="stable")
    covered = np.zeros((R, d), dtype=bool)
    k = np.zeros(R, dtype=np.int16)
    rows = np.arange(R)
    for pos in range(d):
        j = order[:, pos]
        new = ~covered[rows, j]
        k += new
        covered |= new[:, None] & (DL[rows, j, :] == M)
    return k


class RunningMax:
    """Running coordinatewise maximum with the index of its last attainer and the attainer's demand vector."""

    def __init__(self, R, d):
        self.M = np.full((R, d), -32768, dtype=np.int16)
        self.L = np.full((R, d), -1, dtype=np.int64)
        self.DL = np.zeros((R, d, d), dtype=np.int16)

    def update_block(self, D, tglob):
        """D: demands (T, R, d) at the increasing times tglob (T,)."""
        if D.shape[0] == 0:
            return
        T, R, d = D.shape
        bmax = D.max(axis=0)
        rev = (D[::-1] == bmax[None]).argmax(axis=0)
        tloc = T - 1 - rev
        upd = bmax >= self.M
        rows = np.arange(R)[:, None]
        vec = D[tloc, rows, :]
        self.M = np.where(upd, bmax, self.M)
        self.L = np.where(upd, tglob[tloc], self.L)
        self.DL = np.where(upd[:, :, None], vec, self.DL)


def simulate_chunk(rng, theta, N, R, gaps):
    """R stationary trajectories of length N, generated in blocks of 200 steps. Returns per trajectory the complexity
    k, the risk V and the cost of the full-sample decision, and the same for the sample thinned with each gap in gaps
    (keys kthin_<beta>, Vthin_<beta>, costthin_<beta>), where the thinned sample keeps the times 1, 1+g, 1+2g, ..."""
    cdf_s = np.cumsum(Q_S)
    cdf_s[-1] = 1.0
    reg = np.searchsorted(np.cumsum(PI_R), rng.random(R), side="right").astype(np.int8)
    full = RunningMax(R, D_DIM)
    thin = {b: RunningMax(R, D_DIM) for b in gaps}
    t0 = 0
    while t0 < N:
        T = min(BLOCK, N - t0)
        refresh = rng.random((T, R)) < theta
        if t0 == 0:
            refresh[0] = False
        draws = np.searchsorted(np.cumsum(PI_R), rng.random((T, R)), side="right").astype(np.int8)
        idx = np.where(refresh, np.arange(T)[:, None], -1)
        last = np.maximum.accumulate(idx, axis=0)
        regs = np.where(last >= 0, draws[np.clip(last, 0, None), np.arange(R)[None, :]], reg[None, :])
        reg = regs[-1].copy()
        S = (np.searchsorted(cdf_s, rng.random((T, R)), side="right") - 10).astype(np.int16)
        u = rng.random((T, R, D_DIM))
        E = np.where(u < 0.25, -1, np.where(u < 0.75, 0, 1)).astype(np.int16)
        D = MU[regs][:, :, None] + S[:, :, None] + E
        tglob = np.arange(t0, t0 + T)
        full.update_block(D, tglob)
        for b, g in gaps.items():
            sel = (tglob % g) == 0
            thin[b].update_block(D[sel], tglob[sel])
        t0 += T
    out = {"k": greedy_k(full.M, full.L, full.DL), "V": risk(full.M), "cost": full.M.astype(np.int32).sum(1)}
    for b in gaps:
        tm = thin[b]
        out[f"kthin_{b:g}"] = greedy_k(tm.M, tm.L, tm.DL)
        out[f"Vthin_{b:g}"] = risk(tm.M)
        out[f"costthin_{b:g}"] = tm.M.astype(np.int32).sum(1)
    return out


def gaps_for(N, tbar):
    """Thinning gap for each confidence level, fixed before the data (Section 5.2, complexity d = 5)."""
    return {b: C.thinning_gap(N, b, tbar, D_DIM) for b in BETAS}
