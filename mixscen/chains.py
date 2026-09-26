"""The chains of Section 5.3 and exact exceedance probabilities for the largest observed value.

The decision is the threshold program min{x : x >= delta_t for all t} with A_0 = 0 (Example 2.6), so theta*_N is the
largest observed value and its risk is V(x) = pi{delta > x}. For a finite chain the scenario of state s is its label
phi(s) = s. The greedy rule reports k_N = 0 when every observed value is the bottom label 0 and k_N = 1 otherwise; on the
chain with a law without atoms k_N = 1 almost surely.

build_chains()                          the seven families for T = 5 and T = 20 (22 chain / sample-size cases)
exceed_fixed(ch, eps, n, g)             P{V(max of the retained values) > eps} for a fixed level eps
exceed_adaptive(ch, eps0, eps1, n, g)   the same for a certificate eps(k) evaluated at the observed complexity k
Retained values: Z_1, Z_{1+g}, ..., n of them (g = 1 keeps the whole trajectory of length n).
"""
import math

import numpy as np
from scipy import stats

from . import certificates as C
from . import sticky as S

BETA = 0.05


def stationary(P):
    w, v = np.linalg.eig(P.T)
    i = np.argmin(np.abs(w - 1.0))
    pi = np.real(v[:, i])
    pi = pi / pi.sum()
    return np.clip(pi, 0, None) / np.clip(pi, 0, None).sum()


def tmix_exact(P, pi=None, tmax=100000):
    """min{t >= 1 : max_z ||P^t(z, .) - pi||_TV <= 1/4}."""
    if pi is None:
        pi = stationary(P)
    Pt = np.eye(P.shape[0])
    for t in range(1, tmax + 1):
        Pt = Pt @ P
        if 0.5 * np.abs(Pt - pi[None, :]).sum(axis=1).max() <= 0.25:
            return t
    raise RuntimeError("t_mix > tmax")


def two_state_chain(p, lam):
    """Two states, state 1 rare with stationary mass p, second eigenvalue lam."""
    a, b = p * (1 - lam), (1 - p) * (1 - lam)
    return np.array([[1 - a, a], [b, 1 - b]])


def lazy_cycle(m):
    """Hold with probability 1/2, otherwise move to either neighbour on the cycle with equal probability."""
    P = np.zeros((m, m))
    for i in range(m):
        P[i, i] += 0.5
        P[i, (i + 1) % m] += 0.25
        P[i, (i - 1) % m] += 0.25
    return P


def perturbed_cycle(m, p):
    """Move to the next state with probability 1 - p, otherwise to a uniform state."""
    P = np.full((m, m), p / m)
    for i in range(m):
        P[i, (i + 1) % m] += 1 - p
    return P


def discretized_ar1(m, rho):
    """Gaussian AR(1) Z' = rho Z + sqrt(1 - rho^2) xi on m bins of equal N(0,1) probability; the transition from bin i
    starts from the median of bin i. Rows renormalised."""
    edges = stats.norm.ppf(np.linspace(0, 1, m + 1))
    mids = stats.norm.ppf((np.arange(m) + 0.5) / m)
    s = math.sqrt(1 - rho ** 2)
    P = np.empty((m, m))
    for i in range(m):
        cdf = stats.norm.cdf((edges - rho * mids[i]) / s)
        P[i] = np.diff(cdf)
    P /= P.sum(axis=1, keepdims=True)
    return P


def compression_plugin(N):
    """Compression certificate with equal weights at k = 1, applied as if the data were independent."""
    return C.eps_compression_equal(1, N, BETA)


def beta_plugin(N):
    return C.eps_beta_plugin(N, BETA)


def build_chains():
    """The chains of Table 3. Each certificate receives the exact mixing time of its chain as tbar. The tiny offsets of
    size 1e-9 make the computed mixing time equal the target T exactly despite floating-point rounding."""
    chains = []
    for T in (5, 20):
        q = S.q_for_tmix(T)
        chains.append({"family": "C1", "kind": "sticky_cont", "q": q, "T": T, "tmix": S.sticky_tmix(q),
                       "Ns": (1000, 10000)})
        m = 100
        qa = S.q_for_tmix(T, atoms=m) + 1e-9
        P = (1 - qa) * np.eye(m) + qa * np.full((m, m), 1.0 / m)
        pi = np.full(m, 1.0 / m)
        chains.append({"family": "C2", "kind": "finite", "P": P, "pi": pi, "phi": np.arange(m, dtype=float), "T": T,
                       "tmix": tmix_exact(P, pi), "Ns": (1000, 10000)})
        for fam, base in (("C3", compression_plugin), ("C3b", beta_plugin)):
            for N in (1000, 10000):
                p = 1.5 * base(N)
                lam = (4 * (1 - p)) ** (-1.0 / T) * (1 - 1e-9)
                P2 = two_state_chain(p, lam)
                pi2 = np.array([1 - p, p])
                chains.append({"family": fam, "kind": "finite", "P": P2, "pi": pi2, "phi": np.array([0.0, 1.0]),
                               "T": T, "tmix": tmix_exact(P2, pi2), "Ns": (N,)})
        mm = max(m_ for m_ in range(3, 40) if tmix_exact(lazy_cycle(m_), np.full(m_, 1.0 / m_)) <= T)
        P4 = lazy_cycle(mm)
        chains.append({"family": "C4", "kind": "finite", "P": P4, "pi": np.full(mm, 1.0 / mm),
                       "phi": np.arange(mm, dtype=float), "T": T, "tmix": tmix_exact(P4, np.full(mm, 1.0 / mm)),
                       "Ns": (1000,)})
        m5 = 10
        p5 = 1 - (0.25 / (1 - 1.0 / m5)) ** (1.0 / T) + 1e-9
        P5 = perturbed_cycle(m5, p5)
        chains.append({"family": "C5", "kind": "finite", "P": P5, "pi": np.full(m5, 1.0 / m5),
                       "phi": np.arange(m5, dtype=float), "T": T, "tmix": tmix_exact(P5, np.full(m5, 1.0 / m5)),
                       "Ns": (1000,)})
        lo, hi = 0.0, 0.9995
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if tmix_exact(discretized_ar1(50, mid)) <= T:
                lo = mid
            else:
                hi = mid
        P6 = discretized_ar1(50, lo)
        pi6 = stationary(P6)
        chains.append({"family": "C6", "kind": "finite", "P": P6, "pi": pi6, "phi": np.arange(50, dtype=float),
                       "rho": lo, "T": T, "tmix": tmix_exact(P6, pi6), "Ns": (1000,)})
    return chains


def _risk_levels(ch):
    pi, phi = ch["pi"], ch["phi"]
    return np.array([pi[phi > phi[s]].sum() for s in range(len(phi))])


def _prob_all_in(P, pi, idx, n):
    """P{the n retained states all lie in idx}, retained chain with kernel P (stationary start)."""
    return float((pi[idx] @ np.linalg.matrix_power(P[np.ix_(idx, idx)], n - 1)).sum())


def exceed_fixed(ch, eps, n, g=1):
    """P{V(max of the n retained values) > eps} for a fixed level eps."""
    if ch["kind"] == "sticky_cont":
        qg = 1 - (1 - ch["q"]) ** g
        return (1 - eps) * (1 - qg * eps) ** (n - 1)
    phi, pi = ch["phi"], ch["pi"]
    Vs = _risk_levels(ch)
    ok = Vs <= eps
    if not ok.any():
        return 1.0
    crit = phi[ok].min()
    idx = np.where(phi < crit)[0]
    if len(idx) == 0:
        return 0.0
    return _prob_all_in(np.linalg.matrix_power(ch["P"], g), pi, idx, n)


def exceed_adaptive(ch, eps0, eps1, n, g=1):
    """P{V(theta*) > eps(k)} with k = 0 when every retained value is the bottom label and k = 1 otherwise."""
    if ch["kind"] == "sticky_cont":
        qg = ch["q"] if g == 1 else 1 - (1 - ch["q"]) ** g
        return (1 - eps1) * (1 - qg * eps1) ** (n - 1) if eps1 < 1 else 0.0
    P = ch["P"] if g == 1 else np.linalg.matrix_power(ch["P"], g)
    pi, phi = ch["pi"], ch["phi"]
    Vs = _risk_levels(ch)
    bottom = phi.min()
    S0 = np.where(phi == bottom)[0]
    p0 = _prob_all_in(P, pi, S0, n)
    ex = p0 if Vs[S0].max() > eps0 else 0.0
    okset = np.where((phi > bottom) & (Vs > eps1))[0]
    if len(okset):
        crit = phi[okset].max()
        idx = np.where(phi <= crit)[0]
        ex += _prob_all_in(P, pi, idx, n) - p0
    return ex
