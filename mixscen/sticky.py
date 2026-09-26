"""The sticky chain of Section 4: it keeps its state with probability 1 - q and otherwise draws a fresh state from pi.

qbar(tbar)                     the refresh probability (4.1) of the witnesses, 1 - 4^(-1/floor(tbar))
sticky_tmix(q, atoms)          its mixing time for a law without atoms (atoms=None) or with m equiprobable atoms
q_for_tmix(tbar, atoms)        the smallest refresh probability with t_mix <= tbar
quantile_d1(N, beta, q)        (1-beta)-quantile of the risk of max_t delta_t: root of (1-u)(1-qu)^(N-1) = beta
eps_sharp(N, d, beta, tbar)    the threshold of Theorem 4.3(a): inf{u : G_{N,d}(u) <= beta}
"""
import math
from functools import lru_cache

import numpy as np
from scipy import optimize, stats


def qbar(tbar):
    return 1 - 4 ** (-1.0 / math.floor(tbar))


def sticky_tmix(q, atoms=None):
    c = 1.0 if atoms is None else 1.0 - 1.0 / atoms
    if q >= 1.0:
        return 1
    return max(1, math.ceil(math.log(0.25 / c) / math.log(1.0 - q) - 1e-12))


def q_for_tmix(tbar, atoms=None):
    c = 1.0 if atoms is None else 1.0 - 1.0 / atoms
    return 1.0 - (0.25 / c) ** (1.0 / tbar)


def quantile_d1(N, beta, q, lower=1e-15):
    """Root u of log(1-u) + (N-1) log(1-qu) = log(beta)."""
    f = lambda u: math.log1p(-u) + (N - 1) * math.log1p(-q * u) - math.log(beta)  # noqa: E731
    return optimize.brentq(f, lower, 1 - 1e-15, xtol=1e-16, rtol=1e-13)


@lru_cache(maxsize=None)
def _refresh_support(N, q):
    """Values j of the number M = 1 + Bin(N-1, q) of fresh draws and their probabilities (tails below 1e-17 dropped)."""
    kd = stats.binom(N - 1, q)
    lo, hi = int(kd.ppf(1e-17)), int(kd.isf(1e-17))
    j1 = np.arange(max(0, lo - 1), min(N - 1, hi + 1) + 1)
    return j1 + 1, kd.pmf(j1)


def G(u, N, d, q):
    """Joint tail P{V > u, k_N = d} = sum_{j >= d} P(M = j) P{Beta(d, j-d+1) > u} of (4.4)."""
    j, w = _refresh_support(N, q)
    keep = j >= d
    return float(np.sum(w[keep] * stats.binom.cdf(d - 1, j[keep], u)))


def eps_sharp(N, d, beta, tbar):
    """The threshold of Theorem 4.3(a) for the witness with refresh probability qbar(tbar)."""
    q = qbar(tbar)
    f = lambda u: G(u, N, d, q) - beta  # noqa: E731
    if f(0.0) <= 0:
        return 0.0
    if f(1 - 1e-15) > 0:
        return 1.0
    return optimize.brentq(f, 0.0, 1 - 1e-15, xtol=1e-15, rtol=1e-12)


def beta_quantile(N, d, beta):
    """(1-beta)-quantile of Beta(d, N-d+1), the exact risk law for fully supported problems on independent data."""
    return float(stats.beta.isf(beta, d, N - d + 1))
