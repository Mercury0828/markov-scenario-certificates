"""Risk certificates used in the paper. Natural logarithms throughout; tbar >= 1 is the mixing-time bound and
m = ceil(tbar).

eps_ub(k, N, beta, tbar)             the certificate of Theorem 3.1
eps_compression(k, n, beta)          the compression certificate for n independent scenarios with the weights
                                     w_k = 1/((k+1)(k+2)) (Remark 3.5)
eps_compression_equal(k, n, beta)    the same with equal weights w_k = 1/n (Campi, Garatti and Ramponi 2018, Theorem 1)
eps_beta_plugin(N, beta)             1 - beta^(1/N), the (1-beta)-quantile of Beta(1, N)
thinning_level, eps_thinning,
thinning_gap                         the thinning baseline of Section 5.2
"""
import math

import numpy as np
from scipy import special

RSTAR = (1 + math.sqrt(17)) / 8  # value at p = 1 of the avoidance factor of Lemma 3.3
LOG_INV_RSTAR = -math.log(RSTAR)


def log_binom(n, k):
    """log of the binomial coefficient C(n, k)."""
    return special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)


def F(a):
    """F(a) = (1 - e^-a)(e^-a - 1/4)/(e^-a - 1/2) for 0 <= a < log(1/r*), and 1 otherwise."""
    if a >= LOG_INV_RSTAR:
        return 1.0
    e = math.exp(-a)
    return (1 - e) * (e - 0.25) / (e - 0.5)


def ub_parts(k, N, beta, tbar):
    """(m, b_k, B_N(k), L_k) of the certificate of Theorem 3.1, for 0 <= k < N."""
    m = max(1, math.ceil(tbar))
    w = 1.0 / ((k + 1) * (k + 2))
    if k == 0:
        return m, 0, 0, math.log(2.0 / beta)
    lc = log_binom(N, k)
    b = m * math.ceil((math.log(4 * k) + lc - math.log(beta * w)) / math.log(2))
    return m, b, k * (2 * b + 1), math.log(2) + lc - math.log(beta * w)


def eps_ub(k, N, beta, tbar):
    """The certificate eps_UB(k) of Theorem 3.1 (equal to one when k >= N or when the buffer B_N(k) is at least N)."""
    if k >= N:
        return 1.0
    m, _, B, L = ub_parts(k, N, beta, tbar)
    if N <= B:
        return 1.0
    a = m * L / (N - B)
    return 1.0 if a >= LOG_INV_RSTAR else F(a)


def eps_compression(k, n, beta):
    """1 - (beta w_k / C(n,k))^(1/(n-k)) with w_k = 1/((k+1)(k+2)); one when k >= n or beta <= 0."""
    if beta <= 0 or k >= n:
        return 1.0
    w = 1.0 / ((k + 1) * (k + 2))
    return -math.expm1((math.log(beta * w) - log_binom(n, k)) / (n - k))


def eps_compression_equal(k, n, beta):
    """1 - (beta / (n C(n,k)))^(1/(n-k)), the equal allocation w_k = 1/n; one when k >= n or beta <= 0."""
    if beta <= 0 or k >= n:
        return 1.0
    kk = np.asarray([k], dtype=float)
    logc = special.gammaln(n + 1) - special.gammaln(kk + 1) - special.gammaln(n - kk + 1)
    return float(-np.expm1((math.log(beta) - math.log(n) - logc) / (n - kk))[0])


def eps_beta_plugin(N, beta):
    """1 - beta^(1/N)."""
    return float(-math.expm1(math.log(beta) / N))


def thinning_level(N, beta, tbar, g):
    """Retained sample size n = ceil(N/g) and confidence level beta - (n-1) 2^(-floor(g/tbar)) of the thinned sample."""
    n = math.ceil(N / g)
    return n, beta - (n - 1) * 2.0 ** (-math.floor(g / tbar))


def eps_thinning(k, N, beta, tbar, g, equal_weights=False):
    """Certificate of the thinned decision at thinned complexity k (one when the level is not positive)."""
    n, level = thinning_level(N, beta, tbar, g)
    if equal_weights:
        return eps_compression_equal(k, n, level)
    return eps_compression(k, n, level)


def thinning_gap(N, beta, tbar, d, equal_weights=False, gmax_factor=40):
    """Gap fixed before the data: the minimiser of the thinned certificate at complexity d over
    g in {ceil(tbar), ..., 40 ceil(tbar)}, the smallest such g on ties."""
    t = max(1, math.ceil(tbar))
    return min(range(t, gmax_factor * t + 1),
               key=lambda g: (eps_thinning(d, N, beta, tbar, g, equal_weights), g))
