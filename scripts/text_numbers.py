"""Numbers quoted in the text outside the figures and tables (deterministic).

Example 2.6: the threshold decision on a sticky chain with t_mix = 5, N = 10^4, beta = 1e-3.
Section 3.1: the worked computation of eps_UB(2) for N = 10^4, beta = 1e-3, tbar = 5.
Section 4.4: the limit of the path-dependent floor over the independent risk quantile (beta = 1e-3, gamma = 3/4).
Remark 4.7: the thresholds of Theorem 8 of Lauer (arXiv:2303.11650v1) for n = 1000, confidence level 0.05 and the
            exceedance probabilities on three stationary chains.
Section 5.2: size of the latent state space and mixing times of the reserve model.
Output: results/text_numbers.json.
"""
import math

import _common as U
import numpy as np

from mixscen import certificates as C
from mixscen import chains as CH
from mixscen import reserve as RS
from mixscen import sticky as S


def main():
    out = {"environment": U.environment()}
    beta, N, tbar = 1e-3, 10 ** 4, 5
    q = S.qbar(tbar)
    out["example_2_6"] = {"q": q, "tmix": S.sticky_tmix(q), "sticky_quantile": S.quantile_d1(N, beta, q),
                          "independent_quantile": C.eps_beta_plugin(N, beta), "eps_ub1": C.eps_ub(1, N, beta, tbar)}
    k = 2
    m, b, B, L = C.ub_parts(k, N, beta, tbar)
    out["worked_computation"] = {"m": m, "b_2": b, "B_N_2": B, "L_2": L, "N_minus_B": N - B, "argument": m * L / (N - B),
                                 "eps_ub2": C.eps_ub(k, N, beta, tbar), "eps_compression2": C.eps_compression(k, N, beta),
                                 "ratio": C.eps_ub(k, N, beta, tbar) / C.eps_compression(k, N, beta),
                                 "first_order_ratio": 1.5 * m * N / (N - B)}
    gamma = 0.75
    out["path_floor_limit"] = {t: math.log((1 - gamma) / beta) / (S.qbar(t) * math.log(1 / beta)) for t in (5, 20)}
    n, delta = 1000, 0.05
    growth = 2 * n + 1
    printed = (4 * math.log(growth) + math.log(4 / delta)) / n
    proof = 4 * (math.log(growth) + math.log(4 / delta)) / n
    P2 = np.array([[1 - 5e-4, 5e-4], [5e-4, 1 - 5e-4]])

    def exceed(u):
        return {"constant": 1 - u, "sticky_q0.01": (1 - u) * (1 - 0.01 * u) ** (n - 1),
                "two_state": 0.5 * (1 - 5e-4) ** (n - 1) if u < 0.5 else 0.0}
    out["remark_4_7"] = {"threshold_printed": printed, "threshold_proof": proof, "exceed_printed": exceed(printed),
                         "exceed_proof": exceed(proof), "tmix_sticky_q0.01": S.sticky_tmix(0.01),
                         "tmix_two_state": CH.tmix_exact(P2, np.array([0.5, 0.5]))}
    out["reserve_model"] = {"latent_states": 3 * len(RS.S_VALS) * 3 ** RS.D_DIM,
                            "tmix": {theta: RS.tmix_formula(theta) for theta in (0.25, 0.066)},
                            "tmix_exact_regime_chain": {theta: CH.tmix_exact(RS.regime_chain(theta), RS.PI_R)
                                                        for theta in (0.25, 0.066)}}
    U.save_json("text_numbers.json", out)
    e, w, r = out["example_2_6"], out["worked_computation"], out["remark_4_7"]
    print(f"Example 2.6: q = {e['q']:.3f}, t_mix = {e['tmix']}, sticky quantile {100 * e['sticky_quantile']:.3f}%, "
          f"independent {100 * e['independent_quantile']:.3f}%, eps_UB(1) = {100 * e['eps_ub1']:.2f}%")
    print(f"Worked computation: m = {w['m']}, b_2 = {w['b_2']}, B_N(2) = {w['B_N_2']}, L_2 = {w['L_2']:.1f}, argument "
          f"{w['argument']:.4f}, eps_UB(2) = {100 * w['eps_ub2']:.2f}%, compression {100 * w['eps_compression2']:.2f}%, "
          f"ratio {w['ratio']:.1f}, first-order ratio {w['first_order_ratio']:.1f}")
    print("Path floor limits:", {t: round(v, 1) for t, v in out["path_floor_limit"].items()})
    print(f"Remark 4.7: thresholds {r['threshold_printed']:.6f} and {r['threshold_proof']:.6f}; exceedances "
          f"{ {k_: round(v, 4) for k_, v in r['exceed_printed'].items()} } and "
          f"{ {k_: round(v, 4) for k_, v in r['exceed_proof'].items()} }; t_mix {r['tmix_sticky_q0.01']} and "
          f"{r['tmix_two_state']}")
    print("Reserve model:", out["reserve_model"])


if __name__ == "__main__":
    main()
