"""Compare the regenerated results with the values printed in the paper.

Every number is rounded as printed in the paper before the comparison; Tables 2 and 3 are compared as LaTeX text with the
copies in expected/. Run after the other scripts (run_all.py does this). Prints one line per check and a summary.
"""
import io
import os

import _common as U

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append((label, bool(ok), detail))


def same(label, computed, printed, digits):
    """Pass when the computed value rounds to the printed one."""
    c = round(computed, digits)
    check(label, abs(c - printed) < 10 ** (-digits) / 2, f"computed {computed:.6g}, printed {printed}")


def text_file(path):
    with io.open(path, encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n").strip()


def main():
    t = U.load_json("text_numbers.json")
    e, w, r, rm = t["example_2_6"], t["worked_computation"], t["remark_4_7"], t["reserve_model"]
    same("Example 2.6: q", e["q"], 0.242, 3)
    check("Example 2.6: t_mix = 5", e["tmix"] == 5)
    same("Example 2.6: sticky quantile (percent)", 100 * e["sticky_quantile"], 0.285, 3)
    same("Example 2.6: independent quantile (percent)", 100 * e["independent_quantile"], 0.069, 3)
    same("Example 2.6: eps_UB(1) (percent)", 100 * e["eps_ub1"], 1.44, 2)
    check("Section 3.1: m, b_2, B_N(2)", (w["m"], w["b_2"], w["B_N_2"]) == (5, 215, 862), str((w["m"], w["b_2"], w["B_N_2"])))
    same("Section 3.1: L_2", w["L_2"], 27.8, 1)
    check("Section 3.1: N - B_N(2) = 9138", w["N_minus_B"] == 9138)
    same("Section 3.1: argument of F", w["argument"], 0.0152, 4)
    same("Section 3.1: eps_UB(2) (percent)", 100 * w["eps_ub2"], 2.29, 2)
    same("Section 3.1: compression certificate (percent)", 100 * w["eps_compression2"], 0.27, 2)
    same("Section 3.1: ratio", w["ratio"], 8.5, 1)
    same("Section 3.1: (3/2) m N / (N - B_N(2))", w["first_order_ratio"], 8.2, 1)
    same("Section 4.4: limit for tbar = 5", t["path_floor_limit"]["5"], 3.3, 1)
    same("Section 4.4: limit for tbar = 20", t["path_floor_limit"]["20"], 11.9, 1)
    same("Remark 4.7: printed threshold", r["threshold_printed"], 0.034788, 6)
    same("Remark 4.7: proof threshold", r["threshold_proof"], 0.047934, 6)
    for key, a, b in (("constant", 0.9652, 0.9521), ("sticky_q0.01", 0.6818, 0.5897), ("two_state", 0.3034, 0.3034)):
        same(f"Remark 4.7: {key}, printed threshold", r["exceed_printed"][key], a, 4)
        same(f"Remark 4.7: {key}, proof threshold", r["exceed_proof"][key], b, 4)
    check("Remark 4.7: t_mix 138 and 693", (r["tmix_sticky_q0.01"], r["tmix_two_state"]) == (138, 693))
    check("Section 5.2: 15,309 latent states", rm["latent_states"] == 15309)
    check("Section 5.2: mixing times 5 and 20", list(rm["tmix"].values()) == [5, 20]
          and list(rm["tmix_exact_regime_chain"].values()) == [5, 20])

    f1 = U.load_json("figure1_sticky_path.json")
    check("Figure 1: 60 and 5 fresh values", (f1["fresh_a"], f1["fresh_b"]) == (60, 5))

    f3 = U.load_json("figure3_certificate.json")["numbers"]
    for tb, printed in (("5", {"1": 1.44, "2": 2.29, "3": 3.30, "5": 6.64}), ("20", {"1": 6.34, "2": 12.99})):
        for k, v in printed.items():
            same(f"Section 5.1: eps_UB({k}), tbar = {tb}, N = 1e4 (percent)", 100 * f3["eps_ub_N1e4"][tb][k], v, 2)
    same("Section 5.1: eps_UB(3), tbar = 20, N = 1e4 (percent)", 100 * f3["eps_ub_N1e4"]["20"]["3"], 36.5, 1)
    check("Section 5.1: eps_UB(5) vacuous at tbar = 20, N = 1e4", f3["eps_ub_N1e4"]["20"]["5"] >= 1)
    check("Section 5.1: vacuous at N = 1e3", f3["vacuous_k_at_N1e3"] == {"5": [3, 5], "20": [1, 2, 3, 5]},
          str(f3["vacuous_k_at_N1e3"]))
    same("Section 5.1: qbar, tbar = 5", f3["qbar"]["5"], 0.242, 3)
    same("Section 5.1: qbar, tbar = 20", f3["qbar"]["20"], 0.067, 3)
    for tb, key, printed in (("5", "min_ratio", (5.0, 6.0, 6.8, 8.2)), ("5", "ratio_N1e6", (6.1, 7.3, 8.1, 9.3)),
                             ("20", "min_ratio", (6.0, 7.3, 8.4, 10.2))):
        for k, v in zip(("1", "2", "3", "5"), printed):
            same(f"Section 5.1: {key}, tbar = {tb}, k = {k}", f3[key][tb][k], v, 1)
    for k, a, b in (("1", 2.6, 3.3), ("5", 3.5, 5.1)):
        same(f"Section 5.1: independent-data ratio, k = {k}, N = 1e4", f3["iid_ratio"][k]["N1e4"], a, 1)
        same(f"Section 5.1: independent-data ratio, k = {k}, N = 1e6", f3["iid_ratio"][k]["N1e6"], b, 1)

    t3 = U.load_json("table3_validity.json")["summary"]
    check("Table 3: identical to the paper", text_file(os.path.join(U.RESULTS, "table3_validity.tex"))
          == text_file(os.path.join(U.ROOT, "expected", "table3_validity.tex")))
    check("Section 5.3: 22 cases", t3["cases"] == 22)
    check("Section 5.3: exceedance of eps_UB(k_N) at most 4.3e-4", round(t3["max_exceed_ub"], 5) <= 4.3e-4 + 1e-12,
          f"{t3['max_exceed_ub']:.3g}")
    check("Section 5.3: Beta plug-in fails in 12 cases", t3["beta_plugin_failures"] == 12)
    same("Section 5.3: Beta plug-in, largest exceedance", t3["beta_plugin_max"], 0.82, 2)
    check("Section 5.3: compression plug-in fails in 5 cases", t3["compression_plugin_failures"] == 5)
    same("Section 5.3: compression plug-in, largest exceedance", t3["compression_plugin_max"], 0.32, 2)
    check("Section 5.3: thinning valid in all cases", t3["thinning_max"] <= 0.05, f"{t3['thinning_max']:.3g}")
    check("Section 5.3: AR(1) coefficients 0.775 and 0.940", t3["ar1_coefficients"] == [0.775, 0.94])
    check("Section 5.3: lazy cycles with 7 and 14 states", t3["lazy_cycle_states"] == [7, 14])

    if os.path.exists(os.path.join(U.RESULTS, "table2_reserve.json")):
        rows = U.load_json("table2_reserve.json")["rows"]
        check("Table 2: identical to the paper", text_file(os.path.join(U.RESULTS, "table2_reserve.tex"))
              == text_file(os.path.join(U.ROOT, "expected", "table2_reserve.tex")))
        share = [100 * r["share_k_1_or_2"] for r in rows]
        check("Section 5.2: k_N in {1, 2} in 75 to 83 percent", round(min(share)) == 75 and round(max(share)) == 83,
              str([round(s, 1) for s in share]))
        check("Section 5.2: median k_N = 2", all(r["k_median"] == 2 for r in rows))
        b3 = [r["by_beta"]["0.001"] for r in rows]
        b5 = [r["by_beta"]["0.05"] for r in rows]
        check("Section 5.2: no exceedance of eps_UB(k_N)", sum(d["exceed_ub"] for d in b3 + b5) == 0)
        check("Section 5.3: no exceedance of the compression plug-in in the reserve problem",
              sum(d["exceed_plugin"] for d in b3 + b5) == 0)
        for lab, key, dd, lo, hi in (("matched", "ratio_thin_to_ub", b3, 2.3, 8.5), ("matched", "ratio_thin_to_ub", b5, 1.7, 6.2),
                                     ("equal", "ratio_thin_equal_to_ub", b3, 2.5, 10.5),
                                     ("equal", "ratio_thin_equal_to_ub", b5, 2.0, 7.9)):
            v = [d[key] for d in dd]
            same(f"Section 5.2: thinning ({lab} weights), smallest factor", min(v), lo, 1)
            same(f"Section 5.2: thinning ({lab} weights), largest factor", max(v), hi, 1)
        v = [d["ratio_thin_risk_to_full_risk"] for d in b3]
        check("Section 5.2: thinned mean risk 80 to 360 times larger", round(min(v), -1) == 80 and round(max(v), -1) == 360,
              str([round(x) for x in v]))
        check("Section 5.2: conservatism 27, 48, 179, 200", [round(d["conservatism"]) for d in b3] == [27, 48, 179, 200],
              str([round(d["conservatism"], 1) for d in b3]))
        v = [d["ratio_dimension_bound"] for d in b3 if d["ratio_dimension_bound"] is not None]
        check("Section 5.2: about two to three times below the dimension bound", len(v) == 3 and 1.95 <= min(v)
              and max(v) <= 3.1, str([round(x, 2) for x in v]))
        check("Section 5.2: full-sample risk never above thinned risk",
              sum(d["full_risk_above_thinned_risk"] for d in b3 + b5) == 0)
    else:
        check("Table 2: results present (run scripts/reserve_simulate.py and scripts/table2_reserve.py)", False)

    for label, ok, detail in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not ok else ""))
    failed = sum(not ok for _, ok, _ in CHECKS)
    print(f"\n{len(CHECKS) - failed} of {len(CHECKS)} checks agree with the paper.")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
