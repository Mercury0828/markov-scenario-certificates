"""Table 2 and the numbers of Section 5.2 from the runs of reserve_simulate.py.

For each setting: shares of the observed complexity k_N, the mean certificate eps_UB(k_N) at beta = 1e-3, the constant
certificate eps_UB(d), the mean risk, and for the thinned sample the gap g, the mean thinned certificate (weights w_k of
eps_UB) and the mean risk of the thinned decision. Also the ratios quoted in the text, the exceedance counts and the
conservatism of the certificate relative to the 0.999-quantile of the risk.
Output: results/table2_reserve.tex and results/table2_reserve.json.
Usage: python scripts/table2_reserve.py [--dir results/reserve]
"""
import argparse
import os

import _common as U
import numpy as np

from mixscen import certificates as C
from mixscen import reserve as RS

D = RS.D_DIM


def load(directory, name):
    parts = []
    for c in range(RS.CHUNKS):
        p = os.path.join(directory, f"{name}_chunk{c}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}: run scripts/reserve_simulate.py first")
        parts.append(np.load(p))
    return {f: np.concatenate([q[f] for q in parts]) for f in parts[0].files}


def pct(x):
    return "100\\%" if x >= 1 else (f"{100 * x:.2f}\\%" if x >= 1e-3 else f"{100 * x:.3f}\\%")


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\cdot10^{{{int(e)}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(U.RESULTS, "reserve"))
    a = ap.parse_args()
    rows, lines = [], [r"\begin{tabular}{rrccccccc}", r"\toprule",
                       r" & & \multicolumn{4}{c}{all $N$ scenarios} & \multicolumn{3}{c}{thinning, same $N$} \\",
                       r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
                       r"$\bar t$ & $N$ & $k_N$ (1/2/3/$\ge$4) & $\overline{\varepsilon_{\rm UB}(k_N)}$ & "
                       r"$\varepsilon_{\rm UB}(d)$ & $\overline{V}$ & $g$ & $\overline{\varepsilon}$ & $\overline{V}$ \\",
                       r"\midrule"]
    for name, theta, N in RS.SETTINGS:
        data = load(a.dir, name)
        tbar = RS.tmix_formula(theta)
        gaps = RS.gaps_for(N, tbar)
        k = data["k"].astype(int)
        V = data["V"]
        R = len(k)
        dist = np.bincount(k, minlength=D + 1)
        row = {"setting": name, "theta": theta, "tbar": tbar, "N": N, "runs": R, "k_dist": dist.tolist(),
               "k_median": float(np.median(k)), "share_k_1_or_2": float((dist[1] + dist[2]) / R),
               "V_mean": float(V.mean()), "by_beta": {}}
        for b in RS.BETAS:
            bk = f"{b:g}"
            ub = np.array([C.eps_ub(j, N, b, tbar) for j in range(D + 1)])[k]
            g = gaps[b]
            n, level = C.thinning_level(N, b, tbar, g)
            kt = data[f"kthin_{bk}"].astype(int)
            Vt = data[f"Vthin_{bk}"]
            thin_w = np.array([C.eps_compression(j, n, level) for j in range(D + 1)])[kt]
            thin_eq = np.array([C.eps_compression_equal(j, n, level) for j in range(D + 1)])[kt]
            plugin = np.array([C.eps_compression_equal(j, N, b) for j in range(D + 1)])[k]
            row["by_beta"][bk] = {
                "eps_ub_mean": float(ub.mean()), "eps_ub_d": C.eps_ub(D, N, b, tbar),
                "exceed_ub": int((V > ub).sum()), "exceed_plugin": int((V > plugin).sum()),
                "q_1mbeta_V": float(np.quantile(V, 1 - b)),
                "gap": g, "thinned_n": n, "thinned_level": level,
                "thin_eps_mean": float(thin_w.mean()), "thin_eps_equal_mean": float(thin_eq.mean()),
                "thin_V_mean": float(Vt.mean()),
                "ratio_thin_to_ub": float(thin_w.mean() / ub.mean()),
                "ratio_thin_equal_to_ub": float(thin_eq.mean() / ub.mean()),
                "ratio_thin_risk_to_full_risk": float(Vt.mean() / V.mean()),
                "exceed_thin_by_thinned_decision": int((Vt > thin_w).sum()),
                "exceed_thin_by_full_decision": int((V > thin_w).sum()),
                "full_risk_above_thinned_risk": int((V > Vt + 1e-15).sum())}
            d = row["by_beta"][bk]
            d["conservatism"] = d["eps_ub_mean"] / d["q_1mbeta_V"]
            d["ratio_dimension_bound"] = d["eps_ub_d"] / d["eps_ub_mean"] if d["eps_ub_d"] < 1 else None
        rows.append(row)
        d = row["by_beta"]["0.001"]
        kd = dist / R
        share = f"{100 * kd[1]:.0f}/{100 * kd[2]:.0f}/{100 * kd[3]:.0f}/{100 * kd[4:].sum():.0f}"
        lines.append(f"{tbar} & {N:,} & {share} & {pct(d['eps_ub_mean'])} & {pct(d['eps_ub_d'])} & "
                     f"{sci(row['V_mean'])} & {d['gap']} & {pct(d['thin_eps_mean'])} & "
                     f"{sci(d['thin_V_mean'])} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    U.write_text("table2_reserve.tex", "\n".join(lines) + "\n")
    U.save_json("table2_reserve.json", {"environment": U.environment(), "rows": rows})
    print("\n".join(lines))
    for bk in ("0.001", "0.05"):
        r_thin = [r["by_beta"][bk]["ratio_thin_to_ub"] for r in rows]
        r_eq = [r["by_beta"][bk]["ratio_thin_equal_to_ub"] for r in rows]
        print(f"beta = {bk}: mean thinned / mean eps_UB = {min(r_thin):.2f} to {max(r_thin):.2f}; equal weights "
              f"{min(r_eq):.2f} to {max(r_eq):.2f}; exceedances of eps_UB(k_N): "
              f"{sum(r['by_beta'][bk]['exceed_ub'] for r in rows)}; of the compression plug-in: "
              f"{[r['by_beta'][bk]['exceed_plugin'] for r in rows]}")
    d3 = [r["by_beta"]["0.001"] for r in rows]
    print("share of k_N in {1, 2}:", [round(100 * r["share_k_1_or_2"]) for r in rows], "percent; medians",
          [r["k_median"] for r in rows])
    print("thinned / full mean risk at beta = 1e-3:", [round(x["ratio_thin_risk_to_full_risk"]) for x in d3])
    print("mean eps_UB(k_N) / 0.999-quantile of the risk:", [round(x["conservatism"]) for x in d3])
    print("eps_UB(d) / mean eps_UB(k_N) where eps_UB(d) < 1:",
          [None if x["ratio_dimension_bound"] is None else round(x["ratio_dimension_bound"], 2) for x in d3])
    print("full-sample risk above thinned risk in", sum(r["by_beta"][bk]["full_risk_above_thinned_risk"]
                                                        for r in rows for bk in ("0.001", "0.05")), "runs")


if __name__ == "__main__":
    main()
