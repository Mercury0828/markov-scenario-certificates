"""Table 3 and the numbers of Section 5.3 (exact computation, no simulation).

For each chain of Section 5.3 and N in {10^3, 10^4}: the exact probability that the risk of the largest observed value
exceeds (i) eps_UB(k_N) of Theorem 3.1 with tbar = the exact mixing time of the chain, (ii) the Beta plug-in 1 - beta^(1/N),
(iii) the compression plug-in (equal weights, k = 1), (iv) the thinned certificate of Section 5.2 at the thinned
complexity, with its gap fixed before the data at complexity d = 1. beta = 0.05.
Output: results/table3_validity.tex and results/table3_validity.json.
"""
import _common as U

from mixscen import certificates as C
from mixscen import chains as CH

BETA = CH.BETA
NAMES = {"C1": r"sticky, uniform on $[0,1]$", "C2": r"sticky, 100 points", "C3": r"two-state (a)",
         "C3b": r"two-state (b)", "C4": r"lazy cycle", "C5": r"perturbed cycle", "C6": r"discretized AR(1)"}


def fmt(x):
    if x is None:
        return "--"
    return "0" if x < 5e-5 else f"{x:.4f}"


def main():
    rows = []
    for ch in CH.build_chains():
        for N in ch["Ns"]:
            tb = ch["tmix"]
            g = C.thinning_gap(N, BETA, tb, 1)
            n, level = C.thinning_level(N, BETA, tb, g)
            rows.append({"family": ch["family"], "T": ch["T"], "tmix": tb, "N": N,
                         "rho": ch.get("rho"), "states": len(ch["phi"]) if ch["kind"] == "finite" else None,
                         "eps_ub1": C.eps_ub(1, N, BETA, tb),
                         "ub": CH.exceed_adaptive(ch, C.eps_ub(0, N, BETA, tb), C.eps_ub(1, N, BETA, tb), N, 1),
                         "beta_plugin": CH.exceed_fixed(ch, CH.beta_plugin(N), N, 1),
                         "compression_plugin": CH.exceed_fixed(ch, CH.compression_plugin(N), N, 1),
                         "thin_gap": g,
                         "thinning": CH.exceed_adaptive(ch, C.eps_compression(0, n, level),
                                                        C.eps_compression(1, n, level), n, g)})
    by = {}
    for r in rows:
        by.setdefault((r["T"], r["family"]), {})[r["N"]] = r
    lines = [r"\begin{tabular}{lrcccccccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{$\varepsilon_{\rm UB}(k_N)$} & \multicolumn{2}{c}{Beta plug-in} & "
             r"\multicolumn{2}{c}{compression plug-in} & \multicolumn{2}{c}{thinning} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
             r"chain & $t_{\rm mix}$ & $10^3$ & $10^4$ & $10^3$ & $10^4$ & $10^3$ & $10^4$ & $10^3$ & $10^4$ \\",
             r"\midrule"]
    for T in (5, 20):
        for fam in NAMES:
            cell = by[(T, fam)]
            r3, r4 = cell.get(1000), cell.get(10000)
            vals = []
            for key in ("ub", "beta_plugin", "compression_plugin", "thinning"):
                vals += [fmt(r3[key]), fmt(r4[key]) if r4 else "--"]
            lines.append(f"{NAMES[fam]} & {r3['tmix']} & " + " & ".join(vals) + r" \\")
        if T == 5:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    U.write_text("table3_validity.tex", "\n".join(lines) + "\n")
    summary = {"cases": len(rows), "max_exceed_ub": max(r["ub"] for r in rows),
               "beta_plugin_failures": sum(r["beta_plugin"] > BETA for r in rows),
               "beta_plugin_max": max(r["beta_plugin"] for r in rows),
               "compression_plugin_failures": sum(r["compression_plugin"] > BETA for r in rows),
               "compression_plugin_max": max(r["compression_plugin"] for r in rows),
               "thinning_max": max(r["thinning"] for r in rows),
               "ar1_coefficients": sorted({round(r["rho"], 3) for r in rows if r["rho"] is not None}),
               "lazy_cycle_states": sorted({r["states"] for r in rows if r["family"] == "C4"})}
    U.save_json("table3_validity.json", {"environment": U.environment(), "beta": BETA, "rows": rows,
                                         "summary": summary})
    print("\n".join(lines))
    print(summary)


if __name__ == "__main__":
    main()
