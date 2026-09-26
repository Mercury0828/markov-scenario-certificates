"""Figure 3 and the numbers of Section 5.1 (deterministic).

Panel (a): the certificate eps_UB(k) of Theorem 3.1 for beta = 1e-3, k in {1, 2, 3, 5}, tbar in {5, 20}, N from 10^3 to
10^6. Panel (b): its ratio to the threshold of Theorem 4.3(a) where eps_UB(k) < 1, and (dotted) the ratio of the compression
certificate for independent data (Remark 3.5) to the exact Beta(k, N-k+1) quantile.
Output: results/figure3_certificate.pdf/.png and results/figure3_certificate.json.
"""
import logging
import os

import _common as U
import matplotlib
import numpy as np

from mixscen import certificates as C
from mixscen import sticky as S

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)  # font subsetting notes on the bundled cmr10 font
import matplotlib.pyplot as plt  # noqa: E402

BETA = 1e-3
KS = (1, 2, 3, 5)
TBARS = (5, 20)
NGRID = sorted(set(np.round(np.logspace(3, 6, 61)).astype(int).tolist()) | {1000, 10000, 100000, 1000000})
SHADES = {1: "#08306b", 2: "#2171b5", 3: "#4292c6", 5: "#6baed6"}
MARK = {1: "o", 2: "s", 3: "^", 5: "D"}
STYLE = {5: "-", 20: "--"}


def compute():
    rows = []
    for tbar in TBARS:
        for k in KS:
            for N in NGRID:
                rows.append({"tbar": tbar, "k": k, "N": N, "eps_ub": C.eps_ub(k, N, BETA, tbar),
                             "eps_sharp": S.eps_sharp(N, k, BETA, tbar)})
    iid = [{"k": k, "N": N, "eps_compression": C.eps_compression(k, N, BETA),
            "beta_quantile": S.beta_quantile(N, k, BETA)} for k in KS for N in NGRID]
    return rows, iid


def numbers(rows, iid):
    def get(tbar, k, N):
        return next(r for r in rows if r["tbar"] == tbar and r["k"] == k and r["N"] == N)
    out = {"qbar": {t: S.qbar(t) for t in TBARS},
           "eps_ub_N1e4": {t: {k: get(t, k, 10 ** 4)["eps_ub"] for k in KS} for t in TBARS},
           "vacuous_k_at_N1e3": {t: [k for k in KS if get(t, k, 10 ** 3)["eps_ub"] >= 1] for t in TBARS},
           "min_ratio": {}, "ratio_N1e6": {}, "iid_ratio": {}}
    for t in TBARS:
        out["min_ratio"][t], out["ratio_N1e6"][t] = {}, {}
        for k in KS:
            rat = [r["eps_ub"] / r["eps_sharp"] for r in rows if r["tbar"] == t and r["k"] == k and r["eps_ub"] < 1]
            out["min_ratio"][t][k] = min(rat)
            out["ratio_N1e6"][t][k] = get(t, k, 10 ** 6)["eps_ub"] / get(t, k, 10 ** 6)["eps_sharp"]
    for k in KS:
        r4 = next(r for r in iid if r["k"] == k and r["N"] == 10 ** 4)
        r6 = next(r for r in iid if r["k"] == k and r["N"] == 10 ** 6)
        out["iid_ratio"][k] = {"N1e4": r4["eps_compression"] / r4["beta_quantile"],
                               "N1e6": r6["eps_compression"] / r6["beta_quantile"]}
    return out


def plot(rows, iid):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
                         "lines.linewidth": 1.1, "lines.markersize": 3.2, "xtick.minor.size": 0, "ytick.minor.size": 0,
                         "pdf.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.06, 2.35), constrained_layout=True)
    for tbar in TBARS:
        for k in KS:
            s = sorted((r for r in rows if r["tbar"] == tbar and r["k"] == k), key=lambda r: r["N"])
            N = [r["N"] for r in s]
            every = max(1, len(N) // 8)
            ax1.plot(N, [r["eps_ub"] for r in s], STYLE[tbar], color=SHADES[k], marker=MARK[k], markevery=every)
            ok = [(r["N"], r["eps_ub"] / r["eps_sharp"]) for r in s if r["eps_ub"] < 1]
            if ok:
                ax2.plot(*zip(*ok), STYLE[tbar], color=SHADES[k], marker=MARK[k], markevery=every)
    for k in KS:
        s = sorted((r for r in iid if r["k"] == k), key=lambda r: r["N"])
        ax2.plot([r["N"] for r in s], [r["eps_compression"] / r["beta_quantile"] for r in s], ":", color="#737373",
                 linewidth=0.9)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator([1e3, 1e4, 1e5, 1e6]))
        ax.xaxis.set_major_formatter(FixedFormatter(["1", "10", "100", "1000"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"sample size $N$ (thousands)")
    ax1.set_ylim(1.2e-4, 1.4)
    ax1.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1]))
    ax1.yaxis.set_major_formatter(FixedFormatter(["0.001", "0.01", "0.1", "1"]))
    ax1.set_ylabel("certificate")
    ax1.set_title("(a) certificate", fontsize=8)
    ax2.set_ylim(2, 80)
    ax2.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50]))
    ax2.yaxis.set_major_formatter(FixedFormatter(["2", "5", "10", "20", "50"]))
    ax2.set_ylabel("ratio to the lower bound")
    ax2.set_title("(b) ratio to a lower bound", fontsize=8)
    h1 = [Line2D([], [], color=SHADES[k], marker=MARK[k], linestyle="-", label=f"$k={k}$") for k in KS]
    h2 = [Line2D([], [], color="black", linestyle=STYLE[t], label=rf"$\overline{{t}}={t}$") for t in TBARS]
    h2.append(Line2D([], [], color="#737373", linestyle=":", label="independent data"))
    ax1.legend(handles=h1, loc="lower left", frameon=False, handlelength=2.0)
    ax2.legend(handles=h2, loc="upper right", frameon=False, handlelength=2.0)
    os.makedirs(U.RESULTS, exist_ok=True)
    out = os.path.join(U.RESULTS, "figure3_certificate.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06, metadata={"Creator": None, "Producer": None})
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight", pad_inches=0.06, metadata={"Software": None})
    return out


def main():
    rows, iid = compute()
    nums = numbers(rows, iid)
    out = plot(rows, iid)
    U.save_json("figure3_certificate.json", {"environment": U.environment(), "beta": BETA, "grid": NGRID,
                                             "rows": rows, "independent": iid, "numbers": nums})
    e = nums["eps_ub_N1e4"]
    print("Section 5.1: eps_UB at N = 1e4 (percent): tbar 5:", [round(100 * e[5][k], 2) for k in KS],
          "tbar 20:", [round(100 * e[20][k], 2) for k in KS])
    print("  vacuous at N = 1e3:", nums["vacuous_k_at_N1e3"], " qbar:", {t: round(v, 3) for t, v in nums["qbar"].items()})
    for t in TBARS:
        print(f"  tbar {t}: smallest ratio to the threshold", [round(nums['min_ratio'][t][k], 1) for k in KS],
              " at N = 1e6", [round(nums['ratio_N1e6'][t][k], 1) for k in KS])
    print("  independent data, ratio at N = 1e4 -> 1e6:",
          {k: (round(v["N1e4"], 1), round(v["N1e6"], 1)) for k, v in nums["iid_ratio"].items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
