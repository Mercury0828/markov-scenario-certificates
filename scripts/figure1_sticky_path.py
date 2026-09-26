"""Figure 1: one path of 60 independent uniform scenarios and one path of a sticky chain with t_mix = 20.

Panel (a) uses seed 0; panel (b) uses the first seed s = 0, 1, 2, ... whose path has round(1 + 59 q) fresh values, the mean
number of fresh values. Output: results/figure1_sticky_path.pdf/.png and results/figure1_sticky_path.json.
"""
import logging
import os

import _common as U  # noqa: F401  (sets the import path)
import matplotlib
import numpy as np

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)  # font subsetting notes on the bundled cmr10 font
import matplotlib.pyplot as plt  # noqa: E402

N = 60
TBAR = 20


def path(seed, q):
    rng = np.random.default_rng(seed)
    fresh = rng.random(N) < q
    fresh[0] = True
    vals = rng.random(N)
    idx = np.maximum.accumulate(np.where(fresh, np.arange(N), 0))
    return vals[idx], fresh


def main():
    q = 1.0 - 4.0 ** (-1.0 / TBAR)
    target = round(1 + (N - 1) * q)
    seed_b = next(s for s in range(10_000) if int(path(s, q)[1].sum()) == target)
    xa, fa = path(0, 1.0)
    xb, fb = path(seed_b, q)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(5.06, 1.95), constrained_layout=True, sharey=True)
    t = np.arange(1, N + 1)
    for ax, x, fr, title in ((axes[0], xa, fa, f"(a) independent, {int(fa.sum())} fresh values"),
                             (axes[1], xb, fb, f"(b) sticky, {int(fb.sum())} fresh values")):
        top = x.max()
        last = int(np.flatnonzero(x == top)[-1]) + 1
        ax.fill_between([-1.0, N + 2.5], top, 1.0, facecolor="#c6dbef", linewidth=0.0)
        ax.step(t, x, where="post", color="#bdbdbd", linewidth=0.5, zorder=1)
        ax.plot(t[~fr], x[~fr], ".", color="#737373", markersize=2.5, zorder=2)
        ax.plot(t[fr], x[fr], "o", color="#08306b", markersize=3.0, zorder=3)
        ax.plot([last], [top], "s", markerfacecolor="none", markeredgecolor="black", markersize=7, zorder=4)
        ax.axhline(top, color="black", linewidth=0.8, linestyle="--", zorder=2)
        ax.set_xlim(-1.0, N + 2.5)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([1, 20, 40, 60])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_xlabel("time $t$")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("scenario value")
    os.makedirs(U.RESULTS, exist_ok=True)
    out = os.path.join(U.RESULTS, "figure1_sticky_path.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06, metadata={"Creator": None, "Producer": None})
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight", pad_inches=0.06, metadata={"Software": None})
    rec = {"environment": U.environment(), "N": N, "tbar": TBAR, "q": q, "seed_a": 0, "seed_b": seed_b,
           "fresh_a": int(fa.sum()), "fresh_b": int(fb.sum()), "risk_a": float(1 - xa.max()), "risk_b": float(1 - xb.max()),
           "path_a": xa.tolist(), "path_b": xb.tolist()}
    U.save_json("figure1_sticky_path.json", rec)
    print(f"Figure 1: seed (b) = {seed_b}, fresh values {rec['fresh_a']} / {rec['fresh_b']}, risks "
          f"{rec['risk_a']:.4f} / {rec['risk_b']:.4f}; wrote {out}")


if __name__ == "__main__":
    main()
