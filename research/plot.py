"""Figures from the held-out test results (results/test/*.jsonl, see evaluate.py).

figures/ratio_n3.png    fair-rule MSE / sampling MSE vs budget, one panel per noise level
figures/mse_<d>_3.png   fair-rule MSE (a step function) and sampling MSE vs budget, one model
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize import load, order_key

# categorical palette, fixed order (blue, orange, aqua, yellow, magenta, green, violet)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID_C, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
DS = [3, 4, 5, 6, 8, 16, 32]
SIGMAS = ["0.002", "0.005", "0.01"]
B = np.logspace(0, 4, 2001)


def path(d, sigma):
    return f"results/test/M_{d}_3_{sigma}_lguard.jsonl"


def fair_curve(names, truth, est, cost):
    """MSE of the fair-rule rung in use at each budget in B, and sampling MSE."""
    var = float((truth * (1 - truth)).mean())
    mse = ((est - truth[:, None]) ** 2).mean(0)
    c = cost.mean(0)
    ex = names.index("exact")
    rungs = sorted([i for i in range(len(names)) if i == ex or c[i] < c[ex]],
                   key=lambda i: order_key(names[i]))
    use = np.full(len(B), np.nan)
    for i in rungs:  # later rungs override earlier ones once affordable
        use[B >= c[i]] = mse[i]
    return use, var / B, c[ex]


def style(ax):
    ax.grid(True, which="major", color=GRID_C, lw=0.6)
    ax.tick_params(labelsize=7, colors=INK2)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)


def ratio_figure(out):
    fig, axes = plt.subplots(1, len(SIGMAS), figsize=(10.5, 3.6), sharey=True)
    for ax, sigma in zip(axes, SIGMAS):
        ax.axhline(1.0, color=REF, lw=1.2, ls=(0, (4, 3)))
        for d, col in zip(DS, SERIES):
            if not os.path.exists(path(d, sigma)):
                continue
            names, truth, est, cost = load(path(d, sigma))
            use, samp, exact_cost = fair_curve(names, truth, est, cost)
            r = use / samp
            r[B >= exact_cost] = np.nan  # exact: MSE ~ 0
            ax.plot(B, r, color=col, lw=1.6, label=f"d={d}")
            ax.axvline(exact_cost, color=col, lw=0.6, ls=":", alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 1e4)
        ax.set_ylim(1e-2, 1e2)
        ax.set_title(f"noise σ = {sigma}", fontsize=8.5, color=INK)
        ax.set_xlabel("compute budget (forward-pass equivalents)", fontsize=8, color=INK)
        ax.text(1.3, 1.2, "random sampling", fontsize=7, color=INK2)
        style(ax)
    axes[0].set_ylabel("MSE of our estimate / MSE of sampling", fontsize=8, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, fontsize=8, frameon=False,
               title="2nd-argmax RNNs with n = 3 inputs and hidden size d (200 held-out noisy models each)",
               title_fontsize=8.5)
    fig.text(0.5, 0.005, "Below 1 = lower MSE than random sampling at that budget (fair rule: most "
             "expensive affordable rung). Dotted lines: exact computation affordable (MSE ≈ 0) from there on.",
             ha="center", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    fig.savefig(out, dpi=170)
    print("saved", out)


def mse_figure(d, out):
    fig, axes = plt.subplots(1, len(SIGMAS), figsize=(10.5, 3.3), sharey=True)
    for ax, sigma in zip(axes, SIGMAS):
        if not os.path.exists(path(d, sigma)):
            continue
        names, truth, est, cost = load(path(d, sigma))
        use, samp, exact_cost = fair_curve(names, truth, est, cost)
        ax.plot(B, samp, color=INK2, lw=1.5, ls=(0, (4, 3)), label="random sampling")
        before = B < exact_cost
        ax.plot(B[before], use[before], color=SERIES[0], lw=1.8, label="our estimator (fair rule)")
        ax.scatter([exact_cost], [1e-11], color=SERIES[0], marker="v", s=30, zorder=3)
        ax.annotate("exact from here\n(MSE ≈ 0)", (exact_cost, 1e-11), textcoords="offset points",
                    xytext=(6, 2), fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 1e4)
        ax.set_ylim(3e-12, 1e-1)
        ax.set_title(f"σ = {sigma}, mean accuracy {truth.mean():.4f}", fontsize=8.5, color=INK)
        ax.set_xlabel("compute budget (forward-pass equivalents)", fontsize=8, color=INK)
        style(ax)
    axes[0].set_ylabel("MSE of accuracy estimate", fontsize=8, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle(f"2nd-argmax RNN d={d}, n=3: MSE vs compute over 200 held-out noisy models per panel",
                 fontsize=9, color=INK)
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out, dpi=170)
    print("saved", out)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    ratio_figure("figures/ratio_n3.png")
    for d in [int(a) for a in sys.argv[1:]] or [32, 16]:
        mse_figure(d, f"figures/mse_{d}_3.png")
