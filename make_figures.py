"""Draws the result figures from the tables: main.pdf (accuracy and gap vs epsilon) and curves.pdf (accuracy per round).

usage: python make_figures.py [--tables results/tables] [--out results/figures]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

TEXT_W = 7.16
FONT = 7.5
SUBFL, SUBFL_LIGHT, DENSE = "#1f4e9e", "#a9c1e8", "#d1541f"
INK = "#000000"
EPS = [4, 8, 16, 32]
X = np.log2(EPS)
SEEDS = ("1234", "2345", "3456", "4567", "5678")
NAME = {"scratch": "From scratch", "warm": "Warm-started"}
DENSE_LABEL = "Dense (client-side DP-SGD)"
ARM = {"subfl": dict(color=SUBFL, marker="o", ls="-", label="SubFL"),
       "dense": dict(color=DENSE, marker="s", ls=(0, (4, 1.5)), label=DENSE_LABEL)}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": FONT,
    "axes.labelsize": FONT, "axes.titlesize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
    "legend.fontsize": FONT, "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.top": True, "ytick.right": True, "axes.labelpad": 1.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "legend.frameon": False, "legend.handlelength": 2.4, "legend.columnspacing": 1.6, "legend.borderaxespad": 0.2,
    "lines.linewidth": 1.0, "lines.markersize": 2.8, "errorbar.capsize": 1.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 100, "savefig.dpi": 300,
})


def table(tables, name):
    with open(tables / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def signed(v):
    return format(v, "+.1f").replace("-", "−")


def gap_label(t):
    return (f"$\\Delta={signed(100 * float(t['diff']))}$ "
            f"[{signed(100 * float(t['ci_lo']))}, {signed(100 * float(t['ci_hi']))}]")


def save(fig, path):
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    print("wrote", path)


def caption_below(ax, text, y=-0.3):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top")


def main_figure(tables, out):
    cells = {(r["init"], int(r["eps"]), r["arm"]): r for r in table(tables, "main_cells")}
    tests = {(r["init"], int(r["eps"])): r for r in table(tables, "main_tests")}
    ceil = {r["init"]: 100 * float(r["mean"]) for r in table(tables, "ceilings")}
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 1.9))
    fig.subplots_adjust(left=0.052, right=0.985, top=0.885, bottom=0.26, wspace=0.17)
    for ax, init, tag in ((axes[0], "scratch", "(a)"), (axes[1], "warm", "(b)")):
        for arm, z in (("subfl", 3), ("dense", 2)):
            m = 100 * np.array([float(cells[(init, e, arm)]["mean"]) for e in EPS])
            sd = 100 * np.array([float(cells[(init, e, arm)]["sd"]) for e in EPS])
            ax.errorbar(X, m, yerr=sd, ms=2.0, zorder=z, **ARM[arm])
        ax.set_ylabel("Test accuracy (%)")
        ax.text(0.97, 0.06, f"Non-private: {ceil[init]:.1f}% (off scale)", transform=ax.transAxes, ha="right")
        caption_below(ax, f"{tag} {NAME[init]}", y=-0.27)
    axes[0].set_ylim(40, 62)
    axes[1].set_ylim(65, 80)

    # gap bars in SubFL's blue: dark from scratch, light warm-started
    ax = axes[2]
    for init, color, off in (("scratch", SUBFL, -0.15), ("warm", SUBFL_LIGHT, 0.15)):
        m = 100 * np.array([float(tests[(init, e)]["diff"]) for e in EPS])
        lo = 100 * np.array([float(tests[(init, e)]["ci_lo"]) for e in EPS])
        hi = 100 * np.array([float(tests[(init, e)]["ci_hi"]) for e in EPS])
        ax.bar(X + off, m, width=0.3, color=color, ec=INK, lw=0.5, label=NAME[init], zorder=2)
        eb = ax.errorbar(X + off, m, yerr=[m - lo, hi - m], color=INK, ls="none", lw=0.7, zorder=3)
        if init == "scratch":  # white edge keeps the whiskers readable on the dark bars
            for a in (*eb.lines[1], *eb.lines[2]):
                a.set_path_effects([patheffects.withStroke(linewidth=1.3, foreground="white")])
    ax.set_ylabel("SubFL $-$ Dense (%)")
    ax.set_ylim(0, 9.2)
    ax.legend(loc="upper left", ncol=2, handlelength=1.4, columnspacing=1.0)
    caption_below(ax, "(c) Gap, with 95% CI", y=-0.27)
    for ax in axes:
        ax.set_xticks(X)
        ax.set_xticklabels([str(e) for e in EPS])
        ax.set_xlim(X[0] - 0.25, X[-1] + 0.25)
        ax.set_xlabel("Privacy budget $\\varepsilon$")
    axes[2].set_xlim(X[0] - 0.55, X[-1] + 0.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.355, 1.015))
    save(fig, out / "main.pdf")


def curves_figure(tables, out):
    tests = {(r["init"], int(r["eps"])): r for r in table(tables, "main_tests")}
    niid = {(r["init"], r["alpha"]): r for r in table(tables, "noniid_tests")}
    curves = defaultdict(lambda: defaultdict(dict))
    for r in table(tables, "curves"):
        curves[(r["setting"], r["arm"])][r["seed"]][int(r["round"])] = float(r["acc"])
    rounds = np.array([*range(1, 100, 2), 100])
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W, 3.15), sharex=True)
    fig.subplots_adjust(left=0.052, right=0.985, top=0.93, bottom=0.155, wspace=0.17, hspace=0.27)
    n = 0
    for row, (init, ylim) in enumerate([("scratch", (10, 62)), ("warm", (60, 80))]):
        columns = [(f"iid_{init}_e16", tests[(init, 16)]),
                   (f"noniid_{init}_a0.5", niid[(init, "0.5")]),
                   (f"noniid_{init}_a0.1", niid[(init, "0.1")])]
        for col, (setting, t) in enumerate(columns):
            ax = axes[row, col]
            split = "IID" if col == 0 else f"$\\alpha={t['alpha']}$"
            for arm in ("subfl", "dense"):
                per_seed = np.array([[curves[(setting, arm)][s][r] for r in rounds] for s in SEEDS])
                m, s = 100 * per_seed.mean(0), 100 * per_seed.std(0, ddof=1)
                ax.fill_between(rounds, m - s, m + s, color=ARM[arm]["color"], alpha=0.18, lw=0, zorder=2)
                ax.plot(rounds, m, markevery=6, ms=2.2, lw=1.0, zorder=3, **ARM[arm])
            ax.set_xlim(0, 100)
            ax.set_ylim(*ylim)
            ax.text(0.97, 0.06, gap_label(t), transform=ax.transAxes, ha="right")
            if col == 0:
                ax.set_ylabel("Test accuracy (%)")
            if row == 1:
                ax.set_xlabel("Communication round")
            n += 1
            caption_below(ax, f"({'abcdef'[n - 1]}) {NAME[init]}, {split}", y=-0.05 if row == 0 else -0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.004))
    save(fig, out / "curves.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables", type=Path, default=Path("results/tables"), help="the CSVs written by make_tables.py")
    ap.add_argument("--out", type=Path, default=Path("results/figures"), help="where the PDFs are written")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    main_figure(args.tables, args.out)
    curves_figure(args.tables, args.out)
