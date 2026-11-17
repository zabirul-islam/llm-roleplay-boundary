#!/usr/bin/env python3
"""Paper figures (Nature Communications) from the committed results JSON only. figures4papers house style
(PALETTE, FigureStyle, apply_publication_style, finalize_figure; scientific-figure-making skill) at Nature print size:
7 pt type, 183 mm double-column width. PDF + PNG 300 dpi.

    python3 figures/make_figs.py            # writes figures/fig2..fig5 (fig5_replication).{pdf,png}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_1": "#DDF3DE", "green_2": "#AADCA9",
           "green_3": "#8BCF8B", "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342", "neutral": "#CFCECE",
           "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E"}
MM = 1 / 25.4
# role-play arms = blues (the method under test); direct rating = red (comparator); text baselines = neutral
COL = {"A2": PALETTE["blue_main"], "A1": PALETTE["blue_secondary"], "A0": PALETTE["teal"], "B4": PALETTE["red_strong"],
       "B1": PALETTE["neutral"], "B2": PALETTE["neutral"], "B3": PALETTE["neutral"]}


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 7
    axes_linewidth: float = 0.6
    font_family: tuple = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def apply_publication_style(s: FigureStyle = FigureStyle()) -> None:
    plt.rcParams.update({"font.size": s.font_size, "font.family": "sans-serif", "font.sans-serif": list(s.font_family),
                         "axes.linewidth": s.axes_linewidth, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": s.font_size, "axes.labelsize": s.font_size, "xtick.labelsize": s.font_size - 0.5,
                         "ytick.labelsize": s.font_size - 0.5, "legend.fontsize": s.font_size - 0.5, "legend.frameon": False,
                         "xtick.major.width": s.axes_linewidth, "ytick.major.width": s.axes_linewidth,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight"})


def finalize_figure(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, pad_inches=0.02)
    plt.close(fig)


def panel(ax, letter: str) -> None:
    ax.text(-0.18, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def load(p):
    return json.load(open(ROOT / p))


H1, SEC, DBE, EXP = load("results/hh/h1.json"), load("results/hh/secondary.json"), load("results/dbe/results-dbe.json"), \
    load("results/explore/explore-postlock.json")
AS = load("results/assist/results-assist.json")
LAB = {"A2": "Role-play A2\n(anchored)", "A1": "Role-play A1\n(population)", "A0": "Role-play A0\n(fixed level)",
       "B4": "Direct rating\nB4", "B1": "Text 2PL\nB1", "B2": "Embedding\nB2", "B3": "TF-IDF\nB3"}


def fig2():
    d = SEC["difficulty"]
    rho = {"A2": d["A2 t0.6"]["rho"], "A1": d["A1 t0.8"]["rho"], "A0": d["A0 t0.8"]["rho"], "B4": d["B4"]["rho"],
           "B1": d["B1_anchor"]["rho"], "B2": d["B2"]["rho"], "B3": d["B3"]["rho"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM), gridspec_kw={"width_ratios": [1.25, 1]})
    keys = list(rho); x = np.arange(len(keys))
    a.bar(x, [rho[k] for k in keys], color=[COL[k] for k in keys], width=0.7, edgecolor="none")
    reps = H1["estimates"]["rhoS_r"]
    a.scatter(np.full(len(reps), 0) + np.linspace(-0.18, 0.18, len(reps)), reps, s=6, color="black", zorder=3, label="replicates")
    a.set_xticks(x); a.set_xticklabels(keys); a.set_ylabel("Spearman ρ with real item difficulty")
    a.text(0.0, -0.17, "Role-play: A2 anchored, A1 population, A0 fixed level · B4 direct rating · B1–B3 text baselines", transform=a.transAxes, fontsize=6, color="0.3")
    a.set_ylim(0, 0.5); a.legend(loc="upper right"); panel(a, "a")
    a.set_title("EEDI held-out items (n = 130)", loc="left")
    rows = [("A2 − B4 (H1)", SEC["E-anchor A2 vs baselines"]["B4"]), ("A1 − B4", SEC["E-pop A1 vs B4"]),
            ("A2 − B1", SEC["E-anchor A2 vs baselines"]["B1_anchor"]), ("A2 − B2", SEC["E-anchor A2 vs baselines"]["B2"]),
            ("A2 − B3", SEC["E-anchor A2 vs baselines"]["B3"])]
    for i, (lab, r) in enumerate(rows):
        lo, hi = r["ci95"]; y = len(rows) - 1 - i
        c = PALETTE["blue_main"] if i == 0 else PALETTE["blue_secondary"]
        b.plot([lo, hi], [y, y], color=c, lw=1.4 if i == 0 else 0.9); b.plot(r["delta"], y, "o", color=c, ms=3.5 if i == 0 else 2.5)
    b.axvline(0, color="grey", lw=0.5); b.axvline(0.10, color=PALETTE["red_strong"], lw=0.8, ls="--")
    b.text(0.105, len(rows) - 0.55, "registered margin 0.10", color=PALETTE["red_strong"], fontsize=6)
    b.set_yticks(range(len(rows))); b.set_yticklabels([r[0] for r in rows][::-1]); b.set_xlabel("Δρ with 95% basic interval")
    b.set_xlim(-0.35, 0.5); b.set_ylim(-0.6, len(rows) - 0.2); panel(b, "b")
    b.set_title(f"H1 p = {H1['p']['H1']:.2f} (not rejected); others secondary", loc="left")
    fig.tight_layout(); finalize_figure(fig, "fig2_eedi_h1")


def fig3():
    d = SEC["difficulty"]; keys = [("A2", "A2 t0.6"), ("A1", "A1 t0.8"), ("A0", "A0 t0.8"), ("B4", "B4"), ("B1", "B1_anchor"),
                                   ("B2", "B2"), ("B3", "B3")]
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 52 * MM))
    for ax, (m, lab, ref) in zip(axs, [("calSlope", "Calibration slope", 1.0),
                                       ("mae", "Mean absolute error", None), ("bias", "Mean bias (pred. − obs.)", 0.0)]):
        x = np.arange(len(keys)); v = [d[k][m] for _, k in keys]
        ax.bar(x, v, color=[COL[s] for s, _ in keys], width=0.7)
        if ref is not None:
            ax.axhline(ref, color="grey", lw=0.6, ls="--")
        ax.set_xticks(x); ax.set_xticklabels([s for s, _ in keys]); ax.set_ylabel(lab)
    for ax, l in zip(axs, "abc"):
        panel(ax, l)
    fig.suptitle("EEDI: role-play overestimates accuracy and is poorly calibrated", x=0.01, ha="left", fontsize=7)
    fig.tight_layout(); finalize_figure(fig, "fig3_calibration")


def fig4():
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 55 * MM), gridspec_kw={"width_ratios": [1, 0.85, 1.35]})
    a, b, c = axs
    # each arm paired with uniform guessing on the same (arm-estimable) items, per replicate (distractor_pairs.py)
    dp = load("results/hh/distractor-pairs.json")["arms"]
    keys = [("A2", "A2 t0.6"), ("A1", "A1 t0.8"), ("A0", "A0 t0.8")]
    for i, (lab, k) in enumerate(keys):
        a.bar(i - 0.19, dp[k]["armJS"], width=0.36, color=COL[lab])
        a.bar(i + 0.19, dp[k]["uniformJS"], width=0.36, color=PALETTE["neutral"])
        a.scatter(i - 0.19 + np.linspace(-0.08, 0.08, 5), dp[k]["armJS_r"], s=4, color="black", zorder=3)
        a.text(i, max(dp[k]["armJS_r"]) + 0.015, f"{min(dp[k]['estimable'])}–{max(dp[k]['estimable'])}", ha="center", fontsize=5.5)
    a.legend(handles=[matplotlib.patches.Patch(color=PALETTE["neutral"], label="uniform guessing,\nsame items")],
             frameon=False, fontsize=5.5, loc="upper right")
    a.set_xticks(range(3)); a.set_xticklabels([k for k, _ in keys]); a.set_ylabel("JS divergence from real wrong answers")
    a.set_ylim(0, 0.42)
    a.set_title(f"Wrong answers ({dp['A2 t0.6']['eligible']} eligible items)", loc="left"); panel(a, "a")
    w = EXP["EEDI wrong answers"]
    b.bar([0, 1], [w["top wrong option share, real (mean)"], w["top wrong option share, sim (mean)"]],
          color=[PALETTE["red_strong"], COL["A2"]], width=0.6)
    b.axhline(1 / 3, color="grey", lw=0.6, ls="--"); b.text(1.35, 1 / 3 + 0.01, "uniform", fontsize=6, color="grey")
    b.set_xticks([0, 1]); b.set_xticklabels(["Real students", "Role-play A2"]); b.set_ylim(0, 1)
    b.set_ylabel("Share of wrong answers on the\nmost common wrong option"); b.set_title("Error concentration (exploratory)", loc="left")
    panel(b, "b")
    h2 = H1["h2"]["arms"]
    ee = {arm: [math.exp(r["medianLogRatio"]) for r in h2[arm]] for arm in ("S", "A0")}
    cv = DBE["cohort variance ratio (sim/real; all held-out students, not classes)"]["primary"]
    groups = [("EEDI\nA2", ee["S"], COL["A2"]), ("EEDI\nA0", ee["A0"], COL["A0"]),
              ("DBE\nA2", cv["a2-t0.6"]["ratioPerReplicate"], COL["A2"]),
              ("DBE\nA0", cv["a0-t0.8"]["ratioPerReplicate"], COL["A0"])]
    acv = AS["cohort variance ratio (sim/real; all held-out students)"]
    groups += [("ASSIST\nA2", acv["a2-t0.6"]["ratioPerReplicate"], COL["A2"]),
               ("ASSIST\nA0", acv["a0-t0.8"]["ratioPerReplicate"], COL["A0"])]
    for i, (lab, v, col) in enumerate(groups):
        c.bar(i, np.mean(v), color=col, width=0.65, alpha=0.85)
        c.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, s=5, color="black", zorder=3)
    c.axhline(1, color="grey", lw=0.6, ls="--"); c.set_xticks(range(len(groups))); c.set_xticklabels([g[0] for g in groups])
    c.axvline(1.5, color="grey", lw=0.5)   # EEDI: exp(class-median log ratio) | DBE/ASSIST: whole-cohort ratio
    c.text(0.5, 1.02, "class median", transform=c.get_xaxis_transform(), ha="center", fontsize=5.5, color="grey")
    c.text(3.5, 1.02, "whole cohort", transform=c.get_xaxis_transform(), ha="center", fontsize=5.5, color="grey")
    c.set_ylabel("Score variance, simulated / real"); c.set_title("Spread of student scores", loc="left", pad=9); panel(c, "c")
    fig.tight_layout(); finalize_figure(fig, "fig4_fidelity")


SHORT = {"A2": "A2\nanchored", "A1": "A1\npopulation", "A0": "A0\nfixed level", "B4": "B4\ndirect rating",
         "B1": "B1\ntext 2PL", "B2": "B2\nembedding", "B3": "B3\nTF-IDF"}


def fig5():
    e, d, s = SEC["difficulty"], DBE["difficulty"], AS["difficulty"]
    keys = [("A2", "A2 t0.6", "a2-t0.6"), ("A1", "A1 t0.8", "a1-t0.8"), ("A0", "A0 t0.8", "a0-t0.8"), ("B4", "B4", "B4"),
            ("B1", "B1_anchor", "B1_anchor"), ("B2", "B2", "B2"), ("B3", "B3", "B3")]
    sets = [(f"EEDI ({e['B4']['items']} items, registered)", lambda k, j: e[k]["rho"], 1.0, None),
            (f"FoundationalASSIST ({s['B4']['items']} items, descriptive)", lambda k, j: s[j]["rho"], 0.6, "...."),
            (f"DBE-KT22 ({DBE['items']} items, descriptive)", lambda k, j: d[j]["rho"], 0.4, "////")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM), gridspec_kw={"width_ratios": [1.9, 1]})
    x = np.arange(len(keys)); wdt = 0.27
    for i, (lab, get, alpha, hatch) in enumerate(sets):
        a.bar(x + (i - 1) * wdt, [get(k, j) for _, k, j in keys], wdt, color=[COL[c] for c, _, _ in keys], alpha=alpha,
              hatch=hatch, edgecolor="white", linewidth=0.3, label=lab)
    a.axhline(0, color="grey", lw=0.5); a.set_xticks(x); a.set_xticklabels([SHORT[c] for c, _, _ in keys])
    a.set_ylabel("Spearman \u03c1 with real item difficulty"); a.set_ylim(-0.05, 0.8)
    a.legend(loc="upper left", handlelength=1.2); panel(a, "a")
    a.set_title("Item ranking by predictor and dataset", loc="left")
    per = EXP["DBE audit"]["per item (real p, sim p, B4)"]
    rp = np.array([v[0] for v in per.values()]); sp = np.array([v[1] for v in per.values()])
    b.scatter(rp, sp, s=8, color=COL["A2"], alpha=0.8)
    b.plot([0, 1], [0, 1], color="grey", lw=0.5, ls="--"); b.set_xlim(0, 1.02); b.set_ylim(-0.02, 1.05)
    b.set_xlabel("Real proportion correct"); b.set_ylabel("Role-play A2 proportion correct")
    b.set_title(f"DBE-KT22: ceiling on {np.mean(sp >= 0.99):.0%} of items (exploratory)", loc="left"); panel(b, "b")
    fig.tight_layout(); finalize_figure(fig, "fig5_replication")


if __name__ == "__main__":
    apply_publication_style()
    OUT.mkdir(exist_ok=True)
    for f in (fig2, fig3, fig4, fig5):
        f()
    print("figures written:", sorted(p.name for p in OUT.glob("fig*.png")))
