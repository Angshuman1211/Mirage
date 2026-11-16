"""
A4 figure. Top, best-fit chi2/N against the error floor, which says what floor the data
supports. Bottom, importance-sampling efficiency against the same axis.

The two rows together carry the finding. The floor the data supports is 0.7 to 1.4
percent, the pipeline inherited 5 percent, and every arm's efficiency collapses once the
floor is pulled back to what the data actually wants.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("data/a4")
rows = json.load(open(OUT / "a4_floor_sweep.json"))
PLANETS = ["wasp39", "wasp96", "k218"]
TLAB = {"wasp39": "WASP-39b", "wasp96": "WASP-96b", "k218": "K2-18b"}
ACOL = {"none": "0.45", "circular": "#8c6bb1", "A2": "#d95f02"}
ALAB = {"none": "no calibration", "circular": "anchor-fitted (circular)",
        "A2": "injection-fitted (A2)"}
PUBLISHED = 5.0

by = {}
for r in rows:
    by.setdefault((r["target"], r["arm"]), []).append(r)
for k in by:
    by[k].sort(key=lambda r: r["floor"])


def crossing(f, c):
    f, c = np.asarray(f, float), np.asarray(c, float)
    for i in range(len(f) - 1):
        if (c[i] - 1) * (c[i + 1] - 1) <= 0 and c[i] != c[i + 1]:
            t = (1 - c[i]) / (c[i + 1] - c[i])
            return float(np.exp(np.log(f[i]) + t * (np.log(f[i + 1]) - np.log(f[i]))))
    return None


fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.4), sharex=True)
for j, p in enumerate(PLANETS):
    ax0, ax1 = axes[0, j], axes[1, j]
    xs = [100 * r["floor"] for r in by[(p, "none")]]
    cross = [crossing([r["floor"] for r in by[(p, a)]], [r["chi2_n"] for r in by[(p, a)]])
             for a in ACOL]
    cross = [c for c in cross if c]
    if cross:
        ax0.axvspan(100 * min(cross), 100 * max(cross), color="#2a9d8f", alpha=0.16,
                    zorder=0)
        ax1.axvspan(100 * min(cross), 100 * max(cross), color="#2a9d8f", alpha=0.16,
                    zorder=0)
    for ax in (ax0, ax1):
        ax.axvline(PUBLISHED, color="#b22222", lw=1.3, ls="--", zorder=1)
    ax0.axhline(1.0, color="0.25", lw=1.1, zorder=1)
    for a in ACOL:
        d = by[(p, a)]
        ax0.plot(xs, [r["chi2_n"] for r in d], "-o", ms=4, color=ACOL[a], zorder=3,
                 label=ALAB[a] if j == 0 else None)
        e = np.array([r["eps"] for r in d])
        lo = np.array([r["eps_lo"] for r in d])
        hi = np.array([r["eps_hi"] for r in d])
        ax1.plot(xs, e, "-o", ms=4, color=ACOL[a], zorder=3,
                 label=ALAB[a] if j == 0 else None)
        ax1.fill_between(xs, lo, hi, color=ACOL[a], alpha=0.16, lw=0, zorder=2)
    ax0.set_xscale("log"); ax0.set_yscale("log")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax0.set_title(TLAB[p], fontsize=11)
    ax1.set_xlabel("error floor, percent of depth")
    ax0.set_ylim(8e-3, 8)
    ax1.set_ylim(0.03, 9)
    for ax in (ax0, ax1):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([0.5, 1, 2, 3, 5, 7, 10])
        ax.set_xticklabels(["0.5", "1", "2", "3", "5", "7", "10"])
axes[0, 0].set_ylabel("best-fit $\\chi^2/N$")
axes[1, 0].set_ylabel("IS efficiency $\\epsilon$ = ESS/N, percent")
axes[0, 0].text(0.52, 1.25, "$\\chi^2/N=1$", fontsize=8.4, color="0.25")
axes[0, 2].text(PUBLISHED * 1.1, 3.0, "published\nfloor", fontsize=8.2, color="#b22222")
axes[0, 0].text(0.92, 0.012, "floor the\ndata supports", fontsize=8.2, color="#1d6f66")
axes[0, 0].legend(fontsize=8.2, loc="upper right", framealpha=0.95)
fig.suptitle("A4   the published 5 percent error floor is 4 to 7 times what the data supports, "
             "and efficiency collapses at the honest floor", fontsize=11.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = Path("figures") / "fig_a4_floor.png"   # all figures live in figures/
fig.savefig(out, dpi=200)
print(f"[fig] {out}")
