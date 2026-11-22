"""V5 / H7 figure. Every one of the 21 G395H reduction pairs, in units of the
single-bin error, with the cross-team pairs marked.

All 21 are plotted rather than a selected 8, because P6-D36 was a selected subset
drifting from its own JSON. Shape RMS leads because a constant offset is absorbed
by the fitted nuisance offset every retrieval already carries, so it is the shape
that reaches the science.

    conda run -n mirage python scripts/fig_v5h7_pairs.py
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHIFT = json.load(open("configs/v5_k218_h7/spectral_shift.json"))
LBL = {"schmidt_exotedrf": "S exoTEDRF", "schmidt_eureka_ahrer": "S Eureka! Ahrer",
       "schmidt_eureka_bell": "S Eureka! Bell", "schmidt_firefly": "S FIREFLy",
       "luque_exotedrf": "L exoTEDRF", "luque_eureka": "L Eureka!",
       "luque_sparta": "L SPARTA"}
C_IN, C_X = "#4c72b0", "#c44e52"

rows = []
for k, v in SHIFT["pairs_g395h"].items():
    a, b = k.split("__vs__")
    rows.append(dict(a=a, b=b, cross=a.split("_")[0] != b.split("_")[0],
                     shape=v["shape_rms_over_err"], off=v["abs_offset_over_err"],
                     shape_ppm=v["shape_rms_ppm"], off_ppm=v["offset_ppm"]))
rows.sort(key=lambda r: r["shape"])
y = np.arange(len(rows))
names = [f"{LBL[r['a']]}  vs  {LBL[r['b']]}" for r in rows]
cols = [C_X if r["cross"] else C_IN for r in rows]
med = st.median([r["shape"] for r in rows])
# counted, not asserted
n_over = sum(r["shape"] >= 1.0 for r in rows)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.6, 7.2), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.06))

ax.axvline(1.0, color="0.35", lw=1.2, ls="--", zorder=1)
ax.axvline(med, color="#2a9d8f", lw=1.3, zorder=1)
ax.barh(y, [r["shape"] for r in rows], height=0.62, color=cols, alpha=0.9, zorder=3)
for i, r in enumerate(rows):
    ax.text(r["shape"] + 0.03, i, f"{r['shape_ppm']:.1f} ppm", va="center",
            fontsize=8.4, color="0.25", zorder=5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.2))
ax.set_xlim(0, max(r["shape"] for r in rows) * 1.30)
ax.set_xlabel("shape RMS / single-bin error", fontsize=11)
ax.set_title("shape, the part a fitted offset cannot absorb", fontsize=11)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.2)
ax.set_ylim(-1.0, len(rows) - 0.2)
ax.text(0.99, -0.72, "one error bar", fontsize=8.4, color="0.35", ha="right")
ax.text(med + 0.03, -0.72, f"median {med:.2f}", fontsize=8.4, color="#1d6f66")

bx.axvline(1.0, color="0.35", lw=1.2, ls="--", zorder=1)
bx.barh(y, [r["off"] for r in rows], height=0.62, color=cols, alpha=0.55, zorder=3)
for i, r in enumerate(rows):
    bx.text(r["off"] + 0.025, i, f"{r['off_ppm']:+.1f}", va="center",
            fontsize=8.6, color="0.25", zorder=4)
bx.set_xlim(0, 1.25)
bx.set_xlabel("|mean offset| / single-bin error", fontsize=11)
bx.set_title("offset, absorbable by the nuisance offset already fitted", fontsize=11)

h = [plt.Rectangle((0, 0), 1, 1, color=C_X, alpha=0.9),
     plt.Rectangle((0, 0), 1, 1, color=C_IN, alpha=0.9)]
bx.legend(h, ["cross-team pair, two groups, same photons", "within-team pair"],
          fontsize=9.0, loc="lower right", framealpha=0.95)
for a in (ax, bx):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

fig.suptitle("V5 / H7   all 21 pairs of 7 independent reductions of one K2-18b G395H transit\n"
             "the worst shape disagreement is a cross-team pair, and "
             f"{n_over} of {len(rows)} sit at or above one error bar",
             fontsize=12, y=0.978, linespacing=1.5)
fig.subplots_adjust(left=0.175, right=0.985, top=0.858, bottom=0.095)
out = Path("figures") / "fig_v5h7_pairs.png"
fig.savefig(out, dpi=200)
print(f"[fig] {out}")
print(f"  {len(rows)} pairs, {sum(r['cross'] for r in rows)} cross-team")
print(f"  shape/err  {rows[0]['shape']:.3f} to {rows[-1]['shape']:.3f}, median {med:.3f}")
print(f"  worst pair {names[-1]}  ({'cross' if rows[-1]['cross'] else 'within'}-team)")
print(f"  {n_over} of {len(rows)} pairs at or above one error bar")
