"""The astronomy paper's one figure. Three things that can move a JWST spectrum,
measured on one grid against one error bar (P6-D37).

Left, the offset-removed residual against wavelength. Shape is plotted rather than
the raw difference because a constant offset is absorbed by a fitted nuisance offset
or by a floating planet radius, so it is the shape that survives into the science.
Right, the same three effects as a single ordering.

Every number is read from the committed artifacts, never typed in, because P6-D36
was a decision entry drifting from its own JSON.

    conda run -n mirage python scripts/fig_h8_three_level.py
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHIFT = json.load(open("configs/v5_k218_h7/spectral_shift.json"))
RES = np.load("data/h8/h8_residuals_k218.npz")

pairs = list(SHIFT["pairs_g395h"].values())
red = sorted(v["shape_rms_ppm"] for v in pairs)
ERR = st.median([v["median_single_err_ppm"] for v in pairs])
R_MIN, R_MED, R_MAX = red[0], st.median(red), red[-1]
# sigma for the reduction bar comes from the JSON's own per-pair shape_rms_over_err,
# NOT from R_MED/ERR. Each pair carries its own error bar, so the median of the ratios
# (1.105) is not the ratio of the medians (1.15), and 1.105 is what P6-D32 and P6-D36
# quote. The figure has to agree with the text.
rsig = sorted(v["shape_rms_over_err"] for v in pairs)
S_MIN, S_MED, S_MAX = rsig[0], st.median(rsig), rsig[-1]

wl = RES["wl"]
code = RES["code"][0] - RES["code"][0].mean()            # offset removed
lists = RES["line_list"][0] - RES["line_list"][0].mean()
C_RMS, L_RMS = float(code.std()), float(lists.std())

C_RED, C_LST, C_COD = "#2a9d8f", "#d95f02", "#3b4cc0"

fig = plt.figure(figsize=(13.2, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.24)
ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ---- left: residual spectra, with the reduction envelope behind them
ax.axhspan(-R_MED, R_MED, color=C_RED, alpha=0.18, lw=0, zorder=0)
for s in (-1, 1):
    ax.axhline(s * R_MAX, color=C_RED, lw=1.0, ls=":", zorder=1)
ax.axhline(0, color="0.3", lw=0.9, zorder=1)
ax.plot(wl, lists, "-", lw=1.5, color=C_LST, zorder=3,
        label=f"opacity tables, same code   rms {L_RMS:.1f} ppm")
ax.plot(wl, code, "-", lw=1.8, color=C_COD, zorder=4,
        label=f"radiative transfer code   rms {C_RMS:.1f} ppm")
BOX = dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.8)
ax.text(2.83, R_MED * 0.62, f"reduction choice, median {R_MED:.1f} ppm",
        fontsize=9.5, color="#1d6f66", zorder=6, bbox=BOX)
ax.text(2.83, R_MAX * 1.07, f"reduction, worst pair {R_MAX:.1f} ppm",
        fontsize=8.8, color="#1d6f66", zorder=6, bbox=BOX)
ax.set_xlim(wl.min(), wl.max())
ax.set_ylim(-R_MAX * 1.45, R_MAX * 1.45)
ax.set_xlabel("wavelength, $\\mu$m", fontsize=11)
ax.set_ylabel("offset-removed residual, ppm", fontsize=11)
ax.set_title("K2-18b, NIRSpec G395H, 55 log-spaced bins", fontsize=11)
ax.legend(fontsize=9.6, loc="lower right", framealpha=0.95)

# ---- right: the ordering, in units of the single-bin error
names = ["who reduced\nthe data", "which opacity\ntables", "which radiative\ntransfer code"]
vals = [S_MED, L_RMS / ERR, C_RMS / ERR]
lo_hi = [(S_MIN, S_MAX), None, None]
cols = [C_RED, C_LST, C_COD]
y = np.arange(3)[::-1]
bx.barh(y, vals, height=0.52, color=cols, alpha=0.88, zorder=3)
for i, (v, r) in enumerate(zip(vals, lo_hi)):
    if r:
        bx.plot([r[0], r[1]], [y[i], y[i]], color="0.2", lw=1.6, zorder=4)
        bx.plot([r[0], r[1]], [y[i], y[i]], "|", color="0.2", ms=9, zorder=4)
    # when a row carries a span, the value label goes past its right whisker,
    # otherwise it sits just past the bar
    bx.text((r[1] if r else v) + 0.045, y[i], f"{v:.2f}$\\sigma$",
            va="center", fontsize=10.5, zorder=5)
bx.axvline(1.0, color="0.35", lw=1.1, ls="--", zorder=2)
bx.text(1.02, -0.62, "one error bar", fontsize=8.8, color="0.35")
bx.set_yticks(y); bx.set_yticklabels(names, fontsize=10)
bx.set_ylim(-0.85, 2.5)
bx.set_xlim(0, max(S_MAX, 1.0) * 1.24)
bx.set_xlabel(f"shape RMS / single-bin error ({ERR:.1f} ppm)", fontsize=10.5)
bx.set_title(f"reduction = {R_MED/L_RMS:.1f}x the opacity swap, {R_MED/C_RMS:.0f}x the code",
             fontsize=10.5)

for a in (ax, bx):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

fig.suptitle(
    f"H8   which team reduced the data moves a JWST spectrum {R_MED/C_RMS:.0f} times more "
    "than which radiative transfer code models it,\nand twice as much as swapping the "
    "opacity tables for a coarser set", fontsize=12, y=0.975, linespacing=1.45)
fig.subplots_adjust(left=0.055, right=0.985, top=0.800, bottom=0.105)
out = Path("figures") / "fig_h8_three_level.png"
fig.savefig(out, dpi=200)
print(f"[fig] {out}")
print(f"  reduction {R_MED:.1f} ppm ({S_MED:.3f} sigma), span {R_MIN:.1f}-{R_MAX:.1f} ppm "
      f"/ {S_MIN:.3f}-{S_MAX:.3f} sigma")
print(f"  line list {L_RMS:.1f} ppm ({L_RMS/ERR:.2f} sigma)")
print(f"  code      {C_RMS:.1f} ppm ({C_RMS/ERR:.3f} sigma)")
