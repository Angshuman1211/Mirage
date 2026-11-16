"""
A2 figure. Left, the injection-fitted deflation against the anchor-fitted one, per
parameter and target. Right, importance-sampling efficiency by route.

The two panels carry the two separate claims. Left, injections demand far less width
deflation than the real anchor does, so the simulator does not reproduce the real
miscalibration. Right, the anchor-fitted calibration wins on marginal agreement and
loses as a proposal, which is what circularity looks like from the functional side.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from h11_ot_transfer import NAMES, load, calibration          # noqa: E402
from a2_inject_calibrate import A2, ARMS, _pulls, _sk, _boot_k  # noqa: E402

TCOL = {"wasp39": "#1b6ca8", "wasp96": "#c8553d", "k218": "#2a9d8f"}
TLAB = {"wasp39": "WASP-39b", "wasp96": "WASP-96b", "k218": "K2-18b"}
MARK = ["o", "s", "^", "D", "v", "P", "X"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.2))

# ---- A: injection-fitted vs anchor-fitted deflation -------------------------------
lims = [0.04, 1.35]
axA.plot(lims, lims, color="0.35", lw=1.1, zorder=1)
axA.text(0.56, 0.50, "equal", color="0.35", fontsize=9, rotation=38,
         transform=axA.transAxes, ha="center", va="center")
for f, lab in ((3.0, "3x"), (10.0, "10x")):
    axA.plot(lims, [f * v for v in lims], color="0.75", lw=0.9, ls=":", zorder=1)
    xa = lims[1] / f * 0.80
    axA.annotate(lab, (xa, f * xa), color="0.6", fontsize=8.2, rotation=38,
                 ha="center", va="center",
                 bbox=dict(fc="white", ec="none", pad=0.6))
for planet in ARMS:
    p = A2 / f"{planet}_injmoments.npz"
    if not p.exists():
        continue
    d = np.load(p, allow_pickle=True)
    z, _ = _pulls(d)
    m = int(d["n_samples"])
    _, k_i, _, _, _ = _sk(z, m)
    klo, khi = _boot_k(z, m)
    k_a = calibration(load(planet))["k"]
    for i in range(len(NAMES)):
        axA.errorbar(k_a[i], k_i[i],
                     yerr=[[k_i[i] - klo[i]], [khi[i] - k_i[i]]],
                     fmt=MARK[i], ms=7, mfc=TCOL[planet], mec="white", mew=0.8,
                     ecolor=TCOL[planet], elinewidth=1.1, capsize=0, zorder=3)
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlim(*lims); axA.set_ylim(*lims)
axA.set_xlabel("deflation demanded by the real NS anchor   $k_{\\rm anchor}$")
axA.set_ylabel("deflation demanded by injections   $k_{\\rm inj}$")
axA.set_title("A   the simulator understates the miscalibration", loc="left", fontsize=11)
h_t = [plt.Line2D([], [], ls="", marker="o", mfc=TCOL[p], mec="white", ms=7,
                  label=TLAB[p]) for p in ARMS]
h_p = [plt.Line2D([], [], ls="", marker=MARK[i], mfc="0.45", mec="white", ms=6,
                  label=NAMES[i]) for i in range(len(NAMES))]
axA.legend(handles=h_t + h_p, fontsize=7.6, ncol=2, loc="lower right", framealpha=0.95)

# ---- B: IS efficiency by route ----------------------------------------------------
rows = json.load(open(A2 / "a2_scores.json"))
ROUTES = [("no calibration", "0.55"),
          ("circular (anchor-fitted)", "#8c6bb1"),
          ("INJECTION-fitted (A2)", "#d95f02")]
RLAB = {"no calibration": "none", "circular (anchor-fitted)": "anchor-fitted\n(circular)",
        "INJECTION-fitted (A2)": "injection-fitted\n(A2)"}
by = {}
for r in rows:
    by.setdefault(r["target"], {})[r["route"]] = r

W = 0.26
for j, planet in enumerate(ARMS):
    for i, (route, col) in enumerate(ROUTES):
        r = by.get(planet, {}).get(route)
        if not r or r["ess"] is None:
            continue
        x = j + (i - 1) * W
        lo, hi = r["ess_lo"], r["ess_hi"]
        axB.bar(x, r["ess"], W * 0.88, color=col, zorder=2,
                label=RLAB[route] if j == 0 else None)
        if lo is not None:
            axB.errorbar(x, r["ess"], yerr=[[r["ess"] - lo], [hi - r["ess"]]],
                         fmt="none", ecolor="0.15", elinewidth=1.2, capsize=3, zorder=3)
axB.axhline(50, color="#b22222", lw=1.2, ls="--", zorder=1)
axB.text(2.46, 53, "usable, ESS > 50", color="#b22222", fontsize=8.5, ha="right")
axB.set_yscale("log")
axB.set_ylim(3, 160)
axB.set_xticks(range(len(ARMS)))
axB.set_xticklabels([TLAB[p] for p in ARMS])
axB.set_ylabel("effective sample size, out of 2000 draws")
axB.set_title("B   marginal agreement is not sampling efficiency", loc="left", fontsize=11)
axB.legend(fontsize=8.2, loc="upper left", framealpha=0.95)
for s in ("top", "right"):
    axA.spines[s].set_visible(False); axB.spines[s].set_visible(False)

fig.tight_layout()
out = A2 / "fig_a2_injection.png"
fig.savefig(out, dpi=200)
print(f"[fig] {out}")
