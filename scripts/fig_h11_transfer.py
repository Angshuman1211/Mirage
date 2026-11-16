"""
H11 figure — what transfers across targets and what does not.

Left panel is the point of the whole test. W2 to the recipient's held-out anchor splits
exactly into orthogonal location and shape parts (W2^2 = |dmu|^2 + Bures^2), both in
units of the anchor's own sd. A pooled anchor-free width correction removes most of the
SHAPE error on every target and leaves the LOCATION error untouched to three decimal
places, and location is the part that dominates. So the transferable half of the
calibration is the half that does not matter.

Right panel is the hierarchy. Mean marginal JS for four classes of donor, from the
circular case the published result actually measured, through a change of random seed on
the same planet, to a different planet, against doing nothing at all.

    conda run -n mirage python scripts/fig_h11_transfer.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h11_ot_transfer import TARGETS, load, calibration, transfer, corr, _moments, score
from h11_w2 import w2
from h11_seedcheck import SEED_ARMS, load_seed
from h11_ot_transfer import gauss_draws

LBL = {"wasp39": "WASP-39b", "wasp96": "WASP-96b", "k218": "K2-18b"}


def main():
    tg = {k: load(k) for k in TARGETS}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- panel A: W2 decomposition ------------------------------------------
    # Lines, not stacked bars: W2^2 = loc^2 + shape^2 are ORTHOGONAL, and a stacked bar
    # on a log axis misrepresents both the segment heights and SELF's exactly-zero
    # location term.
    arms = [("RAW", "no\ncalibration"), ("pooled", "pooled,\nanchor-free"),
            ("SELF", "own anchor\n(circular)")]
    tcol = {"wasp39": "#2b4c7e", "wasp96": "#d98c3f", "k218": "#4f8a5b"}
    FLOOR = 0.05
    for bname, B in tg.items():
        D = np.diag(1.0 / B["sdn"])
        wmu, wS = D @ B["mun"], D @ B["Sn"] @ D
        muf, Sf = _moments(B["thf"])
        locs, shps = [], []
        for arm, _ in arms:
            if arm == "RAW":
                mu, S = muf, Sf
            elif arm == "pooled":
                others = [calibration(tg[a])["k"] for a in tg if a != bname]
                kp = np.exp(np.mean(np.log(np.array(others)), axis=0))
                sd = kp * B["sdf"]
                mu, S = muf, np.outer(sd, sd) * corr(Sf)
            else:
                mu, S = transfer(calibration(B), B, "M2")
            a, b = D @ mu, D @ S @ D
            locs.append(float(np.sqrt(np.sum((a - wmu) ** 2))))
            shps.append(w2(np.zeros(7), b, np.zeros(7), wS))
        xs = np.arange(3)
        c = tcol[bname]
        axA.plot(xs, np.maximum(locs, FLOOR), "-o", color=c, lw=2.2, ms=7,
                 label=f"{LBL[bname]}  location")
        axA.plot(xs, np.maximum(shps, FLOOR), "--s", color=c, lw=1.6, ms=6,
                 mfc="white", label=f"{LBL[bname]}  shape")
        if locs[-1] < FLOOR:                      # exactly 0 by construction
            axA.annotate("0", (xs[-1], FLOOR), textcoords="offset points",
                         xytext=(0, -13), ha="center", fontsize=8, color=c)
    axA.set_xticks(range(3))
    axA.set_xticklabels([t for _, t in arms], fontsize=9)
    axA.set_yscale("log")
    axA.set_ylim(FLOOR * 0.7, 60)
    axA.set_ylabel("distance to held-out anchor  (anchor sigmas)")
    axA.set_title("A. Solid lines barely move. Dashed lines collapse.\n"
                  "The transferable half of the calibration is the half that does not"
                  " matter.", fontsize=10)
    axA.legend(fontsize=7.5, ncol=1, loc="lower left")
    axA.grid(axis="y", alpha=0.25, which="both")

    # ---- panel B: the JS hierarchy -------------------------------------------
    classes = {"same target,\nsame run\n(circular)": [], "same target,\nnew seed": [],
               "new target": [], "no calibration": []}
    for bname, B in tg.items():
        classes["no calibration"].append(
            score(B["thf"], np.ones(len(B["thf"])), B)["mean_js"])
        for aname, A in tg.items():
            mu, S = transfer(calibration(A), B, "M2")
            x, _ = gauss_draws(mu, S, B["lo"], B["hi"])
            v = score(x, np.ones(len(x)), B)["mean_js"]
            classes["same target,\nsame run\n(circular)" if aname == bname
                    else "new target"].append(v)
    Sd = {k: load_seed(p) for k, p in SEED_ARMS.items()}
    cal = {k: calibration(v) for k, v in Sd.items()}
    for arm in ("rad_nocond",):          # matched to the architecture used above
        ks = [f"{arm}_s{s}" for s in (42, 43, 44)]
        for bk in ks:
            for ak in ks:
                if ak == bk:
                    continue
                mu, S = transfer(cal[ak], Sd[bk], "M2")
                x, _ = gauss_draws(mu, S, Sd[bk]["lo"], Sd[bk]["hi"])
                classes["same target,\nnew seed"].append(
                    score(x, np.ones(len(x)), Sd[bk])["mean_js"])
    rng = np.random.default_rng(1)
    for i, (k, v) in enumerate(classes.items()):
        v = np.array(v)
        axB.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=28, alpha=0.8,
                    color="#2b4c7e", zorder=3)
        axB.plot([i - 0.26, i + 0.26], [v.mean()] * 2, color="#d98c3f", lw=2.4, zorder=4)
    axB.axhline(np.log(2), ls=":", color="0.4", lw=1)
    axB.text(3.35, np.log(2) - 0.02, "ln2, disjoint", fontsize=7.5, ha="right", va="top",
             color="0.35")
    axB.set_xticks(range(len(classes)))
    axB.set_xticklabels(classes.keys(), fontsize=8.5)
    axB.set_ylabel("mean marginal JS to held-out anchor  (nats)")
    axB.set_title("B. The calibration's success is circularity, and it decays\n"
                  "with every step away from the run that fitted it.", fontsize=10)
    axB.grid(axis="y", alpha=0.25)
    axB.set_ylim(0, 0.75)

    fig.tight_layout()
    out = Path("figures/fig_h11_transfer.png")   # all figures live in figures/
    fig.savefig(out, dpi=200)
    print(f"[fig] {out}")
    for k, v in classes.items():
        print(f"  {k.replace(chr(10), ' '):42s} n={len(v):2d}  mean JS = {np.mean(v):.4f}"
              f"  range {np.min(v):.4f}-{np.max(v):.4f}")


if __name__ == "__main__":
    main()
