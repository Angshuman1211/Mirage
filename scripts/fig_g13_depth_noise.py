"""
G13 figure. Depth noise from re-running the light-curve fit, against the flux
approximation Tier 1 currently uses.

Three panels, one per claim:
  left   amplitude. Both depth constructions sit BELOW the published bar while
         the white-noise control sits on it, which is the prayer-bead power
         deficit and not a property of the data.
  middle structure. Correlation versus bin separation. Both depth constructions
         are far more correlated than the flux route, which is the finding.
  right  why the amplitude panel reads low. Residual power against white, with
         the depth Jacobian's own weight overplotted.

    conda run -n mirage python scripts/fig_g13_depth_noise.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402

IN = Path("data/inject")
ARMS = [("wasp96_depth_noise.npz", "depth, prayer bead", "C0", "-"),
        ("wasp96_depth_noise_oot.npz", "depth, OOT blocks", "C2", "-"),
        ("wasp96_depth_noise_iid.npz", "white control", "C3", "--")]


def corr_vs_sep(C, kmax=25):
    return np.array([np.median(np.diag(C, k)) for k in range(1, kmax + 1)])


def main():
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
    z0 = np.load(IN / ARMS[0][0], allow_pickle=True)
    wl, sig_pub = z0["wlen"], z0["sigma_published"]

    # ---- amplitude -------------------------------------------------------
    ax[0].plot(wl, sig_pub * 1e6, color="k", lw=2, label="published error bar")
    for fn, lab, c, ls in ARMS:
        z = np.load(IN / fn, allow_pickle=True)
        r = z["ratio_to_published"]
        ax[0].plot(z["wlen"], z["sigma"] * 1e6, color=c, ls=ls, lw=1.3,
                   label=f"{lab} ({r:.2f}x)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("wavelength (um)")
    ax[0].set_ylabel("depth uncertainty (ppm)")
    ax[0].set_title("amplitude")
    ax[0].legend(fontsize=7.5, loc="upper left")

    # ---- structure -------------------------------------------------------
    for fn, lab, c, ls in ARMS:
        z = np.load(IN / fn, allow_pickle=True)
        ax[1].plot(np.arange(1, 26), corr_vs_sep(z["corr"]), color=c, ls=ls,
                   lw=1.5, label=lab)
    fz = np.load(IN / "wasp96_noise.npz", allow_pickle=True)
    nzf = fz["noise"][:, fz["covered"] & ~fz["gap"]]
    ax[1].plot(np.arange(1, 26), corr_vs_sep(np.corrcoef(nzf, rowvar=False)),
               color="k", lw=2, label="flux route, Tier 1 today")
    ax[1].axhline(0, color="0.6", lw=0.7)
    ax[1].set_xlabel("bin separation")
    ax[1].set_ylabel("median correlation")
    ax[1].set_title("cross-wavelength structure")
    ax[1].legend(fontsize=7.5)

    # ---- mechanism -------------------------------------------------------
    b = ax[2]
    bands = [(0, 4, 0.130, 0.123), (4, 8, 0.589, 0.803),
             (8, 20, 1.577, 0.059), (20, 48, 1.000, 0.016)]
    x = np.arange(len(bands))
    b.bar(x - 0.2, [p for _, _, p, _ in bands], 0.4, color="C0",
          label="residual power / white")
    b.bar(x + 0.2, [w for _, _, _, w in bands], 0.4, color="C1",
          label="depth Jacobian weight")
    b.axhline(1.0, color="k", lw=0.8, ls=":")
    b.set_xticks(x)
    b.set_xticklabels([f"{lo}-{hi}" for lo, hi, _, _ in bands])
    b.set_xlabel("frequency band (cycles / day)")
    b.set_title("why amplitude reads low")
    b.legend(fontsize=7.5)
    b.annotate(f"predicted {z0['bead_iid_predicted']:.3f}\nmeasured 0.68",
               xy=(0.97, 0.62), xycoords="axes fraction", ha="right", fontsize=8)

    fig.tight_layout()
    out = Path("figures") / "fig_g13_depth_noise.png"      # all figures live in figures/
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
