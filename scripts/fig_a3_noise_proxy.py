"""
A3 figure. The conditioning paradox does not come from the noise proxy.

  left   every Sigma-hat tried lands on the same real-data JS, far from the
         unconditioned arm. The paradox is the vertical gap and nothing in the
         proxy closes it.
  middle flat against the structure axis H5 names, with G13's measured depth
         noise marked, so the span is shown to cover the regime in question.
  right  why. The embedding responds strongly, but it is a small block of the
         context, so the posterior cannot follow.

    conda run -n mirage python scripts/fig_a3_noise_proxy.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402

KINDS = ["white", "depth_oot", "real", "depth", "depth_oot_direct",
         "depth_direct", "rank1"]
LABEL = {"white": "white\n(diagonal)", "depth_oot": "depth OOT\ntoeplitz",
         "real": "real OOT flux\n(what Tier 1 uses)", "depth": "depth bead\ntoeplitz",
         "depth_oot_direct": "depth OOT\nmeasured",
         "depth_direct": "depth bead\nmeasured",
         "rank1": "rank 1\n(impossible)"}
SEEDS = (42, 43, 44)


def main():
    z = np.load("data/a3/a3_scores.npz", allow_pickle=True)
    struct = z["structure"][0]
    pr = z["probe"][0]
    js = {k: np.array([z[f"h9_rad_cov_s{s}__{k}"][0] for s in SEEDS]) for k in KINDS}
    nc = np.array([z[f"h9_rad_nocond_s{s}__nocond"][0] for s in SEEDS])

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # ---- left ------------------------------------------------------------
    x = np.arange(len(KINDS))
    for i, k in enumerate(KINDS):
        ax[0].plot([i] * 3, js[k], "o", color="C0", ms=5, alpha=0.75)
        ax[0].plot([i - 0.22, i + 0.22], [js[k].mean()] * 2, color="C0", lw=2.5)
    ax[0].axhspan(nc.mean() - nc.std(ddof=1), nc.mean() + nc.std(ddof=1),
                  color="C2", alpha=0.18)
    ax[0].axhline(nc.mean(), color="C2", lw=2, label="no conditioning")
    ax[0].annotate("", xy=(4.4, nc.mean()), xytext=(4.4, js["real"].mean()),
                   arrowprops=dict(arrowstyle="<->", color="0.3"))
    ax[0].text(4.3, 0.5 * (nc.mean() + js["real"].mean()),
               f"the paradox\n+{js['real'].mean() - nc.mean():.3f}",
               ha="right", va="center", fontsize=8.5)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([LABEL[k] for k in KINDS], fontsize=7,
                          rotation=30, ha="right")
    ax[0].set_ylabel("real WASP-39b mean 1-D JS to NS anchor")
    ax[0].set_title("every $\\hat{\\Sigma}$ gives the same answer")
    ax[0].legend(fontsize=8, loc="center left")

    # ---- middle ----------------------------------------------------------
    sx = np.array([struct[k] for k in KINDS])
    sy = np.array([js[k].mean() for k in KINDS])
    se = np.array([js[k].std(ddof=1) for k in KINDS])
    o = np.argsort(sx)
    ax[1].errorbar(sx[o], sy[o], yerr=se[o], marker="o", color="C0", lw=1.5, capsize=3)
    ax[1].axhline(nc.mean(), color="C2", lw=2, label="no conditioning")
    ax[1].axvspan(0.223, 0.476, color="C3", alpha=0.15,
                  label="G13 measured depth noise")
    ax[1].axvline(0.159, color="k", ls=":", lw=1.2, label="G13 flux route")
    ax[1].axvspan(0.0, np.sqrt(2 / (np.pi * 100)), color="0.6", alpha=0.25,
                  label="K=100 sampling floor")
    ax[1].set_xlabel("mean |off-diagonal| of the $\\hat{\\Sigma}$ fed in")
    ax[1].set_ylabel("mean 1-D JS")
    ax[1].set_title("flat along the axis H5 names")
    ax[1].legend(fontsize=7.5)

    # ---- right -----------------------------------------------------------
    ks = [k for k in KINDS if k != "real"]
    emb = np.array([100 * pr["moves"][k] / pr["norm_cov"] for k in ks])
    pos = np.array([100 * abs(js[k].mean() - js["real"].mean())
                    / (js["real"].mean() - nc.mean()) for k in ks])
    w = 0.38
    xx = np.arange(len(ks))
    ax[2].bar(xx - w / 2, emb, w, color="C0", label="move in the covariance embedding")
    ax[2].bar(xx + w / 2, pos, w, color="C3", label="resulting move in the posterior")
    ax[2].set_yscale("log")
    ax[2].set_xticks(xx)
    ax[2].set_xticklabels([LABEL[k].replace("\n", " ") for k in ks], fontsize=7,
                          rotation=30, ha="right")
    ax[2].set_ylabel("percent (log scale)")
    sh = pr["shares"]
    ax[2].set_title(f"the channel works, the block is "
                    f"{100 * min(sh.values()):.0f}-{100 * max(sh.values()):.0f}% "
                    f"of the context")
    ax[2].legend(fontsize=7.5)

    fig.tight_layout()
    out = Path("figures") / "fig_a3_noise_proxy.png"    # all figures live in figures/
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
