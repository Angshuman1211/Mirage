"""Multi-target figure, v2 — corrects the presentation defect found in P5-D7.

What was wrong with v1 (`fig_multitarget.py`):
  it drew data error bars at the RAW measurement sigma while the fit was evaluated
  against sqrt(sig^2 + (floor*depth)^2). At the 5% floor that is 37.5x larger for
  WASP-39b, and the forward-model systematic dominates 100% of bins on all three
  targets. A residual the likelihood sees as 1 sigma therefore appeared as ~37 on
  the page, which is why a reviewer read good fits (chi2/N 0.66-1.44, zero bins
  past 3 sigma) as "really bad fits, several features completely off".

What v2 does instead:
  - data keeps its RAW sigma, which is the honest measurement precision;
  - the forward-model systematic is added to the MODEL band, where it belongs,
    in quadrature with the posterior-predictive spread (P3-D3 validated the TauREx
    forward model to ~5% median, so the prediction genuinely is not better than that);
  - a residual row in units of sigma_eff makes fit quality directly readable;
  - ESS is annotated per target, so a weight-collapsed posterior is visible on the
    page rather than presented as a credible interval (P5-D7).

Default floor is 1%, standardised in P5-D4.

    conda run -n mirage python scripts/fig_multitarget_v2.py [--floor 0.01]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path("data/real_ess")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
NS_ORDER = ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
ESS_WARN = 20.0

PLANETS = [
    ("WASP-39b", "hot Saturn  ·  NIRSpec PRISM", "abl_rad_nocond", "log_q",
     "wasp39b_ns_posterior_fitrad.npz", 1.279, "#c1440e"),
    ("WASP-96b", "hot Saturn  ·  NIRISS SOSS (self-reduced)", "wasp96", "log_q",
     "wasp39b_ns_posterior_wasp96.npz", 1.200, "#1f6feb"),
    ("K2-18b", "cold sub-Neptune  ·  NIRISS SOSS", "k218cov", "log_q",
     "wasp39b_ns_posterior_k218.npz", 0.2352, "#2ea043"),
]


def wquant(v, w, q):
    i = np.argsort(v)
    return float(np.interp(q, np.cumsum(w[i]), v[i]))


def is_weights(prefix, logp_key, floor):
    s = np.load(OUT / f"{prefix}_samples.npz", allow_pickle=True)
    f = np.load(OUT / f"{prefix}_forward.npz", allow_pickle=True)
    th, x, sig, cov, wl = s["theta"], s["x_obs"], s["sig_obs"], s["covered"], s["wlen"]
    ms = f["model_spec"]
    m = cov & np.isfinite(sig) & (sig > 0)
    se = np.sqrt(sig ** 2 + (floor * x) ** 2)
    val = np.isfinite(ms[:, m]).all(1)
    r = (x[None, m] - ms[:, m]) / se[None, m]
    ll = np.where(val, -0.5 * np.nansum(r ** 2, 1), -np.inf)
    w = np.exp(ll - s[logp_key] - np.nanmax(ll - s[logp_key])); w /= w.sum()
    chi2 = -2 * ll.max() / m.sum()
    return th, w, x, sig, se, m, wl, ms, float(1 / np.sum(w ** 2)), float(chi2)


def ns_radius(npz):
    d = np.load(OUT / npz, allow_pickle=True)
    i = list(d["fit_names"]).index("planet_radius")
    th = d["samples"][:, i]; w = d["weights"] / d["weights"].sum()
    o = np.argsort(th); c = np.cumsum(w[o])
    return (float(np.sum(w * th)), float(np.interp(0.16, c, th[o])), float(np.interp(0.84, c, th[o])))


def main(floor):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11),
                             gridspec_kw={"height_ratios": [2.2, 1.0, 1.6]})

    for c, (name, sub, prefix, lk, nsnpz, truth, col) in enumerate(PLANETS):
        th, w, x, sig, se, m, wl, ms, ess, chi2 = is_weights(prefix, lk, floor)
        wlc = wl[m]; o = np.argsort(wlc)

        # posterior-predictive band from parameter spread, then widened by the
        # forward-model systematic -- the systematic belongs to the MODEL.
        lo = np.array([wquant(ms[:, m][:, k], w, 0.16) for k in range(m.sum())])
        hi = np.array([wquant(ms[:, m][:, k], w, 0.84) for k in range(m.sum())])
        md = np.array([wquant(ms[:, m][:, k], w, 0.50) for k in range(m.sum())])
        # Full posterior-predictive: parameter spread + forward-model systematic +
        # measurement noise. All three must be in it, because the question the band
        # answers is "could this DATA POINT have come from this model", so the
        # fraction of points inside is then directly comparable to 68 / 95%.
        half = np.sqrt(((hi - lo) / 2) ** 2 + (floor * md) ** 2 + sig[m] ** 2)
        # show 1 AND 2 sigma. A 68% band excludes ~1/3 of points BY CONSTRUCTION,
        # which reads as a bad fit even when the fit is good -- the same misreading
        # v1 caused, one level up. The 2-sigma band is what "does the model agree"
        # should be judged against.
        half2 = 2.0 * half

        # ---- row 1: spectrum ----
        ax = axes[0, c]
        ax.errorbar(wlc[o], x[m][o] * 1e2, yerr=sig[m][o] * 1e2, fmt="o", ms=3.5, lw=0.9,
                    color="0.20", ecolor="0.55", capsize=0, zorder=3,
                    label="JWST (real), measurement $\\sigma$")
        ax.fill_between(wlc[o], (md - half2)[o] * 1e2, (md + half2)[o] * 1e2, color=col,
                        alpha=0.14, lw=0, zorder=1, label="MIRAGE 2$\\sigma$")
        ax.fill_between(wlc[o], (md - half)[o] * 1e2, (md + half)[o] * 1e2, color=col,
                        alpha=0.32, lw=0, zorder=2,
                        label=f"MIRAGE 1$\\sigma$ (incl. {floor:.0%} fwd-model syst.)")
        ax.plot(wlc[o], md[o] * 1e2, color=col, lw=1.4, zorder=2)
        ax.set_title(f"{name}\n{sub}", fontsize=13)
        ax.set_xlabel("wavelength [µm]", fontsize=11)
        if c == 0:
            ax.set_ylabel("transit depth  (R$_p$/R$_\\star$)$^2$  [%]", fontsize=11)
        if c == 0:                     # identical across panels; one legend only,
            ax.legend(fontsize=9, loc="upper left", framealpha=0.92)   # rest stay clear of data
        ax.tick_params(labelsize=10)
        ax.text(0.98, 0.04, f"best-fit $\\chi^2$/N = {chi2:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10,
                bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=2.5))

        # ---- row 2: residuals in sigma_eff ----
        ax = axes[1, c]
        resid = (x[m] - md) / se[m]
        ax.axhspan(-1, 1, color="0.88", lw=0, zorder=0)
        for lv, ls in ((2, "--"), (3, ":")):
            ax.axhline(lv, color="0.55", ls=ls, lw=0.8, zorder=1)
            ax.axhline(-lv, color="0.55", ls=ls, lw=0.8, zorder=1)
        ax.axhline(0, color="0.3", lw=0.9, zorder=1)
        ax.plot(wlc[o], resid[o], "o", ms=3.5, color=col, zorder=3)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("wavelength [µm]", fontsize=11)
        if c == 0:
            ax.set_ylabel("residual  [$\\sigma_{\\rm eff}$]", fontsize=11)
        ax.tick_params(labelsize=10)
        n3 = int((np.abs(resid) > 3).sum())
        ax.text(0.98, 0.92, f"|r|>3$\\sigma$: {n3}/{m.sum()}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9.5,
                bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=2.0))

        # ---- row 3: radius posterior ----
        ax = axes[2, c]
        rp = th[:, 0]
        r16, r50, r84 = (wquant(rp, w, q) for q in (0.16, 0.50, 0.84))
        lo_r, hi_r = wquant(rp, w, 0.005), wquant(rp, w, 0.995)
        span = max(hi_r - lo_r, 1e-4)
        edges = np.linspace(min(lo_r, truth) - 0.25 * span, max(hi_r, truth) + 0.25 * span, 60)
        ax.hist(rp, bins=edges, weights=w, color=col, alpha=0.55, density=True,
                label="MIRAGE posterior")
        nsm, ns16, ns84 = ns_radius(nsnpz)
        ax.axvline(nsm, color="0.25", ls="--", lw=1.4, label=f"NS anchor {nsm:.3f}")
        ax.axvspan(ns16, ns84, color="0.25", alpha=0.12, lw=0)
        ax.axvline(truth, color="crimson", ls="-", lw=1.6, label=f"literature {truth:.3f}")
        inside = r16 <= truth <= r84
        ax.set_xlabel("planet radius  [R$_{\\rm Jup}$]", fontsize=11)
        if c == 0:
            ax.set_ylabel("posterior density", fontsize=11)
        ax.set_title(f"MIRAGE {r50:.3f} [{r16:.3f}, {r84:.3f}]   "
                     f"lit {'INSIDE' if inside else 'OUTSIDE'}", fontsize=11.5,
                     color="0.15" if inside else "crimson")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.tick_params(labelsize=10)
        warn = "  — interval not credible" if ess < ESS_WARN else ""
        ax.text(0.02, 0.96, f"ESS = {ess:.1f}{warn}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5,
                color="crimson" if ess < ESS_WARN else "0.25",
                bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=2.0))

    fig.suptitle("One radius-inference method, three regimes — real JWST retrievals "
                 f"(all quantities at a {floor:.0%} forward-model error floor)",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = FIG / "fig6_multitarget_v2.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.01)
    main(ap.parse_args().floor)
