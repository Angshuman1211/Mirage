"""
Phase 4.4 — publication figures for the real-WASP-39b result. Writes PNGs to figures/.
  1. radius_is_the_fix : best-fit χ²/dof across every forward-model lever — all physics
     probes stay cold (~2.5) or forced-hot (~38); freeing the RADIUS drops it to 0.76.
  2. mirage_vs_ns      : OT-calibrated MIRAGE posterior vs the independent NS anchor.
  3. spectrum_fit      : real WASP-39b spectrum + MIRAGE best-fit model.

    conda activate mirage && python scripts/make_figures.py
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("data/real_ess"); FIG = Path("figures"); FIG.mkdir(exist_ok=True)
BLUE, RED, GREY, GREEN = "#2c6fbb", "#c0392b", "#7f8c8d", "#27ae60"


def _chi2(f):
    d = np.load(OUT / f, allow_pickle=True); return float(d["chi2"])


def fig1():
    # (label, chi2, group)  group: physics-lever(cold), forced-hot, radius-fix
    probes = [("isothermal", 2.55, 0), ("+T-gradient", 2.65, 0), ("+clouds+grad", 2.616, 0),
              ("+SO₂", 2.555, 0), ("+hi-fi opacities\n(ExoMolOP R15000)", 2.416, 0),
              ("forced-physical\n(isothermal)", 38.441, 1), ("forced-physical\n(+gradient)", 38.478, 1),
              ("FREE RADIUS", 0.761, 2)]
    labels = [p[0] for p in probes]; vals = [p[1] for p in probes]
    colors = {0: GREY, 1: RED, 2: GREEN}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(range(len(vals)), vals, color=[colors[p[2]] for p in probes], width=0.68)
    ax.set_yscale("log")
    ax.axhline(1.0, ls="--", lw=1, color="k", alpha=0.5)
    ax.text(len(vals) - 0.5, 1.05, "χ²/dof = 1 (good fit)", ha="right", va="bottom", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylabel("best-fit χ²/dof on real WASP-39b")
    ax.set_title("The sim-to-real gap is the radius, not forward-model fidelity", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, RED, GREEN)]
    ax.legend(handles, ["forward-model levers (cold-allowed)", "forced-physical", "free the radius"],
              fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig1_radius_is_the_fix.png", dpi=160); plt.close(fig)
    print("  fig1_radius_is_the_fix.png")


def fig2():
    # OT-calibrated MIRAGE (weighted) vs NS anchor, marginals for radius/T/logH2O
    s = np.load(OUT / "ot_cal_samples.npz"); f = np.load(OUT / "ot_cal_forward.npz")
    th, lg = s["theta"], s["log_g"]; x, sig, cov, ms = s["x_obs"], s["sig_obs"], s["covered"], f["model_spec"]
    m = cov & np.isfinite(sig) & (sig > 0); se = np.sqrt(sig ** 2 + (0.05 * x) ** 2)
    val = np.isfinite(ms[:, m]).all(1); r = (x[None, m] - ms[:, m]) / se[None, m]
    ll = np.where(val, -0.5 * np.nansum(r ** 2, 1), -np.inf)
    w = np.exp(ll - lg - np.nanmax(ll - lg)); w /= w.sum()
    d = np.load(OUT / "wasp39b_ns_posterior_fitrad.npz", allow_pickle=True)
    o = ["planet_radius", "T", "log_H2O"]; idx = [list(d["fit_names"]).index(k) for k in o]
    nsw = d["weights"] / d["weights"].sum(); nss = d["samples"][:, idx]
    cols = [(0, "radius (R$_J$)", 1.27), (1, "T (K)", None), (2, "log H$_2$O", None)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (ci, name, truth) in zip(axes, cols):
        lo = min(th[:, ci].min(), nss[:, {0:0,1:1,2:2}[ci]].min())
        hi = max(th[:, ci].max(), nss[:, {0:0,1:1,2:2}[ci]].max())
        b = np.linspace(lo, hi, 40)
        ax.hist(nss[:, {0:0,1:1,2:2}[ci]], b, weights=nsw, density=True, color=GREY, alpha=0.55, label="NS anchor")
        ax.hist(th[:, ci], b, weights=w, density=True, histtype="step", color=BLUE, lw=2, label="MIRAGE (OT-cal)")
        if truth: ax.axvline(truth, color=GREEN, ls="--", lw=1.5, label="literature")
        ax.set_xlabel(name); ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("OT-calibrated MIRAGE posterior vs independent nested-sampling (7/7 coverage)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig2_mirage_vs_ns.png", dpi=160); plt.close(fig)
    print("  fig2_mirage_vs_ns.png")


def fig3():
    s = np.load(OUT / "ot_cal_samples.npz"); f = np.load(OUT / "ot_cal_forward.npz")
    x, sig, cov, wl, ms = s["x_obs"], s["sig_obs"], s["covered"], s["wlen"], f["model_spec"]
    m = cov & np.isfinite(sig) & (sig > 0); se = np.sqrt(sig ** 2 + (0.05 * x) ** 2)
    val = np.isfinite(ms[:, m]).all(1); r = (x[None, m] - ms[:, m]) / se[None, m]
    ll = np.where(val, -0.5 * np.nansum(r ** 2, 1), -np.inf); bi = np.argmax(ll)
    o = np.argsort(wl)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    mm = m[o]; wlo = wl[o]
    ax.errorbar(wlo[mm], x[o][mm] * 1e6, yerr=sig[o][mm] * 1e6, fmt="o", ms=4, color="k",
                capsize=2, lw=1, label="JWST WASP-39b (real)")
    ax.plot(wlo[mm], ms[bi][o][mm] * 1e6, "-", color=RED, lw=2, label="MIRAGE best-fit (χ²/dof=0.03)")
    ax.set_xlabel("wavelength (µm)"); ax.set_ylabel("transit depth (ppm)")
    ax.set_title("MIRAGE physical fit to real WASP-39b (radius + OT calibration)", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig3_spectrum_fit.png", dpi=160); plt.close(fig)
    print("  fig3_spectrum_fit.png")


def fig4():
    # full 7-param corner: OT-calibrated MIRAGE posterior vs NS anchor
    import corner
    s = np.load(OUT / "ot_cal_samples.npz")          # OT-calibrated proposal = calibrated posterior
    th = s["theta"]
    d = np.load(OUT / "wasp39b_ns_posterior_fitrad.npz", allow_pickle=True)
    o = ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
    idx = [list(d["fit_names"]).index(k) for k in o]
    nsw = d["weights"] / d["weights"].sum(); nss = d["samples"][:, idx]
    labels = ["R (R$_J$)", "T (K)", "log H₂O", "log CO₂", "log CH₄", "log CO", "log NH₃"]
    rng = [(min(th[:, i].min(), nss[:, i].min()), max(th[:, i].max(), nss[:, i].max())) for i in range(7)]
    fig = corner.corner(nss, weights=nsw, labels=labels, color=GREY, range=rng,
                        plot_datapoints=False, hist_kwargs={"density": True})
    corner.corner(th, fig=fig, color=BLUE, labels=labels, range=rng,
                  plot_datapoints=False, hist_kwargs={"density": True})
    fig.suptitle("Calibrated MIRAGE (blue) vs independent NS anchor (grey) — full 7-param posterior",
                 fontsize=13)
    fig.savefig(FIG / "fig4_corner.png", dpi=140); plt.close(fig)
    print("  fig4_corner.png")


def fig5():
    """ABC + radius noise-conditioning ablation, rebuilt from the H9 arms.

    The published version of this figure read `abc_ablation.npz`: ONE seed, and
    with the no-cond arms trained on MPS while the conditioned arms trained on
    CUDA (P6-D1). Both no-cond runs were under-trained, so the 0.587 baseline is
    the coverage of a model that did not train, and the apparent conditioning win
    was a device artifact (P6-D12). This version reads the eighteen device-matched
    H9 arms, three training seeds per family per arm, and plots the seed spread
    the single-seed figure could not show.
    """
    SEEDS = (42, 43, 44)
    ARMS = [("no-cond", "no_cond", GREY),
            ("+σ", "psigma", "#e67e22"),
            ("+σ+cov", "psigmapcov", BLUE)]
    # Published single-seed values, kept on the plot as the claim that did not
    # reproduce rather than deleted. ABC from abc_ablation.npz, radius from the
    # 07 section 4.2 table. The two ABC conditioned arms were identical at 0.687.
    PUB = {"abc": {"no_cond": 0.587, "psigma": 0.687, "psigmapcov": 0.687},
           "rad": {"no_cond": 0.817, "psigmapcov": 0.700}}
    FAM = [("abc", "ABC (6-param)"), ("rad", "radius (7-param, used on real data)")]

    # Measured by re-drawing every posterior for the two arms that existed
    # before this re-run and recomputing coverage. rms 0.0118, max 0.0233 over
    # twelve arm-seed pairs, with the trained weights and the test planets held
    # fixed. It is a floor on what panels (a) and (b) can resolve, and it is the
    # same size as the seed-to-seed spread, so most of that spread is draw noise
    # rather than training-seed variation.
    DRAW_NOISE = 0.0118

    def cov68(s, tr):
        lo = np.quantile(s, 0.16, axis=1); hi = np.quantile(s, 0.84, axis=1)
        return float(np.mean((lo <= tr) & (tr <= hi)))

    data = {}
    for fam, _ in FAM:
        per = {}
        for _, key, _c in ARMS:
            vals, gaps = [], []
            for sd in SEEDS:
                f = OUT / f"h9_{fam}_s{sd}.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)
                if "s__" + key not in z.files:
                    continue
                vals.append(cov68(z["s__" + key], z["truths"]))
                if key != "no_cond":
                    ps = z["planet_sigma"]
                    b = np.digitize(ps, np.quantile(ps, [1 / 3, 2 / 3]))
                    d = z["ld__" + key] - z["ld__no_cond"]
                    gaps.append([d[b == k].mean() for k in range(3)])
            if vals:
                per[key] = (np.array(vals), np.array(gaps) if gaps else None)
        data[fam] = per

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    for a, (fam, title) in zip(ax[:2], FAM):
        present = [(lab, key, c) for lab, key, c in ARMS if key in data[fam]]
        for i, (lab, key, c) in enumerate(present):
            v = data[fam][key][0]
            a.bar(i, v.mean(), color=c, width=0.6, alpha=0.85)
            a.errorbar(i, v.mean(), yerr=DRAW_NOISE, color="k", lw=1.4,
                       capsize=5, zorder=2,
                       label="posterior-draw noise" if i == 0 else None)
            a.plot([i] * len(v), v, "o", color="k", ms=4, zorder=3)
            a.text(i, v.mean() + 0.055, f"{v.mean():.3f}", ha="center", fontsize=9)
            if key in PUB[fam]:
                a.plot(i, PUB[fam][key], "X", color="crimson", ms=11, zorder=4,
                       label="published, 1 seed" if i == 0 else None)
        a.axhline(0.68, ls="--", color="k", lw=1)
        a.text(-0.45, 0.70, "nominal 68%", ha="left", fontsize=8)
        a.set_xticks(range(len(present)))
        a.set_xticklabels([p[0] for p in present])
        a.set_ylabel("coverage @ 68%"); a.set_ylim(0, 1)
        a.set_title(f"{title}\ndots = training seeds, bars = draw noise",
                    fontsize=9.5)
        a.legend(fontsize=7.5, loc="lower left")

    # Four bars per tercile, both conditioned arms of both families, so the
    # gradient is tested three ways and not just on the arm the paper plotted.
    BARS = [("abc", "psigma", "ABC +\u03c3", -0.30, "#f0b27a"),
            ("abc", "psigmapcov", "ABC +\u03c3+cov", -0.10, "#e67e22"),
            ("rad", "psigma", "radius +\u03c3", 0.10, "#85c1e9"),
            ("rad", "psigmapcov", "radius +\u03c3+cov", 0.30, BLUE)]
    for fam, key, lab, dx, col in BARS:
        g = data[fam].get(key, (None, None))[1]
        if g is None:
            continue
        m, e = g.mean(axis=0), g.std(axis=0, ddof=1)
        ax[2].bar(np.arange(3) + dx, m, yerr=e, width=0.19, capsize=2.5,
                  label=lab, color=col, alpha=0.9)
    ax[2].axhline(0, color="k", lw=1)
    ax[2].plot(np.arange(3) - 0.10, [0.642, 0.586, 0.869], "X", color="crimson",
               ms=11, zorder=4, label="published ABC, 1 seed")
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["low-σ", "mid-σ", "high-σ"])
    ax[2].set_ylim(-1.30, 1.05)
    ax[2].set_ylabel("Δ log-density  (conditioned − no-cond)")
    ax[2].set_title("the adaptivity claim does not survive\nerror bars = seed sd",
                    fontsize=9.5)
    ax[2].legend(fontsize=7, loc="lower center", ncol=3)

    fig.suptitle("Noise conditioning on synthetic data, device-matched and across "
                 "three training seeds (P6-D12)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig5_abc_ablation.png", dpi=160); plt.close(fig)
    print("  fig5_abc_ablation.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5"]
    for w in which:
        globals()[w]()
    print(f"→ figures in {FIG}/")
