"""
G13. WASP-96b DEPTH noise, built by re-running my own light-curve fit on
pseudo-transits instead of reading scatter off out-of-transit flux.

WHY THIS EXISTS. `inject_real_noise.py` builds Tier-1 noise as a difference of
means of real OOT flux residuals, rescaled to the published error bar. That
inherits the real cross-wavelength covariance but it cannot inherit anything the
light-curve fit itself introduces: the quadratic baseline that is marginalized
per bin, the limb-darkening and geometry held fixed at the white-light solution,
and the depth-baseline degeneracy those two produce together. Measuring that
needs the fit re-run on data with a KNOWN injected depth, which is possible only
where I own the reduction end to end. That is WASP-96b, 280 integrations through
exoTEDRF plus the batman fit in `wasp96_fit_lightcurves.py`, and it is NOT the
headline target. The job here is to measure how wrong the published-amplitude
flux approximation is on the one target where both routes run.

THE LITERAL RECIPE IN G13 DOES NOT RUN, AND THAT IS A RESULT NOT AN EXCUSE.
G13 says to inject pseudo-transits into OOT segments. The transit is 2.66 h long
and the two OOT blocks are 2.27 h before and 1.44 h after, so no OOT segment can
host a full pseudo-transit. Injecting a partial one would change the fit's
information content and stop being representative. So the pseudo-transit goes on
the real time axis at the real t0 and the NOISE is resampled, which is prayer-bead
injection, the standard red-noise estimator in transit work and the method
`problems_and_decisions.md` already named under P6-D6.

TWO POOLS, BECAUSE THE POOL CHOICE IS THE ONE ARBITRARY PART.
  --pool full  cyclic rotation of the full 280-frame residual series. Uses every
               frame, gives 279 distinct realizations with no resampling at all.
               Justified by measurement: in-transit residual RMS is 139 ppm
               against 143 ppm out of transit, so in-transit frames carry no
               extra transit-model error to leak into the noise.
  --pool oot   contiguous-block bootstrap from the 164 OOT frames only. No
               transit model has ever touched these residuals. Costs frame reuse,
               since 164 frames have to cover a 280-frame axis.
Agreement between the two is the check that the answer is about the fit and not
about which residuals I fed it.

THE SAME ROTATION IS APPLIED TO EVERY BIN. Rotating bins independently would
destroy the cross-wavelength correlation, which is the entire quantity being
measured.

AMPLITUDE IS A MEASUREMENT HERE, NOT A RESCALING. The flux route has to borrow
its scale from the published error bars because 56 frames cannot reach the
published precision. This route gets its scale from the actual fit on the actual
data, so comparing it to the published bar is an audit of that bar rather than a
tautology. `fit_bin` already inflates to red-chi2 = 1, so an honest bar should
land near the measured depth scatter.

THE MODEL GRID IS THE REDUCTION'S OWN GRID. `abc_rad_wasp96` was built from this
spectrum, so its 90 bins match the fitted bins to 1e-7 um and no regridding step
sits between the fit and the covariance the flow conditions on.

    conda run -n mirage-reduce python scripts/depth_noise_wasp96.py
    conda run -n mirage-reduce python scripts/depth_noise_wasp96.py --pool oot --draws 200
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wasp96_fit_lightcurves as w96                                  # noqa: E402

OUT = Path("data/inject")
WHITE = w96.OUTDIR / "white_light_fit.npy"
SPEC = Path("data/jwst_wasp96b_spectrum.csv")
FLUX_NOISE = OUT / "wasp96_noise.npz"
P_ORB, T14_HALF = w96.P_ORB, w96.T14_HALF


# ---------------------------------------------------------------------------
# the real per-bin fit, returning what fit_bin throws away
# ---------------------------------------------------------------------------
def fit_bin_full(t, y, yerr, sol, rp0):
    """w96.fit_bin's model, returning the baseline and residuals as well.

    Deliberately a copy of that residual closure rather than a refactor of it.
    `wasp96_fit_lightcurves.py` produced the published spectrum and I am not
    editing it to serve this diagnostic. `--selfcheck` asserts the two agree.
    """
    dt = t - t.mean()

    def resid(p, e):
        rp, b0, b1, b2 = p
        return (y - w96.lc(t, rp, sol["t0"], sol["ars"], sol["inc"], sol["u1"],
                           sol["u2"], b0 + b1 * dt + b2 * dt ** 2)) / e
    resid.bounds = ([0.02, 0.5 * np.median(y), -np.inf, -np.inf],
                    [0.25, 1.5 * np.median(y), np.inf, np.inf])
    r, err_inf, red = w96._inflate_refit(resid, [rp0, np.median(y), 0.0, 0.0],
                                         yerr, len(t) - 4, 5000)
    rp, b0, b1, b2 = r.x
    model = w96.lc(t, rp, sol["t0"], sol["ars"], sol["inc"], sol["u1"], sol["u2"],
                   b0 + b1 * dt + b2 * dt ** 2)
    try:
        rp_err = float(np.sqrt(np.linalg.inv(r.jac.T @ r.jac)[0, 0]))
    except np.linalg.LinAlgError:
        rp_err = np.nan
    return dict(rp=rp, rp_err=rp_err, b=(b0, b1, b2), model=model,
                resid=y - model, redchi2=red, err_inf=err_inf)


def refit_depth(t, y, yerr, sol, rp0):
    """Depth recovery only, started from the white-light rp so the fit gets no
    hint of the injected value."""
    dt = t - t.mean()

    def resid(p, e):
        rp, b0, b1, b2 = p
        return (y - w96.lc(t, rp, sol["t0"], sol["ars"], sol["inc"], sol["u1"],
                           sol["u2"], b0 + b1 * dt + b2 * dt ** 2)) / e
    resid.bounds = ([0.02, 0.5 * np.median(y), -np.inf, -np.inf],
                    [0.25, 1.5 * np.median(y), np.inf, np.inf])
    r, _, _ = w96._inflate_refit(resid, [rp0, np.median(y), 0.0, 0.0], yerr,
                                 len(t) - 4, 5000)
    return float(r.x[0])


# ---------------------------------------------------------------------------
# noise index generators
# ---------------------------------------------------------------------------
def rotations(nf, n):
    """Prayer bead. Distinct non-trivial cyclic shifts, spread over the series."""
    n = min(n, nf - 1)
    shifts = np.unique(np.round(np.linspace(1, nf - 1, n)).astype(int))
    return [(np.arange(nf) + k) % nf for k in shifts]


def block_boot(oot_idx, nf, n, block, rng):
    """Contiguous-block bootstrap from the OOT frames, wrapping inside the OOT
    index list. Blocks, not single frames, so time correlation survives."""
    m = len(oot_idx)
    out = []
    for _ in range(n):
        idx = []
        while len(idx) < nf:
            s = int(rng.integers(m))
            idx.extend(oot_idx[(s + np.arange(block)) % m])
        out.append(np.asarray(idx[:nf]))
    return out


# ---------------------------------------------------------------------------
def depth_jacobian(t, sol):
    """d(model)/d(rp) with the quadratic baseline projected out, since the fit
    marginalizes it. This is the direction a depth error actually lives in."""
    dt = t - t.mean()
    e = 1e-5
    g = (w96.lc(t, sol["rp"] + e, sol["t0"], sol["ars"], sol["inc"], sol["u1"],
                sol["u2"], 1.0)
         - w96.lc(t, sol["rp"] - e, sol["t0"], sol["ars"], sol["inc"], sol["u1"],
                  sol["u2"], 1.0)) / (2 * e)
    B = np.column_stack([np.ones(len(t)), dt, dt ** 2])
    return g - B @ np.linalg.lstsq(B, g, rcond=None)[0]


def spectral_deficit(R, g, t):
    """WHY PRAYER-BEAD COMES IN BELOW WHITE NOISE, as a number rather than a story.

    Linearizing, the depth error is (r . g)/(g . g), so the ratio of the
    prayer-bead sd to the iid sd is the sd over cyclic shifts of (r_shift . g)
    divided by sd_r * sqrt(g . g). If that reproduces the measured ratio, the
    shortfall is the residuals being power-deficient where g has its weight, and
    NOT a property of the depth noise. The residuals are fit residuals, so
    exoTEDRF's detrending and the four parameters this fit removes have both
    already taken power out at the transit timescale, and a cyclic shift cannot
    put it back.
    """
    nf, nb = R.shape
    gg = np.dot(g, g)
    pred = np.array([np.std([np.dot(np.roll(R[:, j], k), g) for k in range(nf)])
                     / (R[:, j].std() * np.sqrt(gg)) for j in range(nb)])
    f = np.fft.rfftfreq(nf, d=float(np.median(np.diff(t))))
    wt = np.abs(np.fft.rfft(g)) ** 2
    wt /= wt.sum()
    pw = np.mean([np.abs(np.fft.rfft(R[:, j])) ** 2 / (R[:, j].var() * nf)
                  for j in range(nb)], axis=0)
    bands = [(lo, hi, float(pw[(f >= lo) & (f < hi)].mean()),
              float(wt[(f >= lo) & (f < hi)].sum()))
             for lo, hi in ((0, 4), (4, 8), (8, 20), (20, 1e9))
             if ((f >= lo) & (f < hi)).any()]
    return pred, float(np.average(f, weights=wt)), bands


def corr_stats(C):
    """Structure summary of a correlation matrix, comparable between routes."""
    off = ~np.eye(len(C), dtype=bool)
    ev = np.linalg.eigvalsh(C)
    v = np.linalg.eigh(C)[1][:, -1]
    return dict(lag1=float(np.median(np.diag(C, 1))),
                lag7=float(np.median(np.diag(C, 7))),
                top_ev=float(ev[-1]),
                mean_abs_off=float(np.abs(C[off]).mean()),
                # a near-uniform top eigenvector is a common-mode DEPTH offset,
                # which is the free baseline offset H10 proposes as a latent
                ev1_uniformity=float(abs(v.mean()) / np.sqrt((v ** 2).mean())))


def main(a):
    rng = np.random.default_rng(a.seed)
    wave, flux, ferr, t = w96.load()
    good = np.isfinite(flux).all(axis=0) & (np.nanmedian(flux, axis=0) > 0)
    sol = np.load(WHITE, allow_pickle=True).item()
    nf = len(t)
    ph = ((t - sol["t0"] + 0.5 * P_ORB) % P_ORB) - 0.5 * P_ORB
    oot = np.abs(ph) > T14_HALF
    print(f"[data] {nf} integrations, {flux.shape[1]} columns ({good.sum()} good), "
          f"{(t.max() - t.min()) * 24:.2f} h")
    print(f"[mask] {oot.sum()} out of transit ({(ph < -T14_HALF).sum()} pre, "
          f"{(ph > T14_HALF).sum()} post), {(~oot).sum()} in transit")
    print(f"[G13]  transit {2 * T14_HALF * 24:.2f} h needs more baseline than the "
          f"longest OOT block, so a pseudo-transit cannot live inside one")

    # ---- stage A, the real per-bin fit -----------------------------------
    edges = np.geomspace(wave.min(), wave.max(), a.nbins + 1)
    ctr, rp_true, R, models, ys, yes, ncol = [], [], [], [], [], [], []
    t_a = time.time()
    for k in range(a.nbins):
        m = good & (wave >= edges[k]) & (wave < edges[k + 1])
        if m.sum() < 3:
            continue
        yb, yeb = w96.whitelight(flux, ferr, m)
        f = fit_bin_full(t, yb, yeb, sol, sol["rp"])
        if not np.isfinite(f["rp_err"]):
            continue
        ctr.append(np.sqrt(edges[k] * edges[k + 1]))
        rp_true.append(f["rp"]); R.append(f["resid"]); models.append(f["model"])
        ys.append(yb); yes.append(yeb); ncol.append(int(m.sum()))
    ctr = np.asarray(ctr); rp_true = np.asarray(rp_true)
    R = np.column_stack(R)                       # (nf, nbin) residuals
    models = np.column_stack(models)
    nb = len(ctr)
    print(f"\n[fit]  {nb} bins fitted in {time.time() - t_a:.1f} s, "
          f"{np.median(ncol):.0f} columns per bin")
    print(f"       depth median {np.median(rp_true ** 2):.5f}, residual RMS median "
          f"{np.median(R.std(axis=0)) * 1e6:.0f} ppm")
    print(f"       residual RMS in transit {np.median(R[~oot].std(axis=0)) * 1e6:.0f} ppm "
          f"vs out {np.median(R[oot].std(axis=0)) * 1e6:.0f} ppm "
          f"(this is what licenses --pool full)")
    rc = np.corrcoef(R, rowvar=False)
    print(f"       residual cross-wavelength lag-1 corr {np.median(np.diag(rc, 1)):+.3f}, "
          f"time lag-1 corr {np.median([np.corrcoef(R[:-1, j], R[1:, j])[0, 1] for j in range(nb)]):+.3f}")

    if a.selfcheck:
        for j in (0, nb // 2, nb - 1):
            m = good & (wave >= edges[j]) & (wave < edges[j + 1])
            yb, yeb = w96.whitelight(flux, ferr, m)
            rp_ref, _ = w96.fit_bin(t, yb, yeb, sol, rp0=sol["rp"])
            assert abs(rp_ref - rp_true[j]) < 1e-8, (j, rp_ref, rp_true[j])
        print("       selfcheck: fit_bin_full reproduces w96.fit_bin exactly")

    # ---- stage B, pseudo-transit injection -------------------------------
    sdR = R.std(axis=0)
    if a.pool == "full":
        picks = rotations(nf, a.draws)
        tag = f"prayer-bead, {len(picks)} distinct rotations of {nf} frames"
    elif a.pool == "oot":
        picks = block_boot(np.flatnonzero(oot), nf, a.draws, a.block, rng)
        tag = (f"block bootstrap, {len(picks)} draws of {a.block}-frame blocks "
               f"from {int(oot.sum())} OOT frames")
    else:
        # CONTROL, not a depth-noise construction. White noise at the real
        # per-bin residual sd, through the identical fit. Separates what the fit
        # does to any noise from what the real noise structure contributes, and
        # it is the only way to tell "the fit suppresses the published bar" from
        # "the real residuals are correlated in a way that matters".
        picks = [None] * a.draws
        tag = (f"iid CONTROL, {a.draws} white draws at the real per-bin sd "
               f"(median {np.median(sdR) * 1e6:.0f} ppm), no structure in time or "
               f"wavelength")
    print(f"\n[inject] {tag}")
    print(f"         injected depth per bin = the real fitted depth, so transit "
          f"shape and per-bin SNR match the real data")

    d = np.full((len(picks), nb), np.nan)
    t_b = time.time()
    for i, pk in enumerate(picks):
        # SAME rotation for every bin. Rotating bins independently would destroy
        # the cross-wavelength correlation, which is the quantity being measured.
        noise = (rng.normal(size=(nf, nb)) * sdR) if pk is None else R[pk]
        for j in range(nb):
            rp_hat = refit_depth(t, models[:, j] + noise[:, j], yes[j], sol, sol["rp"])
            d[i, j] = rp_hat ** 2 - rp_true[j] ** 2
        if (i + 1) % 25 == 0 or i == len(picks) - 1:
            el = time.time() - t_b
            print(f"         {i + 1}/{len(picks)}  {el:.0f} s elapsed, "
                  f"{el / (i + 1) * (len(picks) - i - 1):.0f} s left")

    # ---- stage C, the comparison -----------------------------------------
    import pandas as pd
    sp = pd.read_csv(SPEC).sort_values("wavelength_um")
    sig_pub = np.interp(ctr, sp["wavelength_um"], sp["depth_error"])
    sig_d = d.std(axis=0)
    bias = d.mean(axis=0)
    Sig = np.cov(d, rowvar=False)
    Cd = Sig / np.outer(np.sqrt(np.diag(Sig)), np.sqrt(np.diag(Sig)))
    sd = corr_stats(Cd)

    fz = np.load(FLUX_NOISE, allow_pickle=True)
    nzf = fz["noise"][:, fz["covered"] & ~fz["gap"]]
    Cf = np.corrcoef(nzf, rowvar=False)
    sf = corr_stats(Cf)
    sig_flux = fz["noise"].std(axis=0)
    sig_flux = np.interp(ctr, fz["wlen"], sig_flux)

    ratio = sig_d / sig_pub
    print(f"\n=== G13. DEPTH NOISE vs THE FLUX APPROXIMATION ({a.pool}) ===")
    print(f"  measured depth sigma   median {np.median(sig_d) * 1e6:7.1f} ppm   "
          f"range {sig_d.min() * 1e6:.0f}-{sig_d.max() * 1e6:.0f}")
    print(f"  published error bar    median {np.median(sig_pub) * 1e6:7.1f} ppm")
    print(f"  flux-route realized    median {np.median(sig_flux) * 1e6:7.1f} ppm   "
          f"(rescaled TO the published bar by construction)")
    print(f"  RATIO depth/published  median {np.median(ratio):.2f}x   "
          f"range {ratio.min():.2f}-{ratio.max():.2f}   "
          f"{int((ratio > 1).sum())}/{nb} bins above 1")
    print(f"  recovery bias          median {np.median(bias) * 1e6:+.1f} ppm "
          f"({np.median(bias / sig_d):+.3f} sigma), so the fit is "
          f"{'biased' if abs(np.median(bias / sig_d)) > 0.1 else 'unbiased'} at this depth")
    print(f"\n  {'structure':<22}{'depth':>10}{'flux':>10}")
    for key, lab in (("lag1", "lag-1 corr"), ("lag7", "lag-7 corr"),
                     ("top_ev", "top eigenvalue"), ("mean_abs_off", "mean |off-diag|"),
                     ("ev1_uniformity", "ev1 uniformity")):
        print(f"  {lab:<22}{sd[key]:>10.3f}{sf[key]:>10.3f}")
    # Bootstrap over realizations, as A4 did. Rotations are not independent
    # samples of the universe, they are an ensemble built from ONE residual
    # series, so this bounds the ensemble estimate and not the visit.
    bs_r, bs_l1 = [], []
    for _ in range(a.boot):
        s = rng.integers(0, len(d), len(d))
        sb = d[s].std(axis=0)
        bs_r.append(np.median(sb / sig_pub))
        Cb = np.corrcoef(d[s], rowvar=False)
        bs_l1.append(np.median(np.diag(Cb, 1)))
    rlo, rhi = np.percentile(bs_r, [16, 84])
    llo, lhi = np.percentile(bs_l1, [16, 84])
    print(f"  bootstrap 68% CI       ratio [{rlo:.2f}, {rhi:.2f}]   "
          f"lag-1 [{llo:.3f}, {lhi:.3f}]  ({a.boot} resamples over draws)")

    cm = d.mean(axis=1)
    cmf_d = float(cm.var() / np.mean(np.var(d, axis=0)))
    cmf_f = float(nzf.mean(axis=1).var() / np.mean(np.var(nzf, axis=0)))
    print(f"\n  common-mode depth offset  sd {cm.std() * 1e6:.1f} ppm, "
          f"{cmf_d:.3f} of mean per-bin variance")
    print(f"  same statistic, flux route {cmf_f:.3f};  white over {nb} bins {1.0 / nb:.3f}")
    print(f"  so this route carries {cmf_d / cmf_f:.1f}x the flux route's common mode "
          f"and {cmf_d * nb:.1f}x white")
    if cmf_d > cmf_f:
        print(f"  a near-uniform depth offset is the free baseline offset H10 proposes "
              f"as a latent and P3-D11's radius stands in for")

    # the mechanism check, which decides how the amplitude number may be read
    pred, cen, bands = spectral_deficit(R, depth_jacobian(t, sol), t)
    print(f"\n  MECHANISM. linearized prayer-bead/iid sd ratio predicted "
          f"{np.median(pred):.3f}, measured ratio is the number above over the iid "
          f"control's 0.99")
    print(f"  depth-Jacobian weight centroid {cen:.1f} cycles/day "
          f"({24 / cen:.2f} h, against a {2 * T14_HALF * 24:.2f} h transit)")
    print(f"  {'band (c/d)':<14}{'resid power / white':>21}{'Jacobian weight':>18}")
    for lo, hi, pwb, wtb in bands:
        print(f"  {str(lo) + '-' + ('max' if hi > 1e8 else str(hi)):<14}"
              f"{pwb:>21.3f}{wtb:>18.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"wasp96_depth_noise{'' if a.pool == 'full' else '_' + a.pool}.npz"
    np.savez(out, noise=d, sigma=sig_d, bias=bias, Sigma=Sig, corr=Cd, wlen=ctr,
             covered=np.ones(nb, bool), gap=np.zeros(nb, bool),
             sigma_published=sig_pub, sigma_flux_route=sig_flux,
             depth_true=rp_true ** 2, n_frames=np.array([nf]), n_oot=int(oot.sum()),
             planet="wasp96", amplitude="measured", pool=a.pool, block=a.block,
             route="pseudo-transit injection through the light-curve fit",
             ratio_to_published=float(np.median(ratio)),
             ratio_ci=np.array([rlo, rhi]), lag1_ci=np.array([llo, lhi]),
             common_mode_frac=cmf_d, common_mode_frac_flux=cmf_f,
             bead_iid_predicted=float(np.median(pred)), jac_centroid_cpd=cen,
             stats_depth=np.array([sd], dtype=object),
             stats_flux=np.array([sf], dtype=object))
    print(f"\nSaved -> {out}  ({d.shape[0]} realizations x {nb} bins)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool", choices=["full", "oot", "iid"], default="full")
    p.add_argument("--draws", type=int, default=279)   # not --n, conda run eats it
    p.add_argument("--block", type=int, default=20, help="OOT bootstrap block length")
    p.add_argument("--nbins", type=int, default=w96.__dict__.get("NBINS", 90))
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selfcheck", action="store_true", default=True)
    main(p.parse_args())
