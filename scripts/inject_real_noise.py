"""
A1 Tier-1 injections: prior draws pushed through TauREx and contaminated with
REAL JWST noise, so calibration can be scored against labels no transport can
game (G3, G5). Replaces the circular OT check (P5-D5).

Two stages, because the noise construction needs no forward model and must be
validated on its own.

  --noise-only   build depth-noise realizations from held-out OOT frames and
                 check their amplitude against the published error bars.
                 Runs in `mirage`. Writes data/inject/<planet>_noise.npz.

  --build        draw theta from the prior, TauREx forward, add the noise, write
                 an HDF in generate_training_data.py's layout so the existing
                 eval path reads it unchanged. Runs in `mirage-taurex`.

    conda run -n mirage        python scripts/inject_real_noise.py --planet wasp39 --noise-only
    conda run -n mirage-taurex python scripts/inject_real_noise.py --planet wasp39 --build --n 1000

G13, AND IT IS NOT COSMETIC. OOT residuals are FLUX noise; Tier 1 needs DEPTH
noise. A transit depth is delta = 1 - F_in/F_out, so a fractional flux error `a`
on the in-transit mean and `b` on the out-of-transit mean give

    delta_measured ~= delta_true - (a - b)        for delta << 1

which is why a depth-noise realization is built here as a DIFFERENCE OF MEANS of
real frame residuals, drawn disjointly from a held-out pool. That inherits the
real cross-wavelength covariance, and the 1/sqrt(N) averaging, without assuming
either.

AMPLITUDE COMES FROM THE PUBLISHED ERROR BARS, NOT THE FRAMES (P6-D6). Measured
on WASP-39b PRISM, the frames cannot reach the published precision by any amount
of averaging: per-bin frame scatter is 4.6e-3 after detrending, so matching the
30 ppm rebinned error bar would need ~52,000 frames per side and only 810 exist.
Removing the common mode, a linear trend and three PCA components moves it from
9.1x to 6.8x too noisy, so this is not removable systematics. Injecting the raw
amplitude would make the test set ~9x noisier than real data, and a flow scores
as "well calibrated" on noise it will never see.

So the frames set the STRUCTURE and the published errors set the SCALE, which is
the same split P3-D1 already made for the covariance embedding (condition on
structure, not scale). The structure is worth keeping: lag-1 correlation across
wavelength is +0.31 at 121 sigma from a column-shuffled null, still +0.18 at lag
7, and the correlation matrix's top eigenvalue is 8.09 where white noise gives
~1. Real-noise injection is therefore not a repackaged white-noise test.

What it does NOT inherit is anything the light-curve fit itself introduces:
detrending, limb-darkening degeneracy, transit-shape covariance. Capturing those
needs the fit re-run on injected data, which is possible only where we own the
reduction. That is WASP-96b (280 integrations, our own exoTEDRF plus batman
pipeline), NOT the headline target. So this approximation is what WASP-39b and
K2-18b get, it is stated as such, and `depth_noise_wasp96.py` exists to measure
how wrong it is on the one target where both routes are available. Do not report
Tier-1 numbers on the three targets as if they rest on the same construction.

Frames are split CONTIGUOUSLY, not by alternating index. JWST systematics are
time-correlated, so an even/odd split would put the same drift on both sides and
the "held-out" noise would not be held out at all.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

OUT = Path("data/inject")

# per-planet: OOT file, the training-set stats that fix the model grid, the
# published spectrum used as the amplitude cross-check, and the theta prior.
PLANETS = {
    "wasp39": dict(
        oot=["data/jwst_wasp39b_oot/NIRSPEC_CLEAR_PRISM_out_of_transit_full.csv"],
        stats="data/abc_rad/abc_rad_train.hdf",
        spec="data/jwst_wasp39b_spectrum.csv",
        rp=(0.8, 1.6), T=(300.0, 2500.0), geom="WASP39"),
    "wasp96": dict(
        oot=["data/jwst_wasp96b_oot/NIRISS_CLEAR_N_A_out_of_transit_full.csv"],
        stats="data/abc_rad_wasp96/abc_rad_train.hdf",
        spec="data/jwst_wasp96b_spectrum.csv",
        rp=(0.8, 1.6), T=(300.0, 2500.0), geom="WASP96"),
    "k218": dict(
        # NIRISS alone leaves 33% of the 0.85-5.17 um grid without real frames.
        # These are SEPARATE VISITS, so there is no cross-instrument noise
        # correlation to preserve, which is also physically true of the published
        # spectrum: it is stitched from independent observations.
        oot=["data/jwst_k2_18b_oot/NIRISS_CLEAR_N_A_out_of_transit_full.csv",
             "data/jwst_k2_18b_oot/NIRSPEC_F290LP_G395H_out_of_transit_full.csv"],
        oot_extra=["data/jwst_k2_18b_oot/MIRI_P750L_N_A_out_of_transit_full.csv"],
        stats="data/abc_rad_k218/abc_rad_train.hdf",
        spec="data/jwst_k2_18b_spectrum.csv",
        rp=(0.10, 0.40), T=(100.0, 600.0), geom="K2_18b"),
}
LOGX_RANGE = (-9.0, -1.0)

# Absolute instrument guards. The pathological-BIN filter is relative to each
# instrument's own median, so a uniformly bad instrument passes it untouched.
# K2-18b/MIRI is the case that found this: 40% of its flux values are negative,
# median flux is 4000x fainter than NIRISS, per-bin scatter runs 95-427%, and
# dq_flag == 0 on every row, so nothing downstream would have objected.
MAX_REL_SD = 0.10      # median per-bin relative scatter; real instruments are <1%
MAX_NEG_FRAC = 0.05    # negative flux means background-dominated, not observed


def model_grid(stats_hdf: str):
    """Ascending model wavelength grid and its bin edges, from the training set."""
    with h5py.File(stats_hdf) as f:
        wl = np.sort(f["wlen"][0].astype(float))
    edges = np.empty(len(wl) + 1)
    edges[1:-1] = np.sqrt(wl[:-1] * wl[1:])
    edges[0] = wl[0] ** 2 / edges[1]
    edges[-1] = wl[-1] ** 2 / edges[-2]
    return wl, edges


def oot_instruments(files, edges: np.ndarray):
    """-> list of (name, rel, covered, n_frames), one per usable instrument file.

    Instruments failing the absolute guards are refused here with the reason, not
    silently rescaled into the set.
    """
    out = []
    for f in files:
        rel, covered, nf, neg = oot_residuals(f, edges, with_neg=True)
        name = Path(f).name.split("_out_of")[0]
        if not covered.any():
            print(f"  REFUSED {name}: covers no model bin")
            continue
        sd = float(np.nanmedian(np.nanstd(rel[:, covered], axis=0)))
        if neg > MAX_NEG_FRAC:
            print(f"  REFUSED {name}: {100 * neg:.0f}% of flux values are negative "
                  f"(limit {100 * MAX_NEG_FRAC:.0f}%), so it is background-dominated "
                  f"rather than observed")
            continue
        if sd > MAX_REL_SD:
            print(f"  REFUSED {name}: median per-bin scatter {sd:.2f} "
                  f"({100 * sd:.0f}% relative flux, limit {100 * MAX_REL_SD:.0f}%). "
                  f"Rescaling that to the published sigma would inject its pathology "
                  f"as if it were instrument correlation")
            continue
        out.append((name, rel, covered, nf))
    if not out:
        raise SystemExit("no usable OOT instrument for this target")
    return out


def oot_residuals(oot_csv: str, edges: np.ndarray, with_neg: bool = False):
    """-> (n_frames, n_bins) relative flux residuals, ascending bins, NaN where uncovered."""
    df = pd.read_csv(oot_csv)
    df = df[df["dq_flag"] == 0]
    neg = float((df["flux_jy"].to_numpy() < 0).mean()) if len(df) else 1.0
    n_bins = len(edges) - 1
    ints = np.sort(df["integration_num"].unique())
    M = np.full((len(ints), n_bins), np.nan)
    b = np.digitize(df["wavelength_um"].to_numpy(), edges) - 1
    df = df.assign(_bin=b)
    df = df[(df._bin >= 0) & (df._bin < n_bins)]
    row = {ii: r for r, ii in enumerate(ints)}
    for (ii, bb), v in df.groupby(["integration_num", "_bin"])["flux_jy"].mean().items():
        M[row[ii], bb] = v
    covered = np.isfinite(M).all(axis=0)
    rel = np.zeros_like(M)
    rel[:, covered] = M[:, covered] / M[:, covered].mean(axis=0) - 1.0
    return (rel, covered, len(ints), neg) if with_neg else (rel, covered, len(ints))


def published_sigma(spec_csv: str, wl_asc: np.ndarray):
    """Published per-bin depth error on the model grid, for the amplitude check."""
    from mirage.datasets.real_jwst import rebin_spectrum
    _, err, _ = rebin_spectrum(pd.read_csv(spec_csv), wl_asc)
    return np.asarray(err, dtype=float)


def build_noise(planet: str, n: int, n_avg: int | None, split: str,
                amplitude: str, seed: int, extra: bool = False):
    """Depth-noise realizations from held-out OOT frames, with an amplitude audit."""
    P = PLANETS[planet]
    wl, edges = model_grid(P["stats"])
    sig_pub = published_sigma(P["spec"], wl)

    # The model grid can extend past the published spectrum (WASP-39b: grid to
    # 7.28 um, spectrum stops at 5.33), leaving bins with NO published sigma. Those
    # were being written to the HDF as NaN, which the flow ingests as input. Fill
    # them from the nearest valid published bin and mark them, so they are noisy
    # rather than NaN and are never counted as real-noise coverage.
    extrap = ~(np.isfinite(sig_pub) & (sig_pub > 0))
    if extrap.all():
        raise SystemExit(f"{planet}: published spectrum gives no usable sigma")
    if extrap.any():
        good = np.flatnonzero(~extrap)
        sig_pub = sig_pub.copy()
        sig_pub[extrap] = sig_pub[good[np.argmin(
            np.abs(wl[extrap, None] - wl[good][None, :]), axis=1)]]

    files = list(P["oot"]) + (list(P.get("oot_extra", [])) if extra else [])
    insts = oot_instruments(files, edges)
    rng = np.random.default_rng(seed)

    # Drop pathological bins per instrument: a few sit 30x above the median with
    # dq_flag == 0 (1.16 and 1.27 um on PRISM) and would dominate every draw.
    dropped = np.zeros(len(wl), bool)
    for _, rel, covered, _ in insts:
        if not covered.any():
            continue
        ps = np.full(len(wl), np.nan)
        ps[covered] = np.nanstd(rel[:, covered], axis=0)
        bad = covered & (ps > 5.0 * np.nanmedian(ps[covered]))
        covered &= ~bad
        dropped |= bad

    # Each bin is served by the instrument with the most frames among those
    # covering it. Separate visits carry no shared noise, so realizations are
    # drawn independently per instrument and stitched.
    nf_all = np.array([nf for _, _, _, nf in insts])
    owner = np.full(len(wl), -1)
    for b in range(len(wl)):
        have = [k for k, (_, _, cov, _) in enumerate(insts) if cov[b]]
        if have:
            owner[b] = have[int(np.argmax(nf_all[have]))]

    raw = np.zeros((n, len(wl)))
    per_inst = []
    for k, (name, rel, covered, nf) in enumerate(insts):
        bins = (owner == k)
        if not bins.any():
            per_inst.append((name, nf, 0, 0, float("nan")))
            continue
        half = nf // 2
        pool = np.arange(0, half) if split == "calib" else np.arange(half, nf)
        if len(pool) < 4:
            raise SystemExit(f"{planet}/{name}: {nf} frames leaves {len(pool)} in "
                             f"the '{split}' half, too few to resample.")
        navg = (len(pool) // 2) if n_avg is None else min(n_avg, len(pool) // 2)
        for i in range(n):
            pick = rng.choice(pool, size=2 * navg, replace=False)
            a = rel[pick[:navg]][:, bins].mean(axis=0)
            b = rel[pick[navg:]][:, bins].mean(axis=0)
            raw[i, bins] = -(a - b)
        pb = float(np.median(np.nanstd(rel[:, bins], axis=0)))
        per_inst.append((name, nf, int(bins.sum()), navg, pb))

    served = owner >= 0
    sig_raw = np.zeros(len(wl))
    sig_raw[served] = raw[:, served].std(axis=0)
    target = float(np.nanmedian(sig_pub[served])) if served.any() else float("nan")
    ratio_raw = float(np.median(sig_raw[served]) / target) if target > 0 else float("nan")

    noise = raw.copy()
    if amplitude == "published":
        # per-bin rescale to the published sigma. Dividing by the realized per-bin
        # std and multiplying by sigma_pub leaves the CORRELATION matrix untouched.
        sc = np.where(sig_raw[served] > 0, sig_pub[served] / sig_raw[served], 0.0)
        noise[:, served] = raw[:, served] * sc[None, :]

    # Bins no instrument serves would otherwise get EXACTLY zero noise while still
    # carrying a stated error bar, which is an impossible observation rather than a
    # conservative one. White noise at the published sigma, fraction recorded.
    gap = ~served
    if gap.any():
        noise[:, gap] = rng.normal(0.0, 1.0, size=(n, int(gap.sum()))) * sig_pub[gap]
    # denominator is every bin the model ingests, not just the ones with a
    # published error bar; otherwise extrapolated bins vanish from the accounting
    real_frac = float(served.sum() / len(wl))
    sig_emp = noise.std(axis=0)

    def lag1(m, mask):
        if mask.sum() < 3:
            return float("nan")
        c = np.corrcoef(m[:, mask][:, np.argsort(wl[mask])], rowvar=False)
        return float(np.mean([c[i, i + 1] for i in range(len(c) - 1)]))

    print(f"[{planet}] grid {len(wl)} bins, served {served.sum()}/{len(wl)}"
          f"  ({wl[served].min():.2f}-{wl[served].max():.2f} um)")
    print(f"  {'instrument':<24}{'frames':>7}{'bins':>6}{'n_avg':>7}{'per-bin sd':>12}")
    for name, nf, nb, navg, pb in per_inst:
        print(f"  {name:<24}{nf:>7}{nb:>6}{navg:>7}{pb:>12.3e}")
    if dropped.any():
        print(f"  dropped {dropped.sum()} pathological bin(s) at "
              f"{[round(float(x), 2) for x in wl[dropped]]} (sd > 5x median)")
    if extrap.any():
        print(f"  {extrap.sum()} bin(s) past the published spectrum "
              f"({wl[extrap].min():.2f}-{wl[extrap].max():.2f} um) took sigma from "
              f"the nearest published bin; they are white-filled, never 'real'")
    print(f"  raw amplitude {np.median(sig_raw[served]):.3e} = {ratio_raw:.1f}x "
          f"published {target:.3e}")
    print(f"  amplitude mode '{amplitude}' -> median sigma "
          f"{np.median(sig_emp[served]):.3e} "
          f"({np.median(sig_emp[served]) / target:.2f}x published)")
    print(f"  lag-1 wavelength correlation {lag1(noise, served):+.3f}  "
          f"(white noise would give ~0)")
    if gap.any():
        gw = np.sort(wl[gap])
        tail = gw[gw > wl[served].max()]
        where = (f"beyond {wl[served].max():.2f} um" if len(tail) == len(gw)
                 else f"at {[round(float(x), 2) for x in gw[:6]]}"
                      + (" ..." if len(gw) > 6 else ""))
        print(f"  !! {gap.sum()} of {len(wl)} bins "
              f"({100 * (1 - real_frac):.0f}%) have NO real frames, filled with "
              f"WHITE noise at the published sigma ({where}). Real-noise fraction "
              f"{real_frac:.2f}.")
    else:
        print(f"  every bin served by real frames. Real-noise fraction 1.00.")
    if amplitude == "raw" and ratio_raw > 1.4:
        print(f"  !! raw mode keeps injections {ratio_raw:.1f}x noisier than real "
              f"data; calibration measured here is pessimistic.")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{planet}_noise.npz"
    np.savez(f, noise=noise, covered=served, wlen=wl, sigma=sig_emp,
             sigma_published=sig_pub, n_frames=nf_all, split=split, planet=planet,
             amplitude=amplitude, ratio_raw=ratio_raw, dropped=wl[dropped],
             gap=gap, real_frac=real_frac, owner=owner, extrap=extrap,
             instruments=np.array([i[0] for i in per_inst]))
    print(f"  -> {f}")
    return f


def sigma_scale_for_training(ref_dir: str = "configs/h9_rad_cov_s42") -> float:
    """Scale that puts the injected sigma in the middle of the TRAINING range.

    The models perturb flux with InjectCorrelatedNoise at sigma 0.05-0.3 in
    NORMALISED units, which is 23-139x the real published error bar (P6-D7). So a
    published-amplitude injection set is out of distribution for the sigma and
    covariance embeddings in exactly the way real data is. That is what Tier 1 is
    for, but it cannot be read without an in-distribution control, otherwise an
    intrinsic flow defect and a scale mismatch look identical.
    """
    import yaml
    with open(Path(ref_dir) / "config.yaml") as fh:
        kw = yaml.safe_load(fh)["training"]["stage_0"]["data_transforms"][0]
        kw = kw["kwargs"]["config"]["kwargs"]
    return float(np.sqrt(kw["sigma_min"] * kw["sigma_max"]))


def build_set(planet: str, n: int, seed: int, out: str | None,
              sigma_scale: str = "1.0"):
    """Prior draws + TauREx forward + the held-out real noise, in training layout."""
    import taurex_forward as tf          # PYTHONPATH=scripts, mirage-taurex env

    P = PLANETS[planet]
    nz = np.load(OUT / f"{planet}_noise.npz", allow_pickle=True)
    noise_pool, covered = nz["noise"], nz["covered"]
    wl = nz["wlen"]
    if len(noise_pool) < n:
        raise SystemExit(f"only {len(noise_pool)} noise realizations; "
                         f"rerun --noise-only --n {n}")

    with h5py.File(P["stats"]) as h:
        _fstd_med = float(np.median(h["flux_std"][:]))
    if sigma_scale == "train":
        want = sigma_scale_for_training()                 # normalised sigma target
        have = float(np.nanmedian(nz["sigma_published"][nz["covered"]])) / _fstd_med
        scale = want / have
        tag = f"train (normalised sigma -> {want:.3f})"
    else:
        scale = float(sigma_scale)
        tag = f"{scale:g}x published"
    print(f"[build] {planet}: sigma scale = {tag}  (factor {scale:.3g})")

    width_asc = np.diff(np.concatenate([
        [wl[0] ** 2 / np.sqrt(wl[0] * wl[1])],
        np.sqrt(wl[:-1] * wl[1:]),
        [wl[-1] ** 2 / np.sqrt(wl[-2] * wl[-1])]]))
    pgeom = getattr(tf, P["geom"])
    rp_lo, rp_hi = P["rp"]
    T_lo, T_hi = P["T"]
    rng = np.random.default_rng(seed)

    theta = np.empty((n, 7), np.float32)
    flux = np.empty((n, len(wl)), np.float32)
    sig = np.empty((n, len(wl)), np.float32)
    geom = np.empty((n, 5), np.float32)
    clean = np.empty((n, len(wl)), np.float32)

    for i in range(n):
        rp = rng.uniform(rp_lo, rp_hi)
        T = rng.uniform(T_lo, T_hi)
        logx = rng.uniform(*LOGX_RANGE, size=5)
        g = dict(rp_rj=rp, mp_mj=pgeom["mp_mj"], rs_rsun=pgeom["rs_rsun"],
                 ms_msun=pgeom["ms_msun"], ts=pgeom["ts"])
        spec = tf.spectrum(tf.build_model(**g), np.concatenate([[T], logx]),
                           wl, width_asc)
        clean[i] = spec
        flux[i] = spec + noise_pool[i] * scale  # REAL noise, not a kernel draw
        # sigma the analysis would report: published everywhere, which is what the
        # realizations were scaled to. Already complete, including white-filled bins.
        sig[i] = nz["sigma_published"] * scale
        theta[i] = np.concatenate([[rp, T], logx])
        geom[i] = [rp, g["mp_mj"], g["rs_rsun"], g["ms_msun"], g["ts"]]
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}")

    # NORMALISE exactly as prepare_abc_hdf5.py does, with the arm's own training
    # stats: flux -> (flux - flux_mean)/flux_std, noise -> noise/flux_std, per bin.
    # The training and test HDFs store z-scored flux (mean ~-0.13, std ~0.89) while
    # TauREx returns raw transit depth (~0.021). Writing raw depth here would hand
    # the flow an input ~100 sigma from anything it saw in training, and it would
    # not error, it would just return nonsense.
    with h5py.File(P["stats"]) as h:
        wl_stored = h["wlen"][0].astype(np.float64)
        fmean = h["flux_mean"][:].astype(np.float64)
        fstd = h["flux_std"][:].astype(np.float64)
    if fmean.shape[0] != flux.shape[1]:
        raise SystemExit(f"stats grid {fmean.shape[0]} != injection grid {flux.shape[1]}")

    # REORDER to the training set's STORED bin order before normalising. The
    # training and test HDFs store wavelength DESCENDING (7.276 -> 0.55); every
    # computation above works ascending. Writing ascending flux would hand the
    # flow a reversed spectrum, and normalising it with the stored-order
    # flux_mean/flux_std would mismatch the stats bin by bin as well. Neither
    # raises an error; both silently produce nonsense.
    asc = np.argsort(wl_stored)                 # stored -> ascending
    inv = np.argsort(asc)                       # ascending -> stored
    if not np.allclose(wl_stored[asc], wl, rtol=1e-5):
        raise SystemExit("stats grid wavelengths do not match the injection grid")
    flux, clean, sig = flux[:, inv], clean[:, inv], sig[:, inv]
    covered_out = np.asarray(covered)[inv]
    wl_out = wl_stored

    flux_raw, sig_raw_out = flux.copy(), sig.copy()
    flux = ((flux - fmean) / fstd).astype(np.float32)
    clean_n = ((clean - fmean) / fstd).astype(np.float32)
    sig = (sig / fstd).astype(np.float32)

    out = Path(out or OUT / f"{planet}_tier1.hdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("theta", data=theta)
        f.create_dataset("flux", data=flux)
        f.create_dataset("noise", data=sig)
        f.create_dataset("flux_mean", data=fmean.astype(np.float32))
        f.create_dataset("flux_std", data=fstd.astype(np.float32))
        f.create_dataset("flux_depth", data=flux_raw)        # raw depth, for plots
        f.create_dataset("noise_depth", data=sig_raw_out)
        f.create_dataset("geometry", data=geom)
        f.create_dataset("wlen", data=wl_out[None, :].astype(np.float32))
        f.create_dataset("clean", data=clean_n)     # normalised, for injection-recovery
        f.create_dataset("covered", data=covered_out)
        f.attrs["noise_source"] = "real_oot_resampled_difference_of_means"
        f.attrs["planet"] = planet
        f.attrs["instruments"] = ", ".join(str(x) for x in nz["instruments"])
        f.attrs["frame_split"] = str(nz["split"])
        f.attrs["amplitude"] = str(nz["amplitude"])
        f.attrs["real_noise_fraction"] = float(nz["real_frac"])
        f.attrs["sigma_scale"] = scale
        f.attrs["sigma_scale_mode"] = tag
        f.attrs["sigma_normalised_median"] = float(np.median(sig[0] ))
    print(f"[build] {n} Tier-1 injections -> {out}")
    print(f"  raw depth range [{flux_raw.min():.5f}, {flux_raw.max():.5f}]")
    print(f"  normalised flux mean {flux.mean():+.4f} std {flux.std():.4f}  "
          f"(training set is 0.00 / 1.00 by construction)")
    _sm = float(np.median(sig))
    print(f"  normalised sigma median {_sm:.5f}  vs the TRAINING PERTURBATION range "
          f"0.05-0.30  -> "
          + ("INSIDE" if 0.05 <= _sm <= 0.30 else
             f"{0.05 / max(_sm, 1e-12):.0f}x BELOW" if _sm < 0.05 else "ABOVE"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planet", default="wasp39", choices=list(PLANETS))
    ap.add_argument("--noise-only", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-avg", type=int, default=None,
                    help="frames averaged per side; default solves for the published amplitude")
    ap.add_argument("--split", default="inject", choices=["inject", "calib"])
    ap.add_argument("--amplitude", default="published", choices=["published", "raw"],
                    help="'published' rescales to the published error bars, keeping the "
                         "real wavelength correlation; 'raw' keeps the frame amplitude")
    ap.add_argument("--extra", action="store_true",
                    help="also use oot_extra instruments (K2-18b: MIRI, 18 frames)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sigma-scale", default="1.0",
                    help="'1.0' = published error bars (Tier 1, realistic but OOD for "
                         "the sigma embedding); 'train' = scaled into the training "
                         "sigma range as an in-distribution control")
    a = ap.parse_args()
    if a.noise_only:
        build_noise(a.planet, a.n, a.n_avg, a.split, a.amplitude, a.seed, a.extra)
    elif a.build:
        build_set(a.planet, a.n, a.seed, a.out, a.sigma_scale)
    else:
        ap.error("pass --noise-only or --build")
