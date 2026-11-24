"""WISE 0855 GO 2327 extraction with the background subtraction knob isolated.

Spike for the brown dwarf reduction-shift study (P6-D39). Miles et al. 2026
(arXiv:2609.20664) section 3.3 states their reduction differs from Rowland et al.
2024 and Wilson et al. 2025 in that they "remove the excess background and use
JWST-pipeline-derived errors". The reduced spectra behind that sentence were
never deposited, so the between-team comparison is impossible (P6-D39). What is
possible is flipping that one named knob on the public stage-2 products and
measuring what it moves.

Both arms share the same calints, so the measurement is insensitive to the
pipeline version and CRDS context MAST happened to use. Only the knob differs.

Stages 1 and 2 are default pipeline per Miles sections 3.1 and 3.2, so the
custom work starts at the stage-2 calibrated per-integration images.
"""
import glob
import numpy as np
from astropy.io import fits
from numpy.polynomial import chebyshev as C

CAL = "data/jwst_w0855_cal/*_nrs1_calints.fits"
HALFWIDTH = 4        # Miles: "we extract a 4-pixel wide radius"
NBKG = 3             # Miles: "median of the outer 3-pixels from both sides"
OUT = "data/jwst_w0855_cal/w0855_knob.npz"


def load():
    sci, err, wav = [], [], None
    for f in sorted(glob.glob(CAL)):
        with fits.open(f) as h:
            sci.append(h["SCI"].data)
            err.append(h["ERR"].data)
            if wav is None:
                wav = h["WAVELENGTH"].data
    return np.concatenate(sci, 0), np.concatenate(err, 0), wav


def trace_center(mean2d):
    """Flux-weighted centroid per column, then a 2nd-order Chebyshev through the
    columns that can actually support one. Miles fit a Gaussian per column and
    then a 2nd-order Chebyshev; a centroid is the same statistic for a symmetric
    profile and does not fail on the low-SNR blue columns, which below 3.6 um
    have SNR under 5 by their own figure 1."""
    nrow, ncol = mean2d.shape
    rows = np.arange(nrow)
    cen = np.full(ncol, np.nan)
    for j in range(ncol):
        col = mean2d[:, j]
        good = np.isfinite(col)
        if good.sum() < 10:
            continue
        w = np.where(col > 0, col, 0.0)
        w[~good] = 0.0
        # centroid in a window around the global peak so wings do not drag it
        pk = int(np.nanargmax(np.where(good, col, -np.inf)))
        lo, hi = max(0, pk - 6), min(nrow, pk + 7)
        ww = w[lo:hi]
        if ww.sum() <= 0:
            continue
        cen[j] = np.sum(rows[lo:hi] * ww) / ww.sum()

    cols = np.arange(ncol)
    ok = np.isfinite(cen)
    # keep only centroids near the global trace, reject blue-end noise jumps
    med = np.nanmedian(cen)
    ok &= np.abs(cen - med) < 3.0
    fit = C.chebfit(cols[ok], cen[ok], 2)
    return C.chebval(cols, fit), ok.sum()


def extract(sci, err, centers, subtract_bkg):
    """Box extract on the trace, optionally removing the outer-3-pixel background.

    Returns flux and propagated error per integration per column."""
    nint, nrow, ncol = sci.shape
    rows = np.arange(nrow)[:, None]                      # (nrow, 1)
    lo = np.rint(centers - HALFWIDTH).astype(int)[None, :]
    hi = np.rint(centers + HALFWIDTH).astype(int)[None, :]
    inap = (rows >= lo) & (rows <= hi)                   # (nrow, ncol)
    bkg_lo = (rows >= lo - NBKG) & (rows < lo)
    bkg_hi = (rows > hi) & (rows <= hi + NBKG)
    inbkg = bkg_lo | bkg_hi

    f = np.full((nint, ncol), np.nan)
    e = np.full((nint, ncol), np.nan)
    npix = inap.sum(0).astype(float)
    for i in range(nint):
        s = sci[i]
        v = err[i] ** 2
        a = np.where(inap, s, np.nan)
        f[i] = np.nansum(a, 0)
        e[i] = np.sqrt(np.nansum(np.where(inap, v, np.nan), 0))
        if subtract_bkg:
            b = np.where(inbkg, s, np.nan)
            per_pix = np.nanmedian(b, 0)                 # median, per Miles
            f[i] -= per_pix * npix
    return f, e, npix, inbkg.sum(0)


def main():
    sci, err, wav = load()
    mean2d = np.nanmean(sci, 0)
    centers, nfit = trace_center(mean2d)
    print(f"cube {sci.shape}, trace fit on {nfit} columns, "
          f"center {centers.min():.2f} to {centers.max():.2f}")

    f_on, e_on, npix, nbkgpix = extract(sci, err, centers, True)
    f_off, e_off, _, _ = extract(sci, err, centers, False)

    # wavelength at the trace row
    row = int(round(float(np.median(centers))))
    wl = wav[row]

    m_on = np.nanmean(f_on, 0)
    m_off = np.nanmean(f_off, 0)
    # error on the time-averaged spectrum
    err_mean = np.sqrt(np.nansum(e_on ** 2, 0)) / np.sum(np.isfinite(f_on), 0)

    np.savez(OUT, wl=wl, m_on=m_on, m_off=m_off, err_mean=err_mean,
             f_on=f_on, f_off=f_off, centers=centers, npix=npix, nbkgpix=nbkgpix)
    print(f"wrote {OUT}")
    print(f"aperture pixels {npix.min():.0f} to {npix.max():.0f}, "
          f"background pixels {nbkgpix.min():.0f} to {nbkgpix.max():.0f}")


if __name__ == "__main__":
    main()
