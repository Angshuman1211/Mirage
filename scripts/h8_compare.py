"""
H8 part 3 of 3 - score TauREx against petitRADTRANS.

Both native spectra are binned onto the 90 observed WASP-96b bins with the SAME
TauREx FluxBinner, so the binning cannot be the thing that differs.

The question is not "do the two codes differ" (they always do, at some ppm) but
"do they differ by enough to matter for what the anchor is used for". Two
separate numbers answer that, and they are NOT interchangeable:

  1. the MEAN depth offset. The NS retrieval floats planet_radius, so a constant
     offset between codes is absorbed by the radius. It therefore does not touch
     the abundances, and it DOES bias the retrieved radius directly. This is the
     number that speaks to P5-D7 and to the 1 sigma gap to the literature value.
  2. the residual SHAPE difference once that offset is removed. This is the part
     a floating radius cannot absorb, so it is what could bias temperature and
     abundances, and it is the number that speaks to forward-model
     misspecification proper.

Reporting only the raw RMS would conflate the two and overstate the damage.

    conda run -n mirage-taurex python scripts/h8_compare.py
"""

import argparse

import numpy as np
from taurex.binning import FluxBinner


def binned(native_wl_um, native_r2, wl, width):
    """Bin a native spectrum onto (wl, width). native_wl may be descending."""
    o = np.argsort(native_wl_um)
    fb = FluxBinner(wl, width)
    _, b, _, _ = fb.bindown(native_wl_um[o], native_r2[o])
    return b


def g395h_grid():
    """The grid V5 / H7 measures reduction shift on, rebuilt analytically.

    scripts/v5_h7_pipeline_shift.py:203 is np.geomspace(2.80, 5.10, 56), so 55
    log-spaced bins of which 53 are populated in every reduction pair. Rebuilding
    it here rather than reading data/v5_k218/canonical means the code residual and
    the reduction shift land on the same bins without needing that directory, which
    is not on this machine. The docstring in that file says 60, which is the NIRISS
    grid and is wrong for G395H (P6-D36).
    """
    e = np.geomspace(2.80, 5.10, 56)
    return np.sqrt(e[:-1] * e[1:]), np.diff(e)


def main(dump, prt, grid="obs", ref_err_ppm=None, planet=None):
    d = np.load(dump, allow_pickle=True)
    p = np.load(prt, allow_pickle=True)
    if grid == "g395h":
        wl, width = g395h_grid()
        depth = err = None
    else:
        wl, width = d["obs_wl"], d["obs_width"]
        depth, err = d["obs_depth"], d["obs_err"]
    tx_wl = 10000.0 / d["native_wn"]
    rs_rj = float(d["rs_rsun"]) * float(d["RSOL"]) / float(d["RJUP"])
    n = len(d["thetas"])

    rows = []
    for k in range(n):
        bt = binned(tx_wl, d["rprs2"][k], wl, width)
        bp = binned(p["wl_um"], p["rprs2"][k], wl, width)
        diff = (bp - bt) * 1e6                         # ppm
        off = float(np.mean(diff))
        shape = diff - off
        # equivalent radius: Rp_eff = sqrt(depth) * Rs
        rp_t = np.sqrt(np.clip(bt, 0, None)) * rs_rj
        rp_p = np.sqrt(np.clip(bp, 0, None)) * rs_rj
        d_rp = float(np.mean(rp_p - rp_t))
        if depth is None:
            chi2_t = chi2_p = float("nan")
        else:
            chi2_t = float(np.mean(((bt - depth) / err) ** 2))
            chi2_p = float(np.mean(((bp - depth) / err) ** 2))
        rows.append((off, float(np.sqrt(np.mean(shape ** 2))), float(np.max(np.abs(shape))),
                     d_rp, chi2_t, chi2_p))

    rows = np.array(rows)
    # In g395h mode the denominator is V5 / H7's own median single-bin error, so the
    # ratio is matched by construction instead of being rederived from a different
    # observation file on a different grid.
    med_err = ref_err_ppm if ref_err_ppm is not None else float(np.median(err)) * 1e6
    err_src = ("V5 / H7 median single-bin error" if ref_err_ppm is not None
               else "median observed error on these bins")
    label = planet or str(d["planet"])

    print("\n" + "=" * 86)
    print("  H8 - TauREx vs petitRADTRANS, matched atmosphere and matched line lists (by DOI)")
    print("=" * 86)
    print(f"  {label}, grid={grid}, {len(wl)} bins, {wl.min():.3f}-{wl.max():.3f} um")
    print(f"  denominator {med_err:.1f} ppm ({err_src})")
    print(f"  {n-1} posterior draws plus the weighted mean (row 'mean')\n")
    print(f"  {'draw':>6}{'offset ppm':>12}{'shape rms':>11}{'shape max':>11}"
          f"{'dRp (RJ)':>11}{'chi2 taurex':>13}{'chi2 prt':>10}")
    for k in range(n):
        lab = "mean" if k == 0 else str(k)
        o, sr, sm, dr, ct, cp = rows[k]
        print(f"  {lab:>6}{o:>12.1f}{sr:>11.1f}{sm:>11.1f}{dr:>11.5f}{ct:>13.3f}{cp:>10.3f}")

    post = rows[1:]
    print("\n" + "-" * 86)
    print(f"  at the posterior mean:")
    print(f"    constant depth offset   {rows[0,0]:>9.1f} ppm   "
          f"= {abs(rows[0,0])/med_err:.3f} x the median error bar")
    print(f"    shape residual (rms)    {rows[0,1]:>9.1f} ppm   "
          f"= {rows[0,1]/med_err:.3f} x the median error bar")
    print(f"    equivalent radius shift {rows[0,3]:>9.5f} RJ")
    print(f"  across {len(post)} posterior draws:")
    print(f"    offset      {post[:,0].min():>8.1f} to {post[:,0].max():.1f} ppm")
    print(f"    shape rms   {post[:,1].min():>8.1f} to {post[:,1].max():.1f} ppm")
    print(f"    dRp         {post[:,3].min():>8.5f} to {post[:,3].max():.5f} RJ")
    if depth is not None:
        print(f"    chi2/N      taurex {post[:,4].mean():.3f}, prt {post[:,5].mean():.3f}")
    print("-" * 86)

    # The comparison that decides whether this matters: the A5 anchor's own
    # 68% radius width, and its distance to the literature value.
    A5_WIDTH, A5_SIGMA_GAP = 0.0057, 0.0029   # RJ: 68% width, and 1.0 sigma half-width
    dr = abs(rows[0, 3])
    if label == "wasp96":
        print(f"\n  A5 anchor 68% radius width = {A5_WIDTH:.4f} RJ (P6-D18)")
        print(f"  code-choice radius shift   = {dr:.5f} RJ = {dr/A5_WIDTH:.2f} x that width")
        print(f"                             = {dr/A5_SIGMA_GAP:.2f} x the 1 sigma gap to the literature 1.20")
    else:
        print(f"\n  code-choice radius shift   = {dr:.5f} RJ")
        print("  (the A5 radius-width comparison is WASP-96b specific, P6-D18, so it is not printed here)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="data/h8/taurex_dump.npz")
    ap.add_argument("--prt", default="data/h8/prt_spectra.npz")
    ap.add_argument("--grid", default="obs", choices=["obs", "g395h"],
                    help="obs = the observation's own bins; g395h = V5/H7's 2.80-5.10 um grid")
    ap.add_argument("--ref-err-ppm", type=float, default=None,
                    help="denominator for the x-error-bar figures, e.g. 50.01 for V5/H7")
    ap.add_argument("--planet", default=None)
    a = ap.parse_args()
    main(a.dump, a.prt, grid=a.grid, ref_err_ppm=a.ref_err_ppm, planet=a.planet)
