"""
H8 - TauREx's FluxBinner as an explicit linear operator, usable outside TauREx.

The pRT retrieval runs in `mirage-prt`, where taurex.binning does not exist, but
it has to bin its model spectra EXACTLY the way the TauREx retrieval did or the
two posteriors are not comparable and H8 proves nothing. Binning is linear in
the spectrum, so it can be precomputed once as a weight matrix W and applied as
W @ spectrum.

This reimplements taurex.binning.FluxBinner.bindown: overlap-weighted mean, with
native bin widths from taurex.util.compute_bin_edges. It is NOT trusted on the
strength of reading the source. `--validate` checks it against the real
FluxBinner on real spectra, and the retrieval refuses to start unless that
passes (see h8_prt_retrieve.py).
"""

import numpy as np


def native_widths(grid):
    """taurex.util.compute_bin_edges widths, for a sorted grid."""
    d = np.diff(grid) / 2
    edges = np.concatenate([[grid[0] - d[0]], grid[:-1] + d, [grid[-1] + d[-1]]])
    return np.diff(edges)


def bin_matrix(native, target, target_width):
    """W with shape (len(target), len(native)) so that W @ f == FluxBinner(f)."""
    o = np.argsort(native)
    native = native[o]
    w = native_widths(native)
    lo, hi = native - w / 2, native + w / 2
    tlo = target - target_width / 2
    thi = target + target_width / 2

    W = np.zeros((len(target), len(native)))
    for i in range(len(target)):
        ov = np.minimum(thi[i], hi) - np.maximum(tlo[i], lo)
        ov = np.clip(ov, 0.0, None)
        s = ov.sum()
        if s > 0:
            W[i] = ov / s
    # undo the sort so W applies to the caller's original ordering
    Wu = np.zeros_like(W)
    Wu[:, o] = W
    return Wu


def _validate():
    """Compare against the real FluxBinner on the H8 spectra. mirage-taurex only."""
    from taurex.binning import FluxBinner
    d = np.load("data/h8/taurex_dump.npz", allow_pickle=True)
    p = np.load("data/h8/prt_spectra.npz", allow_pickle=True)
    wl, width = d["obs_wl"], d["obs_width"]
    worst = 0.0
    for name, grid, spectra in (("taurex", 10000.0 / d["native_wn"], d["rprs2"]),
                                ("prt", p["wl_um"], p["rprs2"])):
        W = bin_matrix(np.asarray(grid), wl, width)
        fb = FluxBinner(wl, width)
        o = np.argsort(grid)
        for k in range(len(spectra)):
            _, ref, _, _ = fb.bindown(np.asarray(grid)[o], spectra[k][o])
            mine = W @ spectra[k]
            e = float(np.max(np.abs(mine - ref))) * 1e6      # ppm
            worst = max(worst, e)
        print(f"  {name:<8} {len(spectra)} spectra, worst |mine - FluxBinner| = {e:.3e} ppm")
    print(f"\n  worst overall: {worst:.3e} ppm")
    ok = worst < 1e-6
    print("  PASS: the matrix reproduces FluxBinner" if ok else "  FAIL: binning differs")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _validate() else 1)
