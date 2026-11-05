"""Build the CycleGAN's REAL domain (B) from genuinely independent JWST frames.

Why. The published ablation drew domain B from ONE reduced WASP-39b spectrum
bootstrap-expanded with 1% Gaussian jitter to 500 samples. A discriminator shown
500 near-copies of a single spectrum learns that spectrum, so the D4 comparison
was never a fair test -- reviewer Viae raised exactly this. Every target has real
out-of-transit frames on disk (WASP-39b PRISM alone has 3,241), which are genuine
independent realisations of the instrument's own correlated noise.

Construction. For target t:
    real_sample = depth_t * (1 + r_i),   r_i = frame_i / mean(frames) - 1
so the signal is the real reduced spectrum and the scatter is a REAL noise
realisation rather than synthetic jitter. This also sharpens the question the
ablation asks: the domain-randomisation baseline emulates this noise with a
parametric SE+OU kernel, so the comparison becomes "can a learned translator
reproduce real correlated noise better than a hand-designed one".

Caveat to state in the paper, not hide: OOT residuals characterise FLUX noise and
are used here as a proxy for depth noise. Still far closer to reality than Gaussian
jitter, but it is an approximation.

Grids:
    abc        all 52 ABC model bins -- WASP-39b/PRISM only, the single-target run
    abcniriss  the 26 NATIVE ABC bins inside 0.85-2.83um, shared by all three
               targets for the cross-target comparison

Why the shared grid restricts the ABC grid instead of resampling onto a new one.
Only 26 of the 52 ABC bins fall in the NIRISS range, and they are coarse below
1.9um (~0.2um spacing) then dense above (~0.04um). Interpolating ABC UP onto a
uniform 52-bin grid would invent 2x resolution and leave domain A almost linear
across the blue half, which a discriminator separates on smoothness alone rather
than on noise structure -- the same free-shortcut defect as filling uncovered
bins with a constant. Binning the REAL frames DOWN onto native ABC bins invents
nothing on either side.

    python scripts/build_real_domain.py --target wasp39 --grid abc --n 2000
    python scripts/build_real_domain.py --target wasp96 --grid abcniriss
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_real_covariance import ABC as ABC_HDF, rebin_frames   # noqa: E402

OUT = Path("data/real_domain"); OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "wasp39": dict(spec="data/jwst_wasp39b_spectrum.csv",
                   oot="data/jwst_wasp39b_oot/NIRSPEC_CLEAR_PRISM_out_of_transit_full.csv"),
    "wasp96": dict(spec="data/jwst_wasp96b_spectrum.csv",
                   oot="data/jwst_wasp96b_oot/NIRISS_CLEAR_N_A_out_of_transit_full.csv"),
    "k218":   dict(spec="data/jwst_k2_18b_spectrum.csv",
                   oot="data/jwst_k2_18b_oot/NIRISS_CLEAR_N_A_out_of_transit_full.csv"),
}


NIRISS_LO, NIRISS_HI = 0.85, 2.83


def abc_stored_grid():
    """The 52-bin ABC grid in the HDF's STORED order, which is descending.

    This is the order the hdf's `flux` columns sit in, and the order the network
    is fed: mirage/datasets/real_jwst.py bins a real spectrum on the ascending
    grid and then maps it BACK to stored order before building the context.
    Domain B has to match it, or domain A and domain B are wavelength-reversed
    relative to each other.
    """
    with h5py.File(ABC_HDF) as f:
        return f["wlen"][0].astype(float)


def log_edges(wl_asc):
    e = np.empty(len(wl_asc) + 1)
    e[1:-1] = np.sqrt(wl_asc[:-1] * wl_asc[1:])
    e[0] = wl_asc[0] ** 2 / e[1]
    e[-1] = wl_asc[-1] ** 2 / e[-2]
    return e


def fill_uncovered(x, wl, cov):
    """Interpolate uncovered bins PER SAMPLE. Done once here, at build time.

    A constant fill would give every real sample identical values in those bins
    while the sim side varies, and a discriminator takes that as a free
    real-vs-fake shortcut instead of learning noise structure. Interpolating per
    sample keeps each frame's own variation and leaves no constant anywhere.

    Done here rather than in the loaders because np.interp needs INCREASING xp
    while `wl` is in stored (descending) order, so a loader that passed `wl`
    straight to np.interp would silently return garbage.
    """
    if cov.all():
        return x
    o = np.argsort(wl)                       # ascending view for np.interp
    wa, ca = wl[o], cov[o]
    for i in range(x.shape[0]):
        xa = x[i, o]
        xa[~ca] = np.interp(wa[~ca], wa[ca], xa[ca])
        x[i, o] = xa
    return x


def rebin_spectrum(df, edges):
    """Inverse-variance weighted depth per bin (same weighting as taurex_retrieve)."""
    n = len(edges) - 1
    b = np.digitize(df["wavelength_um"].to_numpy(), edges) - 1
    d = df["transit_depth"].to_numpy()
    e = np.clip(df["depth_error"].to_numpy(), 1e-12, None)
    depth = np.full(n, np.nan)
    for k in range(n):
        m = b == k
        if not m.any():
            continue
        w = 1.0 / e[m] ** 2
        depth[k] = np.sum(w * d[m]) / np.sum(w)
    return depth


def main(target, grid, n, seed, suffix=""):
    t = TARGETS[target]
    wl_s = abc_stored_grid()                 # the model's bin order (descending)
    order = np.argsort(wl_s)                 # -> ascending, which binning needs
    inv = np.argsort(order)                  # ascending -> stored
    edges = log_edges(wl_s[order])

    depth_a = rebin_spectrum(pd.read_csv(t["spec"]), edges)

    oot = pd.read_csv(t["oot"], usecols=["wavelength_um", "flux_jy", "dq_flag",
                                         "integration_num"])
    oot = oot[oot["dq_flag"] == 0]
    M_a, ints = rebin_frames(oot, edges)                    # (n_frames, 52) ascending

    # Back to the model's stored bin order, the step real_jwst.py takes before
    # handing a real spectrum to the network.
    depth, M = depth_a[inv], M_a[:, inv]

    # Bin on the FULL grid, then slice. The edges are log midpoints of the full
    # grid either way, so slicing after binning gives exactly the wanted bins at
    # their native widths, and no contiguity assumption is needed.
    cols = (np.arange(len(wl_s)) if grid == "abc"
            else np.where((wl_s >= NIRISS_LO) & (wl_s <= NIRISS_HI))[0])
    wl, depth, M = wl_s[cols], depth[cols], M[:, cols]

    covered = np.isfinite(M).all(axis=0) & np.isfinite(depth)
    if covered.sum() == 0:
        raise SystemExit(f"[guard] no bins covered by BOTH spectrum and OOT frames for "
                         f"{target} on grid '{grid}' — wrong grid for this instrument?")

    # real fractional noise per frame, on covered bins only
    rel = np.zeros_like(M)
    rel[:, covered] = M[:, covered] / M[:, covered].mean(axis=0) - 1.0

    rng = np.random.default_rng(seed)
    idx = np.arange(len(ints)) if n is None or n >= len(ints) else \
        rng.choice(len(ints), size=n, replace=False)
    samples = np.full((len(idx), len(depth)), np.nan)
    samples[:, covered] = depth[covered][None, :] * (1.0 + rel[idx][:, covered])
    samples = fill_uncovered(samples, wl, covered)
    if not np.isfinite(samples).all():
        raise SystemExit("[guard] non-finite values survived the uncovered-bin fill")

    out = OUT / f"real_domain_{target}_{grid}{suffix}.npz"
    # `samples` is already gap-filled and finite, so loaders just load it.
    # `covered` stays for bookkeeping: which bins carry real measurements.
    # `abc_cols` indexes the STORED-order ABC grid, letting the trainer slice
    # domain A to exactly these wavelengths. Its absence is how downstream
    # scripts detect a file with no ABC correspondence and refuse, rather than
    # silently pairing mismatched bins.
    np.savez(out, samples=samples.astype(np.float32), wlen=wl, covered=covered,
             depth=depth, frame_ids=ints[idx], target=target, grid=grid,
             abc_cols=cols, source=Path(t["oot"]).name)

    scat = np.nanstd(rel[idx][:, covered], axis=0).mean()
    print(f"  target     {target}  ({Path(t['oot']).name})")
    print(f"  grid       {grid}, {len(wl)} bins, {wl.min():.2f}-{wl.max():.2f} µm "
          f"(stored order, descending — matches the hdf's flux columns)")
    print(f"  frames     {len(ints)} available -> {len(idx)} used "
          f"({'all' if len(idx) == len(ints) else f'random subset, seed {seed}'})")
    print(f"  covered    {covered.sum()}/{len(covered)} bins")
    print(f"  depth      median {np.nanmedian(depth[covered]):.5f}")
    print(f"  real noise mean per-bin fractional scatter = {scat:.4f} "
          f"(vs 0.01 synthetic jitter in the published run)")
    print(f"  saved      {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS), required=True)
    ap.add_argument("--grid", choices=["abc", "abcniriss"], default="abc")
    ap.add_argument("--n", type=int, default=None, help="frames to use (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="appended to the output "
                    "filename, so a second frame count does not overwrite the first")
    a = ap.parse_args()
    main(a.target, a.grid, a.n, a.seed, a.suffix)
