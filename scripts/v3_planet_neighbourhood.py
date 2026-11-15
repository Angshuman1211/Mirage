"""V3 / H1 (closes G7): restrict CycleGAN domain A to one planet's neighbourhood.

H1, from P5-D11: domain A is a distribution over ~73k different PLANETS, while
domain B is a distribution over NOISE REALISATIONS OF ONE PLANET. A translator
asked to map the first onto the second has to collapse a population spread into
a single-target noise spread, which would explain marginals that agree alongside
a mean displaced by 0.308 per-bin sigma (tier 1).

The test: rebuild domain A from only the ABC training spectra nearest the real
WASP-39b spectrum, retrain once, and see whether the mean shift collapses. If it
does, the negative gains a mechanism. If not, H1 is struck from the decision log.

Nearness is measured in DEPTH space and offset-invariant. The ABC grid fixes the
planet radius, so a real spectrum and its closest ABC atmosphere can differ by a
near-constant depth offset (the missing-radius story, P3-D11). Subtracting each
spectrum's own mean over the covered bins before the distance removes that
offset and leaves spectral shape, which is what "the same planet" should mean
here. The fill bins outside the reduced spectrum are excluded.

    python scripts/v3_planet_neighbourhood.py --k 2000
writes data/real_domain/v3_wasp39_neighbourhood_k2000.npz with `idx` (abc_train
rows) and `dist`, read by `train_cyclegan.py --abc-index`.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_real_domain import TARGETS, log_edges, rebin_spectrum  # noqa: E402

ABC_TRAIN = Path("data/abc/abc_train.hdf")
OUT = Path("data/real_domain")


def main(target: str, k: int) -> None:
    with h5py.File(ABC_TRAIN) as f:
        flux = f["flux"][:].astype(np.float64)
        mean, std = f["flux_mean"][:], f["flux_std"][:]
        wl_s = f["wlen"][0].astype(float)
    depth_abc = flux * std + mean                       # back to transit depth

    order = np.argsort(wl_s)
    inv = np.argsort(order)
    edges = log_edges(wl_s[order])
    real = rebin_spectrum(pd.read_csv(TARGETS[target]["spec"]), edges)[inv]
    cov = np.isfinite(real)
    if cov.sum() < 10:
        raise SystemExit(f"[guard] only {cov.sum()} bins covered; wrong grid?")

    a = depth_abc[:, cov] - depth_abc[:, cov].mean(axis=1, keepdims=True)
    r = real[cov] - real[cov].mean()
    d = np.sqrt(((a - r) ** 2).mean(axis=1))
    idx = np.argsort(d)[:k]
    offset = depth_abc[idx][:, cov].mean(axis=1) - real[cov].mean()

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"v3_{target}_neighbourhood_k{k}.npz"
    np.savez(out, idx=idx, dist=d[idx], all_dist_median=np.median(d),
             depth_offset=offset, covered=cov, target=target)
    print(f"  {target}: {cov.sum()}/{len(cov)} bins covered by the reduced spectrum")
    print(f"  shape RMS distance, depth units: nearest {d[idx[0]]:.2e}, k-th "
          f"{d[idx[-1]]:.2e}, population median {np.median(d):.2e}")
    print(f"  removed level offset over the {k}: median {np.median(offset):+.2e} "
          f"(real mean depth {real[cov].mean():.4f})")
    print(f"  saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="wasp39", choices=["wasp39"])
    ap.add_argument("--k", type=int, default=2000,
                    help="neighbourhood size; 2000 matches the published run's n_abc")
    a = ap.parse_args()
    main(a.target, a.k)
