"""
A5 part 1 (G4): does the NS anchor move with live points?

G4 says the NS anchors were never shown stable and everything cites them. This
scores the WASP-96b sweep at 150 / 300 / 600 / 1000 live points on the three
things that would make the anchor unusable if they moved:

  1. the radius 68% interval, because P5-D7 makes WASP-96b the ONE target whose
     interval contains the literature value, so the whole multi-target claim
     rests on this interval and not on a mean
  2. posterior width per parameter, since a reference whose width is a function
     of a sampler setting cannot calibrate anything
  3. log Z, the most live-point-sensitive quantity in nested sampling, parsed
     from the run logs because taurex_retrieve.py does not save it

Read: if widths and the radius interval are flat across a 6.7x change in live
points, the anchor is stable at the setting the paper used and G4 closes. If they
drift monotonically, the published 150-point anchor is under-converged and every
number anchored to it inherits that.

    python scripts/a5_livesweep_table.py
"""

import re
from pathlib import Path

import numpy as np

OUT = Path("data/real_ess")
LIVES = [150, 300, 600, 1000]
WASP96_LIT_RADIUS = 1.20   # RJ, geometry truth used throughout P5


def wq(v, w, q):
    """Weighted quantile, same convention as run_wasp96_eval.sh."""
    i = np.argsort(v)
    return float(np.interp(q, np.cumsum(w[i]), v[i]))


def logz_from_log(path):
    """Final logz from a nestle log. Lines are \\r-separated and prefixed [Kit=."""
    if not path.exists():
        return float("nan")
    txt = path.read_bytes().decode("utf-8", "replace").replace("\r", "\n")
    hits = re.findall(r"logz=(-?\d+\.?\d*)", txt)
    return float(hits[-1]) if hits else float("nan")


def load(live):
    f = OUT / f"wasp39b_ns_posterior_wasp96_live{live}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    s, w = d["samples"], d["weights"].astype(float)
    w = w / w.sum()
    names = [str(x) for x in d["fit_names"]]
    return dict(live=live, s=s, w=w, names=names, chi2=float(d["chi2"]),
                nsamp=len(s), ess=float(w.sum() ** 2 / np.sum(w ** 2)),
                logz=logz_from_log(OUT / f"ns_wasp96_live{live}.log"))


def main():
    runs = [r for r in (load(l) for l in LIVES) if r is not None]
    if not runs:
        raise SystemExit("no sweep outputs found; run scripts/run_a5_livesweep.sh first")
    names = runs[0]["names"]
    ri = names.index("planet_radius")

    print("\n" + "=" * 78)
    print("  A5 / G4 - WASP-96b NS anchor stability against live points")
    print("=" * 78)
    print(f"{'live':>6}{'niter-ish':>11}{'ESS':>9}{'logZ':>11}{'chi2/N':>9}"
          f"{'radius 16%':>12}{'84%':>9}{'width':>8}")
    for r in runs:
        lo = wq(r["s"][:, ri], r["w"], 0.16)
        hi = wq(r["s"][:, ri], r["w"], 0.84)
        print(f"{r['live']:>6}{r['nsamp']:>11}{r['ess']:>9.1f}{r['logz']:>11.3f}"
              f"{r['chi2']:>9.3f}{lo:>12.4f}{hi:>9.4f}{hi - lo:>8.4f}")

    print(f"\n  literature radius {WASP96_LIT_RADIUS} RJ inside the 68% interval?")
    for r in runs:
        lo = wq(r["s"][:, ri], r["w"], 0.16)
        hi = wq(r["s"][:, ri], r["w"], 0.84)
        inside = lo <= WASP96_LIT_RADIUS <= hi
        print(f"    live {r['live']:>4}:  [{lo:.4f}, {hi:.4f}]  -> {'YES' if inside else 'NO'}")

    print("\n  68% width per parameter, and drift across the sweep")
    print(f"    {'param':<16}" + "".join(f"{('live '+str(r['live'])):>12}" for r in runs)
          + f"{'max/min':>10}")
    for j, nm in enumerate(names):
        wd = [wq(r["s"][:, j], r["w"], 0.84) - wq(r["s"][:, j], r["w"], 0.16) for r in runs]
        ratio = max(wd) / min(wd) if min(wd) > 0 else float("inf")
        print(f"    {nm:<16}" + "".join(f"{x:>12.4f}" for x in wd) + f"{ratio:>10.2f}")

    print("\n  weighted mean per parameter")
    print(f"    {'param':<16}" + "".join(f"{('live '+str(r['live'])):>12}" for r in runs)
          + f"{'spread':>10}")
    for j, nm in enumerate(names):
        mu = [float(np.average(r["s"][:, j], weights=r["w"])) for r in runs]
        print(f"    {nm:<16}" + "".join(f"{x:>12.4f}" for x in mu)
              + f"{max(mu) - min(mu):>10.4f}")

    # The verdict is about the radius interval, because that is what P5-D7 made
    # load-bearing. Width ratio over a 6.7x live-point change is the test.
    wd = [wq(r["s"][:, ri], r["w"], 0.84) - wq(r["s"][:, ri], r["w"], 0.16) for r in runs]
    mu = [float(np.average(r["s"][:, ri], weights=r["w"])) for r in runs]
    print("\n" + "-" * 78)
    print(f"  radius 68% width: {min(wd):.4f} to {max(wd):.4f}, ratio {max(wd)/min(wd):.2f}x")
    print(f"  radius mean drift: {max(mu)-min(mu):.4f} RJ over live {runs[0]['live']}-{runs[-1]['live']}")
    print(f"  logZ drift: {max(r['logz'] for r in runs)-min(r['logz'] for r in runs):.3f}")
    print("-" * 78)


if __name__ == "__main__":
    main()
