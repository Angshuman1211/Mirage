"""Fair real-data metric: MIRAGE's posterior on a real target vs the nested-sampling
ANCHOR, compared parameter-by-parameter and by 1-D Jensen-Shannon divergence.

Why this metric. Raw IS-ESS is destroyed by real JWST's tiny sigma against the
forward-model systematic (P3-D5), and P5-D7 showed the importance weights collapse
onto a single draw (ESS 1.0-12), so anything computed THROUGH those weights is not
trustworthy. This comparison uses the RAW, UNWEIGHTED flow samples, so it never
touches importance sampling and is immune to that collapse. That makes it the
defensible real-side metric for the transfer claim.

Fixed 2026-10-24 (P5-D10). The previous version had three defects:
  - it loaded `wasp39b_ns_posterior.npz`, which is the T-GRADIENT run
    (fit_names = [T_surface, T_top, ...]), not the free-radius anchor;
  - it indexed the anchor BY POSITION against a hardcoded 6-param name list, so
    with that anchor it silently compared temperature against T_surface and water
    against T_top;
  - it read the generic `real_ess_samples.npz`, which is shared across planets and
    arms (P5-D6), so the arm being compared was whatever ran last.
Parameters are now matched BY NAME and a mismatch is a hard error.

    python scripts/real_compare.py --samples data/real_ess/abl_rad_nocond_samples.npz
    python scripts/real_compare.py --samples ... --ns wasp39b_ns_posterior_wasp96.npz
"""
import argparse
from pathlib import Path

import numpy as np

OUT = Path("data/real_ess")

# theta layout of the flow, keyed by dimension. 7 = the radius models (P3-D11).
THETA_LAYOUT = {
    7: ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"],
    6: ["T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"],
}


def wquantile(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    c /= w.sum()
    return np.interp(q, c, x)


def js_1d(xa, wa, xb, wb, bins=40):
    lo = min(xa.min(), xb.min()); hi = max(xa.max(), xb.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(xa, edges, weights=wa, density=True)
    pb, _ = np.histogram(xb, edges, weights=wb, density=True)
    pa = pa / pa.sum() + 1e-12; pb = pb / pb.sum() + 1e-12
    m = 0.5 * (pa + pb)
    kl = lambda p, q: np.sum(p * np.log(p / q))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)      # nats; 0 = identical, ln2 = disjoint


def compare(samples_path, ns_path, quiet=False, allow_partial=False):
    mi = np.load(samples_path, allow_pickle=True)
    ns = np.load(ns_path, allow_pickle=True)

    mi_th = mi["theta"]
    layout = THETA_LAYOUT.get(mi_th.shape[1])
    if layout is None:
        raise SystemExit(f"[guard] unknown theta width {mi_th.shape[1]} in {samples_path}")

    ns_names = [str(n) for n in ns["fit_names"]]
    shared = [n for n in layout if n in ns_names]
    missing = [n for n in layout if n not in ns_names]
    if not shared:
        raise SystemExit(
            f"[guard] no shared parameters between flow {layout} and anchor {ns_names}. "
            f"Wrong anchor for this arm — the free-radius anchor is "
            f"wasp39b_ns_posterior_fitrad.npz.")
    if missing and not allow_partial:
        # A mean JS over 5 parameters is not comparable to one over 7. Quoting them
        # side by side is the same class of error as the P5-D4 floor mismatch, so
        # refuse by default rather than emit a number that looks comparable.
        raise SystemExit(
            f"[guard] anchor {Path(ns_path).name} lacks {missing}, so only "
            f"{len(shared)}/{len(layout)} parameters can be compared and the mean JS "
            f"would NOT be comparable to a full-parameter run. The free-radius anchor "
            f"is wasp39b_ns_posterior_fitrad.npz. Pass --allow-partial to override.")
    if missing and not quiet:
        print(f"  [note] PARTIAL: anchor lacks {missing} — mean JS is over "
              f"{len(shared)}/{len(layout)} parameters and is not comparable to a full run")

    ns_th = ns["samples"]; ns_w = ns["weights"] / ns["weights"].sum()
    mi_w = np.full(len(mi_th), 1.0 / len(mi_th))       # UNWEIGHTED: no importance sampling

    rows, jss = [], []
    for nm in shared:
        a = ns_th[:, ns_names.index(nm)]
        b = mi_th[:, layout.index(nm)]
        js = js_1d(a, ns_w, b, mi_w)
        jss.append(js)
        rows.append((nm, wquantile(a, ns_w, [0.16, 0.5, 0.84]),
                     wquantile(b, mi_w, [0.16, 0.5, 0.84]), js))
    return rows, float(np.mean(jss))


def main(samples, ns, quiet, allow_partial):
    sp = Path(samples) if "/" in samples else OUT / samples
    np_ = Path(ns) if "/" in ns else OUT / ns
    rows, mean_js = compare(sp, np_, quiet, allow_partial)
    print(f"\nMIRAGE posterior vs NS anchor   (unweighted flow samples, IS-free)")
    print(f"  flow   : {sp.name}")
    print(f"  anchor : {np_.name}\n")
    print(f"  {'param':<15}{'NS median [16,84]':>26}{'MIRAGE median [16,84]':>28}{'JS':>7}")
    print("  " + "-" * 76)
    for nm, (a, b, c), (d, e, f), js in rows:
        print(f"  {nm:<15}{b:>10.3f} [{a:.2f},{c:.2f}]{e:>14.3f} [{d:.2f},{f:.2f}]{js:>7.3f}")
    print("  " + "-" * 76)
    print(f"  mean 1-D JS = {mean_js:.3f} nats   (0 = identical, {np.log(2):.2f} = disjoint)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="abl_rad_nocond_samples.npz",
                    help="per-arm posterior npz (NOT the shared real_ess_samples.npz)")
    ap.add_argument("--ns", default="wasp39b_ns_posterior_fitrad.npz",
                    help="nested-sampling anchor; must share parameter names with the flow")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="permit an anchor missing some flow parameters (mean JS then NOT comparable)")
    a = ap.parse_args()
    main(a.samples, a.ns, a.quiet, a.allow_partial)
