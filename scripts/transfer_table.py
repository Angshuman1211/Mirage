"""The transfer table: does noise-conditioning survive the move from synthetic
benchmarks to real instrument noise?

This is the central evidence artifact for the sim-to-real claim, built because the
numbers behind it were previously scattered across an ad-hoc shell heredoc and a
single-arm script that read a shared filename (P5-D6, P5-D8, P5-D10).

Two sides, both deliberately free of importance sampling, because the IS weights on
real data collapse onto one draw (ESS 1.0-12, P5-D7) and nothing computed through
them is trustworthy:

  SYNTHETIC  central-interval coverage at 68% against KNOWN truth, from the saved
             ablation arrays. Reported for BOTH model families, because the published
             win was measured on the 6-param ABC arms while the real-data test used
             the 7-param radius arms -- a cross-family comparison until now (P5-D10).

  REAL       mean 1-D Jensen-Shannon divergence between the RAW unweighted flow
             posterior and the independent nested-sampling anchor on real WASP-39b.
             Lower = closer to the trusted answer.

    python scripts/transfer_table.py
"""
import numpy as np
from pathlib import Path

from real_compare import compare

OUT = Path("data/real_ess")
NOMINAL = 0.68

ARMS = ["no-cond", "+sigma", "+sigma+cov"]
SYN_KEY = {"no-cond": "s__no_cond", "+sigma": "s__psigma", "+sigma+cov": "s__psigmapcov"}
SYNTHETIC = [("ABC (6-param)", "abc_ablation.npz"),
             ("radius (7-param)", "abc_rad_ablation.npz")]
REAL = {"no-cond": "abl_rad_nocond_samples.npz",
        "+sigma": "abl_rad_sigma_samples.npz",
        "+sigma+cov": "abl_rad_samples.npz"}
ANCHOR = "wasp39b_ns_posterior_fitrad.npz"


def coverage68(samples, truths):
    """Fraction of (planet, parameter) pairs whose truth lies in the central 68%."""
    hits = []
    for s, t in zip(samples, truths):
        lo = np.quantile(s, (1 - NOMINAL) / 2, axis=0)
        hi = np.quantile(s, 1 - (1 - NOMINAL) / 2, axis=0)
        hits.append((lo <= t) & (t <= hi))
    return float(np.mean(hits))


def main():
    syn = {}
    for fam, fn in SYNTHETIC:
        p = OUT / fn
        if not p.exists():
            print(f"  [skip] {fam}: {fn} not found")
            continue
        d = np.load(p, allow_pickle=True)
        truths = list(d["truths"])
        syn[fam] = {a: coverage68(list(d[SYN_KEY[a]]), truths)
                    for a in ARMS if SYN_KEY[a] in d}

    real = {}
    for a, fn in REAL.items():
        try:
            _, mean_js = compare(OUT / fn, OUT / ANCHOR, quiet=True)
            real[a] = mean_js
        except Exception as exc:
            print(f"  [skip] real {a}: {type(exc).__name__}: {exc}")

    fams = list(syn)
    print(f"\n{'='*94}")
    print("  NOISE-CONDITIONING: SYNTHETIC BENCHMARK  vs  REAL INSTRUMENT NOISE")
    print(f"{'='*94}")
    head = f"  {'arm':<13}"
    for fam in fams:
        head += f"{fam + ' cov@68':>26}"
    head += f"{'real JS to NS':>18}"
    print(head)
    print("  " + "-" * 90)
    for a in ARMS:
        row = f"  {a:<13}"
        for fam in fams:
            c = syn[fam].get(a)
            row += f"{c:>18.3f} (err {abs(c-NOMINAL):.3f})" if c is not None else f"{'--':>26}"
        row += f"{real.get(a, float('nan')):>18.3f}"
        print(row)
    print("  " + "-" * 90)
    print(f"  synthetic: coverage vs KNOWN truth, nominal {NOMINAL:.2f} — closer to nominal is better")
    print(f"  real     : mean 1-D JS to {ANCHOR} — LOWER is closer to the trusted answer")

    # the two transfer failures, stated from the numbers rather than asserted
    print()
    if {"no-cond", "+sigma+cov"} <= set(real):
        d_real = real["+sigma+cov"] - real["no-cond"]
        verdict = "WORSE" if d_real > 0 else "better"
        print(f"  [real] +sigma+cov is {verdict} than no-cond by {abs(d_real):.3f} nats")
    for fam in fams:
        if {"no-cond", "+sigma+cov"} <= set(syn[fam]):
            e0 = abs(syn[fam]["no-cond"] - NOMINAL)
            e1 = abs(syn[fam]["+sigma+cov"] - NOMINAL)
            print(f"  [synthetic/{fam}] +sigma+cov calibration error "
                  f"{e0:.3f} -> {e1:.3f} ({'improved' if e1 < e0 else 'degraded'})")
    if len(fams) == 2 and all("+sigma" in syn[f] for f in fams):
        e = {f: abs(syn[f]["+sigma"] - NOMINAL) for f in fams}
        print(f"  [replication] +sigma calibration error differs across families: "
              + " vs ".join(f"{f} {e[f]:.3f}" for f in fams))


if __name__ == "__main__":
    main()
