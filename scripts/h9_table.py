"""H9 result table: the conditioning effect against training-seed dispersion.

Reads the npz files written by run_noisecond_ablation.py --train-seed and
compares them to the published numbers.

Why the published numbers cannot be used as a baseline (P6-D1): both no-cond
arms were trained on MPS and under-trained there. ABC reached best val 1.923
against 1.390 for the identical config and seed on CUDA, and its train loss
moved 1.9945 to 1.9365 across 125 epochs. The published 0.587 and 0.817 are
the coverage of models that did not train.

    python scripts/h9_table.py
"""
import numpy as np
from pathlib import Path

OUT = Path("data/real_ess")
NOMINAL = 0.68
SEEDS = (42, 43, 44)
ARMS = {"no-cond": "s__no_cond", "+sigma+cov": "s__psigmapcov"}
LD = {"no-cond": "ld__no_cond", "+sigma+cov": "ld__psigmapcov"}

# Published table, Documentation/07_state_gaps_and_research_agenda.md section 4.2.
PUBLISHED = {"abc": {"no-cond": 0.587, "+sigma+cov": 0.687},
             "rad": {"no-cond": 0.817, "+sigma+cov": 0.700}}


def coverage68(samples, truths):
    lo = np.quantile(samples, (1 - NOMINAL) / 2, axis=1)
    hi = np.quantile(samples, 1 - (1 - NOMINAL) / 2, axis=1)
    return float(np.mean((lo <= truths) & (truths <= hi)))


def load(fam, seed):
    p = OUT / f"h9_{fam}_s{seed}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    t = d["truths"]
    return {a: (coverage68(d[k], t), float(np.mean(d[LD[a]]))) for a, k in ARMS.items()}


def main():
    for fam, label in (("abc", "ABC (6-param)"), ("rad", "radius (7-param)")):
        runs = {s: load(fam, s) for s in SEEDS}
        have = [s for s in SEEDS if runs[s]]
        if not have:
            print(f"\n{label}: no runs yet")
            continue

        print(f"\n{'='*78}\n  {label}   coverage@68, nominal {NOMINAL}\n{'='*78}")
        hdr = f"  {'arm':<13}" + "".join(f"{'s'+str(s):>9}" for s in have)
        print(hdr + f"{'mean':>9}{'sd':>8}{'|err|':>8}{'published':>11}{'Δ':>8}")
        print("  " + "-" * (len(hdr) + 42))

        err = {}
        for a in ARMS:
            v = np.array([runs[s][a][0] for s in have])
            err[a] = abs(v.mean() - NOMINAL)
            pub = PUBLISHED[fam][a]
            print(f"  {a:<13}" + "".join(f"{x:>9.3f}" for x in v) +
                  f"{v.mean():>9.3f}{v.std(ddof=1) if len(v) > 1 else 0:>8.3f}"
                  f"{err[a]:>8.3f}{pub:>11.3f}{v.mean()-pub:>+8.3f}")

        print(f"\n  {'logdens':<13}" + "".join(f"{'s'+str(s):>9}" for s in have) + f"{'mean':>9}")
        for a in ARMS:
            v = np.array([runs[s][a][1] for s in have])
            print(f"  {a:<13}" + "".join(f"{x:>9.2f}" for x in v) + f"{v.mean():>9.2f}")

        # The H9 test: is the conditioning effect bigger than the spread it sits in?
        if len(have) > 1:
            d0 = np.array([runs[s]["no-cond"][0] for s in have])
            d1 = np.array([runs[s]["+sigma+cov"][0] for s in have])
            pooled = float(np.sqrt((d0.var(ddof=1) + d1.var(ddof=1)) / 2))
            delta = err["no-cond"] - err["+sigma+cov"]
            pub_delta = abs(PUBLISHED[fam]["no-cond"] - NOMINAL) - abs(PUBLISHED[fam]["+sigma+cov"] - NOMINAL)
            print(f"\n  calibration error {err['no-cond']:.3f} -> {err['+sigma+cov']:.3f} "
                  f"(conditioning gains {delta:+.3f})")
            print(f"  published gain was {pub_delta:+.3f}, so the effect shrank "
                  f"{abs(pub_delta/delta) if delta else float('inf'):.0f}x")
            print(f"  pooled seed sd {pooled:.3f} -> effect is {abs(delta)/pooled:.1f}x the spread: "
                  f"{'SURVIVES' if abs(delta) > 2*pooled else 'NOT RESOLVED BY'} seed noise")
    print()


if __name__ == "__main__":
    main()
