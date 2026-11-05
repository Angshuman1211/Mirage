"""Seed dispersion for the noise-conditioning claim.

Every headline number in this project has been a single run (dQk2 W2, and the
v2 plan's Track-1 item 6). Under framing B, a venue that evaluates rigour
rather than novelty, single-run point values are the most likely reason a
reviewer stops reading. This runs the conditioning ablation over several seeds
and reports mean, spread and range instead of one number.

WHAT THE DISPERSION COVERS, precisely, because the distinction is load-bearing.
`--test-seed` seeds the correlated noise injected into the EVALUATION spectra,
so this measures dispersion over noise realisations at fixed trained weights.
It does NOT cover training-run variance, which would need each arm retrained
per seed. The claim this supports is therefore "the conditioning effect is
stable across noise draws", not "stable across training runs". Say it that way.

    python scripts/seed_dispersion.py --seeds 1234 1235 1236 1237 1238
    python scripts/seed_dispersion.py --families abc rad --n-planets 50
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT = Path("data/real_ess")
NOMINAL = 0.68
ARMS = ["no-cond", "+sigma", "+sigma+cov"]
SYN_KEY = {"no-cond": "s__no_cond", "+sigma": "s__psigma", "+sigma+cov": "s__psigmapcov"}


def coverage68(samples, truths):
    hits = []
    for s, t in zip(samples, truths):
        lo = np.quantile(s, (1 - NOMINAL) / 2, axis=0)
        hi = np.quantile(s, 1 - (1 - NOMINAL) / 2, axis=0)
        hits.append((lo <= t) & (t <= hi))
    return float(np.mean(hits))


def run_one(family, seed, n_planets, n_samples, force):
    path = OUT / f"seeddisp_{family}_{seed}.npz"
    if path.exists() and not force:
        print(f"  [cached] {path.name}")
        return path
    cmd = [sys.executable, "scripts/run_noisecond_ablation.py",
           "--family", family, "--test-seed", str(seed),
           "--n-planets", str(n_planets), "--n-samples", str(n_samples),
           "--save", str(path)]
    print(f"  running {family} seed {seed} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not path.exists():
        print(f"  [FAIL] {family} seed {seed}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return None
    return path


def main(families, seeds, n_planets, n_samples, force):
    table = {}
    for fam in families:
        per_arm = {a: [] for a in ARMS}
        used = []
        for seed in seeds:
            p = run_one(fam, seed, n_planets, n_samples, force)
            if p is None:
                continue
            d = np.load(p, allow_pickle=True)
            truths = list(d["truths"])
            ok = False
            for a in ARMS:
                if SYN_KEY[a] in d:
                    per_arm[a].append(coverage68(list(d[SYN_KEY[a]]), truths))
                    ok = True
            if ok:
                used.append(seed)
        table[fam] = (per_arm, used)

    print(f"\n{'='*86}")
    print("  NOISE CONDITIONING, COVERAGE@68 OVER SEEDS   (nominal 0.68)")
    print(f"{'='*86}")
    print("  dispersion is over EVALUATION-NOISE draws at fixed weights,")
    print("  not over training runs. See the module docstring.")
    for fam, (per_arm, used) in table.items():
        if not used:
            print(f"\n  {fam}: no successful runs")
            continue
        print(f"\n  family {fam}   ({len(used)} seeds: {', '.join(map(str, used))})")
        print(f"    {'arm':<13}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}{'|err|':>8}")
        print("    " + "-" * 53)
        for a in ARMS:
            v = np.asarray(per_arm[a], dtype=float)
            if v.size == 0:
                print(f"    {a:<13}{'--':>8}")
                continue
            print(f"    {a:<13}{v.mean():>8.3f}{v.std(ddof=1) if v.size > 1 else 0.0:>8.3f}"
                  f"{v.min():>8.3f}{v.max():>8.3f}{abs(v.mean()-NOMINAL):>8.3f}")

        # Is the conditioning effect larger than the seed spread it sits in?
        a0, a1 = np.asarray(per_arm["no-cond"]), np.asarray(per_arm["+sigma+cov"])
        if a0.size > 1 and a1.size > 1:
            e0, e1 = abs(a0.mean() - NOMINAL), abs(a1.mean() - NOMINAL)
            pooled = float(np.sqrt((a0.var(ddof=1) + a1.var(ddof=1)) / 2))
            delta = e0 - e1
            print(f"    calibration error {e0:.3f} -> {e1:.3f} (delta {delta:+.3f}), "
                  f"pooled seed sd {pooled:.3f}")
            if pooled > 0:
                print(f"    effect is {abs(delta)/pooled:.1f}x the seed spread -> "
                      f"{'SURVIVES' if abs(delta) > 2*pooled else 'NOT RESOLVED BY'} seed noise")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["abc", "rad"])
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[1234, 1235, 1236, 1237, 1238])
    ap.add_argument("--n-planets", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.families, a.seeds, a.n_planets, a.n_samples, a.force)
