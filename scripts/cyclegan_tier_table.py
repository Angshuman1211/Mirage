"""Pillar-2 summary: learned translation vs structured randomisation, on real noise.

Reads the per-run ablation.json files written by run_cyclegan_tiers.sh and prints
one table, so the claim has a single reproducible artifact instead of a number
copied out of a console log (the failure mode behind the published MMD 1.22).

Tier 1 is the headline run, WASP-39b/PRISM on all 52 ABC bins against 3,241 real
out-of-transit frames. Tier 2 repeats the comparison on the 26 native ABC bins
NIRISS and PRISM share, for three targets whose real per-bin noise spans 28x, to
ask whether the tier-1 conclusion is a property of the method or of one target.

    python scripts/cyclegan_tier_table.py
"""
import json
from pathlib import Path

import numpy as np

CFG = Path("configs")
RUNS = [("tier 1", "WASP-39b PRISM 52b", "cyclegan_t1_wasp39"),
        ("tier 2", "WASP-39b PRISM 26b", "cyclegan_t2_wasp39"),
        ("tier 2", "WASP-96b NIRISS 26b", "cyclegan_t2_wasp96"),
        ("tier 2", "K2-18b NIRISS 26b", "cyclegan_t2_k218")]


def load(slug):
    p = CFG / slug / "ablation.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def main():
    rows = []
    for tier, label, slug in RUNS:
        d = load(slug)
        if d is None:
            print(f"  [skip] {label}: {slug}/ablation.json not found")
            continue
        c = d["conditions"]
        rows.append((tier, label, c["sim_baseline"], c["domain_random"],
                     c["cyclegan_trans"], d["conclusion"]))

    if not rows:
        raise SystemExit("no ablation.json files found — run scripts/run_cyclegan_tiers.sh")

    print(f"\n{'='*104}")
    print("  LEARNED TRANSLATION vs STRUCTURED RANDOMISATION, against real JWST noise")
    print(f"{'='*104}")
    print(f"  {'':<22}{'MMD to real (lower better)':>34}{'KL to real (lower better)':>32}")
    print(f"  {'run':<22}{'sim':>11}{'random':>11}{'cycleGAN':>12}"
          f"{'sim':>11}{'random':>10}{'cycleGAN':>11}{'winner':>12}")
    print("  " + "-" * 100)
    last_tier = None
    for tier, label, s, r, g, concl in rows:
        if tier != last_tier:
            print(f"  {tier}")
            last_tier = tier
        # Declared on BOTH metrics. Calling it on MMD alone would contradict the
        # per-run conclusion, which weighs each delta against its own scale.
        w_mmd = r["mmd_to_real"] < g["mmd_to_real"]
        w_kl = r["kl_to_real"] < g["kl_to_real"]
        win = "random" if (w_mmd and w_kl) else ("cycleGAN" if not (w_mmd or w_kl)
                                                 else "SPLIT")
        print(f"  {'  ' + label:<22}"
              f"{s['mmd_to_real']:>11.4f}{r['mmd_to_real']:>11.4f}{g['mmd_to_real']:>12.4f}"
              f"{s['kl_to_real']:>11.4f}{r['kl_to_real']:>10.4f}{g['kl_to_real']:>11.4f}"
              f"{win:>12}")
    print("  " + "-" * 100)
    print("  winner = both metrics agree, else SPLIT. 'sim' is no adaptation at all,")
    print("  so a translator scoring WORSE than sim has actively hurt.")

    # Consistency across tier 2, stated from the numbers rather than asserted.
    t2 = [r for r in rows if r[0] == "tier 2"]
    if len(t2) > 1:
        def rel(a, b):
            return (a - b) / max((a + b) / 2, 1e-12)
        rm = [rel(r["mmd_to_real"], g["mmd_to_real"]) for _, _, _, r, g, _ in t2]
        rk = [rel(r["kl_to_real"], g["kl_to_real"]) for _, _, _, r, g, _ in t2]
        print(f"\n  relative margin, negative favours randomisation")
        print(f"    MMD  " + ", ".join(f"{v:+.3f}" for v in rm))
        print(f"    KL   " + ", ".join(f"{v:+.3f}" for v in rk))
        agree = [(m < 0) == (k < 0) for m, k in zip(rm, rk)]
        if all((m < 0) and (k < 0) for m, k in zip(rm, rk)):
            print(f"  CONSISTENT across all {len(t2)} targets: randomisation closer on "
                  f"both metrics, median MMD margin {np.median(rm):+.3f}")
        else:
            print(f"  NOT consistent. The two metrics disagree on "
                  f"{sum(not a for a in agree)}/{len(t2)} targets, and the MMD winner "
                  f"flips sign across targets.")
            print(f"  This is CONFOUNDED with sample size and must not be read as a "
                  f"target effect on its own: every tier-2 arm is capped at 56 real "
                  f"frames, where both metrics are noisy and a translator can map onto "
                  f"a small real set. The control that separates the two is the same "
                  f"target and grid at full frame count.")

    print("\n  conclusions as each run recorded them")
    for _, label, _, _, _, concl in rows:
        print(f"    {label:<22} {concl}")
    print()


if __name__ == "__main__":
    main()
