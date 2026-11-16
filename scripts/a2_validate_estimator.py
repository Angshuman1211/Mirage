"""
A2 estimator validation — recover KNOWN (s, k) from synthetic flows before trusting the
estimator on real injections.

Same discipline P6-D5 applied to `mirage/eval/sbc.py`, which was checked against five
known-wrong samplers before use. The A2 estimator is new and it carries a finite-sample
correction that is comparable in size to the signal on the radius parameter, so it does not
get used on real data until it demonstrably recovers planted values.

Construction. For injection i the TRUE posterior is N(mu_i, sp_i^2) and the label is drawn
from it, which is the SBC property the estimator relies on. The flow reports
N(mu_i - s*sf_i, sf_i^2) with sf_i = sp_i / k, so by H11's definitions the planted values are
exactly s and k. Means and widths are deliberately heterogeneous across injections, because
real injections are, and a pooled estimator has to be insensitive to that.

    conda run -n mirage python scripts/a2_validate_estimator.py
"""
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_inject_calibrate import _sk, _boot_k

CASES = [
    ("perfect flow", 0.0, 1.0),
    ("overdispersed, radius-like", 0.0, 0.075),
    ("overdispersed, T-like", 0.0, 0.110),
    ("biased only", 1.5, 1.0),
    ("biased AND overdispersed (wasp39 T)", -1.05, 0.110),
    ("UNDERdispersed, flow too narrow", 0.0, 2.0),
    ("severe bias, narrow", 3.0, 0.3),
]


def run(n=250, m=256, seed=0, heterogeneous=True):
    rng = np.random.default_rng(seed)
    print("=" * 100)
    print(f"A2 estimator validation.  n={n} injections, m={m} posterior draws each.")
    print("Planted values must come back. k_uncorr shows what the finite-sample bias would")
    print("have done if the correction were not applied.")
    print("=" * 100)
    print(f"  {'case':38s} {'s_true':>7s} {'s_hat':>7s}   {'k_true':>7s} {'k_hat':>7s} "
          f"{'68% CI':>16s} {'k_uncorr':>9s}")
    worst = 0.0
    for name, s_t, k_t in CASES:
        mu = rng.normal(0, 10, n) if heterogeneous else np.zeros(n)
        sp = rng.uniform(0.5, 5.0, n) if heterogeneous else np.ones(n)
        sf = sp / k_t
        theta = rng.normal(mu, sp)                    # label ~ TRUE posterior (SBC property)
        mu_f = mu - s_t * sf
        draws = rng.normal(mu_f[:, None], sf[:, None], size=(n, m))
        mu_hat = draws.mean(1)
        sd_hat = draws.std(1, ddof=1)
        z = ((theta - mu_hat) / sd_hat)[:, None]      # one "parameter"
        s_h, k_h, _, _, k_raw = _sk(z, m)
        lo, hi = _boot_k(z, m)
        covered = lo[0] <= k_t <= hi[0]
        print(f"  {name:38s} {s_t:7.2f} {s_h[0]:7.2f}   {k_t:7.3f} {k_h[0]:7.3f} "
              f"{f'[{lo[0]:.3f}, {hi[0]:.3f}]':>16s} {k_raw[0]:9.3f}"
              f"{'' if covered else '   <-- CI MISSES'}")
        worst = max(worst, abs(k_h[0] - k_t) / k_t)
    print(f"\n  worst relative error on k: {worst * 100:.1f}%")

    # the bias correction is the thing most likely to be wrong, so show it alone
    print("\n" + "=" * 100)
    print("Size of the finite-sample correction, k_uncorrected / k_true, by planted k.")
    print("This is why the correction exists: at small k the uncorrected estimator is")
    print("dominated by the noise in the posterior mean, not by the flow's actual width.")
    print("=" * 100)
    print(f"  {'k_true':>8s} " + "".join(f"{'m=' + str(mm):>12s}" for mm in (64, 128, 256, 1024)))
    for k_t in (0.05, 0.075, 0.11, 0.25, 0.5, 1.0):
        row = []
        for mm in (64, 128, 256, 1024):
            sp = np.ones(n)
            sf = sp / k_t
            theta = rng.normal(0, sp)
            draws = rng.normal(0.0, sf[:, None], size=(n, mm))
            z = ((theta - draws.mean(1)) / draws.std(1, ddof=1))[:, None]
            _, _, _, _, k_raw = _sk(z, mm)
            row.append(k_raw[0] / k_t)
        print(f"  {k_t:8.3f} " + "".join(f"{v:12.2f}" for v in row))
    print("\n  A value of 1.00 means no correction needed. The radius parameter sits at")
    print("  k ~ 0.075, so at m=256 the uncorrected estimator overstates it by the factor")
    print("  in that row, and that error would land on the one parameter the radius story")
    print("  depends on.")


def repeat_study(reps=120, n=250, m=256):
    """Coverage and bias of k_hat over independent repetitions.

    Three misses out of seven at a 68 percent interval is not evidence of a broken
    estimator (2.2 are expected), so eyeballing single draws cannot settle it. This
    measures the sampling distribution directly: the bias of k_hat, and whether the
    bootstrap interval is the right WIDTH, which is what a single run cannot show.
    """
    print("\n" + "=" * 100)
    print(f"Repetition study, {reps} independent draws per case. Bias must be small and")
    print("the bootstrap width must match the actual spread of k_hat.")
    print("=" * 100)
    print(f"  {'case':38s} {'k_true':>7s} {'E[k_hat]':>9s} {'bias %':>7s} "
          f"{'sd(k_hat)':>10s} {'boot sd':>8s} {'cover68':>8s}")
    for name, s_t, k_t in CASES:
        ks, widths, cov = [], [], 0
        for r in range(reps):
            rng = np.random.default_rng(10_000 + r)
            sp = rng.uniform(0.5, 5.0, n)
            sf = sp / k_t
            mu = rng.normal(0, 10, n)
            theta = rng.normal(mu, sp)
            draws = rng.normal((mu - s_t * sf)[:, None], sf[:, None], size=(n, m))
            z = ((theta - draws.mean(1)) / draws.std(1, ddof=1))[:, None]
            _, k_h, _, _, _ = _sk(z, m)
            ks.append(k_h[0])
            lo, hi = _boot_k(z, m, n_boot=120, seed=r)
            widths.append((hi[0] - lo[0]) / 2)
            cov += int(lo[0] <= k_t <= hi[0])
        ks = np.array(ks)
        print(f"  {name:38s} {k_t:7.3f} {ks.mean():9.3f} "
              f"{(ks.mean() / k_t - 1) * 100:7.1f} {ks.std():10.4f} "
              f"{np.mean(widths):8.4f} {cov / reps * 100:7.0f}%")
    print("\n  cover68 should sit near 68 percent. Far below means the interval is too")
    print("  narrow or the estimator is biased; far above means it is too conservative.")


if __name__ == "__main__":
    run()
    repeat_study()
