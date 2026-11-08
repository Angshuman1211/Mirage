"""
A1 validation — do the calibration tests in mirage/eval/sbc.py actually detect
miscalibration, and do they stay quiet on an exact posterior?

This has to come before any result. The whole point of A1 is replacing a
calibration check that was wrong (P5-D5, the circular OT validation), so shipping
an unvalidated replacement would repeat the mistake with new code.

Conjugate Gaussian toy where the exact posterior is known in closed form:

    theta ~ N(0, tau^2)      x | theta ~ N(theta, sigma^2)
    theta | x ~ N(x * tau^2/(tau^2+sigma^2),  tau^2 sigma^2/(tau^2+sigma^2))

Five sampler variants, four of them wrong in a known direction:

  exact          draws from that posterior            all tests must PASS
  overconfident  same mean, sd x 0.5                  SBC U-shaped
  overdispersed  same mean, sd x 2                    SBC bell-shaped
  biased         mean shifted by +1.0 sd              SBC sloped
  decorrelated   correct MARGINALS, correlation       SBC must PASS (it is
                 dropped from 0.9 to 0                marginal and blind here),
                                                      TARP and HPD must FAIL

That last row is the one that matters. It is the reason three tests are reported
instead of one, and if TARP cannot catch it the joint test is decorative.

Each case runs at several seeds and the MEDIAN p-value decides, because TARP's
reference draw is random and a single seed is not a stable regression test.

Measured sensitivity, 30 seeds at 400 sims x 999 samples, D=3 (P6-D5):

    exact          median p 0.52,  false alarms at 0.01: 0/30, p-values uniform
                   (KS p=0.66), so the null is calibrated
    biased 0.5 sd  median p 6.1e-3,  detected 57% of the time
    biased 1.0 sd  median p 6.6e-22, detected 100%
    decorrelated   median p 1.6e-4,  detected 90%

So TARP's floor here is a bias of roughly half a sigma, at which it is a coin
flip. The bias case below uses 1.0 sd to keep this file deterministic. Do not
read a single passing TARP p-value as proof of calibration at N this small.

Run from Project/:
    python scripts/validate_sbc.py
"""

import numpy as np

from mirage.eval.sbc import hpd_ranks, sbc, tarp

TAU, SIGMA = 1.0, 1.0
N_SIMS, N_SAMPLES = 400, 999        # 999 so n_bins=20 divides n_samples+1
ALPHA = 0.01                        # p below this = detected
BIAS = 1.0                          # in posterior sd; 0.5 leaves TARP at 57% power
N_SEEDS = 3                         # median over seeds, see the power table above


def posterior(x, d):
    """Exact conjugate posterior mean and sd, elementwise."""
    w = TAU ** 2 / (TAU ** 2 + SIGMA ** 2)
    var = TAU ** 2 * SIGMA ** 2 / (TAU ** 2 + SIGMA ** 2)
    return w * x, np.full(d, np.sqrt(var))


def make_case(kind: str, d: int, rho: float, rng):
    """-> (samples_list, truths, logq_samples, logq_truth)."""
    truths, samples, lq_s, lq_t = [], [], [], []

    # correlated prior/posterior only for the decorrelated case, else independent
    corr = np.eye(d)
    if kind == "decorrelated":
        corr = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    chol_true = np.linalg.cholesky(corr)

    for _ in range(N_SIMS):
        theta = chol_true @ rng.normal(0, TAU, size=d)
        x = theta + chol_true @ rng.normal(0, SIGMA, size=d)
        mu, sd = posterior(x, d)

        if kind == "overconfident":
            sd = sd * 0.5
        elif kind == "overdispersed":
            sd = sd * 2.0
        elif kind == "biased":
            mu = mu + BIAS * sd

        # the approximate sampler's correlation: identity unless it is exact
        chol_q = chol_true if kind != "decorrelated" else np.eye(d)
        z = rng.normal(size=(N_SAMPLES, d))
        s = mu[None, :] + (z @ chol_q.T) * sd[None, :]

        # log q under the sampler's own Gaussian, up to an additive constant
        cov_q = (chol_q * sd[None, :]) @ (chol_q * sd[None, :]).T
        prec = np.linalg.inv(cov_q)
        ds = s - mu[None, :]
        dt = (theta - mu)[None, :]
        lq_s.append(-0.5 * np.einsum("ij,jk,ik->i", ds, prec, ds))
        lq_t.append(float(-0.5 * np.einsum("ij,jk,ik->i", dt, prec, dt)[0]))

        truths.append(theta)
        samples.append(s)

    return samples, np.stack(truths), lq_s, lq_t


def main(d: int = 3, rho: float = 0.9, seed: int = 0):
    print(f"{N_SIMS} sims x {N_SAMPLES} samples, D={d}, bias={BIAS} sd, "
          f"median of {N_SEEDS} seeds, detected at p < {ALPHA}\n")
    print(f"{'case':<15}{'SBC min p':>12}{'shape':>14}"
          f"{'TARP p':>10}{'TARP dev':>10}{'HPD p':>10}   verdict")
    print("-" * 86)

    expectations = {
        # case:          (sbc_should_fail, tarp_should_fail, hpd_should_fail)
        "exact":          (False, False, False),
        "overconfident":  (True,  True,  True),
        "overdispersed":  (True,  True,  True),
        "biased":         (True,  True,  True),
        "decorrelated":   (False, True,  True),
    }

    failures = []
    for kind, (want_sbc, want_tarp, want_hpd) in expectations.items():
        sbc_ps, tarp_ps, hpd_ps, devs, shapes = [], [], [], [], []
        for k in range(N_SEEDS):
            rng = np.random.default_rng(seed + k)
            samples, truths, lq_s, lq_t = make_case(kind, d, rho, rng)

            r = sbc(samples, truths, n_bins=20)
            t = tarp(samples, truths, random_seed=seed + 100 + k)
            h = hpd_ranks(lq_s, lq_t)

            sbc_ps.append(float(r.pvalue.min()))
            tarp_ps.append(t.pvalue)
            hpd_ps.append(h.pvalue)
            devs.append(t.max_dev)

            hist = r.counts.sum(axis=0)
            edge = hist[[0, -1]].mean()
            mid = hist[len(hist) // 2 - 1:len(hist) // 2 + 1].mean()
            sh = ("U-shaped" if edge > 1.3 * mid else
                  "bell" if mid > 1.3 * edge else "flat")
            if abs(r.skew()).max() > 0.08:
                sh = "sloped"
            shapes.append(sh)

        sbc_p = float(np.median(sbc_ps))
        tarp_p = float(np.median(tarp_ps))
        hpd_p = float(np.median(hpd_ps))
        shape = max(set(shapes), key=shapes.count)

        got = (sbc_p < ALPHA, tarp_p < ALPHA, hpd_p < ALPHA)
        ok = got == (want_sbc, want_tarp, want_hpd)
        if not ok:
            failures.append((kind, got, (want_sbc, want_tarp, want_hpd)))

        print(f"{kind:<15}{sbc_p:>12.2e}{shape:>14}"
              f"{tarp_p:>10.2e}{float(np.median(devs)):>10.3f}{hpd_p:>10.2e}"
              f"   {'ok' if ok else 'MISMATCH'}")

    print()
    if failures:
        for kind, got, want in failures:
            print(f"  {kind}: detected {got}, expected {want}")
        raise SystemExit("VALIDATION FAILED")
    print("All five cases behaved as expected.")
    print("The decorrelated row is the load-bearing one: SBC ranks stayed flat "
          "while TARP and HPD fired,\nwhich is the joint-vs-marginal distinction "
          "the three-test report exists to make.")


if __name__ == "__main__":
    main()
