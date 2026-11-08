"""
Simulation-based calibration for MIRAGE (A1, closes G3).

The Phase-3 OT calibration was circular: it calibrated against the TauREx-NS
posterior and was then scored by coverage against that same posterior, which is
what two Sim2Science reviewers independently caught (P5-D5). Everything here
takes its labels from a *prior draw* instead, so no transport can game them.

Three tests, deliberately not interchangeable.

  SBC RANKS (Talts et al. 2018, arXiv:1804.06788). Rank the true value among
  posterior samples, one parameter at a time. Uniform under exact inference.
  Shape is diagnostic: U-shaped ranks mean intervals too narrow (overconfident),
  bell-shaped means too wide, sloped means biased. Marginal, so it cannot see a
  wrong correlation structure.

  TARP (Lemos et al. 2023, arXiv:2302.03026). Tests the JOINT posterior by
  asking how often the truth is closer to a random reference point than the
  samples are. Catches the dependence errors SBC ranks miss. Needs a metric, so
  theta is standardized first.

  HPD RANKS. Fraction of samples with higher flow density than the truth. Also
  joint, and it uses the flow's exact log-density rather than a metric, so it is
  the sharpest of the three where log q is affordable.

All three reduce to "is this statistic uniform on [0,1]", which is the only
reason it is safe to report them together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SBCResult:
    ranks: np.ndarray        # (n_sims, D) int, in [0, n_samples]
    n_samples: int           # L, posterior draws per simulation
    n_bins: int
    chi2: np.ndarray         # (D,) uniformity statistic per parameter
    pvalue: np.ndarray       # (D,) p-value per parameter
    counts: np.ndarray       # (D, n_bins) binned rank histogram

    @property
    def n(self) -> int:
        return int(self.ranks.shape[0])

    def worst(self) -> tuple[int, float]:
        """(parameter index, p-value) of the least uniform parameter."""
        i = int(np.argmin(self.pvalue))
        return i, float(self.pvalue[i])

    def skew(self) -> np.ndarray:
        """Mean normalised rank minus 0.5. Negative = posterior biased high."""
        return self.ranks.mean(axis=0) / self.n_samples - 0.5


@dataclass(frozen=True)
class UniformityResult:
    stat: np.ndarray         # (n_sims,) the statistic, uniform on [0,1] if calibrated
    alpha: np.ndarray        # credibility grid
    ecp: np.ndarray          # expected coverage probability at each alpha
    max_dev: float           # sup |ecp - alpha|, 0 is perfect
    pvalue: float            # KS against U(0,1)
    ecp_lo: np.ndarray | None = None   # bootstrap band, TARP only
    ecp_hi: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.stat.shape[0])


def sbc_ranks(samples_list, theta_true_list) -> np.ndarray:
    """
    Rank of the truth among posterior samples, per parameter.

    Args:
        samples_list: sequence of (L, D) posterior-sample arrays, same L.
        theta_true_list: sequence of (D,) prior draws that generated them.
    Returns:
        (n_sims, D) integer ranks in [0, L].
    """

    ranks = []
    for samples, theta in zip(samples_list, theta_true_list):
        s = np.asarray(samples)
        t = np.asarray(theta)
        if s.ndim != 2 or t.shape[0] != s.shape[1]:
            raise ValueError(f"shape mismatch: samples {s.shape}, theta {t.shape}")
        ranks.append((s < t[None, :]).sum(axis=0))
    if not ranks:
        raise ValueError("no simulations given")
    return np.stack(ranks).astype(int)


def rank_uniformity(ranks: np.ndarray, n_samples: int, n_bins: int = 20):
    """
    Chi-square test that ranks are DiscreteUniform{0..n_samples}.

    Binning is not cosmetic. Talts et al. §4 shows that bins which do not divide
    the n_samples+1 possible rank values evenly put a sawtooth in the histogram
    that looks like a calibration failure, so that case is rejected rather than
    silently fudged.
    """

    ranks = np.asarray(ranks)
    n, d = ranks.shape
    if (n_samples + 1) % n_bins:
        raise ValueError(
            f"n_bins={n_bins} must divide n_samples+1={n_samples + 1} evenly, "
            "otherwise unequal bin widths fake a calibration failure "
            "(Talts et al. 2018, sec. 4)"
        )
    if n / n_bins < 5:
        raise ValueError(
            f"{n} simulations over {n_bins} bins is {n / n_bins:.1f} expected per "
            "bin; chi-square needs >= 5. Run more simulations or pass fewer bins."
        )

    edges = np.linspace(0, n_samples + 1, n_bins + 1)
    chi2 = np.empty(d)
    pval = np.empty(d)
    counts = np.empty((d, n_bins), dtype=int)
    for j in range(d):
        c, _ = np.histogram(ranks[:, j], bins=edges)
        counts[j] = c
        res = stats.chisquare(c)           # expected = n/n_bins, uniform
        chi2[j], pval[j] = res.statistic, res.pvalue
    return chi2, pval, counts


def sbc(samples_list, theta_true_list, n_bins: int = 20) -> SBCResult:
    """Marginal SBC: ranks plus a per-parameter uniformity test."""

    ranks = sbc_ranks(samples_list, theta_true_list)
    n_samples = int(np.asarray(samples_list[0]).shape[0])
    chi2, pval, counts = rank_uniformity(ranks, n_samples, n_bins)
    return SBCResult(ranks=ranks, n_samples=n_samples, n_bins=n_bins,
                     chi2=chi2, pvalue=pval, counts=counts)


def _ecp_curve(stat: np.ndarray, n_grid: int = 101):
    """Expected coverage probability: fraction of stat <= alpha, on a grid."""
    alpha = np.linspace(0.0, 1.0, n_grid)
    ecp = (stat[None, :] <= alpha[:, None]).mean(axis=1)
    return alpha, ecp


def tarp(
    samples_list,
    theta_true_list,
    reference: str = "posterior",
    bounds=None,
    scale=None,
    n_bootstrap: int = 0,
    random_seed: int = 0,
) -> UniformityResult:
    """
    TARP: distance-to-random-point coverage, a joint-posterior test.

    For each simulation draw a reference point and compute the fraction of
    posterior samples lying closer to it than the truth is. That fraction is a
    probability integral transform, so it is uniform on [0,1] for an exact
    posterior and ANY reference that does not depend on the truth
    (Lemos et al. 2023, thm. 1).

    Args:
        reference: where to draw the reference point.
            "posterior" (default) draws from a Gaussian matched to each
                simulation's own posterior mean and covariance.
            "box" draws uniformly over `bounds`.
            The theorem allows either, because sample mean and covariance are
            functions of the posterior draws and so are independent of the truth
            given x. It only constrains validity, not power, and the two differ a
            lot in power. Measured on the conjugate toy in
            `scripts/validate_sbc.py`, a 0.5-sigma bias gives p = 0.52 with "box"
            and p = 5.7e-3 with "posterior", while the exact case stays null under
            both (0.84 and 0.70). Diffuse references put the truth deep inside a
            huge ball whose mass barely moves when the posterior is wrong.
        bounds: (2, D) low/high for reference="box". Defaults to the bounding box
            of theta_true_list, which is the prior support when the truths are
            prior draws. Passing it implies reference="box".
        scale: (D,) divisor applied before distances. Defaults to the std of
            theta_true. Without it the widest parameter sets the metric and the
            test mostly ignores the others.
        n_bootstrap: repeat with fresh reference points for an ECP band. The
            reference draw is random, so one curve carries sampling noise that is
            easy to over-read.
    """

    truths = np.asarray([np.asarray(t) for t in theta_true_list], dtype=float)
    n, d = truths.shape
    if bounds is not None:
        reference = "box"
    if reference not in ("posterior", "box"):
        raise ValueError(f"reference must be 'posterior' or 'box', got {reference!r}")

    if bounds is None:
        lo, hi = truths.min(axis=0), truths.max(axis=0)
        pad = 0.05 * (hi - lo)
        bounds = np.stack([lo - pad, hi + pad])
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, d):
        raise ValueError(f"bounds must be (2, {d}), got {bounds.shape}")

    if scale is None:
        scale = truths.std(axis=0)
    scale = np.asarray(scale, dtype=float)
    scale = np.where(scale > 0, scale, 1.0)

    # posterior-matched reference geometry, precomputed once per simulation
    chols = None
    if reference == "posterior":
        chols = []
        for samples in samples_list:
            s = np.asarray(samples, dtype=float)
            c = np.cov(s, rowvar=False)
            c = np.atleast_2d(c) + 1e-12 * np.eye(d)
            try:
                chols.append((s.mean(axis=0), np.linalg.cholesky(c)))
            except np.linalg.LinAlgError:       # degenerate posterior
                chols.append((s.mean(axis=0), np.diag(np.sqrt(np.diag(c)))))

    rng = np.random.default_rng(random_seed)

    def one_pass() -> np.ndarray:
        if reference == "box":
            refs = rng.uniform(bounds[0], bounds[1], size=(n, d))
        else:
            z = rng.normal(size=(n, d))
            refs = np.stack([m + L @ zi for (m, L), zi in zip(chols, z)])
        out = np.empty(n)
        for i, samples in enumerate(samples_list):
            s = np.asarray(samples, dtype=float) / scale
            r = refs[i] / scale
            d_true = np.linalg.norm(truths[i] / scale - r)
            d_samp = np.linalg.norm(s - r[None, :], axis=1)
            out[i] = (d_samp < d_true).mean()
        return out

    stat = one_pass()
    alpha, ecp = _ecp_curve(stat)
    lo_band = hi_band = None
    if n_bootstrap > 0:
        curves = np.stack([_ecp_curve(one_pass())[1] for _ in range(n_bootstrap)])
        lo_band = np.quantile(curves, 0.16, axis=0)
        hi_band = np.quantile(curves, 0.84, axis=0)

    return UniformityResult(
        stat=stat, alpha=alpha, ecp=ecp,
        max_dev=float(np.abs(ecp - alpha).max()),
        pvalue=float(stats.kstest(stat, "uniform").pvalue),
        ecp_lo=lo_band, ecp_hi=hi_band,
    )


def hpd_ranks(logq_samples_list, logq_truth_list) -> UniformityResult:
    """
    Joint test from the flow's own density: the fraction of posterior samples
    with log q above log q at the truth. Uniform on [0,1] for an exact posterior,
    and equal to the HPD credible level that just contains the truth.

    Args:
        logq_samples_list: sequence of (L,) log-densities of the samples.
        logq_truth_list: sequence of floats, log-density at the truth.
    """

    stat = np.array([
        float((np.asarray(ls) > float(lt)).mean())
        for ls, lt in zip(logq_samples_list, logq_truth_list)
    ])
    if stat.size == 0:
        raise ValueError("no simulations given")
    alpha, ecp = _ecp_curve(stat)
    return UniformityResult(
        stat=stat, alpha=alpha, ecp=ecp,
        max_dev=float(np.abs(ecp - alpha).max()),
        pvalue=float(stats.kstest(stat, "uniform").pvalue),
    )
