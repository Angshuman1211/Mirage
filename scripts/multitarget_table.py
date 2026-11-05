"""Regenerate the multi-target results table (paper Table 2) from saved posteriors.

Why this exists: nothing in the repo computed these numbers. `benchmark_table.py`
covers only the WASP-39b arms against the WASP-39b anchor, and `fig_multitarget.py`
computes no coverage at all. The published 7/7 / 7/7 / 6/7 were produced ad hoc,
while the paper's reproducibility checklist claims otherwise (P5-D7).

Two things this reports that the paper did not:
  - the ERROR FLOOR every number is computed at (P5-D4: the paper silently mixed
    a 1% floor for NS with a 5% floor for the flow),
  - the EFFECTIVE SAMPLE SIZE behind each interval (P5-D7: at ESS ~1 the weights
    collapse onto a single draw and the "68% interval" is a point).

Usage:
    python scripts/multitarget_table.py                # both floors
    python scripts/multitarget_table.py --floor 0.01   # one floor
    python scripts/multitarget_table.py --markdown     # paste-ready table
"""
import argparse
from pathlib import Path

import numpy as np

OUT = Path("data/real_ess")
ORDER = ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
ESS_WARN = 20.0          # below this an interval is not meaningfully a credible interval

# (label, instrument, samples/forward prefix, log-prob key, NS anchor, literature radius RJ)
# NOTE: WASP-39b's figure/table row uses the UNCALIBRATED nocond arm, matching
# fig_multitarget.py. `ot_cal` is listed separately because §4's prose credits the
# OT calibration for a row the figure never took from it.
TARGETS = [
    ("WASP-39b", "NIRSpec PRISM", "abl_rad_nocond", "log_q",
     "wasp39b_ns_posterior_fitrad.npz", 1.279),
    ("WASP-96b", "NIRISS SOSS",   "wasp96",         "log_q",
     "wasp39b_ns_posterior_wasp96.npz", 1.200),
    ("K2-18b",   "NIRISS SOSS",   "k218cov",        "log_q",
     "wasp39b_ns_posterior_k218.npz",   0.2352),
]
EXTRA = [
    ("WASP-39b [OT-cal]", "NIRSpec PRISM", "ot_cal", "log_g",
     "wasp39b_ns_posterior_fitrad.npz", 1.279),
]


def wquant(v, w, q):
    o = np.argsort(v)
    return float(np.interp(q, np.cumsum(w[o]), v[o]))


def ns_means(npz):
    """Weighted posterior mean per parameter from the nested-sampling anchor."""
    d = np.load(OUT / npz, allow_pickle=True)
    idx = [list(d["fit_names"]).index(k) for k in ORDER]
    w = d["weights"] / d["weights"].sum()
    return np.sum(w[:, None] * d["samples"][:, idx], axis=0)


def evaluate(prefix, logp_key, ns_npz, truth, floor):
    s = np.load(OUT / f"{prefix}_samples.npz", allow_pickle=True)
    f = np.load(OUT / f"{prefix}_forward.npz", allow_pickle=True)
    th, x, sig, cov, ms = s["theta"], s["x_obs"], s["sig_obs"], s["covered"], f["model_spec"]

    m = cov & np.isfinite(sig) & (sig > 0)
    se = np.sqrt(sig ** 2 + (floor * x) ** 2)
    val = np.isfinite(ms[:, m]).all(1)
    r = (x[None, m] - ms[:, m]) / se[None, m]
    ll = np.where(val, -0.5 * np.nansum(r ** 2, 1), -np.inf)

    w = np.exp(ll - s[logp_key] - np.nanmax(ll - s[logp_key]))
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)

    nsm = ns_means(ns_npz)
    lo = np.array([wquant(th[:, i], w, 0.16) for i in range(7)])
    hi = np.array([wquant(th[:, i], w, 0.84) for i in range(7)])
    med = np.array([wquant(th[:, i], w, 0.50) for i in range(7)])

    # IS-FREE interval: the raw flow posterior, equal weights. P5-D7 found the
    # importance weights collapse onto one draw (a single sample carries 99.95%
    # for WASP-39b-nocond), which makes the weighted 68% interval a point mass.
    # Equal weights cannot collapse, so this interval is reportable whatever the
    # ESS. It is the same IS-free move P5-D10 made for the real-side JS metric.
    u = np.full(len(w), 1.0 / len(w))
    lo_u = np.array([wquant(th[:, i], u, 0.16) for i in range(7)])
    hi_u = np.array([wquant(th[:, i], u, 0.84) for i in range(7)])
    med_u = np.array([wquant(th[:, i], u, 0.50) for i in range(7)])

    width = float(hi[0] - lo[0])
    width_u = float(hi_u[0] - lo_u[0])
    return dict(
        nbins=int(m.sum()),
        chi2=float(-2 * ll.max() / m.sum()),
        ess=float(ess),
        n=len(w),
        coverage=int(np.sum((lo <= nsm) & (nsm <= hi))),   # vs NS = the CIRCULAR metric
        rad_med=float(med[0]), rad_lo=float(lo[0]), rad_hi=float(hi[0]),
        rad_med_u=float(med_u[0]), rad_lo_u=float(lo_u[0]), rad_hi_u=float(hi_u[0]),
        width=width, width_u=width_u,
        inflation=(width_u / width if width > 0 else float("inf")),
        truth=truth,
        truth_inside=bool(lo[0] <= truth <= hi[0]),        # vs LITERATURE = non-circular
        truth_inside_u=bool(lo_u[0] <= truth <= hi_u[0]),
    )


def main(floors, markdown, include_otcal, ess_min):
    rows = TARGETS + (EXTRA if include_otcal else [])
    for floor in floors:
        print(f"\n{'='*104}\nERROR FLOOR = {floor:.0%}   "
              f"(coverage-vs-NS is circular, see P5-D5; truth-vs-literature is not)\n{'='*104}")
        hdr = (f"{'planet':<20}{'bins':>5}{'chi2/N':>8}{'ESS':>8}"
               f"{'radius 68% (IS-weighted)':>28}{'lit in?':>8}"
               f"{'radius 68% (IS-free)':>26}{'lit in?':>8}{'x wider':>8}")
        print(hdr + "\n" + "-" * len(hdr))
        out = []
        for lbl, inst, prefix, lk, ns, truth in rows:
            try:
                e = evaluate(prefix, lk, ns, truth, floor)
            except Exception as exc:
                print(f"{lbl:<20}  ERROR {type(exc).__name__}: {exc}")
                continue
            collapsed = e["ess"] < ess_min
            # P5-D7's resolution: do not quote an IS-weighted interval from a
            # collapsed arm at all. Withhold it rather than print it with a note,
            # because a printed number gets copied and the note does not.
            iv = ("WITHHELD (ESS)" if collapsed else
                  f"{e['rad_med']:.4f} [{e['rad_lo']:.4f},{e['rad_hi']:.4f}]")
            lit = "--" if collapsed else ("yes" if e["truth_inside"] else "NO")
            iv_u = f"{e['rad_med_u']:.4f} [{e['rad_lo_u']:.4f},{e['rad_hi_u']:.4f}]"
            print(f"{lbl:<20}{e['nbins']:>5}{e['chi2']:>8.3f}{e['ess']:>8.2f}"
                  f"{iv:>28}{lit:>8}{iv_u:>26}"
                  f"{('yes' if e['truth_inside_u'] else 'NO'):>8}{e['inflation']:>8.1f}")
            out.append((lbl, inst, e, collapsed))

        n_collapsed = sum(1 for *_, c in out if c)
        print(f"\n  ESS threshold {ess_min:g} of {out[0][2]['n'] if out else 0} draws. "
              f"{n_collapsed}/{len(out)} arms are collapsed and have their "
              f"IS-weighted interval withheld.")
        print("  IS-free = the raw flow posterior at equal weights, which cannot")
        print("  collapse. It is the reportable interval for a collapsed arm.")
        agree = [e["truth_inside_u"] for *_, e, _c in [(a, b, c, d) for a, b, c, d in out]]
        if agree:
            print(f"  literature radius inside the IS-free interval: "
                  f"{sum(agree)}/{len(agree)} targets.")

        if markdown:
            print("\nmarkdown:\n")
            print("| Planet | Instrument | Radius 68% (IS-free) | Literature | Lit. inside? | ESS | IS-weighted |")
            print("|---|---|---|---|---|---|---|")
            for lbl, inst, e, collapsed in out:
                isw = ("withheld, ESS too low" if collapsed else
                       f"{e['rad_lo']:.3f}, {e['rad_hi']:.3f}")
                print(f"| {lbl} | {inst} | {e['rad_med_u']:.3f} "
                      f"[{e['rad_lo_u']:.3f}, {e['rad_hi_u']:.3f}] | {e['truth']:.3f} | "
                      f"{'yes' if e['truth_inside_u'] else '**no**'} | "
                      f"{e['ess']:.1f} | {isw} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=None,
                    help="single error floor; default reports both 0.01 and 0.05")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--no-otcal", action="store_true",
                    help="omit the OT-calibrated WASP-39b row")
    ap.add_argument("--ess-min", type=float, default=ESS_WARN,
                    help="below this ESS the IS-weighted interval is withheld")
    a = ap.parse_args()
    main([a.floor] if a.floor else [0.01, 0.05], a.markdown, not a.no_otcal, a.ess_min)
