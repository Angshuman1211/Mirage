"""
H11 (A6/G9) — does the optimal-transport calibration TRANSFER to a new target
without that target's own nested-sampling anchor?

Why this exists. A6 found that the per-target NS anchor is cheap (5 min 24 s), so
"NS is unaffordable" cannot motivate amortization at this problem size. What is left
of the amortization argument is a conditional: it survives only if the calibration
learned on one target fixes the flow posterior on a NEW target, because then the
anchor is paid once rather than per target. That was written down as untested. This
tests it.

It is also the non-circular version of the calibration evaluation. P5-D5 killed the
published calibration because the OT pushforward of N(mu_f,S_f) onto N(mu_n,S_n) IS
N(mu_n,S_n), so measuring it against the anchor it was fitted to is guaranteed to
pass. Here the donor target supplies the calibration and the recipient's anchor is
held out, so every off-diagonal cell is an honest out-of-sample measurement.

Three nested modes, each asking LESS of the calibration than the one before, so a
failure can be attributed rather than just reported:

  M1 absolute   the literal pushforward, N(mu_n^A, S_n^A). Asks: is the anchor itself
                target-portable?
  M2 shift+scale  mu_f^B + s_A*sd_f^B, widths k_A*sd_f^B, B's own correlations, where
                s_A=(mu_n^A-mu_f^A)/sd_f^A and k_A=sd_n^A/sd_f^A are DIMENSIONLESS.
                Asks: is the correction portable? Scales differ hugely across targets
                (T ~ 600 K vs 260 K, radius 1.23 vs 0.229 RJ) so a transferable
                calibration has to be stated this way or it fails on units alone.
  M3 scale-only   B's own centre, widths k_A*sd_f^B. Asks: is just the overdispersion
                fix portable? This is the one that matters for importance sampling,
                since ESS collapsed from overdispersion and not from bias.

  M2c            M2 with the DONOR's correlation matrix instead of B's, to separate
                 "the correlations did not transfer" from "the widths did not".

Baselines. RAW = B's uncalibrated flow posterior (the floor). RAWG = a Gaussian fit to
it, so the cost of the Gaussian assumption is visible separately from the cost of
transfer. SELF = donor is B itself, the circular ceiling P5-D5 objected to.

ARCHITECTURE MUST BE MATCHED ACROSS TARGETS, and this nearly broke the test. The
published WASP-39b arm is noisecond_rad_cov while the two per-planet arms are
noisecond_rad_nocond. h11_seedcheck.py shows those two architectures carry DIFFERENT
calibrations on the same planet against the same anchor (radius deflation 0.138 vs
0.075, temperature bias -5.56 vs -1.05 flow sd), so a cov-vs-nocond comparison would
charge an architecture difference to the target. Default is therefore the matched
rad_nocond arm on all three. Set H11_W39_ARM=cov to reproduce the mixed-architecture
run, which is kept only as a negative control on the test itself.

Per-target priors are NOT shared: K2-18b is a cold sub-Neptune with rp in [0.10,0.40]
and T in [100,600], against [0.8,1.6] and [300,2500] for the hot Jupiters. The
in-prior fraction of each transferred proposal is reported because a proposal with no
prior support cannot be used for importance sampling whatever its divergence says.

    conda run -n mirage python scripts/h11_ot_transfer.py
"""
import json
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_compare import js_1d

OUT = Path("data/real_ess")
NAMES = ["radius", "T", "logH2O", "logCO2", "logCH4", "logCO", "logNH3"]
_ORDER = ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]

# per-target (flow posterior, NS anchor, training prior) — priors from
# generate_training_data.py PLANETS, which are per-planet and NOT interchangeable
TARGETS = {
    "wasp39": dict(fmpe=OUT / f"real_ess_samples_h9_rad_{os.environ.get('H11_W39_ARM', 'nocond')}_s42.npz",
                   ns=OUT / "wasp39b_ns_posterior_fitrad.npz",
                   lo=np.array([0.8, 300., -9, -9, -9, -9, -9]),
                   hi=np.array([1.6, 2500., -1, -1, -1, -1, -1])),
    "wasp96": dict(fmpe=OUT / "wasp96_samples.npz",
                   ns=OUT / "wasp39b_ns_posterior_wasp96.npz",
                   lo=np.array([0.8, 300., -9, -9, -9, -9, -9]),
                   hi=np.array([1.6, 2500., -1, -1, -1, -1, -1])),
    "k218": dict(fmpe=OUT / "k218cov_samples.npz",
                 ns=OUT / "wasp39b_ns_posterior_k218.npz",
                 lo=np.array([0.10, 100., -9, -9, -9, -9, -9]),
                 hi=np.array([0.40, 600., -1, -1, -1, -1, -1])),
}
NBINS_OF = {"wasp39": 52, "wasp96": 90, "k218": 150}


def _moments(th, w=None):
    if w is None:
        return th.mean(0), np.cov(th.T)
    mu = (w[:, None] * th).sum(0)
    d = th - mu
    S = (w[:, None, None] * (d[:, :, None] * d[:, None, :])).sum(0)
    return mu, S


def load(name):
    T = TARGETS[name]
    f = np.load(T["fmpe"], allow_pickle=True)
    thf = f["theta"]
    if len(f["wlen"]) != NBINS_OF[name]:
        raise SystemExit(f"[guard] {T['fmpe']} has {len(f['wlen'])} bins, {name} expects "
                         f"{NBINS_OF[name]}. Wrong target's posterior.")
    n = np.load(T["ns"], allow_pickle=True)
    nsn = [str(x) for x in n["fit_names"]]
    idx = [nsn.index(o) for o in _ORDER]
    thn = n["samples"][:, idx]
    wn = n["weights"] / n["weights"].sum()
    muf, Sf = _moments(thf)
    mun, Sn = _moments(thn, wn)
    return dict(name=name, thf=thf, thn=thn, wn=wn, muf=muf, Sf=Sf, mun=mun, Sn=Sn,
                sdf=np.sqrt(np.diag(Sf)), sdn=np.sqrt(np.diag(Sn)),
                lo=T["lo"], hi=T["hi"])


def corr(S):
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def calibration(A):
    """The donor's calibration in dimensionless form: bias in flow-sd units, width
    deflation as a ratio, plus the anchor's correlation structure."""
    return dict(s=(A["mun"] - A["muf"]) / A["sdf"], k=A["sdn"] / A["sdf"],
                Cn=corr(A["Sn"]), mun=A["mun"], Sn=A["Sn"])


def transfer(cal, B, mode):
    if mode == "M1":
        return cal["mun"].copy(), cal["Sn"].copy()
    sd = cal["k"] * B["sdf"]
    C = cal["Cn"] if mode == "M2c" else corr(B["Sf"])
    S = np.outer(sd, sd) * C
    mu = B["muf"] + (cal["s"] * B["sdf"] if mode in ("M2", "M2c") else 0.0)
    return mu, S


def gauss_draws(mu, S, lo, hi, M=20000, seed=0):
    """Draw from N(mu,S); return the draws and what fraction land in the recipient's
    prior. Jitter only if the covariance is numerically indefinite."""
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-18, None)
    L = V * np.sqrt(w)
    rng = np.random.default_rng(seed)
    x = mu + rng.standard_normal((M, 7)) @ L.T
    inside = np.all((x >= lo) & (x <= hi), axis=1)
    return x, float(inside.mean())


def score(x, wx, B):
    """Divergence and moment agreement against the RECIPIENT's held-out anchor."""
    js = np.array([js_1d(x[:, i], wx, B["thn"][:, i], B["wn"]) for i in range(7)])
    mu, S = _moments(x, wx / wx.sum() if wx is not None else None)
    sd = np.sqrt(np.diag(S))
    return dict(js=js, mean_js=float(js.mean()),
                dmu_sig=np.abs(mu - B["mun"]) / B["sdn"], sd_ratio=sd / B["sdn"])


def main():
    tg = {k: load(k) for k in TARGETS}
    rows = []
    print("=" * 108)
    print("H11  OT calibration transfer. Every cell is scored against the RECIPIENT's"
          " own NS anchor,")
    print("     which the transfer never sees except where donor == recipient (SELF,"
          " circular).")
    print("=" * 108)

    for bname, B in tg.items():
        # --- baselines -------------------------------------------------------
        raw = score(B["thf"], np.ones(len(B["thf"])), B)
        mu_g, S_g = B["muf"], B["Sf"]
        xg, frac_g = gauss_draws(mu_g, S_g, B["lo"], B["hi"])
        rawg = score(xg, np.ones(len(xg)), B)
        print(f"\n--- recipient {bname}  (anchor held out) "
              f"{'-' * 60}")
        print(f"  {'donor':8s} {'mode':5s} {'meanJS':>8s} {'maxJS':>7s} "
              f"{'|dmu|/sig med':>14s} {'max':>7s} {'sd ratio med':>13s} {'in-prior':>9s}")
        print(f"  {'-':8s} {'RAW':5s} {raw['mean_js']:8.4f} {raw['js'].max():7.4f} "
              f"{np.median(raw['dmu_sig']):14.2f} {raw['dmu_sig'].max():7.2f} "
              f"{np.median(raw['sd_ratio']):13.2f} {'-':>9s}")
        print(f"  {'-':8s} {'RAWG':5s} {rawg['mean_js']:8.4f} {rawg['js'].max():7.4f} "
              f"{np.median(rawg['dmu_sig']):14.2f} {rawg['dmu_sig'].max():7.2f} "
              f"{np.median(rawg['sd_ratio']):13.2f} {frac_g:9.3f}")
        rows.append(dict(recipient=bname, donor="-", mode="RAW", **_ser(raw), in_prior=None))
        rows.append(dict(recipient=bname, donor="-", mode="RAWG", **_ser(rawg), in_prior=frac_g))

        for aname, A in tg.items():
            cal = calibration(A)
            for mode in ("M1", "M2", "M2c", "M3"):
                mu, S = transfer(cal, B, mode)
                x, frac = gauss_draws(mu, S, B["lo"], B["hi"])
                sc = score(x, np.ones(len(x)), B)
                tagself = " SELF" if aname == bname else ""
                print(f"  {aname:8s} {mode:5s} {sc['mean_js']:8.4f} {sc['js'].max():7.4f} "
                      f"{np.median(sc['dmu_sig']):14.2f} {sc['dmu_sig'].max():7.2f} "
                      f"{np.median(sc['sd_ratio']):13.2f} {frac:9.3f}{tagself}")
                rows.append(dict(recipient=bname, donor=aname, mode=mode, **_ser(sc),
                                 in_prior=frac, self=(aname == bname)))

    # --- the dimensionless calibrations, side by side ------------------------
    print("\n" + "=" * 108)
    print("The calibrations themselves, dimensionless. If these three columns do not"
          " agree, there is")
    print("nothing to transfer. s = bias in flow-sd units, k = width deflation"
          " sd_NS/sd_flow.")
    print("=" * 108)
    cals = {k: calibration(v) for k, v in tg.items()}
    print(f"  {'param':8s} " + "".join(f"{'s_' + k:>12s}" for k in cals)
          + "   " + "".join(f"{'k_' + k:>12s}" for k in cals))
    for i, nm in enumerate(NAMES):
        print(f"  {nm:8s} " + "".join(f"{cals[k]['s'][i]:12.2f}" for k in cals)
              + "   " + "".join(f"{cals[k]['k'][i]:12.3f}" for k in cals))
    sm = np.array([cals[k]["s"] for k in cals])
    km = np.array([cals[k]["k"] for k in cals])
    print(f"\n  bias s:        across-target spread / mean |s| = "
          f"{np.mean(sm.std(0) / (np.abs(sm).mean(0) + 1e-12)):.2f}")
    print(f"  deflation k:   across-target spread / mean k     = "
          f"{np.mean(km.std(0) / km.mean(0)):.2f}")
    print(f"  k range per param (max/min across the three donors): "
          f"{np.max(km.max(0) / km.min(0)):.1f}x worst, "
          f"{np.median(km.max(0) / km.min(0)):.1f}x median")

    rows += pooled_block(tg)

    Path("data/h11").mkdir(parents=True, exist_ok=True)
    with open("data/h11/transfer_matrix.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n[h11] wrote data/h11/transfer_matrix.json")



def pooled_block(tg):
    """The fairest version of transfer, and the strongest honest claim available.

    M2/M3 transfer ONE donor's calibration. A practitioner with several anchored targets
    would instead POOL them, and would leave the recipient out of the pool. So:

      M4  leave-one-out pooled deflation. Widths = geometric mean of the other targets'
          k vectors times the recipient's own flow sd, recipient's own centre.
      M5  one scalar. A single number, the geometric mean of the other targets' k over
          every parameter. This needs no anchor on the recipient and almost no
          information at all, which is the point: if M5 captures most of the available
          gain then the "calibration" is just "the flow is too wide, divide by about
          three", and no amortized anchor is being transferred.

    Geometric mean because k is a ratio.
    """
    print("\n" + "=" * 108)
    print("Leave-one-out POOLED calibration. The recipient is never in the pool, so these"
          " are honest.")
    print("M4 = pooled per-parameter deflation. M5 = ONE scalar for all seven parameters.")
    print("=" * 108)
    print(f"  {'recipient':10s} {'mode':5s} {'meanJS':>8s} {'vs RAW':>8s} "
          f"{'sd ratio med':>13s} {'in-prior':>9s}   scalar")
    out = []
    for bname, B in tg.items():
        raw = score(B["thf"], np.ones(len(B["thf"])), B)
        others = [calibration(tg[a])["k"] for a in tg if a != bname]
        kpool = np.exp(np.mean(np.log(np.array(others)), axis=0))
        for mode, kv in (("M4", kpool), ("M5", np.full(7, float(np.exp(np.mean(np.log(kpool))))))):
            sd = kv * B["sdf"]
            S = np.outer(sd, sd) * corr(B["Sf"])
            x, frac = gauss_draws(B["muf"], S, B["lo"], B["hi"])
            sc = score(x, np.ones(len(x)), B)
            gain = raw["mean_js"] - sc["mean_js"]
            sc_txt = f"{kv[0]:.3f}" if mode == "M5" else "-"
            print(f"  {bname:10s} {mode:5s} {sc['mean_js']:8.4f} {gain:+8.4f} "
                  f"{np.median(sc['sd_ratio']):13.2f} {frac:9.3f}   {sc_txt}")
            out.append(dict(recipient=bname, donor="pooled-LOO", mode=mode,
                            in_prior=frac, **_ser(sc)))
    return out


def _ser(sc):
    return dict(mean_js=sc["mean_js"], js=sc["js"].tolist(),
                dmu_sig=sc["dmu_sig"].tolist(), sd_ratio=sc["sd_ratio"].tolist())


if __name__ == "__main__":
    main()
