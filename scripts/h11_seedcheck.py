"""
H11 control — is the across-target variation in the calibration real, or just estimation
noise from retraining?

The objection this answers. The three flow posteriors in h11_ot_transfer.py come from
three SEPARATELY TRAINED networks (noisecond_rad_cov for WASP-39b, and the two
per-planet rad_nocond arms), each with its own grid and prior box. So a calibration that
fails to transfer across targets could be failing across MODELS instead, which would be
a different and much less interesting finding.

The control. WASP-39b has three training seeds of the same arm against one anchor. Those
differ only by training run, so the spread of the calibration vector across them is the
NOISE FLOOR on the calibration. If the across-target spread is large compared to that
floor, the across-target variation is target-driven. If it is comparable, the whole H11
result is confounded and has to be withdrawn.

It also runs seed-to-seed transfer, a strictly easier ask than target-to-target: same
planet, same anchor, same architecture, different random seed.

    conda run -n mirage python scripts/h11_seedcheck.py
"""
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h11_ot_transfer import (NAMES, _ORDER, _moments, corr, calibration, transfer,
                             gauss_draws, score, load, TARGETS)

OUT = Path("data/real_ess")
LO = np.array([0.8, 300., -9, -9, -9, -9, -9])
HI = np.array([1.6, 2500., -1, -1, -1, -1, -1])
SEED_ARMS = {f"{arm}_s{s}": OUT / f"real_ess_samples_h9_{arm}_s{s}.npz"
             for arm in ("rad_cov", "rad_nocond") for s in (42, 43, 44)}


def load_seed(path):
    f = np.load(path, allow_pickle=True)
    if len(f["wlen"]) != 52:
        raise SystemExit(f"[guard] {path} is not a 52-bin WASP-39b posterior")
    n = np.load(OUT / "wasp39b_ns_posterior_fitrad.npz", allow_pickle=True)
    nsn = [str(x) for x in n["fit_names"]]
    thn = n["samples"][:, [nsn.index(o) for o in _ORDER]]
    wn = n["weights"] / n["weights"].sum()
    muf, Sf = _moments(f["theta"])
    mun, Sn = _moments(thn, wn)
    return dict(thf=f["theta"], thn=thn, wn=wn, muf=muf, Sf=Sf, mun=mun, Sn=Sn,
                sdf=np.sqrt(np.diag(Sf)), sdn=np.sqrt(np.diag(Sn)), lo=LO, hi=HI)


def main():
    S = {k: load_seed(p) for k, p in SEED_ARMS.items()}
    cals = {k: calibration(v) for k, v in S.items()}

    for arm in ("rad_cov", "rad_nocond"):
        keys = [f"{arm}_s{s}" for s in (42, 43, 44)]
        km = np.array([cals[k]["k"] for k in keys])
        sm = np.array([cals[k]["s"] for k in keys])
        print(f"\n=== {arm}, three seeds, same target and same anchor "
              f"{'=' * 30}")
        print(f"  {'param':8s} {'k s42':>9s} {'k s43':>9s} {'k s44':>9s} "
              f"{'k max/min':>10s}    {'s s42':>8s} {'s s43':>8s} {'s s44':>8s} "
              f"{'s spread':>9s}")
        for i, nm in enumerate(NAMES):
            print(f"  {nm:8s} " + "".join(f"{km[j, i]:9.3f}" for j in range(3))
                  + f"{km[:, i].max() / km[:, i].min():10.2f}    "
                  + "".join(f"{sm[j, i]:8.2f}" for j in range(3))
                  + f"{sm[:, i].std():9.2f}")
        print(f"  seed noise floor:  k spread/mean = {np.mean(km.std(0) / km.mean(0)):.3f}"
              f"   k max/min = {np.max(km.max(0) / km.min(0)):.2f}x worst, "
              f"{np.median(km.max(0) / km.min(0)):.2f}x median")
        print(f"                     s spread / mean|s| = "
              f"{np.mean(sm.std(0) / (np.abs(sm).mean(0) + 1e-12)):.3f}")

    # the across-TARGET numbers, for the comparison that matters
    tg = {k: load(k) for k in TARGETS}
    tcal = np.array([calibration(v)["k"] for v in tg.values()])
    tsal = np.array([calibration(v)["s"] for v in tg.values()])
    print(f"\n=== the comparison {'=' * 60}")
    print(f"  across-TARGET   k spread/mean = {np.mean(tcal.std(0) / tcal.mean(0)):.3f}"
          f"   s spread/mean|s| = "
          f"{np.mean(tsal.std(0) / (np.abs(tsal).mean(0) + 1e-12)):.3f}")
    for arm in ("rad_cov", "rad_nocond"):
        keys = [f"{arm}_s{s}" for s in (42, 43, 44)]
        km = np.array([cals[k]["k"] for k in keys])
        sm = np.array([cals[k]["s"] for k in keys])
        print(f"  across-SEED     k spread/mean = {np.mean(km.std(0) / km.mean(0)):.3f}"
              f"   s spread/mean|s| = "
              f"{np.mean(sm.std(0) / (np.abs(sm).mean(0) + 1e-12)):.3f}   ({arm})")

    # seed-to-seed transfer, an easier ask than target-to-target
    print(f"\n=== seed-to-seed transfer, same planet and same anchor "
          f"{'=' * 30}")
    print("  (if the calibration cannot even survive a change of random seed, "
          "target transfer is moot)")
    for arm in ("rad_cov", "rad_nocond"):
        keys = [f"{arm}_s{s}" for s in (42, 43, 44)]
        print(f"\n  --- {arm}")
        print(f"    {'recipient':12s} {'donor':12s} {'mode':5s} {'meanJS':>8s} "
              f"{'in-prior':>9s}")
        for bk in keys:
            B = S[bk]
            raw = score(B["thf"], np.ones(len(B["thf"])), B)
            print(f"    {bk:12s} {'-':12s} {'RAW':5s} {raw['mean_js']:8.4f} {'-':>9s}")
            for ak in keys:
                for mode in ("M2", "M3"):
                    mu, Sg = transfer(cals[ak], B, mode)
                    x, frac = gauss_draws(mu, Sg, B["lo"], B["hi"])
                    sc = score(x, np.ones(len(x)), B)
                    tag = " SELF" if ak == bk else ""
                    print(f"    {bk:12s} {ak:12s} {mode:5s} {sc['mean_js']:8.4f} "
                          f"{frac:9.3f}{tag}")


if __name__ == "__main__":
    main()
