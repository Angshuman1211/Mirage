"""
H11 guard — the JS ranking, re-derived with a metric that cannot saturate.

Why. js_1d saturates at ln2 = 0.6931 for disjoint marginals and many H11 cells are
already there, so a reviewer can fairly say the ordering among the bad cells is an
artefact of the ceiling. The 2-Wasserstein distance between Gaussians has no ceiling.
Everything is first divided by the RECIPIENT ANCHOR's per-parameter sd, so the number
is dimensionless and comparable across targets, and reads as "how many anchor sigmas
away is this proposal, counting both location and shape".

    conda run -n mirage python scripts/h11_w2.py
"""
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h11_ot_transfer import TARGETS, load, calibration, transfer, corr, _moments, NAMES


def _msqrt(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def w2(mu1, S1, mu2, S2):
    S2h = _msqrt(S2)
    cross = _msqrt(S2h @ S1 @ S2h)
    v = float(np.sum((mu1 - mu2) ** 2) + np.trace(S1 + S2 - 2 * cross))
    return np.sqrt(max(v, 0.0))


def main():
    tg = {k: load(k) for k in TARGETS}
    print("=" * 96)
    print("W2 to the recipient's held-out anchor, in units of the anchor's own sd"
          " (no saturation ceiling).")
    print("=" * 96)
    print(f"  {'recipient':10s} {'donor':12s} {'mode':5s} {'W2/sig_NS':>10s} "
          f"{'loc part':>9s} {'shape part':>11s}")
    for bname, B in tg.items():
        D = np.diag(1.0 / B["sdn"])                  # whiten by the ANCHOR's scale
        wmu, wS = D @ B["mun"], D @ B["Sn"] @ D
        def row(donor, mode, mu, S):
            a, b = D @ mu, D @ S @ D
            tot = w2(a, b, wmu, wS)
            loc = float(np.sqrt(np.sum((a - wmu) ** 2)))
            shp = w2(np.zeros(7), b, np.zeros(7), wS)
            print(f"  {bname:10s} {donor:12s} {mode:5s} {tot:10.2f} {loc:9.2f} {shp:11.2f}")
        muf, Sf = _moments(B["thf"])
        row("-", "RAW", muf, Sf)
        others = [calibration(tg[a])["k"] for a in tg if a != bname]
        kp = np.exp(np.mean(np.log(np.array(others)), axis=0))
        sd = kp * B["sdf"]
        row("pooled-LOO", "M4", muf, np.outer(sd, sd) * corr(Sf))
        for aname, A in tg.items():
            cal = calibration(A)
            for mode in ("M1", "M2", "M3"):
                mu, S = transfer(cal, B, mode)
                row(aname + (" SELF" if aname == bname else ""), mode, mu, S)
        print()


if __name__ == "__main__":
    main()
