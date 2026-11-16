"""
A4 - H3, the error-floor sweep.

WHY THIS IS CHEAP. The floor enters the likelihood only through

    se = sqrt(sigma^2 + (floor * x)^2)

so changing it re-weights forward spectra that already exist. Every cell below reuses a
stored forward run from H11 or A2. No TauREx is called.

WHY IT MATTERS. Every ESS in P3-D12, P6-D21 and P6-D22 was computed at ONE floor, 5
percent, inherited from the published pipeline and never justified. Three things follow
that the single-floor numbers cannot answer.

  1. Is 5 percent favourable, neutral or costly for the headline numbers.
  2. What floor the data actually wants, read off where best-fit chi2/N crosses 1.
  3. Does A2's "WASP-96b is the only arm above ESS 50" survive a change of floor.

EFFICIENCY, NOT ESS, IS THE COMPARABLE QUANTITY. WASP-39b's uncalibrated arm has 1000
draws and every other cell has 2000 (corrected in P6-D22). ESS is an absolute count, so
epsilon = ESS/N is what gets plotted and compared.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from h11_ot_transfer import NAMES, load                       # noqa: E402
from real_compare import js_1d                                # noqa: E402

FLOORS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
PUBLISHED = 0.05
PLANETS = ["wasp39", "wasp96", "k218"]
ARMS = {
    "none":     ("data/h11/h11_{p}_raw_samples.npz",     "data/h11/h11_{p}_raw_forward.npz"),
    "circular": ("data/h11/h11_{p}_self_M2_samples.npz", "data/h11/h11_{p}_self_M2_forward.npz"),
    "A2":       ("data/a2/a2_{p}_injcal_samples.npz",    "data/a2/a2_{p}_injcal_forward.npz"),
}
OUT = Path("data/a4")


def cell(d, ms, floor):
    """Weights, best-fit chi2/N and ESS at one floor. Pure re-weighting."""
    x, sig, cov = d["x_obs"], d["sig_obs"], d["covered"]
    msk = cov & np.isfinite(sig) & (sig > 0)
    se = np.sqrt(sig ** 2 + (floor * x) ** 2)
    valid = np.isfinite(ms[:, msk]).all(axis=1)
    r = (x[None, msk] - ms[:, msk]) / se[None, msk]
    chi2 = np.nansum(r ** 2, axis=1)
    ll = np.where(valid, -0.5 * chi2, -np.inf)
    lw = ll - d["log_g"]
    lw -= np.nanmax(lw)
    w = np.exp(lw)
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    nb = int(msk.sum())
    best = int(np.argmax(ll))
    rg = np.random.default_rng(0)
    bs = []
    for _ in range(400):
        wb = w[rg.integers(0, len(w), len(w))]
        if wb.sum() > 0:
            bs.append(1.0 / np.sum((wb / wb.sum()) ** 2))
    lo, hi = np.percentile(bs, [16, 84])
    return dict(ess=ess, n=len(w), eps=100.0 * ess / len(w),
                eps_lo=100.0 * lo / len(w), eps_hi=100.0 * hi / len(w),
                chi2_n=float(chi2[best] / nb), nbins=nb, w=w)


def crossing(floors, chi2n):
    """Floor at which best-fit chi2/N first reaches 1, linear in log(floor)."""
    f, c = np.asarray(floors, float), np.asarray(chi2n, float)
    for i in range(len(f) - 1):
        if (c[i] - 1.0) * (c[i + 1] - 1.0) <= 0 and c[i] != c[i + 1]:
            t = (1.0 - c[i]) / (c[i + 1] - c[i])
            return float(np.exp(np.log(f[i]) + t * (np.log(f[i + 1]) - np.log(f[i]))))
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print("=" * 104)
    print("A4  error-floor sweep. Stored forwards re-weighted at each floor, no TauREx.")
    print("    epsilon = ESS/N is the comparable column; N differs between arms.")
    print("=" * 104)
    for p in PLANETS:
        B = load(p)
        thn, wn = B["thn"], B["wn"]
        print(f"\n--- {p} {'-' * 72}")
        print(f"  {'arm':9s} {'floor':>7s} {'chi2/N':>8s} {'ESS':>8s} {'N':>6s} "
              f"{'eps %':>7s} {'68% CI on eps':>16s} {'meanJS':>8s}")
        for arm, (sp, fp) in ARMS.items():
            sp, fp = Path(sp.format(p=p)), Path(fp.format(p=p))
            if not (sp.exists() and fp.exists()):
                print(f"  {arm:9s} missing artefact, skipped")
                continue
            d = np.load(sp, allow_pickle=True)
            ms = np.load(fp)["model_spec"]
            th = d["theta"]
            for fl in FLOORS:
                c = cell(d, ms, fl)
                js = float(np.mean([js_1d(th[:, i], c["w"], thn[:, i], wn)
                                    for i in range(len(NAMES))]))
                ci = f"[{c['eps_lo']:.2f}, {c['eps_hi']:.2f}]"
                star = " <- published" if abs(fl - PUBLISHED) < 1e-9 else ""
                print(f"  {arm:9s} {100 * fl:6.1f}% {c['chi2_n']:8.2f} {c['ess']:8.1f} "
                      f"{c['n']:6d} {c['eps']:7.2f} {ci:>16s} {js:8.4f}{star}")
                rows.append(dict(target=p, arm=arm, floor=fl, mean_js=js,
                                 **{k: v for k, v in c.items() if k != "w"}))
            sub = [r for r in rows if r["target"] == p and r["arm"] == arm]
            x = crossing([r["floor"] for r in sub], [r["chi2_n"] for r in sub])
            print(f"  {'':9s} chi2/N reaches 1 at floor = "
                  f"{(f'{100 * x:.2f}%' if x else 'outside the swept range')}")
    (OUT / "a4_floor_sweep.json").write_text(json.dumps(rows, indent=1))
    print(f"\n[a4] {len(rows)} cells -> {OUT / 'a4_floor_sweep.json'}")


if __name__ == "__main__":
    main()
