"""
H8 final - TauREx-NS against petitRADTRANS-NS on real WASP-96b.

This is the comparison G6/H8 is literally phrased as: "if TauREx-NS and pRT-NS
disagree by more than the flow-to-NS gap, the reference is not a reference".
Both posteriors come from nestle, method multi, dlogz 0.5, 150 live points, the
same priors, the same 1 percent error floor and the same binning. The only
difference between them is the radiative transfer code.

The yardstick is the published flow-to-NS JS on real data, 0.352 / 0.371 /
0.428 for the three arms (P6-D6). JS is computed with the SAME js_1d used for
those numbers, so the two are directly comparable.

    conda run -n mirage-taurex python scripts/h8_ns_compare.py
"""

import numpy as np

from real_compare import js_1d

FLOW_TO_NS = (0.352, 0.428)     # best and worst published arm, real data (P6-D6)
NAMES = ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]


def wq(v, w, q):
    i = np.argsort(v)
    return float(np.interp(q, np.cumsum(w[i]), v[i]))


def load(path):
    d = np.load(path, allow_pickle=True)
    w = d["weights"].astype(float)
    w = w / w.sum()
    return d["samples"], w, [str(x) for x in d["fit_names"]], float(d["chi2"])


def main():
    sa, wa, na, chi_a = load("data/real_ess/wasp39b_ns_posterior_wasp96_live150.npz")
    sb, wb, nb, chi_b = load("data/h8/prt_ns_posterior_wasp96.npz")
    if na != nb:
        raise SystemExit(f"parameter mismatch:\n  taurex {na}\n  prt    {nb}")

    print("\n" + "=" * 90)
    print("  H8 - TauREx-NS vs petitRADTRANS-NS, real WASP-96b, both 150 live points")
    print("=" * 90)
    print(f"  {'param':<15}{'TauREx mean':>13}{'pRT mean':>11}{'diff':>10}"
          f"{'diff/sigma':>12}{'JS (nats)':>11}")
    js_all, nsig = [], []
    for j, nm in enumerate(NAMES):
        ma = float(np.average(sa[:, j], weights=wa))
        mb = float(np.average(sb[:, j], weights=wb))
        sd = float(np.sqrt(np.average((sa[:, j] - ma) ** 2, weights=wa)))
        js = js_1d(sa[:, j], wa, sb[:, j], wb)
        js_all.append(js)
        nsig.append(abs(mb - ma) / sd if sd > 0 else np.nan)
        print(f"  {nm:<15}{ma:>13.4f}{mb:>11.4f}{mb-ma:>10.4f}{nsig[-1]:>12.2f}{js:>11.4f}")

    js_all = np.array(js_all)
    print("\n" + "-" * 90)
    print(f"  mean JS across parameters   {js_all.mean():.4f} nats")
    print(f"  max  JS (worst parameter)   {js_all.max():.4f} nats  ({NAMES[int(js_all.argmax())]})")
    print(f"  flow-to-NS JS, published    {FLOW_TO_NS[0]:.3f} (best arm) to {FLOW_TO_NS[1]:.3f} (worst)")
    print(f"  ratio, code gap / flow gap  {js_all.mean()/FLOW_TO_NS[0]:.3f}x (vs the BEST arm)")
    print(f"  chi2/N                      TauREx {chi_a:.3f}, pRT {chi_b:.3f}")

    ra = (wq(sa[:, 0], wa, 0.16), wq(sa[:, 0], wa, 0.84))
    rb = (wq(sb[:, 0], wb, 0.16), wq(sb[:, 0], wb, 0.84))
    print(f"\n  radius 68% interval")
    print(f"    TauREx  [{ra[0]:.4f}, {ra[1]:.4f}]  width {ra[1]-ra[0]:.4f}")
    print(f"    pRT     [{rb[0]:.4f}, {rb[1]:.4f}]  width {rb[1]-rb[0]:.4f}")
    print(f"    literature 1.20 inside: TauREx {'YES' if ra[0]<=1.20<=ra[1] else 'NO'}, "
          f"pRT {'YES' if rb[0]<=1.20<=rb[1] else 'NO'}")
    # the forward-model test predicted this shift before the retrieval was run
    print(f"\n  radius mean shift    {float(np.average(sb[:,0],weights=wb))-float(np.average(sa[:,0],weights=wa)):+.5f} RJ")
    print(f"  forward-model predicted  -0.00055 RJ (h8_compare.py, before this ran)")
    print("-" * 90)
    verdict = js_all.mean() < FLOW_TO_NS[0]
    print(f"\n  VERDICT: the two codes' posteriors are {'MUCH CLOSER' if verdict else 'NOT closer'} "
          f"to each other than the flow is to either.")
    print(f"  => the NS anchor survives as a reference. G6 closes.\n" if verdict else
          "  => the reference is not a reference. G6 stands.\n")


if __name__ == "__main__":
    main()
