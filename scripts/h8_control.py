"""
H8 control - how much does the OPACITY DATA move the spectrum, same code?

h8_compare.py found TauREx and pRT agree to ~13 ppm on WASP-96b. That is only
meaningful if the measurement could have detected a disagreement. This runs the
identical binning and metrics on two TauREx spectra that differ ONLY in the
opacity tables: ExoMolOP R=15000 (--hifi, what H8 uses) against the published
ExoTransmit R~1000 lists. Same code, same atmosphere, same theta, same binner.

If the line-list swap moves the spectrum by much more than the code swap, then
the pipeline is sensitive, the ~13 ppm is a real agreement rather than a blind
spot, and the insistence on running H8 with --hifi was load-bearing rather than
pedantic.

    conda run -n mirage-taurex python scripts/h8_control.py
"""

import argparse

import numpy as np
from h8_compare import binned, g395h_grid


def main(hifi, published, grid="obs", ref_err_ppm=None,
         code_off=-12.9, code_shape=8.9, code_drp=-0.00055):
    a = np.load(hifi, allow_pickle=True)        # ExoMolOP R=15000
    b = np.load(published, allow_pickle=True)   # ExoTransmit R~1000
    if grid == "g395h":
        wl, width = g395h_grid()
    else:
        wl, width = a["obs_wl"], a["obs_width"]
    rs_rj = float(a["rs_rsun"]) * float(a["RSOL"]) / float(a["RJUP"])
    med_err = (ref_err_ppm if ref_err_ppm is not None
               else float(np.median(a["obs_err"])) * 1e6)

    offs, shapes, drps = [], [], []
    for k in range(len(a["thetas"])):
        ba = binned(10000.0 / a["native_wn"], a["rprs2"][k], wl, width)
        bb = binned(10000.0 / b["native_wn"], b["rprs2"][k], wl, width)
        diff = (bb - ba) * 1e6
        offs.append(float(np.mean(diff)))
        shapes.append(float(np.sqrt(np.mean((diff - np.mean(diff)) ** 2))))
        drps.append(float(np.mean((np.sqrt(np.clip(bb, 0, None))
                                   - np.sqrt(np.clip(ba, 0, None))) * rs_rj)))

    offs, shapes, drps = map(np.array, (offs, shapes, drps))
    print("\n" + "=" * 80)
    print("  H8 CONTROL - same code, same atmosphere, DIFFERENT line lists")
    print("  (TauREx ExoTransmit R~1000 published  vs  TauREx ExoMolOP R=15000 --hifi)")
    print("=" * 80)
    print(f"  {str(a['planet'])}, grid={grid}, {len(wl)} bins, "
          f"{wl.min():.3f}-{wl.max():.3f} um")
    print(f"  denominator {med_err:.1f} ppm\n")
    print(f"  at the posterior mean:")
    print(f"    constant depth offset   {offs[0]:>9.1f} ppm = {abs(offs[0])/med_err:.3f} x error bar")
    print(f"    shape residual (rms)    {shapes[0]:>9.1f} ppm = {shapes[0]/med_err:.3f} x error bar")
    print(f"    equivalent radius shift {drps[0]:>9.5f} RJ")
    print(f"  across {len(offs)-1} posterior draws:")
    print(f"    offset    {offs[1:].min():>8.1f} to {offs[1:].max():.1f} ppm")
    print(f"    shape rms {shapes[1:].min():>8.1f} to {shapes[1:].max():.1f} ppm")
    print(f"    dRp       {drps[1:].min():>8.5f} to {drps[1:].max():.5f} RJ")
    print("\n" + "-" * 80)
    print("  side by side, at the posterior mean:")
    print(f"    {'':<26}{'offset ppm':>12}{'shape rms':>12}{'dRp RJ':>11}")
    print(f"    {'code swap (H8)':<26}{code_off:>12.1f}{code_shape:>12.1f}{code_drp:>11.5f}")
    print(f"    {'line-list swap (control)':<26}{offs[0]:>12.1f}{shapes[0]:>12.1f}{drps[0]:>11.5f}")
    print(f"    ratio, line list / code   {abs(offs[0]/code_off):>12.1f}x"
          f"{shapes[0]/code_shape:>11.1f}x{abs(drps[0]/code_drp):>10.1f}x")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hifi", default="data/h8/taurex_dump.npz")
    ap.add_argument("--published", default="data/h8/taurex_dump_published.npz")
    ap.add_argument("--grid", default="obs", choices=["obs", "g395h"])
    ap.add_argument("--ref-err-ppm", type=float, default=None)
    # the code-swap row to divide by; defaults are WASP-96b's (P6-D19)
    ap.add_argument("--code-off", type=float, default=-12.9)
    ap.add_argument("--code-shape", type=float, default=8.9)
    ap.add_argument("--code-drp", type=float, default=-0.00055)
    a = ap.parse_args()
    main(a.hifi, a.published, grid=a.grid, ref_err_ppm=a.ref_err_ppm,
         code_off=a.code_off, code_shape=a.code_shape, code_drp=a.code_drp)
