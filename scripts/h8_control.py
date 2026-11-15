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

import numpy as np
from h8_compare import binned


def main():
    a = np.load("data/h8/taurex_dump.npz", allow_pickle=True)            # ExoMolOP R=15000
    b = np.load("data/h8/taurex_dump_published.npz", allow_pickle=True)  # ExoTransmit R~1000
    wl, width, err = a["obs_wl"], a["obs_width"], a["obs_err"]
    rs_rj = float(a["rs_rsun"]) * float(a["RSOL"]) / float(a["RJUP"])
    med_err = float(np.median(err)) * 1e6

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
    print(f"  median obs error {med_err:.1f} ppm\n")
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
    print(f"    {'code swap (H8)':<26}{-12.9:>12.1f}{8.9:>12.1f}{-0.00055:>11.5f}")
    print(f"    {'line-list swap (control)':<26}{offs[0]:>12.1f}{shapes[0]:>12.1f}{drps[0]:>11.5f}")
    print(f"    ratio, line list / code   {abs(offs[0])/12.9:>12.1f}x{shapes[0]/8.9:>11.1f}x"
          f"{abs(drps[0]/-0.00055):>10.1f}x")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
