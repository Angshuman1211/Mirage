"""
H8 part 2 of 3 - the petitRADTRANS side.

Reads the atmosphere TauREx actually integrated (h8_dump_atmosphere.py) and
re-integrates it with pRT. Nothing about the atmosphere is recomputed here:
pressure, temperature, mixing ratios, mean molecular weight, and the reference
radius/gravity/pressure triple all come from the dump. Only the radiative
transfer and the opacity tables are pRT's own, which is exactly what H8 is
trying to isolate.

Line lists are matched to data/opacity_hifi by the DOI stored inside each file.
The CO entry is the one that needed care: pRT's 12C-16O__HITEMP carries the
HITEMP 2010 DOI, NOT Li et al. 2015, despite the isotopologue name matching
TauREx's 12C-16O__Li2015. C-O-NatAbund__Chubb carries the Li 2015 DOI and is
the correct match, at the cost of being natural abundance rather than the main
isotopologue alone.

Runs in the `mirage-prt` env, which exists because the pRT in `mirage` cannot
run at all: see scripts/setup_prt_env.sh.

    conda run -n mirage-prt python scripts/h8_prt_forward.py
"""

import argparse
from pathlib import Path

import numpy as np
from petitRADTRANS.radtrans import Radtrans

# TauREx gas name -> pRT correlated-k species, matched on the DOI inside the file
SPECIES = {
    "H2O": "1H2-16O__POKAZATEL",
    "CO2": "12C-16O2__UCL-4000",
    "CH4": "12C-1H4__YT34to10",
    "CO":  "C-O-NatAbund__Chubb",
    "NH3": "14N-1H3__CoYuTe",
}
MOLAR_MASS = {       # g/mol
    "H2O": 18.01528, "CO2": 44.0095, "CH4": 16.04246, "CO": 28.0101,
    "NH3": 17.03052, "H2": 2.01588, "He": 4.002602,
}
NA = 6.02214076e26   # per kg-mol, to turn TauREx's mu (kg) into g/mol


def main(dump, out, wl_lo, wl_hi):
    d = np.load(dump, allow_pickle=True)
    gases = [str(x) for x in d["gas_names"]]
    p_pa = d["p_pa"]
    # TauREx layer 0 is the BOTTOM (highest pressure); pRT wants top first.
    order = np.argsort(p_pa)
    p_bar = p_pa[order] / 1e5

    rt = Radtrans(
        pressures=p_bar,
        line_species=[SPECIES[g] for g in gases if g in SPECIES],
        gas_continuum_contributors=["H2--H2", "H2--He"],
        rayleigh_species=["H2", "He"],
        wavelength_boundaries=[wl_lo, wl_hi],
        line_opacity_mode="c-k",
    )

    rs_cm = float(d["rs_rsun"]) * float(d["RSOL"]) * 100.0
    ref_p_bar = float(d["ref_pressure_pa"]) / 1e5
    rjup_cm = float(d["RJUP"]) * 100.0

    out_r2, mu_err = [], []
    for k in range(len(d["thetas"])):
        vmr = d["vmr"][k][:, order]                 # (7, nlayers), top first
        temp = d["temperature"][k][order]
        vmr_of = {g: vmr[i] for i, g in enumerate(gases)}

        # mass fractions from volume mixing ratios, with the MMW implied by the
        # same numbers. Checked against TauREx's own mu below rather than
        # assumed: a silent MMW mismatch rescales the scale height and would
        # look exactly like a radiative-transfer disagreement.
        mmw = sum(vmr_of[g] * MOLAR_MASS[g] for g in gases)
        mass_fractions = {}
        for g in gases:
            key = SPECIES.get(g, g)                 # H2 and He keep their names
            mass_fractions[key] = vmr_of[g] * MOLAR_MASS[g] / mmw
        mu_taurex = d["mu_kg"][k][order] * NA
        mu_err.append(float(np.max(np.abs(mmw - mu_taurex) / mu_taurex)))

        wl_cm, radii_cm, _ = rt.calculate_transit_radii(
            temperatures=temp,
            mass_fractions=mass_fractions,
            mean_molar_masses=mmw,
            # per draw: TauREx recomputes gravity from the mass and the draw's
            # own radius, so a single scalar taken from draw 0 is wrong for
            # every other draw (it was, and the h8_prt_retrieve self-test caught
            # it at 8 ppm, the same size as the shape residual being measured).
            reference_gravity=float(d["gravity_ms2"][k][0]) * 100.0,
            reference_pressure=ref_p_bar,
            planet_radius=float(d["rp_rj"][k]) * rjup_cm,
            variable_gravity=True,
        )
        out_r2.append((radii_cm / rs_cm) ** 2)
        print(f"  [{k:2d}] Rp={float(d['rp_rj'][k]):.4f} T={d['temperature'][k][0]:7.2f} "
              f"mu rel-err {mu_err[-1]:.2e}")

    wl_um = wl_cm * 1e4
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, wl_um=wl_um, rprs2=np.asarray(out_r2),
                        mu_rel_err=np.asarray(mu_err),
                        species=np.array([SPECIES[g] for g in gases if g in SPECIES], dtype=object))
    print(f"\n[h8] pRT: {len(out_r2)} spectra, {len(wl_um)} native pts, "
          f"{wl_um.min():.3f}-{wl_um.max():.3f} um -> {out}")
    print(f"  worst MMW disagreement vs TauREx: {max(mu_err):.2e} (relative)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="data/h8/taurex_dump.npz")
    ap.add_argument("--out", default="data/h8/prt_spectra.npz")
    ap.add_argument("--wl-lo", type=float, default=0.80)
    ap.add_argument("--wl-hi", type=float, default=2.90)
    a = ap.parse_args()
    main(a.dump, a.out, a.wl_lo, a.wl_hi)
