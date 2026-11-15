"""
H8 part 1 of 3 - dump the TauREx side of the cross-code check.

H8 asks whether the TauREx forward model that every MIRAGE number is anchored
to is misspecified. A5 part 1 (P6-D18) showed the nested sampler is stable, so
"the anchor is under-converged noise" is closed. "The forward model is wrong"
is not, and a sampler can converge tightly onto a wrong model.

The check is to recompute the same spectrum with an independent radiative
transfer code (petitRADTRANS) and see whether the two agree to better than the
observational error bar. For that to mean anything, EVERYTHING except the
radiative transfer has to be held fixed. So this does not just record theta and
let pRT rebuild the atmosphere from it: it dumps TauREx's own pressure,
temperature, per-gas volume mixing ratio and mean-molecular-weight profiles,
and the reference radius/gravity/pressure triple, so pRT integrates through the
identical atmosphere. Anything the two codes still disagree about is then the
radiative transfer and the opacity treatment, which is the thing being tested.

Runs with --hifi so the TauREx side uses the ExoMolOP R=15000 lists, which are
the same line lists (checked by DOI inside each file, not by filename) as the
pRT correlated-k tables. Without that the comparison is unattributable between
radiative transfer and opacity data, which would void the whole result.

    conda run -n mirage-taurex python scripts/h8_dump_atmosphere.py
"""

import argparse
from pathlib import Path

import numpy as np

import taurex_forward as tf                      # PYTHONPATH=scripts
from taurex.constants import RJUP, MJUP, RSOL, MSOL

OUT = Path("data/h8")
PLANET_GEOM = {"wasp96": lambda: tf.WASP96, "wasp39": lambda: tf.WASP39,
               "k218": lambda: tf.K2_18b}


def main(planet, post, ndraw, seed, out, hifi=True, floor=0.01):
    geom = PLANET_GEOM[planet]()
    d = np.load(post, allow_pickle=True)
    names = [str(x) for x in d["fit_names"]]
    s, w = d["samples"], d["weights"].astype(float)
    w = w / w.sum()
    if names != ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]:
        raise SystemExit(f"unexpected fit_names {names}; this script assumes the 7-param radius model")

    # theta[0] is the posterior weighted mean (the point every headline number
    # quotes). The draws are there so a single lucky/unlucky point cannot carry
    # the conclusion: if the two codes agree at the mean but not across the
    # posterior, that is a different and worse finding.
    wmean = np.average(s, axis=0, weights=w)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(s), size=ndraw, replace=True, p=w)
    thetas = np.vstack([wmean, s[idx]])

    obs = np.load(f"data/real_ess/{planet}_samples.npz", allow_pickle=True)
    cov = obs["covered"]
    wl, depth, err = obs["wlen"][cov], obs["x_obs"][cov], obs["sig_obs"][cov]
    o = np.argsort(wl)
    wl, depth, err = wl[o], depth[o], err[o]
    err = np.sqrt(err ** 2 + (floor * depth) ** 2)     # same floor as the NS run
    edges = np.empty(len(wl) + 1)
    edges[1:-1] = np.sqrt(wl[:-1] * wl[1:])
    edges[0] = wl[0] ** 2 / edges[1]
    edges[-1] = wl[-1] ** 2 / edges[-2]
    width = np.diff(edges)

    tm = tf.build_model(**geom, hifi=hifi)
    gases = list(tf.MOLECULES) + ["H2", "He"]

    rprs2, temps, vmrs, mus, gravs = [], [], [], [], []
    for k, th in enumerate(thetas):
        tm["planet_radius"] = float(th[0])
        tm["T"] = float(th[1])
        for mol, logx in zip(tf.MOLECULES, th[2:7]):
            tm[mol] = float(10.0 ** logx)
        wn, r2, _, _ = tm.model()
        act = np.asarray(tm.chemistry.activeGasMixProfile)      # (5, nlayers)
        ina = np.asarray(tm.chemistry.inactiveGasMixProfile)    # (2, nlayers)
        rprs2.append(r2)
        temps.append(np.array(tm.temperatureProfile))
        vmrs.append(np.vstack([act, ina]))
        mus.append(np.array(tm.chemistry.muProfile))            # kg
        gravs.append(np.array(tm.gravity_profile))              # m/s^2
        print(f"  [{k:2d}/{len(thetas)-1}] Rp={th[0]:.4f} T={th[1]:7.2f} done")

    p_pa = np.array(tm.pressureProfile)           # descending, layer 0 = bottom
    alt = np.array(tm.altitudeProfile)
    if alt[0] != 0.0:
        raise SystemExit(f"altitude[0]={alt[0]}, expected 0; the reference radius convention moved")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        planet=planet, thetas=thetas, fit_names=np.array(names, dtype=object),
        native_wn=np.asarray(wn), rprs2=np.asarray(rprs2),
        p_pa=p_pa, temperature=np.asarray(temps), vmr=np.asarray(vmrs),
        gas_names=np.array(gases, dtype=object),
        mu_kg=np.asarray(mus), gravity_ms2=np.asarray(gravs), altitude_m=alt,
        # reference triple: TauREx puts the planet radius at altitude 0, which is
        # the CENTRE of the bottom layer, not at atm_max_pressure. pRT is given
        # that same pressure as its reference_pressure, or the two codes would be
        # integrating from different radii and every ppm after that is meaningless.
        ref_pressure_pa=float(p_pa[0]), ref_gravity_ms2=float(gravs[0][0]),
        rp_rj=np.asarray([t[0] for t in thetas]),
        rs_rsun=float(geom["rs_rsun"]), mp_mj=float(geom["mp_mj"]),
        RJUP=float(RJUP), RSOL=float(RSOL), MJUP=float(MJUP),
        obs_wl=wl, obs_width=width, obs_depth=depth, obs_err=err,
    )
    print(f"\n[h8] dumped {len(thetas)} spectra + atmospheres -> {out}")
    print(f"  native grid {len(wn)} pts, {10000/wn.max():.3f}-{10000/wn.min():.3f} um")
    print(f"  ref P {p_pa[0]/1e5:.4f} bar, g {gravs[0][0]:.3f} m/s2, mu {mus[0][0]*6.02214076e26:.4f} g/mol")
    print(f"  obs {len(wl)} bins, {wl.min():.3f}-{wl.max():.3f} um, median err {np.median(err)*1e6:.1f} ppm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planet", default="wasp96")
    ap.add_argument("--post", default="data/real_ess/wasp39b_ns_posterior_wasp96_live1000.npz")
    ap.add_argument("--ndraw", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/h8/taurex_dump.npz")
    # --no-hifi reruns the SAME code on the SAME atmosphere with the published
    # ExoTransmit R~1000 lists instead. That is the control for H8: it shows the
    # comparison can see an opacity-data change at all, and it measures how much
    # the line lists move the spectrum relative to the code choice.
    ap.add_argument("--no-hifi", action="store_true")
    a = ap.parse_args()
    main(a.planet, a.post, a.ndraw, a.seed, a.out, hifi=not a.no_hifi)
