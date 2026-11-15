"""
H8 proper - an independent nested-sampling retrieval on real WASP-96b, with
petitRADTRANS as the forward model instead of TauREx.

h8_compare.py already showed the two FORWARD MODELS agree to ~13 ppm against a
193 ppm error bar. This is the stronger version the gap list actually asks for:
G6/H8 is phrased as "if TauREx-NS and pRT-NS disagree by more than the
flow-to-NS gap, the reference is not a reference". So run the second retrieval.

Everything that is not the radiative transfer is held identical to
scripts/taurex_retrieve.py:
  - same observed spectrum and `covered` mask, same 1 percent error floor
  - same priors: radius U(0.8, 1.6) Rjup, T U(110, 5500) K, log VMR U(-9, -3)
  - same sampler: nestle, method "multi", dlogz 0.5, 150 live points (150 is
    what the PUBLISHED anchor used, found during A5 part 1)
  - same binning, via the weight matrix in h8_binmatrix.py, which is checked
    against TauREx's own FluxBinner to 7e-12 ppm
  - same pressure grid and fill-gas algebra (He/H2 = 0.17 by volume)
  - line lists matched to data/opacity_hifi by the DOI inside each file

    conda run -n mirage-prt python scripts/h8_prt_retrieve.py --selftest
    conda run -n mirage-prt python scripts/h8_prt_retrieve.py --live 150
"""

import argparse
import time
from pathlib import Path

import numpy as np
import nestle
from petitRADTRANS.radtrans import Radtrans

from h8_binmatrix import bin_matrix

SPECIES = {"H2O": "1H2-16O__POKAZATEL", "CO2": "12C-16O2__UCL-4000",
           "CH4": "12C-1H4__YT34to10", "CO": "C-O-NatAbund__Chubb",
           "NH3": "14N-1H3__CoYuTe"}
MOLAR_MASS = {"H2O": 18.01528, "CO2": 44.0095, "CH4": 16.04246, "CO": 28.0101,
              "NH3": 17.03052, "H2": 2.01588, "He": 4.002602}
MOLECULES = ["H2O", "CO2", "CH4", "CO", "NH3"]
HE_H2 = 0.17            # volume ratio, == TauREx TaurexChemistry(ratio=[0.17])
GNEWT = 6.67430e-11
RB = (0.8, 1.6)         # wasp96 radius bounds, == _PLANET["wasp96"]["rb"]
T_BOUNDS = (110.0, 5500.0)
LOGX_BOUNDS = (-9.0, -3.0)


class Model:
    def __init__(self, dump, wl_lo=0.80, wl_hi=2.90):
        d = np.load(dump, allow_pickle=True)
        self.d = d
        self.order = np.argsort(d["p_pa"])
        self.p_bar = d["p_pa"][self.order] / 1e5
        self.nlay = len(self.p_bar)
        self.rs_cm = float(d["rs_rsun"]) * float(d["RSOL"]) * 100.0
        self.rjup_m = float(d["RJUP"])
        self.mass_kg = float(d["mp_mj"]) * float(d["MJUP"])
        self.ref_p_bar = float(d["ref_pressure_pa"]) / 1e5
        self.rt = Radtrans(
            pressures=self.p_bar,
            line_species=[SPECIES[m] for m in MOLECULES],
            gas_continuum_contributors=["H2--H2", "H2--He"],
            rayleigh_species=["H2", "He"],
            wavelength_boundaries=[wl_lo, wl_hi],
            line_opacity_mode="c-k",
        )
        self.wl, self.width = d["obs_wl"], d["obs_width"]
        self.depth, self.err = d["obs_depth"], d["obs_err"]
        # one 287 ms forward model to learn the native grid, then freeze W
        wlc, _, _ = self._raw(1.2, 1000.0, [-4.0] * 5)
        self.W = bin_matrix(wlc * 1e4, self.wl, self.width)

    def _raw(self, rp_rj, temp, logx):
        vmr = {m: 10.0 ** lx for m, lx in zip(MOLECULES, logx)}
        rest = 1.0 - sum(vmr.values())
        vmr["H2"] = rest / (1.0 + HE_H2)
        vmr["He"] = HE_H2 * rest / (1.0 + HE_H2)
        mmw = sum(vmr[g] * MOLAR_MASS[g] for g in vmr)
        mf = {SPECIES.get(g, g): np.full(self.nlay, vmr[g] * MOLAR_MASS[g] / mmw)
              for g in vmr}
        r_m = rp_rj * self.rjup_m
        wl_cm, radii_cm, _ = self.rt.calculate_transit_radii(
            temperatures=np.full(self.nlay, temp),
            mass_fractions=mf,
            mean_molar_masses=np.full(self.nlay, mmw),
            # TauREx recomputes g from mass and the CURRENT radius, so this has
            # to follow the fitted radius rather than stay at the dump's value.
            reference_gravity=GNEWT * self.mass_kg / r_m ** 2 * 100.0,
            reference_pressure=self.ref_p_bar,
            planet_radius=r_m * 100.0,
            variable_gravity=True,
        )
        return wl_cm, radii_cm, mmw

    def spectrum(self, theta):
        _, radii_cm, _ = self._raw(theta[0], theta[1], theta[2:7])
        return self.W @ ((radii_cm / self.rs_cm) ** 2)

    def loglike(self, theta):
        m = self.spectrum(theta)
        return -0.5 * float(np.sum(((m - self.depth) / self.err) ** 2))


def prior_transform(u):
    t = np.empty(7)
    t[0] = RB[0] + u[0] * (RB[1] - RB[0])
    t[1] = T_BOUNDS[0] + u[1] * (T_BOUNDS[1] - T_BOUNDS[0])
    t[2:7] = LOGX_BOUNDS[0] + u[2:7] * (LOGX_BOUNDS[1] - LOGX_BOUNDS[0])
    return t


def selftest(M):
    """The model here must reproduce h8_prt_forward.py's spectra, and TauREx's
    own gravity. If either drifts, the retrieval is not the same experiment."""
    d, p = M.d, np.load("data/h8/prt_spectra.npz", allow_pickle=True)
    Wp = bin_matrix(p["wl_um"], M.wl, M.width)
    worst_s, worst_g = 0.0, 0.0
    for k in range(len(d["thetas"])):
        th = d["thetas"][k]
        worst_s = max(worst_s, float(np.max(np.abs(M.spectrum(th) - Wp @ p["rprs2"][k]))) * 1e6)
        g_mine = GNEWT * M.mass_kg / (th[0] * M.rjup_m) ** 2
        worst_g = max(worst_g, abs(g_mine - d["gravity_ms2"][k][0]) / d["gravity_ms2"][k][0])
    print(f"  spectrum vs h8_prt_forward : worst {worst_s:.3e} ppm")
    print(f"  gravity  vs TauREx         : worst {worst_g:.3e} relative")
    ok = worst_s < 1e-6 and worst_g < 1e-6
    print("  PASS" if ok else "  FAIL")
    return ok


def main(live, dump, out, do_selftest):
    M = Model(dump)
    print(f"[h8-ns] {len(M.wl)} bins, {M.nlay} layers, native {M.W.shape[1]} pts")
    print("[h8-ns] self-test")
    if not selftest(M):
        raise SystemExit("[h8-ns] self-test FAILED, refusing to sample")
    if do_selftest:
        return
    t0 = time.time()
    print(f"[h8-ns] nestle: {live} live points, method multi, dlogz 0.5")
    res = nestle.sample(M.loglike, prior_transform, 7, method="multi",
                        npoints=live, dlogz=0.5, callback=None)
    dt = time.time() - t0
    w = res.weights / res.weights.sum()
    mu = np.average(res.samples, axis=0, weights=w)
    chi2 = float(np.mean(((M.spectrum(mu) - M.depth) / M.err) ** 2))
    names = ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, samples=res.samples, weights=res.weights,
             fit_names=np.array(names, dtype=object), chi2=chi2,
             logz=res.logz, logzerr=res.logzerr, niter=res.niter, seconds=dt)
    print(f"[h8-ns] done in {dt/60:.1f} min, niter {res.niter}, logZ {res.logz:.3f}")
    print(f"  weighted means = {np.round(mu, 4)}")
    print(f"  chi2/N = {chi2:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=150)
    ap.add_argument("--dump", default="data/h8/taurex_dump.npz")
    ap.add_argument("--out", default="data/h8/prt_ns_posterior_wasp96.npz")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    main(a.live, a.dump, a.out, a.selftest)
