"""
A2 — rebuild the OT calibration on INJECTION labels, with no nested sampling anywhere.

Why this is the right move now. P5-D5 killed the published calibration for circularity: it
was fitted to the NS anchor and then scored against that same anchor. P6-D21 (H11) then
killed the obvious repair, transferring one target's calibration to another, and in doing so
showed the calibration is specific to BOTH the target and the individual training run. So a
usable calibration has to be derived per target and per trained model, without an anchor.
Injections are the only thing that can do that.

The estimator, and why it lands in exactly H11's units so the two are comparable.

Under correct Bayesian calibration the true label is a draw from the posterior given its own
data, marginally over the prior. That is the SBC argument. So with the flow posterior for
injection i summarised as (mu_i, sd_i), the pulls

    z_i = (theta_i - mu_i) / sd_i

have mean 0 and variance 1 if the flow is right. If the flow is overdispersed by a factor c,
meaning sd_flow = c * sd_true, then std(z) = 1/c, and the deflation that fixes it is
k = sd_true/sd_flow = 1/c = std(z). If the flow is also biased by b, then mean(z) = -b/sd_flow
and the corrected mean is mu + mean(z)*sd.

    s_inj = mean(z_i)     bias, in flow-sd units
    k_inj = std(z_i)      width deflation

Those are the SAME two dimensionless quantities H11 defined as s = (mu_n - mu_f)/sd_f and
k = sd_n/sd_f against the NS anchor. So "does the injection-fitted calibration agree with the
anchor-fitted one" is a direct per-parameter comparison, and it is A2's first measurement
rather than its last (P6-D21 note in 08_work_split_v2.md).

Robust duplicates are reported beside the moment versions because z is not guaranteed
light-tailed: median for the bias and IQR/1.349 for the scale.

HOMOGENEITY IS CHECKED, NOT ASSUMED. A pooled (s, k) applied to one real spectrum assumes the
miscalibration does not depend on where theta sits. The injections span the whole prior while
the real target is one point, so the fit is repeated on the subset of injections nearest the
real posterior and the two are compared.

    conda run -n mirage python scripts/a2_inject_calibrate.py --moments --planet wasp39
    conda run -n mirage python scripts/a2_inject_calibrate.py --fit
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

A2 = Path("data/a2")
NAMES = ["radius", "T", "logH2O", "logCO2", "logCH4", "logCO", "logNH3"]

# arm matched to H11's matched-architecture basis: rad_nocond on all three targets
ARMS = {"wasp39": "configs/h9_rad_nocond_s42",
        "wasp96": "configs/noisecond_rad_nocond_wasp96",
        "k218": "configs/noisecond_rad_nocond_k218"}
TIERS = {"wasp39": "data/inject/wasp39_tier1.hdf",
         "wasp96": "data/inject/wasp96_tier1.hdf",
         "k218": "data/inject/k218_tier1.hdf"}


@torch.no_grad()
def moments(planet, n, n_samples, chunk):
    """Run the arm over the injection set and keep the PER-INJECTION posterior moments.
    sbc_report.run_arm already computes these and throws them away; A2 needs them."""
    from sbc_report import load_arm                # reuse A1's loader, do not re-implement
    model, scaler = load_arm(Path(ARMS[planet]))
    with h5py.File(TIERS[planet]) as f:
        n = min(n, len(f["theta"]))
        theta = f["theta"][:n].astype(np.float64)
        flux = f["flux"][:n].astype(np.float32)
        noise = f["noise"][:n].astype(np.float32)
        wlen = f["wlen"][0].astype(np.float32)
    D = theta.shape[1]
    mu = np.empty((n, D))
    Sig = np.empty((n, D, D))
    wl = torch.from_numpy(wlen)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        S = b - a
        base = {"flux": torch.from_numpy(flux[a:b]),
                "wlen": wl.unsqueeze(0).expand(S, -1),
                "error_bars": torch.from_numpy(noise[a:b])}
        ctx = {k: v.repeat_interleave(n_samples, dim=0).float() for k, v in base.items()}
        th_s, _ = model.sample_and_log_prob_batch(context=ctx, tolerance=1e-3)
        ths = scaler.inverse_array(th_s.cpu().numpy()).reshape(S, n_samples, D)
        mu[a:b] = ths.mean(1)
        for j in range(S):
            Sig[a + j] = np.cov(ths[j].T)
        print(f"    {b}/{n}", end="\r", flush=True)
    print(f"    {n}/{n} done")
    A2.mkdir(parents=True, exist_ok=True)
    out = A2 / f"{planet}_injmoments.npz"
    np.savez(out, mu=mu, Sigma=Sig, theta=theta, planet=planet, n_samples=n_samples)
    print(f"[a2] {out}  n={n} injections, {n_samples} samples each")


def _pulls(d):
    sd = np.sqrt(np.einsum("nii->ni", d["Sigma"]))
    return (d["theta"] - d["mu"]) / sd, sd


def _sk(z, m=None):
    """(s, k) from the pulls, with the finite-sample bias on k removed.

    THE BIAS IS NOT SMALL AND IGNORING IT WOULD HAVE INVALIDATED THE COMPARISON.
    Two finite-sample terms enter, and a2_validate_estimator.py caught the second one by
    planting known values and failing to recover them.

    1. The posterior mean is estimated from m draws, mu_hat = mu + e with var(e) = sd^2/m,
       so z carries an extra 1/m of variance. H11's anchor-fitted k for radius is 0.075,
       giving k^2 = 0.0056 against 1/m = 0.0039 at m=256: the noise term is comparable to
       the signal, and the uncorrected k reads about 1.37x too high (measured, not argued)
       on exactly the parameter the radius story depends on.

    2. z also divides by the NOISY sd_hat. When the bias s is non-zero that scatter acts on
       the offset itself and contributes a further s^2/(2m). Omitting it inflated a planted
       k of 0.300 to 0.352, a 17.5 percent error, and put four of seven planted values
       outside their intervals.

        var(z_observed) = k^2 + (1 + s^2 / 2) / m

    so k^2 = var(z) - (1 + s^2/2)/m. Validated by recovering planted values.
    """
    s = z.mean(0)
    v = z.var(0, ddof=1)
    k_raw = np.sqrt(v)
    k = np.sqrt(np.clip(v - _corr(s, m), 1e-12, None))
    s_rob = np.median(z, 0)
    q1, q3 = np.percentile(z, [25, 75], axis=0)
    k_rob = np.sqrt(np.clip(((q3 - q1) / 1.349) ** 2 - _corr(s_rob, m), 1e-12, None))
    return s, k, s_rob, k_rob, k_raw


def _corr(s, m):
    """The finite-sample variance added to z by estimating the mean from m draws and
    dividing by a noisy sd. See _sk. Zero when m is unknown."""
    return 0.0 if not m else (1.0 + s ** 2 / 2.0) / m


def _boot_k(z, m, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(z), len(z))
        zb = z[idx]
        v = zb.var(0, ddof=1)
        out.append(np.sqrt(np.clip(v - _corr(zb.mean(0), m), 1e-12, None)))
    return np.percentile(np.array(out), [16, 84], axis=0)


def fit():
    from h11_ot_transfer import TARGETS, load, calibration
    print("=" * 104)
    print("A2  injection-fitted calibration against the H11 anchor-fitted one.")
    print("    s = bias in flow-sd units, k = width deflation. Same definition both sides,")
    print("    so these columns are directly comparable. INJ uses no nested sampling.")
    print("=" * 104)
    agree = {}
    for planet in ARMS:
        p = A2 / f"{planet}_injmoments.npz"
        if not p.exists():
            print(f"\n--- {planet}: no moments yet")
            continue
        d = np.load(p, allow_pickle=True)
        z, sd = _pulls(d)
        m = int(d["n_samples"])
        s_i, k_i, s_r, k_r, k_raw = _sk(z, m)
        klo, khi = _boot_k(z, m)
        cal = calibration(load(planet))              # the H11 anchor-fitted calibration
        s_a, k_a = cal["s"], cal["k"]
        print(f"\n--- {planet}   n={len(z)} injections "
              f"{'-' * 55}")
        print(f"  {'param':8s} {'s_INJ':>8s} {'s_anch':>8s}   {'k_INJ':>8s} "
              f"{'68% CI':>14s} {'k_anch':>8s} {'k ratio':>8s} {'k_uncorr':>9s}")
        for i, nm in enumerate(NAMES):
            print(f"  {nm:8s} {s_i[i]:8.2f} {s_a[i]:8.2f}   {k_i[i]:8.3f} "
                  f"{f'[{klo[i]:.3f}, {khi[i]:.3f}]':>14s} {k_a[i]:8.3f} "
                  f"{k_i[i] / k_a[i]:8.2f} {k_raw[i]:9.3f}")
        print(f"  {'-' * 86}")
        # Non-Gaussian pulls make the moment k and the robust k disagree, and the radius
        # pull IS leptokurtic. Report both rather than pick one silently.
        from scipy.stats import kurtosis as _kurt
        ek = _kurt(z, axis=0)
        flag = [NAMES[i] for i in range(7) if abs(ek[i]) > 1.0]
        print(f"  robust k (IQR/1.349):  "
              + "  ".join(f"{NAMES[i][:6]} {k_r[i]:.3f}" for i in range(7)))
        print(f"  excess kurtosis > 1 on: {', '.join(flag) if flag else 'none'}"
              f"   (heavy tails make the moment k and the robust k diverge)")
        print(f"  k: injection-fitted / anchor-fitted, median {np.median(k_i / k_a):.2f}x,"
              f"  range {np.min(k_i / k_a):.2f}-{np.max(k_i / k_a):.2f}x")
        print(f"  s: mean |difference| = {np.mean(np.abs(s_i - s_a)):.2f} flow sd")
        # s is in FLOW-sd units, which flatters whichever side has the wider flow. Divide
        # by k to express the same bias in units of the TRUE posterior width, which is the
        # only scale on which the injection side and the real side are comparable. On the
        # real side s/k is exactly |mu_n - mu_f| / sd_n, H11's |dmu|/sigma_NS.
        bi, ba = np.abs(s_i) / k_i, np.abs(s_a) / k_a
        print(f"  bias in TRUE-posterior sigmas (|s|/k), the comparable scale:")
        print(f"    injections  median {np.median(bi):5.2f}  max {bi.max():5.2f}"
              f"      real data  median {np.median(ba):5.2f}  max {ba.max():5.2f}")
        print(f"    -> the flow is centred on simulated data and off-centre on real data"
              f" by {np.median(ba) / max(np.median(bi), 1e-9):.0f}x")
        agree[planet] = (s_i, k_i, s_a, k_a)

        # homogeneity: refit on the injections nearest the real target's anchor
        B = load(planet)
        dist = np.sqrt(np.sum(((d["theta"] - B["mun"]) / B["sdn"]) ** 2, axis=1))
        near = dist <= np.percentile(dist, 20)
        s_n, k_n, _, _, _ = _sk(z[near], m)
        print(f"  homogeneity, nearest 20% of injections to the anchor (n={near.sum()}):")
        print(f"    k shifts by median {np.median(np.abs(k_n - k_i) / k_i) * 100:.0f}%,"
              f"  s by mean {np.mean(np.abs(s_n - s_i)):.2f} flow sd")
    return agree



def apply_real(inflate=1.5, M=2000, seed=0):
    """Stage 3, the end-to-end payoff: apply the INJECTION-fitted calibration to the real
    spectrum and write an IS proposal. No nested sampling touches the calibration at any
    point, so scoring the result against the NS anchor afterwards is legitimate. That is
    precisely what P5-D5 said the published version could not claim.
    """
    from h11_ot_transfer import TARGETS, load, corr, gauss_draws
    A2.mkdir(parents=True, exist_ok=True)
    cmds = []
    for planet in ARMS:
        pm = A2 / f"{planet}_injmoments.npz"
        if not pm.exists():
            print(f"[apply] {planet}: no moments yet, skipping")
            continue
        d = np.load(pm, allow_pickle=True)
        z, _ = _pulls(d)
        m = int(d["n_samples"])
        s_i, k_i, _, _, _ = _sk(z, m)

        B = load(planet)
        f = np.load(TARGETS[planet]["fmpe"], allow_pickle=True)
        mu = B["muf"] + s_i * B["sdf"]
        sd = k_i * B["sdf"]
        S = (np.outer(sd, sd) * corr(B["Sf"])) * inflate
        th, log_g, n_in = _draw(mu, S, B["lo"], B["hi"], M, seed)
        out = A2 / f"a2_{planet}_injcal_samples.npz"
        np.savez(out, theta=th, log_g=log_g, planet=planet, wlen=f["wlen"],
                 x_obs=f["x_obs"], sig_obs=f["sig_obs"], covered=f["covered"])
        print(f"[apply] {out}  N={n_in}")
        cmds.append(f"python scripts/taurex_forward.py --forward-npz {out} "
                    f"--forward-mode radius --forward-planet {planet}")
    (A2 / "run_forwards.sh").write_text("set -e\n" + "\n".join(cmds) + "\n")
    print(f"[apply] {len(cmds)} forward runs -> {A2 / 'run_forwards.sh'}")


def _draw(mu, S, lo, hi, M, seed):
    w, V = np.linalg.eigh(S)
    L = V * np.sqrt(np.clip(w, 1e-18, None))
    rng = np.random.default_rng(seed)
    Sinv = np.linalg.pinv(S)
    logdet = float(np.sum(np.log(np.clip(w, 1e-18, None))))
    keep_t, keep_g, tries = [], [], 0
    while sum(len(t) for t in keep_t) < M and tries < 200:
        x = mu + rng.standard_normal((4 * M, 7)) @ L.T
        x = x[np.all((x >= lo) & (x <= hi), axis=1)]
        dd = x - mu
        keep_t.append(x)
        keep_g.append(-0.5 * (np.einsum("ij,jk,ik->i", dd, Sinv, dd)
                              + 7 * np.log(2 * np.pi) + logdet))
        tries += 1
    th = np.concatenate(keep_t)[:M] if keep_t else np.zeros((0, 7))
    lg = np.concatenate(keep_g)[:M] if keep_g else np.zeros(0)
    return th, lg, len(th)


def score_real(floor=0.05):
    """Stage 4: how good is the injection-calibrated real posterior, against the anchor it
    never saw. Reported beside H11's numbers so the three routes are side by side:
    no calibration, circular anchor-fitted, and this one."""
    from h11_ot_transfer import (TARGETS, load, score, gauss_draws, calibration,
                                 transfer, corr)
    print("=" * 100)
    print("A2 end to end. JS and ESS against the NS anchor, which the injection-fitted")
    print("calibration never saw. 'circular' is H11's SELF arm, fitted ON that anchor.")
    print("=" * 100)
    print(f"  {'target':9s} {'route':24s} {'meanJS':>8s} {'ESS':>8s} {'eps %':>7s} "
          f"{'68% CI':>12s}")
    dump = []
    for planet in ARMS:
        sp = A2 / f"a2_{planet}_injcal_samples.npz"
        B = load(planet)
        raw = score(B["thf"], np.ones(len(B["thf"])), B)
        mu_c, S_c = transfer(calibration(B), B, "M2")
        xc, _ = gauss_draws(mu_c, S_c, B["lo"], B["hi"])
        circ = score(xc, np.ones(len(xc)), B)
        # H11 measured ESS for the same target's raw and circular arms with the same
        # likelihood, floor and inflate, so quote those rather than leave the column
        # blank and invite a JS-only reading.
        h11 = {}
        try:
            import json
            for r in json.load(open("data/h11/ess_matrix.json")):
                if r["recipient"] == planet and r["arm"] in ("raw", "self_M2"):
                    h11[r["arm"]] = (r["ess"], r["eps"] * 100,
                                     (r.get("ess_lo"), r.get("ess_hi")))
        except (OSError, ValueError):
            pass
        # H11's pooled leave-one-out transfer is the OTHER anchor-free route, so it is
        # the comparison that decides whether A2 was worth building. Same B, same scoring.
        others = [calibration(load(a))["k"] for a in ARMS if a != planet]
        kp = np.exp(np.mean(np.log(np.array(others)), axis=0))
        sdp = kp * B["sdf"]
        xp, _ = gauss_draws(B["muf"], np.outer(sdp, sdp) * corr(B["Sf"]),
                            B["lo"], B["hi"])
        pooled = score(xp, np.ones(len(xp)), B)
        rows = [("no calibration", raw["mean_js"], h11.get("raw")),
                ("H11 pooled transfer", pooled["mean_js"], None),
                ("circular (anchor-fitted)", circ["mean_js"], h11.get("self_M2"))]
        if sp.exists():
            d = np.load(sp, allow_pickle=True)
            sc = score(d["theta"], np.ones(len(d["theta"])), B)
            fp = Path(str(sp).rsplit("samples", 1)[0] + "forward.npz")
            ess = eps = None
            if fp.exists():
                ms = np.load(fp)["model_spec"]
                x, sig, cov = d["x_obs"], d["sig_obs"], d["covered"]
                msk = cov & np.isfinite(sig) & (sig > 0)
                se = np.sqrt(sig ** 2 + (floor * x) ** 2)
                valid = np.isfinite(ms[:, msk]).all(axis=1)
                r = (x[None, msk] - ms[:, msk]) / se[None, msk]
                ll = np.where(valid, -0.5 * np.nansum(r ** 2, axis=1), -np.inf)
                lw = ll - d["log_g"]
                lw -= np.nanmax(lw)
                w = np.exp(lw)
                w /= w.sum()
                ess = float(1.0 / np.sum(w ** 2))
                eps = ess / len(w) * 100
                rg = np.random.default_rng(0)
                bs = []
                for _ in range(400):
                    wb = w[rg.integers(0, len(w), len(w))]
                    if wb.sum() > 0:
                        bs.append(1.0 / np.sum((wb / wb.sum()) ** 2))
                ess_ci = tuple(np.percentile(bs, [16, 84]))
            rows.append(("INJECTION-fitted (A2)", sc["mean_js"],
                         (ess, eps, ess_ci) if ess is not None else None))
        for nm, js, e in rows:
            if e and e[0] is not None:
                ci = e[2] if len(e) > 2 and e[2] and e[2][0] is not None else None
                et = (f"{e[0]:8.1f} {e[1]:7.2f} "
                      f"{f'[{ci[0]:.0f}, {ci[1]:.0f}]':>12s}" if ci
                      else f"{e[0]:8.1f} {e[1]:7.2f} {'-':>12s}")
            else:
                et = f"{'-':>8s} {'-':>7s} {'-':>12s}"
            print(f"  {planet:9s} {nm:24s} {js:8.4f} {et}")
            dump.append(dict(target=planet, route=nm, mean_js=float(js),
                             ess=(float(e[0]) if e and e[0] is not None else None),
                             eps_pct=(float(e[1]) if e and e[0] is not None else None),
                             ess_lo=(float(e[2][0]) if e and e[0] is not None
                                     and len(e) > 2 and e[2] and e[2][0] is not None
                                     else None),
                             ess_hi=(float(e[2][1]) if e and e[0] is not None
                                     and len(e) > 2 and e[2] and e[2][1] is not None
                                     else None)))
        print()
    # Persist so the figure and the write up read one artifact instead of the console.
    import json as _json
    (A2 / "a2_scores.json").write_text(_json.dumps(dump, indent=1))
    print(f"[score] {len(dump)} rows -> {A2 / 'a2_scores.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--moments", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--planet", default="wasp39", choices=list(ARMS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    if a.moments:
        moments(a.planet, a.n, a.samples, a.chunk)
    elif a.fit:
        fit()
    elif a.apply:
        apply_real()
    elif a.score:
        score_real()
