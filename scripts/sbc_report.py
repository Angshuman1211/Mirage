"""
A1 calibration report — run trained arms over a Tier-1 injection set and score
SBC ranks, TARP and HPD ranks against prior-draw labels (G3, G5).

The labels come from the prior, not from a reference posterior, so this is the
non-circular replacement for the Phase-3 OT check that two Sim2Science reviewers
independently found circular (P5-D5).

    python scripts/sbc_report.py --tier data/inject/wasp39_tier1.hdf \
        --arms h9_rad_nocond_s42 h9_rad_cov_s42 --n 400 --samples 199

READ TIER 1 AGAINST TIER 0 (P6-D7). Real published error bars sit 23-139x below
the sigma range the arms were trained on, so a published-amplitude set is out of
distribution for the sigma and covariance embeddings in exactly the way real data
is. An intrinsic flow defect and a pure scale mismatch produce the SAME SBC
failure. Run both:

    data/inject/wasp39_tier1.hdf          published amplitude, realistic, OOD
    data/inject/wasp39_tier0_indist.hdf   same injections scaled into training range

Fails on Tier 1 and holds on Tier 0 -> the defect is the scale mismatch, and the
fix is retraining over a sigma range that contains real data. Fails on both -> the
flow is miscalibrated. Neither conclusion is available from one set alone.

TARP's power floor at these N is roughly a half-sigma bias, where it is a coin
flip (P6-D5). Do not read one passing p-value as proof of calibration.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from fm4ar.datasets.theta_scalers import get_theta_scaler
from fm4ar.models.build_model import build_model
from mirage.eval.coverage import coverage
from mirage.eval.sbc import hpd_ranks, sbc, tarp

PARAM = ["rp_rj", "T", "logH2O", "logCO2", "logCH4", "logCO", "logNH3"]


def load_arm(ckpt_dir: Path):
    ckpt = ckpt_dir / "model__best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing — train this arm first")
    model = build_model(file_path=ckpt, experiment_dir=ckpt_dir, device="cpu")
    model.network.eval()
    with open(ckpt_dir / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    return model, get_theta_scaler(cfg.get("theta_scaler", {}))


def load_tier(path: Path, n: int):
    with h5py.File(path) as f:
        n = min(n, len(f["theta"]))
        d = dict(theta=f["theta"][:n].astype(np.float64),
                 flux=f["flux"][:n].astype(np.float32),
                 noise=f["noise"][:n].astype(np.float32),
                 wlen=f["wlen"][0].astype(np.float32),
                 planet=str(f.attrs["planet"]),
                 sigma_scale=float(f.attrs["sigma_scale"]),
                 scale_mode=str(f.attrs["sigma_scale_mode"]),
                 real_frac=float(f.attrs["real_noise_fraction"]))
    return d


def oot_frames_for(planet: str, wlen: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Held-out real OOT frames on the tier's grid, stored bin order.

    The covariance embedding whitens internally (P3-D1), so the raw frame scale is
    fine here; what it needs is the real structure. Frames come from the same
    held-out contiguous half that generated the injections, so nothing the arm
    conditions on overlaps the calibration half.
    """
    import sys
    sys.path.insert(0, "scripts")
    from inject_real_noise import PLANETS, model_grid, oot_instruments

    P = PLANETS[planet]
    wl_asc, edges = model_grid(P["stats"])
    insts = oot_instruments(list(P["oot"]), edges)
    name, rel, covered, nf = max(insts, key=lambda t: t[3])     # most frames
    pool = np.arange(nf // 2, nf)
    rng = np.random.default_rng(seed)
    sel = rng.choice(pool, size=min(k, len(pool)), replace=len(pool) < k)
    M = np.nan_to_num(rel[sel], nan=0.0)                        # (k, n_bins) ascending
    inv = np.argsort(np.argsort(wlen))                          # ascending -> stored
    order = np.argsort(wlen)
    out = np.zeros((M.shape[0], len(wlen)), np.float32)
    out[:, order] = M                                           # place ascending into stored
    return out


@torch.no_grad()
def run_arm(arm_dir: Path, tier: dict, n_samples: int, chunk: int,
            oot: np.ndarray | None, hpd: bool):
    model, scaler = load_arm(arm_dir)
    n = len(tier["theta"])
    D = tier["theta"].shape[1]
    samples = np.empty((n, n_samples, D))
    logq_s = np.empty((n, n_samples))
    logq_t = np.empty(n)

    wlen = torch.from_numpy(tier["wlen"])
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        S = b - a
        base = {
            "flux": torch.from_numpy(tier["flux"][a:b]),
            "wlen": wlen.unsqueeze(0).expand(S, -1),
            "error_bars": torch.from_numpy(tier["noise"][a:b]),
        }
        if oot is not None:
            base["oot_frames"] = torch.from_numpy(oot).unsqueeze(0).expand(S, *oot.shape)
        ctx = {k: v.repeat_interleave(n_samples, dim=0).float() for k, v in base.items()}
        th_s, lq = model.sample_and_log_prob_batch(context=ctx, tolerance=1e-3)
        samples[a:b] = scaler.inverse_array(
            th_s.cpu().numpy()).reshape(S, n_samples, D)
        logq_s[a:b] = lq.cpu().numpy().reshape(S, n_samples)

        if hpd:
            tt = scaler.forward_tensor(torch.from_numpy(tier["theta"][a:b]).float())
            ctx1 = {k: v.float() for k, v in base.items()}
            logq_t[a:b] = model.log_prob_batch(
                tt, context=ctx1, tolerance=1e-3).cpu().numpy().reshape(-1)
        print(f"    {b}/{n}", end="\r", flush=True)
    print(f"    {n}/{n} done")
    return samples, logq_s, (logq_t if hpd else None)


def main(tier_path: str, arms: list[str], n: int, n_samples: int, n_bins: int,
         chunk: int, seed: int, no_hpd: bool, out: str | None):
    tier = load_tier(Path(tier_path), n)
    n = len(tier["theta"])
    if (n_samples + 1) % n_bins:
        raise SystemExit(f"--samples {n_samples} + 1 must be divisible by "
                         f"--bins {n_bins} (Talts et al. sec. 4)")
    if n / n_bins < 5:
        raise SystemExit(f"{n} sims over {n_bins} bins is {n / n_bins:.1f} per bin; "
                         f"chi-square needs >= 5")

    print(f"Tier set : {tier_path}")
    print(f"  planet {tier['planet']}  sims {n}  samples {n_samples}  "
          f"bins {tier['flux'].shape[1]}")
    print(f"  sigma scale: {tier['scale_mode']}   real-noise fraction "
          f"{tier['real_frac']:.2f}")
    sig_med = float(np.median(tier["noise"]))
    inside = 0.05 <= sig_med <= 0.30
    print(f"  normalised sigma median {sig_med:.4f} -> "
          f"{'INSIDE' if inside else f'{0.05 / sig_med:.0f}x BELOW'} the training "
          f"perturbation range 0.05-0.30")
    print()

    results = {}
    for a in arms:
        d = Path(a) if Path(a).exists() else Path("configs") / a
        needs_oot = "cov" in d.name
        print(f"[{d.name}]  oot_frames: {'yes' if needs_oot else 'no'}")
        oot = (oot_frames_for(tier["planet"], tier["wlen"], 100, seed)
               if needs_oot else None)
        s, lqs, lqt = run_arm(d, tier, n_samples, chunk, oot, not no_hpd)
        results[d.name] = (s, lqs, lqt)

    print(f"\n{'=' * 78}")
    print(f"  A1 calibration, {n} prior draws injected into real noise")
    print(f"  {'arm':<22}{'SBC min p':>11}{'shape':>10}{'TARP p':>10}"
          f"{'TARP dev':>10}{'HPD p':>9}{'cov@68':>8}")
    print(f"  {'-' * 74}")
    saved, stats = {}, {}
    for name, (s, lqs, lqt) in results.items():
        slist = [s[i] for i in range(n)]
        r = sbc(slist, tier["theta"], n_bins=n_bins)
        stats[name] = r
        t = tarp(slist, tier["theta"], random_seed=seed)
        h = hpd_ranks([lqs[i] for i in range(n)], lqt) if lqt is not None else None
        cv = coverage(slist, tier["theta"], levels=(0.68,))

        hist = r.counts.sum(axis=0)
        edge = hist[[0, -1]].mean()
        mid = hist[len(hist) // 2 - 1:len(hist) // 2 + 1].mean()
        shape = ("U-shaped" if edge > 1.3 * mid else
                 "bell" if mid > 1.3 * edge else "flat")
        if abs(r.skew()).max() > 0.08:
            shape = "sloped"

        print(f"  {name:<22}{r.pvalue.min():>11.2e}{shape:>10}{t.pvalue:>10.2e}"
              f"{t.max_dev:>10.3f}{(h.pvalue if h else float('nan')):>9.2e}"
              f"{cv.overall[0]:>8.3f}")
        saved[name] = dict(ranks=r.ranks, sbc_p=r.pvalue, sbc_chi2=r.chi2,
                           counts=r.counts, skew=r.skew(), tarp_stat=t.stat,
                           tarp_ecp=t.ecp, tarp_alpha=t.alpha, tarp_p=t.pvalue,
                           tarp_dev=t.max_dev, cov68=cv.per_dim[0],
                           hpd_stat=(h.stat if h else np.array([])),
                           hpd_p=(h.pvalue if h else np.nan))

    D = tier["theta"].shape[1]
    print(f"\n  per-parameter SBC p-values (bold = below 0.01)")
    print(f"  {'arm':<22}" + "".join(f"{q:>9}" for q in PARAM[:D]))
    for name, r in stats.items():
        print(f"  {name:<22}" + "".join(f"{q:>9.1e}" for q in r.pvalue))
    print(f"\n  SBC rank skew (mean normalised rank - 0.5; negative = posterior high)")
    for name, r in stats.items():
        print(f"  {name:<22}" + "".join(f"{q:>+9.3f}" for q in r.skew()))

    if out:
        np.savez(out, theta=tier["theta"], planet=tier["planet"],
                 sigma_scale=tier["sigma_scale"], tier=str(tier_path),
                 **{f"{k}__{kk}": vv for k, d in saved.items()
                    for kk, vv in d.items()})
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--samples", type=int, default=199)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-hpd", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.tier, a.arms, a.n, a.samples, a.bins, a.chunk, a.seed, a.no_hpd, a.out)
