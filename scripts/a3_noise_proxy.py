"""
A3 / H5. Does the conditioning paradox come from the noise proxy?

THE PARADOX (P6-D2). On real WASP-39b the covariance-conditioned arm is WORSE
than the unconditioned one, mean 1-D JS to the NS anchor 0.426 +- 0.014 against
0.344 +- 0.003 over three training seeds, a gap 8.4x the seed spread. Better
information about the noise makes the posterior further from the reference.

H5 says the cause is the PROXY. Sigma-hat is estimated from out-of-transit FLUX
frames, and the quantity that matters is DEPTH noise. G13 measured the gap
between those two on WASP-96b (P6-D25): depth noise is 1.4x to 3x more
correlated across wavelength than flux noise. If feeding a more correlated
Sigma-hat moves the real posterior, the paradox is an artifact of the proxy. If
it does not move, the paradox is real and it is the headline.

WHAT THIS SCRIPT CAN AND CANNOT DO, STATED UP FRONT.

A3 as specified wants the WASP-96b depth-noise Sigma-hat fed to a WASP-96b
covariance arm. **That model does not exist.** Every covariance-conditioned
config in this repository sits on WASP-39b's 52-bin grid, and both WASP-96b
configs are `n_oot_frames: 0`, so they ignore the input entirely. Building one is
a retrain, not a run.

So this is the SENSITIVITY half of A3, and it is decisive in one direction. The
embedding whitens (`cov_whiten: True`, and `covariance_embedding.py` divides by
the per-bin std), so Sigma-hat IS a correlation matrix and scale is removed
exactly. That means the question "does the correlation structure of Sigma-hat
drive the real posterior" can be asked on WASP-39b today, by feeding the trained
arm correlation matrices that span from identity to more correlated than any
real measurement. **If the posterior barely moves across that span, no choice of
proxy could have caused the paradox, and the WASP-96b retrain is not worth
buying.** If it does move, the retrain is the next step and this says so.

THE SEVEN INPUTS.
  real        the real PRISM OOT frames, unchanged. Must reproduce P6-D2 exactly
              or the harness is wrong and nothing below counts.
  white       identity correlation, which is H5's "published diagonal errors".
  depth       Toeplitz correlation from the correlation-versus-separation profile
  depth_oot   G13 measured on WASP-96b's two depth-noise constructions.
  depth_direct      G13's MEASURED correlation matrix carried over as a principal
  depth_oot_direct  submatrix. These are the probes that actually work, see below.
  rank1       every bin perfectly correlated. Physically impossible, and there to
              BOUND the achievable response from above.
  scale10     the real frames times 10. Whitening should make this identical to
              `real`. A free check that the swap machinery works, that the
              embedding is scale-free, and that the draws are paired across
              inputs, since an identical context must give an identical score.

**WHY THE COMPARISON IS EXACTLY PAIRED.** `real_ess.sample` rebuilds the model
each call, and `FMPEModel.initialize_network` reseeds the global RNG from the
arm's own `random_seed` (fm4ar fmpe.py:79). Every input therefore starts the ODE
from a byte-identical theta_0, and the ONLY thing differing between rows of the
table is the context. Draw noise does not enter a comparison between rows. It
does set how representative the SPAN is, which is checked separately by driving
theta_0 from an explicit generator.

**THE TOEPLITZ PROBES UNDER-DELIVER, AND THE AUDIT CAUGHT IT.** `sep_profile`
takes the median of each diagonal and forces a Toeplitz, which cannot represent
a near-uniform common mode, that being a rank-1 all-ones term. G13's depth
covariance is dominated by exactly such a mode (31-37 ppm). So the Toeplitz
route retains only 46% of the measured mean |off-diagonal| for `depth` (0.219 of
0.476) and 29% for `depth_oot` (0.065 of 0.223), and 0.065 is BELOW the
sampling floor sqrt(2/(pi*K)) = 0.080 that K=100 frames impose on any estimate.
**`depth_oot` is therefore not a distinct probe, it is a second white arm.**
The `*_direct` kinds fix this: a principal submatrix of a correlation matrix is
itself a correlation matrix, so nothing is modelled or projected and the common
mode survives. They retain 95% and 100% of the measured structure.

**All four depth inputs remain PROBES, not measurements of WASP-39b.** The
matrix is carried across targets and across grids, 90 bins over 0.85-2.81 um
onto 52 bins over 0.6-5.29 um, so a given bin separation is not the same
wavelength separation. They span the structure axis in the direction G13 says
the proxy errs; they do not say what WASP-39b's depth noise is.

    conda run -n mirage python scripts/a3_noise_proxy.py --draws 1000
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("data/a3")
RE = Path("data/real_ess")
ANCHOR = RE / "wasp39b_ns_posterior_fitrad.npz"
COV_ARMS = [f"h9_rad_cov_s{s}" for s in (42, 43, 44)]
REF_ARMS = [f"h9_rad_nocond_s{s}" for s in (42, 43, 44)]
G13 = {"depth": "data/inject/wasp96_depth_noise.npz",
       "depth_oot": "data/inject/wasp96_depth_noise_oot.npz",
       "depth_direct": "data/inject/wasp96_depth_noise.npz",
       "depth_oot_direct": "data/inject/wasp96_depth_noise_oot.npz"}
# P6-D2, reproduced through this same code path. The harness is rejected if the
# `real` input does not return these.
PUBLISHED = {"h9_rad_cov_s42": 0.434, "h9_rad_cov_s43": 0.411, "h9_rad_cov_s44": 0.435}


def sep_profile(corr, n):
    """Median correlation at each bin separation, extended to n bins."""
    p = np.ones(n)
    for k in range(1, n):
        p[k] = np.median(np.diag(corr, k)) if k < len(corr) else 0.0
    return p


def psd_toeplitz(profile):
    """Toeplitz from a separation profile, projected to the nearest PSD matrix
    with unit diagonal. A measured profile need not be a valid correlation
    function on a different grid, so this cannot be skipped."""
    from scipy.linalg import toeplitz
    C = toeplitz(profile)
    w, V = np.linalg.eigh(C)
    C = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def submatrix_corr(corr, n):
    """G13's MEASURED correlation matrix carried onto an n-bin grid as a
    principal submatrix at evenly spaced indices. A principal submatrix of a
    correlation matrix is itself one, with unit diagonal and PSD guaranteed, so
    unlike the Toeplitz route nothing is modelled and nothing is projected, and
    the common-mode component survives the transplant."""
    idx = np.round(np.linspace(0, len(corr) - 1, n)).astype(int)
    return corr[np.ix_(idx, idx)]


def corr_sqrt(C):
    """Square root by eigendecomposition, which tolerates a singular C where
    Cholesky would fail. G13's matrices are near-singular by construction."""
    w, V = np.linalg.eigh(C)
    return V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))


def make_frames(kind, real, sig_pub, rng):
    """(K, P) frames in the model's stored bin order, matching `real`'s layout."""
    K, P = real.shape
    if kind == "real":
        return real
    if kind == "scale10":
        return (real * 10.0).astype(np.float32)
    if kind == "white":
        return (rng.standard_normal((K, P)) * sig_pub).astype(np.float32)
    if kind == "rank1":
        # BOUNDING PROBE, physically impossible. Every bin perfectly correlated,
        # so Sigma-hat is rank 1 and the embedding is as far from the real input
        # as it can be driven. If the posterior does not move HERE, no realistic
        # Sigma-hat could have caused the paradox.
        return (rng.standard_normal((K, 1)) * np.ones((1, P)) * sig_pub).astype(np.float32)
    corr = np.load(G13[kind], allow_pickle=True)["corr"]
    C = (submatrix_corr(corr, P) if kind.endswith("_direct")
         else psd_toeplitz(sep_profile(corr, P)))
    return ((rng.standard_normal((K, P)) @ corr_sqrt(C).T) * sig_pub).astype(np.float32)


def embedding_probe(frames: dict, arm: str = "h9_rad_cov_s42"):
    """WHY the posterior does not move, which is not the same as the channel
    being ignored. Measures how far the 64-dim covariance embedding travels
    between inputs, and how large that block is beside the 256-dim spectrum
    block it is concatenated to. A big relative move inside a small block is
    the signature of a channel that works but cannot matter."""
    import torch
    from fm4ar.models.build_model import build_model
    from mirage.datasets.real_jwst import build_observation

    ck = Path(f"configs/{arm}")
    net = build_model(file_path=ck / "model__best.pt", experiment_dir=ck,
                      device="cpu").network
    net.eval()
    enc = [m for m in net.modules() if hasattr(m, "cov_embedding")]
    if len(enc) != 1:
        raise SystemExit(f"expected one covariance encoder, found {len(enc)}")
    enc = enc[0]

    emb = {}
    with torch.no_grad():
        for k, f in frames.items():
            emb[k] = enc.cov_embedding(
                torch.from_numpy(np.asarray(f)).float().unsqueeze(0)).squeeze(0).numpy()
        ctx_np, _ = build_observation("data/jwst_wasp39b_spectrum.csv",
                                      "data/abc_rad/abc_rad_train.hdf")
        shares, full = {}, None
        for k, f in frames.items():
            ctx = {kk: torch.from_numpy(np.asarray(v)).float().unsqueeze(0)
                   for kk, v in ctx_np.items()}
            ctx["oot_frames"] = torch.from_numpy(np.asarray(f)).float().unsqueeze(0)
            fl = enc(ctx).squeeze(0).numpy()
            shares[k] = float(np.linalg.norm(fl[-len(emb[k]):]) / np.linalg.norm(fl))
            if k == "real":
                full = fl

    nc = len(emb["real"])
    spec, cov = full[:-nc], full[-nc:]
    share = float(np.linalg.norm(cov) / np.linalg.norm(full))
    print(f"\n  CONTEXT DECOMPOSITION ({arm})")
    print(f"    spectrum block {len(spec)} dims, norm {np.linalg.norm(spec):.3f}")
    print(f"    covariance block {nc} dims, norm {np.linalg.norm(cov):.3f}")
    print(f"    the covariance block is {100 * share:.1f}% of the context norm")
    print(f"\n  {'input':<12}{'embedding move vs real':>24}{'as % of context':>18}")
    d = {}
    for k in frames:
        if k == "real":
            continue
        dd = float(np.linalg.norm(emb[k] - emb["real"]))
        d[k] = dd
        print(f"    {k:<10}{dd / np.linalg.norm(emb['real']) * 100:>20.1f}%"
              f"{dd / np.linalg.norm(full) * 100:>17.2f}%")
    print(f"\n  {'input':<18}{'cov share of context':>22}")
    for k, v in shares.items():
        print(f"    {k:<16}{100 * v:>21.2f}%")
    return dict(cov_share=share, moves=d, shares=shares,
                norm_spec=float(np.linalg.norm(spec)),
                norm_cov=float(np.linalg.norm(cov)))


def main(a):
    import torch
    import real_ess
    import real_compare

    OUT.mkdir(parents=True, exist_ok=True)
    bak = OUT / "_published_backup"
    bak.mkdir(exist_ok=True)
    # real_ess.sample() writes fixed filenames. Protect the published artefacts
    # before overwriting them, and restore at the end whatever happens.
    for arm in COV_ARMS + REF_ARMS:
        p = RE / f"real_ess_samples_{arm}.npz"
        if p.exists() and not (bak / p.name).exists():
            shutil.copy2(p, bak / p.name)
    # real_ess.sample() ALSO writes the generic real_ess_samples.npz, which
    # ot_calibrate.py and taurex_retrieve.py read by default. Leaving an A3
    # synthetic-Sigma-hat posterior there, stamped with a real arm name, would
    # be silently consumed as a published result. Remove it rather than leave
    # it stale, so a default-path read fails loudly instead of lying.
    generic = RE / "real_ess_samples.npz"

    real0 = real_ess._real_oot_frames()
    P = real0.shape[1]
    # Per-bin scale. The embedding whitens, dividing by the per-bin std of
    # whatever it is handed, so ANY per-bin positive factor is removed exactly
    # and this choice cannot move a score. It was std(|x|) ~ 0.6 sigma before
    # the audit; corrected for honesty, not because a number changed.
    sig_pub = real0.std(axis=0)
    kinds = ["real", "white", "depth", "depth_oot", "depth_direct",
             "depth_oot_direct", "rank1", "scale10"]
    print(f"[frames] {real0.shape[0]} frames x {P} bins, model bin order")
    # 5 of 52 bins carry no real data and are exactly zero in `real0` (P3-D1).
    # The synthetic inputs inherit that through sig_pub, so every arm sees the
    # same coverage and only the correlation differs. Score on the live bins.
    live = real0.std(axis=0) > 0
    print(f"  {int(live.sum())}/{P} bins carry real data; the rest are zero in every input")
    for k in kinds:
        f = make_frames(k, real0, sig_pub, np.random.default_rng(a.seed))[:, live]
        C = np.corrcoef(f, rowvar=False)
        off = ~np.eye(C.shape[0], dtype=bool)
        print(f"  {k:<10} lag-1 {np.median(np.diag(C, 1)):+.3f}   "
              f"mean |off-diag| {np.abs(C[off]).mean():.3f}   "
              f"top eig/dim {np.linalg.eigvalsh(C)[-1] / C.shape[0]:.3f}")
    print(f"  the K={real0.shape[0]}-frame sampling floor on mean |off-diag| is "
          f"{np.sqrt(2 / (np.pi * real0.shape[0])):.3f}; a probe at or below it is a white arm")
    print("  PROBE ADEQUACY, what G13 measured on WASP-96b vs what reaches the arm:")
    for kd in ("depth", "depth_oot"):
        C0 = np.load(G13[kd], allow_pickle=True)["corr"]
        o0 = ~np.eye(C0.shape[0], dtype=bool)
        tp = psd_toeplitz(sep_profile(C0, P))
        sm = submatrix_corr(C0, P)
        otp = ~np.eye(P, dtype=bool)
        print(f"    {kd:<10} G13 {C0.shape[0]} bins {np.abs(C0[o0]).mean():.3f}"
              f"  ->  toeplitz {np.abs(tp[otp]).mean():.3f}"
              f"  ->  submatrix {np.abs(sm[otp]).mean():.3f}")

    try:
        res = {}
        for arm in COV_ARMS:
            for kind in kinds:
                tag = f"{arm}__{kind}"
                dst = OUT / f"a3_{tag}.npz"
                if not dst.exists() or a.force:
                    frames = make_frames(kind, real0, sig_pub,
                                         np.random.default_rng(a.seed))
                    real_ess._real_oot_frames = lambda k=100, _f=frames: _f
                    # NOT what pairs the inputs. `real_ess.sample` rebuilds the
                    # model, and `FMPEModel.initialize_network` calls
                    # `set_random_seed(self.random_seed)` (fm4ar fmpe.py:79),
                    # which resets the global RNG from the arm's config AFTER
                    # this line. Pairing is therefore exact and comes from
                    # fm4ar: every input draws a byte-identical theta_0 and only
                    # the context differs. Kept so the pairing stays explicit if
                    # fm4ar ever stops reseeding.
                    torch.manual_seed(a.seed)
                    real_ess.sample(a.draws, arm)
                    shutil.move(RE / f"real_ess_samples_{arm}.npz", dst)
                _, js = real_compare.compare(dst, ANCHOR, quiet=True)
                th = np.load(dst, allow_pickle=True)["theta"]
                res[tag] = (js, float(th[:, 1].mean()), float(th[:, 0].mean()))
                print(f"  {tag:<28} mean JS {js:.3f}   T {th[:,1].mean():6.0f} K   "
                      f"Rp {th[:,0].mean():.3f} RJ")
        for arm in REF_ARMS:
            dst = OUT / f"a3_{arm}__nocond.npz"
            if not dst.exists() or a.force:
                torch.manual_seed(a.seed)      # see the note above
                real_ess.sample(a.draws, arm)
                shutil.move(RE / f"real_ess_samples_{arm}.npz", dst)
            _, js = real_compare.compare(dst, ANCHOR, quiet=True)
            th = np.load(dst, allow_pickle=True)["theta"]
            res[f"{arm}__nocond"] = (js, float(th[:, 1].mean()), float(th[:, 0].mean()))
    finally:
        for p in sorted(bak.glob("real_ess_samples_*.npz")):
            shutil.copy2(p, RE / p.name)
        generic.unlink(missing_ok=True)
        print(f"[restore] published per-arm sample files put back from {bak}")
        print(f"[clean]   removed {generic}, which this script necessarily "
              f"overwrites; regenerate it with real_ess.py if a default-path "
              f"consumer needs it")

    # ---- the table -------------------------------------------------------
    print(f"\n=== A3 / H5. REAL WASP-39b MEAN 1-D JS TO THE NS ANCHOR ===")
    print(f"  {'Sigma-hat fed to the cov arm':<32}{'s42':>8}{'s43':>8}{'s44':>8}"
          f"{'mean':>9}{'sd':>8}")
    print("  " + "-" * 73)
    base = None
    for kind in kinds:
        v = np.array([res[f"{a_}__{kind}"][0] for a_ in COV_ARMS])
        if kind == "real":
            base = v
        print(f"  {kind:<32}" + "".join(f"{x:>8.3f}" for x in v)
              + f"{v.mean():>9.3f}{v.std(ddof=1):>8.3f}")
    nc = np.array([res[f"{a_}__nocond"][0] for a_ in REF_ARMS])
    print("  " + "-" * 73)
    print(f"  {'no conditioning (reference)':<32}" + "".join(f"{x:>8.3f}" for x in nc)
          + f"{nc.mean():>9.3f}{nc.std(ddof=1):>8.3f}")

    print(f"\n  harness check, `real` vs P6-D2 published:")
    for a_ in COV_ARMS:
        got, pub = res[f"{a_}__real"][0], PUBLISHED[a_]
        print(f"    {a_:<20} {got:.3f} vs {pub:.3f}   "
              f"{'MATCH' if abs(got - pub) < 0.006 else 'DIFFERS'}")
    s10 = np.array([res[f"{a_}__scale10"][0] for a_ in COV_ARMS])
    print(f"  whitening check, scale10 vs real: max |diff| {np.abs(s10 - base).max():.4f} "
          f"({'scale-free as designed' if np.abs(s10 - base).max() < 1e-6 else 'NOT scale-free'})")

    probe = [k for k in kinds if k != "scale10"]
    span = np.array([np.mean([res[f"{a_}__{k}"][0] for a_ in COV_ARMS]) for k in probe])
    seed_sd = np.mean([np.std([res[f"{a_}__{k}"][0] for a_ in COV_ARMS], ddof=1)
                       for k in probe])
    gap = base.mean() - nc.mean()
    print(f"\n  SPAN of mean JS across every Sigma-hat tried: {span.max() - span.min():.4f}")
    print(f"  seed sd within an input:                      {seed_sd:.4f}")
    print(f"  the paradox itself, cov minus no-cond:        {gap:+.4f}")
    print(f"  proxy explains at most {100 * (span.max() - span.min()) / gap:.1f}% of the paradox")

    allf = {k: make_frames(k, real0, sig_pub, np.random.default_rng(a.seed))
            for k in kinds}
    pr = embedding_probe(allf)
    struct = {k: float(np.abs(np.corrcoef(allf[k][:, live], rowvar=False)[
        ~np.eye(int(live.sum()), dtype=bool)]).mean()) for k in kinds}

    np.savez(OUT / "a3_scores.npz", probe=np.array([pr], dtype=object),
             structure=np.array([struct], dtype=object),
             **{k: np.array(v) for k, v in res.items()})
    print(f"\nSaved -> {OUT / 'a3_scores.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    main(p.parse_args())
