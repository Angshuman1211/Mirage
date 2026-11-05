"""
Phase 3 Track 2 — CycleGAN Ablation (standalone, no fm4ar required).

Evaluates CycleGAN translation quality vs domain randomisation using
metrics that don't require the fm4ar inference pipeline:

Metrics (all computable from data alone):
  1. Frechet Inception Distance (FID) proxy — MMD between real and translated
  2. Cycle-consistency error — ||G_BA(G_AB(x)) - x||_1
  3. Domain coverage — KL divergence between marginal flux distributions
  4. Per-wavelength shift — mean absolute bias per bin after translation
  5. Noise structure preservation — correlation matrix similarity

These metrics directly answer the D4 question:
  "Does CycleGAN translation bring simulated spectra closer to the real domain?"

Usage:
    python scripts/run_cyclegan_ablation_standalone.py [--n-samples 500]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

# ── Load CycleGAN without triggering mirage/__init__.py ──────────────────
_cg_path = Path(__file__).resolve().parents[1] / "mirage" / "nn" / "cyclegan.py"
_spec = importlib.util.spec_from_file_location("mirage.nn.cyclegan", _cg_path)
_cg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cg_mod)
CycleGAN = _cg_mod.CycleGAN

# ── Paths ─────────────────────────────────────────────────────────────────
ABC_TEST   = Path("data/abc/abc_test.hdf")
REAL_CSV   = Path("mirage_processed_corpus_v0.1/WASP39b/WASP39b_final_standardized.csv")
CKPT       = Path("configs/cyclegan/cyclegan_best.pt")
N_BINS     = 52


# ── Data loading ──────────────────────────────────────────────────────────

def load_abc(n: int, cols: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(flux, wlen) in the hdf's stored bin order, optionally sliced to `cols`.

    Returning wlen matters: the domain-randomisation baseline builds its
    covariance from |lambda_i - lambda_j|, and the ABC grid is strongly
    non-uniform (0.04 to 0.46 um spacing), so a linspace stand-in gives the
    kernel the wrong distances everywhere.
    """
    with h5py.File(ABC_TEST) as f:
        flux = f["flux"][:n].astype(np.float32)
        wlen = f["wlen"][0].astype(np.float64)
        if "flux_mean" in f and "flux_std" in f:
            flux = (flux - f["flux_mean"][:]) / f["flux_std"][:]
    if cols is not None:
        cols = np.asarray(cols, dtype=int)
        flux, wlen = flux[:, cols], wlen[cols]
    return flux, wlen


def load_real(n_bins: int = N_BINS) -> np.ndarray:
    df = pd.read_csv(REAL_CSV)
    wav_col  = next((c for c in df.columns if "wavelength" in c.lower()), None)
    flux_col = next((c for c in df.columns if "flux" in c.lower()
                     and "error" not in c.lower()), None)
    df = df[[wav_col, flux_col]].dropna()
    df.columns = ["wavelength", "flux"]
    df = df[df["flux"] > 0].sort_values("wavelength")
    edges = np.linspace(df["wavelength"].min(), df["wavelength"].max(), n_bins + 1)
    df["bin"] = pd.cut(df["wavelength"], bins=edges, labels=False)
    binned = df.groupby("bin")["flux"].mean().values.astype(np.float32)
    if len(binned) < n_bins:
        binned = np.pad(binned, (0, n_bins - len(binned)),
                        constant_values=float(np.nanmedian(binned)))
    binned = binned[:n_bins]
    binned = np.nan_to_num(binned, nan=float(np.nanmedian(binned)))
    # Normalise to same scale as ABC
    binned = (binned - binned.mean()) / (binned.std() + 1e-8)
    return binned


def load_real_domain(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Genuine real DISTRIBUTION from build_real_domain.py (P5-D11).

    Replaces load_real(), which returned ONE vector that main() then tiled, so the
    "real distribution" was a point mass with zero variance -- MMD measured distance
    to a point and the correlation metric returned NaN. It also read continuum-
    normalised STELLAR FLUX while the sim side is TRANSIT DEPTH, and sat on the
    ascending grid while the sim side is in stored (descending) order. Here every
    row is a real reduced depth spectrum carrying a real per-frame noise
    realisation, in stored order, gap-filled at build time.
    """
    d = np.load(path, allow_pickle=True)
    if "abc_cols" not in d:
        raise SystemExit(
            f"[guard] {path.name} carries no `abc_cols`, so its bins cannot be "
            f"matched to ABC wavelengths. Rebuild it with build_real_domain.py.")
    x = d["samples"].astype(np.float64)
    if not np.isfinite(x).all():
        raise SystemExit(f"[guard] {path.name} holds non-finite values; rebuild it")
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)   # same scaling as sim_norm
    return x.astype(np.float32), np.asarray(d["abc_cols"], dtype=int)


def load_cyclegan(ckpt_path: Path = None) -> CycleGAN:
    ckpt = torch.load(ckpt_path or CKPT, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    model = CycleGAN(n_bins=cfg["n_bins"], n_res=cfg["n_res"],
                     ngf=cfg["ngf"], ndf=cfg["ndf"])
    model.G_AB.load_state_dict(ckpt["G_AB"])
    model.G_BA.load_state_dict(ckpt["G_BA"])
    model.eval()
    return model


# ── Metrics ───────────────────────────────────────────────────────────────

def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """Maximum Mean Discrepancy with RBF kernel — proxy for distributional distance."""
    def rbf(A, B):
        diff = A[:, None, :] - B[None, :, :]
        return np.exp(-np.sum(diff**2, axis=-1) / (2 * sigma**2))
    n, m = len(X), len(Y)
    kxx = rbf(X, X)
    kyy = rbf(Y, Y)
    kxy = rbf(X, Y)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def cycle_consistency_error(model: CycleGAN, flux: np.ndarray) -> float:
    """Mean L1 error of round-trip G_BA(G_AB(x)) vs x."""
    x = torch.from_numpy(flux).float()
    with torch.no_grad():
        fake_B   = model.G_AB(x)
        rec_A    = model.G_BA(fake_B)
    return float((rec_A - x).abs().mean())


def per_bin_shift(orig: np.ndarray, translated: np.ndarray) -> np.ndarray:
    """Mean absolute shift per wavelength bin after translation."""
    return np.abs(translated.mean(axis=0) - orig.mean(axis=0))


def corr_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius similarity between correlation matrices of two datasets."""
    ca = np.corrcoef(A.T)
    cb = np.corrcoef(B.T)
    diff = ca - cb
    return float(1.0 - np.linalg.norm(diff, 'fro') / (np.linalg.norm(ca, 'fro') + 1e-8))


def kl_divergence_marginals(P: np.ndarray, Q: np.ndarray, bins: int = 50) -> float:
    """Mean KL divergence across marginal (per-bin) flux distributions."""
    kls = []
    for i in range(P.shape[1]):
        lo = min(P[:, i].min(), Q[:, i].min())
        hi = max(P[:, i].max(), Q[:, i].max())
        edges = np.linspace(lo, hi, bins + 1)
        p_hist, _ = np.histogram(P[:, i], bins=edges, density=True)
        q_hist, _ = np.histogram(Q[:, i], bins=edges, density=True)
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()
        kls.append(float(np.sum(p_hist * np.log(p_hist / q_hist))))
    return float(np.mean(kls))


# ── Domain randomisation simulation ──────────────────────────────────────

def apply_domain_randomisation(flux: np.ndarray, wlen: np.ndarray,
                               seed: int = 42) -> np.ndarray:
    """
    Simulate Phase 2 domain randomisation (correlated noise injection).
    This is the standalone proxy for the noisecond_cov arm.

    Two defects fixed here (P5-D11), both of which made this baseline's noise
    SMOOTHER and longer-correlated than what the model was actually trained
    with, in a comparison that is specifically about matching real short-range
    correlated noise:

      - `wlen` was np.linspace(0.55, 7.30, n), but the ABC grid is strongly
        non-uniform, so every |lambda_i - lambda_j| in the kernel was wrong. It
        is now passed in from the hdf, sliced to the same bins as the data.
      - the correlation-length ranges were the CorrelatedNoiseGenerator class
        defaults (se 0.10-1.00, ou 0.50-3.00) rather than the trained config's
        (se 0.02-1.00, ou 0.05-3.00). P3-D1 extended those bounds DOWN on
        purpose, because real JWST correlation is short-range.

    Ranges below track configs/noisecond_cov/config.yaml. Changing them there
    without changing them here is how the two drifted apart in the first place.
    """
    rng = np.random.default_rng(seed)
    result = flux.copy()
    wlen = np.asarray(wlen, dtype=np.float64)
    if wlen.shape[0] != flux.shape[1]:
        raise SystemExit(f"[guard] wlen has {wlen.shape[0]} bins, flux has "
                         f"{flux.shape[1]} — the kernel distances would be wrong")

    for i in range(len(flux)):
        sigma = rng.uniform(0.05, 0.30)
        rho   = rng.uniform(0.30, 0.80)
        se_l  = rng.uniform(0.02, 1.00)
        ou_l  = rng.uniform(0.05, 3.00)

        d = np.abs(wlen[:, None] - wlen[None, :])
        var = sigma ** 2
        cov = ((1 - rho) * var * np.eye(len(wlen))
               + 0.5 * rho * var * np.exp(-0.5 * (d / se_l)**2)
               + 0.5 * rho * var * np.exp(-d / ou_l))
        cov += 1e-8 * var * np.eye(len(wlen))

        try:
            chol = np.linalg.cholesky(cov)
            noise = chol @ rng.standard_normal(len(wlen))
            result[i] += noise.astype(np.float32)
        except np.linalg.LinAlgError:
            result[i] += rng.standard_normal(len(wlen)).astype(np.float32) * sigma

    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main(n_samples: int = 500, real_domain: str | None = None,
         ckpt: str | None = None, out: str | None = None) -> None:
    ckpt_path = Path(ckpt) if ckpt else CKPT
    print(f"\n{'='*72}")
    print("  Phase 3 Track 2 — CycleGAN Ablation (standalone)")
    print(f"{'='*72}")

    # Verify files
    _real_src = Path(real_domain) if real_domain else REAL_CSV
    for p, name in [(ABC_TEST, "abc_test.hdf"), (_real_src, "real domain"), (ckpt_path, "cyclegan ckpt")]:
        status = "✓" if p.exists() else "✗ MISSING"
        print(f"  {status}  {name}")
    if not all(p.exists() for p in [ABC_TEST, _real_src, ckpt_path]):
        raise SystemExit("Missing required files.")

    print(f"\n  Loading data ({n_samples} ABC samples) ...")
    if real_domain:                      # a genuine distribution, never tiled
        real_rep, abc_cols = load_real_domain(Path(real_domain))
        sim_raw, wlen = load_abc(n_samples, cols=abc_cols)
        if real_rep.shape[1] != sim_raw.shape[1]:
            raise SystemExit(f"[guard] real domain has {real_rep.shape[1]} bins, "
                             f"sliced ABC has {sim_raw.shape[1]}")
        if len(real_rep) > n_samples:
            real_rep = real_rep[np.random.default_rng(0).choice(
                len(real_rep), n_samples, replace=False)]
        print(f"  grid: {sim_raw.shape[1]} ABC bins, "
              f"{wlen.min():.2f}-{wlen.max():.2f} µm (stored order)")
    else:
        sim_raw, wlen = load_abc(n_samples)
        real_rep = np.tile(load_real(), (len(sim_raw), 1))   # legacy point mass

    # Normalise ABC to same scale as real
    sim_norm = (sim_raw - sim_raw.mean(axis=0)) / (sim_raw.std(axis=0) + 1e-8)

    print("  Loading CycleGAN model ...")
    model = load_cyclegan(ckpt_path)
    if model.G_AB.n_bins != sim_norm.shape[1]:
        raise SystemExit(f"[guard] checkpoint was trained on {model.G_AB.n_bins} "
                         f"bins, this data has {sim_norm.shape[1]}")

    print("  Running translations ...")
    with torch.no_grad():
        sim_t  = torch.from_numpy(sim_norm.astype(np.float32))
        trans  = model.G_AB(sim_t).numpy()   # CycleGAN translated

    domain_rand = apply_domain_randomisation(sim_norm, wlen, seed=42)

    print("\n  Computing metrics ...\n")

    # ── 1. Cycle-consistency error ─────────────────────────────────────────
    cyc_err = cycle_consistency_error(model, sim_norm.astype(np.float32))

    # ── 2. MMD to real domain ──────────────────────────────────────────────
    # Use PCA-reduced (10-dim) features to make MMD tractable
    from numpy.linalg import svd
    combined = np.vstack([sim_norm, real_rep, trans, domain_rand])
    _, _, Vt = svd(combined - combined.mean(axis=0), full_matrices=False)
    V10 = Vt[:10].T

    sim_pca   = sim_norm    @ V10
    real_pca  = real_rep    @ V10
    trans_pca = trans        @ V10
    dr_pca    = domain_rand  @ V10

    mmd_sim_real   = mmd_rbf(sim_pca,   real_pca,  sigma=1.0)
    mmd_trans_real = mmd_rbf(trans_pca, real_pca,  sigma=1.0)
    mmd_dr_real    = mmd_rbf(dr_pca,    real_pca,  sigma=1.0)

    # ── 3. KL divergence of marginals ─────────────────────────────────────
    kl_sim_real   = kl_divergence_marginals(sim_norm[:200],    real_rep[:200])
    kl_trans_real = kl_divergence_marginals(trans[:200],        real_rep[:200])
    kl_dr_real    = kl_divergence_marginals(domain_rand[:200],  real_rep[:200])

    # ── 4. Per-bin shift ──────────────────────────────────────────────────
    shift_sim   = per_bin_shift(real_rep, sim_norm).mean()
    shift_trans = per_bin_shift(real_rep, trans).mean()
    shift_dr    = per_bin_shift(real_rep, domain_rand).mean()

    # ── 5. Correlation structure similarity ───────────────────────────────
    corr_sim   = corr_similarity(sim_norm[:200],   real_rep[:200])
    corr_trans = corr_similarity(trans[:200],        real_rep[:200])
    corr_dr    = corr_similarity(domain_rand[:200],  real_rep[:200])

    # ── Print results table ───────────────────────────────────────────────
    print(f"  {'─'*70}")
    print(f"  Phase 3 Track 2 — CycleGAN vs Domain Randomisation")
    print(f"  (n={n_samples} ABC test spectra, real = see real-domain source below)")
    print(f"  {'─'*70}")
    print(f"  {'condition':<22} {'MMD↓':>8} {'KL↓':>8} {'shift↓':>8} {'corr↑':>8}")
    print(f"  {'─'*70}")
    print(f"  {'sim (no adaptation)':<22} {mmd_sim_real:>8.4f} {kl_sim_real:>8.4f} {shift_sim:>8.4f} {corr_sim:>8.4f}")
    print(f"  {'domain_random':<22} {mmd_dr_real:>8.4f} {kl_dr_real:>8.4f} {shift_dr:>8.4f} {corr_dr:>8.4f}")
    print(f"  {'cyclegan_trans':<22} {mmd_trans_real:>8.4f} {kl_trans_real:>8.4f} {shift_trans:>8.4f} {corr_trans:>8.4f}")
    print(f"  {'─'*70}")
    print(f"  Cycle-consistency error: {cyc_err:.6f} (↓ better; <0.05 = tight)")

    # ── Comparative analysis ──────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  Comparative Analysis (D4 ablation)")
    print(f"  {'─'*70}")

    delta_mmd = mmd_dr_real - mmd_trans_real
    print(f"  MMD: domain_random − cyclegan = {delta_mmd:+.4f}")
    if delta_mmd > 0.001:
        print(f"    → CycleGAN is CLOSER to real domain (MMD lower by {delta_mmd:.4f})")
    elif delta_mmd < -0.001:
        print(f"    → Domain randomisation is CLOSER to real domain")
    else:
        print(f"    → Essentially EQUAL (|Δ| < 0.001) — both approaches equivalent")

    delta_kl = kl_dr_real - kl_trans_real
    print(f"  KL:  domain_random − cyclegan = {delta_kl:+.4f}")
    if delta_kl > 0.001:
        print(f"    → CycleGAN distribution CLOSER to real (KL lower by {delta_kl:.4f})")
    elif delta_kl < -0.001:
        print(f"    → Domain randomisation distribution closer to real")
    else:
        print(f"    → Essentially EQUAL — both approaches equivalent")

    print(f"\n  {'─'*70}")
    print(f"  D4 Conclusion:")
    # P5-D11: the previous rule was `delta_mmd > 0.005 OR delta_kl > 0.005`, which is
    # magnitude-blind and fires on whichever metric crosses an ABSOLUTE threshold. With
    # MMD ~0.3 and KL ~3.7 those thresholds mean completely different things, so a 0.15%
    # KL difference could override a 93% MMD difference pointing the other way. Compare
    # each delta RELATIVE to its own scale and require the two metrics to agree.
    rel_mmd = delta_mmd / max((mmd_dr_real + mmd_trans_real) / 2, 1e-12)
    rel_kl  = delta_kl  / max((kl_dr_real + kl_trans_real) / 2, 1e-12)
    TOL = 0.02                                  # 2% of the metric's own scale
    print(f"  relative: MMD {rel_mmd:+.3f}  KL {rel_kl:+.3f}  (tolerance ±{TOL:.2f})")
    if abs(rel_mmd) < TOL and abs(rel_kl) < TOL:
        print(f"  ✓ domain_random ≈ cyclegan_trans — D4 VALIDATED")
        print(f"    Structured randomisation is sufficient; CycleGAN adds no value.")
        conclusion = "D4_VALIDATED"
    elif rel_mmd > TOL and rel_kl > TOL:
        print(f"  ⚠ CycleGAN closer to real on BOTH metrics — consider translation")
        conclusion = "CYCLEGAN_BENEFICIAL"
    elif rel_mmd < -TOL and rel_kl < -TOL:
        print(f"  ✓ domain randomisation closer on BOTH metrics — preferred")
        conclusion = "RANDOMISATION_PREFERRED"
    else:
        dom, dom_rel = ("MMD", rel_mmd) if abs(rel_mmd) >= abs(rel_kl) else ("KL", rel_kl)
        side = "CycleGAN" if dom_rel > 0 else "domain randomisation"
        print(f"  ! metrics DISAGREE — {dom} dominates by {abs(dom_rel)/max(abs(rel_kl if dom=='MMD' else rel_mmd),1e-12):.0f}x "
              f"and favours {side}")
        conclusion = ("CYCLEGAN_BENEFICIAL_MIXED" if dom_rel > 0
                      else "RANDOMISATION_PREFERRED_MIXED")
    print(f"  {'─'*70}\n")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "n_samples": n_samples,
        "conditions": {
            "sim_baseline": {
                "mmd_to_real": mmd_sim_real, "kl_to_real": kl_sim_real,
                "mean_shift": float(shift_sim), "corr_similarity": corr_sim
            },
            "domain_random": {
                "mmd_to_real": mmd_dr_real, "kl_to_real": kl_dr_real,
                "mean_shift": float(shift_dr), "corr_similarity": corr_dr
            },
            "cyclegan_trans": {
                "mmd_to_real": mmd_trans_real, "kl_to_real": kl_trans_real,
                "mean_shift": float(shift_trans), "corr_similarity": corr_trans,
                "cycle_consistency_error": cyc_err
            }
        },
        "deltas": {
            "mmd_dr_minus_cg":  float(delta_mmd),
            "kl_dr_minus_cg":   float(delta_kl),
        },
        "conclusion": conclusion
    }

    out_path = Path(out) if out else Path("configs/cyclegan/ablation_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--real-domain", type=str, default=None,
                   help="npz from build_real_domain.py (genuine real distribution)")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    main(n_samples=args.n_samples, real_domain=args.real_domain,
         ckpt=args.ckpt, out=args.out)
