"""V5 / H7: Multi-reduction pipeline shift on K2-18b (Schmidt 14735689 & Luque 15685551).

Each pair of reductions shares the exact same JWST NIRSpec G395H (or NIRISS SOSS)
visit and astrophysical truth, isolating pipeline reduction shift from atmosphere.

This script performs two stages:
  1. `--spectral` (runs anywhere with the extracted Zenodo spectra under `data/v5_k218/`):
     - Parses 4 independent Schmidt et al. (2025, Zenodo 14735689) R100 G395H reductions
       (exoTEDRF, Eureka! Ahrer, Eureka! Bell, FIREFLy), 2 Schmidt R100 NIRISS reductions
       (exoTEDRF, FIREFLy), and 3 independent Luque et al. (2025, Zenodo 15685551) R200
       G395H reductions (exoTEDRF, Eureka!, SPARTA).
     - Builds canonical 4-column CSVs (`wavelength_um, transit_depth, depth_error, instrument`)
       pairing each G395H reduction with the SAME fixed NIRISS segment (<2.75 um from
       `data/jwst_k2_18b_spectrum.csv`), so only the G395H pipeline varies.
     - Measures exact pairwise spectral shift: mean offset (ppm), offset-removed shape
       RMS (ppm), offset and shape in median-sigma units, and pairwise chi2/N (raw and
       offset-removed) on a shared 60-bin G395H grid (2.80-5.10 um).
     - Writes `configs/v5_k218_h7/spectral_shift.json`.

  2. `--run-flow` (requires `configs/noisecond_rad_nocond_k218/model__best.pt` AND
     the K2-18b model's training normalisation stats `wlen, flux_mean, flux_std` from
     `data/abc_rad_k218/abc_rad_train.hdf` or a 4 KB `abc_rad_k218_norm_stats.npz`):
     - Evaluates the K2-18b flow on every reduction (N=2000 draws each, identical ODE
       base noise per pair), computes pairwise 1-D JS divergence across all pairs of
       reductions of the same visit, and compares against the noise-model conditioning
       JS (0.083 nats, P6-D2) to answer H7 directly.
     - Writes `configs/v5_k218_h7/h7_posterior_js.json`.

Usage:
    python scripts/v5_h7_pipeline_shift.py --spectral
    python scripts/v5_h7_pipeline_shift.py --run-flow --stats data/abc_rad_k218/abc_rad_train.hdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHMIDT_DIR = Path("data/v5_k218/schmidt_14735689")
LUQUE_DIR = Path("data/v5_k218/luque_15685551")
CANON_DIR = Path("data/v5_k218/canonical")
OUT_DIR = Path("configs/v5_k218_h7")
BASE_K218_CSV = Path("data/jwst_k2_18b_spectrum.csv")
K218_CKPT_DIR = Path("configs/noisecond_rad_nocond_k218")
NOISE_COND_JS_REF = 0.0827  # P6-D2 / P6-D26 real-data conditioning JS shift (nats)


def _read_schmidt_dat(path: Path, instrument: str) -> pd.DataFrame:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if len(parts) < 4:
            continue
        try:
            wl, depth, err = float(parts[0]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if np.isfinite(wl) and np.isfinite(depth) and np.isfinite(err) and err > 0:
            rows.append((wl, depth, err, instrument))
    return pd.DataFrame(rows, columns=["wavelength_um", "transit_depth", "depth_error", "instrument"])


def _read_luque_csv(path: Path, instrument: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    wl = df["wavelength"].to_numpy(dtype=float)
    depth = df["dppm"].to_numpy(dtype=float) * 1e-6
    if "dppm_err" in df.columns:
        err = df["dppm_err"].to_numpy(dtype=float) * 1e-6
    else:
        err = 0.5 * (
            df["dppm_err_neg"].to_numpy(dtype=float)
            + df["dppm_err_pos"].to_numpy(dtype=float)
        ) * 1e-6
    m = np.isfinite(wl) & np.isfinite(depth) & np.isfinite(err) & (err > 0)
    out = pd.DataFrame({
        "wavelength_um": wl[m],
        "transit_depth": depth[m],
        "depth_error": err[m],
        "instrument": instrument,
    })
    return out.sort_values("wavelength_um").reset_index(drop=True)


def load_all_reductions() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return (g395h_reductions, niriss_reductions) as canonical DataFrames."""
    g395h: dict[str, pd.DataFrame] = {}
    niriss: dict[str, pd.DataFrame] = {}

    schmidt_g395h = {
        "schmidt_exotedrf": ("K2-18b_NIRSpec_G395H_NRS1_exoTEDRF_R100.dat",
                             "K2-18b_NIRSpec_G395H_NRS2_exoTEDRF_R100.dat"),
        "schmidt_eureka_ahrer": ("K2-18b_NIRSpec_G395H_NRS1_Eureka_Ahrer_R100.dat",
                                 "K2-18b_NIRSpec_G395H_NRS2_Eureka_Ahrer_R100.dat"),
        "schmidt_eureka_bell": ("K2-18b_NIRSpec_G395H_NRS1_Eureka_Bell_R100.dat",
                                "K2-18b_NIRSpec_G395H_NRS2_Eureka_Bell_R100.dat"),
        "schmidt_firefly": ("K2-18b_NIRSpec_G395H_NRS1_FIREFLy_R100.dat",
                            "K2-18b_NIRSpec_G395H_NRS2_FIREFLy_R100.dat"),
    }
    for name, (f1, f2) in schmidt_g395h.items():
        p1, p2 = SCHMIDT_DIR / f1, SCHMIDT_DIR / f2
        if p1.exists() and p2.exists():
            df = pd.concat([
                _read_schmidt_dat(p1, "G395H_NRS1"),
                _read_schmidt_dat(p2, "G395H_NRS2"),
            ], ignore_index=True).sort_values("wavelength_um").reset_index(drop=True)
            g395h[name] = df

    luque_g395h = {
        "luque_exotedrf": ("exoTEDRF_K2-18b_NIRSpec_G395H_NRS1_R200.csv",
                           "exoTEDRF_K2-18b_NIRSpec_G395H_NRS2_R200.csv"),
        "luque_eureka": ("Eureka_K2-18b_NIRSpec_G395H_NRS1_R200.csv",
                         "Eureka_K2-18b_NIRSpec_G395H_NRS2_R200.csv"),
        "luque_sparta": ("SPARTA_K2-18b_NIRSpec_G395H_NRS1_R200.csv",
                         "SPARTA_K2-18b_NIRSpec_G395H_NRS2_R200.csv"),
    }
    for name, (f1, f2) in luque_g395h.items():
        p1, p2 = LUQUE_DIR / f1, LUQUE_DIR / f2
        if p1.exists() and p2.exists():
            df = pd.concat([
                _read_luque_csv(p1, "G395H_NRS1"),
                _read_luque_csv(p2, "G395H_NRS2"),
            ], ignore_index=True).sort_values("wavelength_um").reset_index(drop=True)
            g395h[name] = df

    schmidt_niriss = {
        "schmidt_niriss_exotedrf": ("K2-18b_NIRISS_SOSS_Ord1_exoTEDRF_R100.dat",
                                    "K2-18b_NIRISS_SOSS_Ord2_exoTEDRF_R100.dat"),
        "schmidt_niriss_firefly": ("K2-18b_NIRISS_SOSS_Ord1_FIREFLy_R100.dat",
                                   "K2-18b_NIRISS_SOSS_Ord2_FIREFLy_R100.dat"),
    }
    for name, (o1, o2) in schmidt_niriss.items():
        p1, p2 = SCHMIDT_DIR / o1, SCHMIDT_DIR / o2
        if p1.exists() and p2.exists():
            df = pd.concat([
                _read_schmidt_dat(p1, "NIRISS_O1"),
                _read_schmidt_dat(p2, "NIRISS_O2"),
            ], ignore_index=True).sort_values("wavelength_um").reset_index(drop=True)
            niriss[name] = df

    return g395h, niriss


def _rebin_common(df: pd.DataFrame, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nb = len(edges) - 1
    b = np.digitize(df["wavelength_um"].to_numpy(), edges) - 1
    d = df.assign(_b=b)
    d = d[(d._b >= 0) & (d._b < nb)]
    w = 1.0 / np.clip(d["depth_error"].to_numpy(), 1e-12, None) ** 2
    d = d.assign(_w=w)
    depth = np.full(nb, np.nan)
    err = np.full(nb, np.nan)
    for bb, grp in d.groupby("_b"):
        ww, dd = grp["_w"].to_numpy(), grp["transit_depth"].to_numpy()
        depth[bb] = np.sum(ww * dd) / np.sum(ww)
        err[bb] = 1.0 / np.sqrt(np.sum(ww))
    return depth, err, np.isfinite(depth)


def _pair_metrics(d1: np.ndarray, e1: np.ndarray, d2: np.ndarray, e2: np.ndarray, mask: np.ndarray) -> dict:
    diff = (d1[mask] - d2[mask]) * 1e6  # ppm
    sig = np.sqrt(e1[mask] ** 2 + e2[mask] ** 2) * 1e6
    med_sig_single = float(np.median(0.5 * (e1[mask] + e2[mask])) * 1e6)
    offset = float(np.mean(diff))
    shape_rms = float(np.sqrt(np.mean((diff - offset) ** 2)))
    total_rms = float(np.sqrt(np.mean(diff ** 2)))
    chi2_raw = float(np.mean((diff / sig) ** 2))
    chi2_noshift = float(np.mean(((diff - offset) / sig) ** 2))
    return {
        "n_bins": int(mask.sum()),
        "offset_ppm": offset,
        "shape_rms_ppm": shape_rms,
        "total_rms_ppm": total_rms,
        "median_single_err_ppm": med_sig_single,
        "abs_offset_over_err": abs(offset) / med_sig_single,
        "shape_rms_over_err": shape_rms / med_sig_single,
        "pair_chi2_per_bin_raw": chi2_raw,
        "pair_chi2_per_bin_offset_removed": chi2_noshift,
    }


def run_spectral_analysis() -> dict:
    g395h, niriss = load_all_reductions()
    if not g395h:
        raise SystemExit("No G395H reductions found under data/v5_k218/.")

    # Save canonical combined spectra (fixed NIRISS anchor <2.75 um + swapped G395H)
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    base_df = pd.read_csv(BASE_K218_CSV)
    niriss_anchor = base_df[base_df["wavelength_um"] < 2.75].copy()
    g395h_anchor = base_df[base_df["wavelength_um"] >= 2.75].copy()

    for name, df in g395h.items():
        combined = pd.concat([niriss_anchor, df], ignore_index=True).sort_values("wavelength_um")
        combined.to_csv(CANON_DIR / f"k218_combined_{name}.csv", index=False)
    for name, df in niriss.items():
        combined = pd.concat([df, g395h_anchor], ignore_index=True).sort_values("wavelength_um")
        combined.to_csv(CANON_DIR / f"k218_combined_{name}.csv", index=False)

    # Common R~100 G395H grid (2.80 - 5.10 um, 55 log-spaced bins)
    g_edges = np.geomspace(2.80, 5.10, 56)
    rebinned_g = {nm: _rebin_common(df, g_edges) for nm, df in g395h.items()}

    names = list(g395h.keys())
    pairs_g395h = {}
    print(f"\n{'=' * 94}")
    print("  V5 / H7 Spectral Pipeline Shift — K2-18b NIRSpec G395H (same visit, 7 reductions)")
    print(f"{'=' * 94}")
    print(f"  {'pair':<44}{'bins':>5}{'offset':>9}{'shapeRMS':>10}{'off/σ':>7}{'shp/σ':>7}{'χ²/N':>7}{'χ²_0/N':>8}")
    print("  " + "-" * 92)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d1, e1, m1 = rebinned_g[a]
            d2, e2, m2 = rebinned_g[b]
            m = m1 & m2
            met = _pair_metrics(d1, e1, d2, e2, m)
            key = f"{a}__vs__{b}"
            pairs_g395h[key] = met
            print(
                f"  {a + ' vs ' + b:<44}{met['n_bins']:>5}"
                f"{met['offset_ppm']:>+9.1f}{met['shape_rms_ppm']:>10.1f}"
                f"{met['abs_offset_over_err']:>7.2f}{met['shape_rms_over_err']:>7.2f}"
                f"{met['pair_chi2_per_bin_raw']:>7.2f}{met['pair_chi2_per_bin_offset_removed']:>8.2f}"
            )

    # NIRISS SOSS pair (Schmidt exoTEDRF vs FIREFLy, 0.85 - 2.75 um, 60 log-spaced bins)
    n_edges = np.geomspace(0.85, 2.75, 61)
    pairs_niriss = {}
    if len(niriss) >= 2:
        n_names = list(niriss.keys())
        reb_n = {nm: _rebin_common(df, n_edges) for nm, df in niriss.items()}
        d1, e1, m1 = reb_n[n_names[0]]
        d2, e2, m2 = reb_n[n_names[1]]
        met = _pair_metrics(d1, e1, d2, e2, m1 & m2)
        pairs_niriss[f"{n_names[0]}__vs__{n_names[1]}"] = met
        print("  " + "-" * 92)
        print(
            f"  {n_names[0] + ' vs ' + n_names[1]:<44}{met['n_bins']:>5}"
            f"{met['offset_ppm']:>+9.1f}{met['shape_rms_ppm']:>10.1f}"
            f"{met['abs_offset_over_err']:>7.2f}{met['shape_rms_over_err']:>7.2f}"
            f"{met['pair_chi2_per_bin_raw']:>7.2f}{met['pair_chi2_per_bin_offset_removed']:>8.2f}"
        )

    # Summary statistics within Schmidt (4 reductions) and within Luque (3 reductions)
    sch_pairs = [v for k, v in pairs_g395h.items() if k.startswith("schmidt_") and "__vs__schmidt_" in k]
    luq_pairs = [v for k, v in pairs_g395h.items() if k.startswith("luque_") and "__vs__luque_" in k]
    all_pairs = list(pairs_g395h.values())

    summary = {
        "g395h_reductions": {k: {"n_points": int(len(df)), "wl_range": [float(df["wavelength_um"].min()), float(df["wavelength_um"].max())]} for k, df in g395h.items()},
        "niriss_reductions": {k: {"n_points": int(len(df)), "wl_range": [float(df["wavelength_um"].min()), float(df["wavelength_um"].max())]} for k, df in niriss.items()},
        "pairs_g395h": pairs_g395h,
        "pairs_niriss": pairs_niriss,
        "summary_stats": {
            "schmidt_4red_max_offset_ppm": max(abs(x["offset_ppm"]) for x in sch_pairs) if sch_pairs else None,
            "schmidt_4red_max_shape_rms_ppm": max(x["shape_rms_ppm"] for x in sch_pairs) if sch_pairs else None,
            "luque_3red_max_offset_ppm": max(abs(x["offset_ppm"]) for x in luq_pairs) if luq_pairs else None,
            "luque_3red_max_shape_rms_ppm": max(x["shape_rms_ppm"] for x in luq_pairs) if luq_pairs else None,
            "all7_max_offset_ppm": max(abs(x["offset_ppm"]) for x in all_pairs),
            "all7_max_shape_rms_ppm": max(x["shape_rms_ppm"] for x in all_pairs),
            "all7_median_shape_rms_over_err": float(np.median([x["shape_rms_over_err"] for x in all_pairs])),
        },
        "canonical_dir": str(CANON_DIR),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "spectral_shift.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Saved spectral pipeline-shift summary -> {out_path}")
    return summary


def _load_norm_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not stats_path.exists():
        raise SystemExit(
            f"[blocked] K2-18b training normalisation stats not found at {stats_path}.\n"
            f"To unblock `--run-flow` for V5/H7, export the 4 KB stats file on Track 1's machine:\n"
            f"  python -c \"import h5py, numpy as np; f=h5py.File('data/abc_rad_k218/abc_rad_train.hdf'); "
            f"np.savez('data/abc_rad_k218/abc_rad_k218_norm_stats.npz', wlen=f['wlen'][0], "
            f"flux_mean=f['flux_mean'][:], flux_std=f['flux_std'][:])\"\n"
            f"and re-run: python scripts/v5_h7_pipeline_shift.py --run-flow --stats data/abc_rad_k218/abc_rad_k218_norm_stats.npz"
        )
    if stats_path.suffix == ".npz":
        d = np.load(stats_path)
        return d["wlen"].astype(float), d["flux_mean"].astype(float), d["flux_std"].astype(float)
    import h5py
    with h5py.File(stats_path, "r") as f:
        return f["wlen"][0].astype(float), f["flux_mean"][:].astype(float), f["flux_std"][:].astype(float)


def run_flow_h7(stats_path: Path, n_samples: int = 2000) -> None:
    import torch
    import yaml
    import mirage  # noqa: F401
    from mirage.datasets.real_jwst import rebin_spectrum
    from fm4ar.models.build_model import build_model
    from fm4ar.datasets.theta_scalers import get_theta_scaler
    from scripts.real_compare import THETA_LAYOUT, js_1d

    wl_s, mean_s, std_s = _load_norm_stats(stats_path)
    order = np.argsort(wl_s)
    inv = np.argsort(order)
    wl_a, mean_a, std_a = wl_s[order], mean_s[order], std_s[order]

    ckpt = K218_CKPT_DIR / "model__best.pt"
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint {ckpt}")
    with open(K218_CKPT_DIR / "config.yaml") as fh:
        scaler = get_theta_scaler(yaml.safe_load(fh).get("theta_scaler", {}))

    if not CANON_DIR.exists():
        run_spectral_analysis()

    csvs = sorted(CANON_DIR.glob("k218_combined_*.csv"))
    posteriors: dict[str, np.ndarray] = {}
    medians: dict[str, dict] = {}
    names = THETA_LAYOUT[7]

    for csv_path in csvs:
        tag = csv_path.stem.replace("k218_combined_", "")
        df = pd.read_csv(csv_path)
        depth_a, err_a, cov_a = rebin_spectrum(df, wl_a)
        flux_a = (depth_a - mean_a) / std_a
        ebar_a = err_a / std_a
        med_sigma = np.nanmedian(ebar_a[cov_a])
        flux_a = np.where(cov_a, flux_a, 0.0)
        ebar_a = np.where(cov_a, ebar_a, med_sigma)
        ctx_np = {
            "flux": flux_a[inv].astype(np.float32),
            "wlen": wl_s.astype(np.float32),
            "error_bars": ebar_a[inv].astype(np.float32),
        }
        # Fresh build_model per reduction so ODE base noise is byte-identical (P6-D26 pairing)
        model = build_model(file_path=ckpt, experiment_dir=K218_CKPT_DIR, device="cpu")
        model.network.eval()
        ctx = {k: torch.from_numpy(v).float().unsqueeze(0).expand(n_samples, *v.shape)
               for k, v in ctx_np.items()}
        with torch.no_grad():
            theta_scaled, _ = model.sample_and_log_prob_batch(context=ctx, tolerance=1e-3)
        th = scaler.inverse_array(theta_scaled.cpu().numpy())
        posteriors[tag] = th
        medians[tag] = {
            nm: [float(np.quantile(th[:, k], q)) for q in (0.16, 0.50, 0.84)]
            for k, nm in enumerate(names)
        }
        print(f"  sampled {tag:<28} ({int(cov_a.sum())}/{len(cov_a)} bins)  "
              f"Rp={medians[tag]['planet_radius'][1]:.4f}  T={medians[tag]['T'][1]:.1f}K  "
              f"CH4={medians[tag]['log_CH4'][1]:.2f}  CO2={medians[tag]['log_CO2'][1]:.2f}")

    w = np.full(n_samples, 1.0 / n_samples)
    g_tags = [t for t in posteriors if "niriss" not in t]
    pair_js = {}
    for i in range(len(g_tags)):
        for j in range(i + 1, len(g_tags)):
            a, b = g_tags[i], g_tags[j]
            per_p = {
                nm: float(js_1d(posteriors[a][:, k], w, posteriors[b][:, k], w))
                for k, nm in enumerate(names)
            }
            pair_js[f"{a}__vs__{b}"] = {
                "mean_js_nats": float(np.mean(list(per_p.values()))),
                "per_param_js": per_p,
            }

    mean_vals = [v["mean_js_nats"] for v in pair_js.values()]
    res = {
        "n_samples": n_samples,
        "checkpoint": str(ckpt),
        "stats_source": str(stats_path),
        "noise_conditioning_js_ref_nats": NOISE_COND_JS_REF,
        "medians_16_50_84": medians,
        "pairwise_g395h_js": pair_js,
        "max_pair_js_nats": float(max(mean_vals)),
        "median_pair_js_nats": float(np.median(mean_vals)),
        "exceeds_noise_cond_js": bool(max(mean_vals) > NOISE_COND_JS_REF),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "h7_posterior_js.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n  H7 max pairwise G395H JS = {res['max_pair_js_nats']:.4f} nats "
          f"(median {res['median_pair_js_nats']:.4f}) vs noise-cond JS = {NOISE_COND_JS_REF:.4f} nats")
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectral", action="store_true", help="Run canonical parsing + spectral pipeline-shift metrics")
    ap.add_argument("--run-flow", action="store_true", help="Run K2-18b FMPE flow across reductions (needs norm stats)")
    ap.add_argument("--stats", default="data/abc_rad_k218/abc_rad_train.hdf",
                    help="Path to abc_rad_train.hdf or exported abc_rad_k218_norm_stats.npz")
    ap.add_argument("--n-samples", type=int, default=2000)
    args = ap.parse_args()
    if not args.spectral and not args.run_flow:
        args.spectral = True
    if args.spectral:
        run_spectral_analysis()
    if args.run_flow:
        run_flow_h7(Path(args.stats), n_samples=args.n_samples)
