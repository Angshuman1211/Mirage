"""V2 inferential half (closes G8): synthetic posterior quality + real WASP-39b posteriors
across all cheap baselines, structured randomisation, and CycleGAN.

Evaluates the 6-parameter ABC arms on:
  1. Held-out `data/abc/abc_test.hdf` (50 planets x 1000 samples, test_seed=1234,
     identical correlated noise per planet — exact P6-D3 protocol):
     coverage@68%, coverage@95%, normalised width@68%, log-density at truth,
     and per-sigma tertiles.
  2. Real WASP-39b (`data/jwst_wasp39b_spectrum.csv` + 100 real PRISM OOT frames,
     1000 unweighted draws per arm): posterior medians/68% intervals, pairwise JS
     to the untranslated control, and — whenever `data/real_ess/wasp39b_ns_posterior_fitrad.npz`
     is present — exact IS-free 1-D JS to the TauREx nested-sampling anchor.

Prints the unified V2 table combining BOTH distributional MMD values (adapted +
no-adaptation baseline, from P6-D13 `v2_baselines_seed*.json`) beside the
inferential metrics, and persists `configs/v2_baselines_wasp39/v2_inferential.json`.

Usage:
    python scripts/v2_inferential_eval.py [--n-planets 50] [--n-samples 1000] [--n-real 1000]
    python scripts/v2_inferential_eval.py --score-ns-only   # recompute JS-to-NS from saved real npzs
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import mirage  # noqa: F401
from mirage.datasets.real_jwst import abc_grid_and_norm, build_observation, rebin_spectrum
from mirage.datasets.transforms import InjectCorrelatedNoise
from mirage.eval.coverage import coverage

from fm4ar.datasets.theta_scalers import get_theta_scaler
from fm4ar.models.build_model import build_model
from real_compare import THETA_LAYOUT, compare as compare_vs_ns, js_1d
from real_ess import _real_oot_frames

ABC_TEST = Path("data/abc/abc_test.hdf")
WASP39_CSV = Path("data/jwst_wasp39b_spectrum.csv")
REAL_ESS_DIR = Path("data/real_ess")
NS_ANCHOR = REAL_ESS_DIR / "wasp39b_ns_posterior_fitrad.npz"
V2_DIR = Path("configs/v2_baselines_wasp39")
OUT_JSON = V2_DIR / "v2_inferential.json"

# Every arm evaluated in V2's inferential half. All `+cov` arms share identical
# architecture, seed 42, and correlated-noise injection; only the training dataset differs.
V2_ARMS: list[tuple[str, Path, bool, str]] = [
    # (label, ckpt_dir, uses_cov_oot, p6d13_key)
    ("no_adapt_clean", Path("configs/transformer_abc"), False, "sim (no adaptation)"),
    ("struct_rand_cov", Path("configs/h9_abc_cov_s42"), True, "domain_random"),
    ("affine_data_cov", Path("configs/v2_affine_cov"), True, "affine"),
    ("quantile_cov", Path("configs/v2_quantile_cov"), True, "quantile"),
    ("reweight_cov", Path("configs/v2_reweight_cov"), True, "reweight"),
    ("cyclegan_cov", Path("configs/cyclegan_posterior_cov"), True, "cyclegan"),
]

# Optional P6-D3 reproduction control (CPU-trained noisecond_cov)
P6D3_CTRL = ("p6d3_ctrl_cov", Path("configs/noisecond_cov"), True, "domain_random")


def load_arm(ckpt_dir: Path, device: str = "cpu"):
    ckpt = ckpt_dir / "model__best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found")
    model = build_model(file_path=ckpt, experiment_dir=ckpt_dir, device=device)
    model.network.eval()
    with open(ckpt_dir / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    return model, get_theta_scaler(cfg.get("theta_scaler", {})), cfg


def noise_transform(ref_dir: Path, test_seed: int) -> InjectCorrelatedNoise:
    with open(ref_dir / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    kw = cfg["training"]["stage_0"]["data_transforms"][0]["kwargs"]
    kw = {
        **kw,
        "config": {
            **kw["config"],
            "kwargs": {**kw["config"]["kwargs"], "random_seed": test_seed},
        },
    }
    return InjectCorrelatedNoise(**kw)


@torch.no_grad()
def sample_posterior(model, scaler, ctx_single: dict, n_samples: int, device: str = "cpu") -> np.ndarray:
    ctx = {k: v.to(device).unsqueeze(0).expand(n_samples, *v.shape) for k, v in ctx_single.items()}
    theta_scaled = model.sample_batch(context=ctx, tolerance=1e-3)
    return scaler.inverse_array(theta_scaled.cpu().numpy())


@torch.no_grad()
def logdensity_at_truth(model, scaler, ctx_single: dict, theta_true: np.ndarray, device: str = "cpu") -> float:
    t_th = torch.from_numpy(np.asarray(theta_true)).float().unsqueeze(0)
    theta_scaled = scaler.forward_tensor(t_th).to(device)
    ctx = {k: v.to(device).unsqueeze(0) for k, v in ctx_single.items()}
    logq = model.log_prob_batch(theta_scaled, context=ctx, tolerance=1e-3)
    return float(logq.reshape(-1)[0])


def norm_width68(samples_list: list[np.ndarray], scaler) -> float:
    w = []
    for s in samples_list:
        lo = np.quantile(s, 0.16, axis=0)
        hi = np.quantile(s, 0.84, axis=0)
        w.append(scaler.forward_array(hi) - scaler.forward_array(lo))
    return float(np.mean(w))


def load_p6d13_mmd() -> tuple[dict[str, dict[str, float]], float]:
    """Mean +- sd held-out MMD & KL across the 3 committed P6-D13 seeds.

    Prefers the fixed-basis rescoring (P6-D33). The pre-fix files fitted the PCA
    basis on [sim, real, cyclegan, randomised], so two of the six arms sat inside
    the space all six were scored in. The originals stay on disk as the record of
    what was published, but a stale MMD column beside a current coverage column
    would be worse than either basis used consistently.
    """
    seeds = sorted(V2_DIR.glob("v2_baselines_fixedbasis_seed*.json"))
    if not seeds:
        seeds = sorted(V2_DIR.glob("v2_baselines_seed*.json"))
        if seeds:
            print("  [warn] no v2_baselines_fixedbasis_seed*.json found, falling back to the\n"
                  "         pre-P6-D33 basis. The MMD column below is the contaminated one.\n"
                  "         Regenerate with: python scripts/v2_cheap_baselines.py --oot wasp39\n"
                  "         --seed {0,1,2} --out .../v2_baselines_fixedbasis_seed{0,1,2}.json")
    if not seeds:
        return {}, float("nan")
    by_arm: dict[str, list[dict]] = {}
    for sp in seeds:
        h = json.load(open(sp))["heldout"]
        for k, v in h.items():
            by_arm.setdefault(k, []).append(v)
    out = {}
    for k, vals in by_arm.items():
        mmds = [v["mmd_to_real"] for v in vals]
        kls = [v["kl_to_real"] for v in vals]
        out[k] = {
            "mmd_mean": float(np.mean(mmds)),
            "mmd_sd": float(np.std(mmds, ddof=1)) if len(mmds) > 1 else 0.0,
            "kl_mean": float(np.mean(kls)),
        }
    mmd_none = out["sim (no adaptation)"]["mmd_mean"]
    return out, mmd_none


def eval_synthetic(arms_spec: list[tuple[str, Path, bool, str]],
                   n_planets: int, n_samples: int, test_seed: int) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {}
    for name, d, uses_cov, _ in arms_spec:
        try:
            loaded[name] = (*load_arm(d, device=device), uses_cov)
        except FileNotFoundError:
            print(f"  [skip] {name}: {d / 'model__best.pt'} missing")

    inject = noise_transform(Path("configs/h9_abc_cov_s42"), test_seed)

    cache_path = V2_DIR / f"synthetic_cache_n{n_planets}_s{test_seed}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        samples = cached["samples"]
        logdens = cached["logdens"]
        truths = cached["truths"]
        planet_sigma = cached["planet_sigma"]
        start_idx = len(truths)
        print(f"  [cache] Loaded {start_idx}/{n_planets} planets from {cache_path}")
    else:
        samples = {nm: [] for nm in loaded}
        logdens = {nm: [] for nm in loaded}
        truths = []
        planet_sigma = []
        start_idx = 0

    with h5py.File(ABC_TEST, "r") as f:
        wlen = f["wlen"][0].astype(np.float32)
        n = min(n_planets, len(f["theta"]))
        print(f"\n[1/2] Synthetic held-out abc_test.hdf: {n} planets x {n_samples} samples, "
              f"{len(loaded)} arms (seed={test_seed}, device={device})")
        for idx in tqdm(range(start_idx, n)):
            flux = f["flux"][idx].astype(np.float32)
            theta_true = f["theta"][idx].astype(np.float64)
            truths.append(theta_true)

            gen = inject.noise_generator
            sigma_mat, params = gen.sample_covariance(wlen, return_params=True)
            planet_sigma.append(float(params["sigma"]))
            noisy = flux + gen.sample_noise_from_covariance(sigma_mat)
            err = np.sqrt(np.diag(sigma_mat)).astype(np.float32)
            oot = gen.sample_oot_frames(sigma_mat, inject.n_oot_frames)

            ctx_cov = {
                "flux": torch.from_numpy(noisy).float(),
                "wlen": torch.from_numpy(wlen).float(),
                "error_bars": torch.from_numpy(err).float(),
                "oot_frames": torch.from_numpy(oot).float(),
            }
            ctx_clean_style = {
                "flux": torch.from_numpy(noisy).float(),
                "wlen": torch.from_numpy(wlen).float(),
            }

            for name, (model, scaler, _cfg, uses_cov) in loaded.items():
                ctx = ctx_cov if uses_cov else ctx_clean_style
                samples[name].append(sample_posterior(model, scaler, ctx, n_samples, device=device))
                logdens[name].append(logdensity_at_truth(model, scaler, ctx, theta_true, device=device))

            if (idx + 1) % 5 == 0 or (idx + 1) == n:
                V2_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as fh:
                    pickle.dump({"samples": samples, "logdens": logdens, "truths": truths, "planet_sigma": planet_sigma}, fh)

    sig = np.asarray(planet_sigma)
    binned = np.digitize(sig, np.quantile(sig, [1 / 3, 2 / 3]))
    out = {}
    for name, (model, scaler, cfg, _uses_cov) in loaded.items():
        cov = coverage(samples[name], truths, levels=(0.68, 0.95))
        w68 = norm_width68(samples[name], scaler)
        ld_arr = np.asarray(logdens[name])
        tertiles = {}
        for b, lab in enumerate(("low_sigma", "mid_sigma", "high_sigma")):
            m = binned == b
            tertiles[lab] = float(ld_arr[m].mean()) if m.any() else float("nan")
        out[name] = {
            "cov68": float(cov.overall[0]),
            "cov95": float(cov.overall[1]),
            "cov68_per_param": [float(x) for x in cov.per_dim[0]],
            "cov95_per_param": [float(x) for x in cov.per_dim[1]],
            "width68": float(w68),
            "logdens": float(ld_arr.mean()),
            "logdens_se": float(ld_arr.std(ddof=1) / np.sqrt(len(ld_arr))),
            "logdens_tertiles": tertiles,
            "device_trained": cfg.get("local", {}).get("device", "?"),
        }
    return out


def eval_real_wasp39(arms_spec: list[tuple[str, Path, bool, str]],
                     n_real: int, score_ns_only: bool = False) -> dict:
    REAL_ESS_DIR.mkdir(parents=True, exist_ok=True)
    ctx_base, covered = build_observation(str(WASP39_CSV), str(ABC_TEST))
    oot_real = _real_oot_frames(100)
    wl_s, _, _ = abc_grid_and_norm(str(ABC_TEST))
    order = np.argsort(wl_s)
    inv = np.argsort(order)
    depth_a, err_a, _ = rebin_spectrum(pd.read_csv(WASP39_CSV), wl_s[order])
    x_obs = depth_a[inv].astype(np.float64)
    sig_obs = err_a[inv].astype(np.float64)

    names = THETA_LAYOUT[6]
    thetas: dict[str, np.ndarray] = {}
    out: dict[str, dict] = {}

    print(f"\n[2/2] Real WASP-39b posteriors (N={n_real} unweighted draws per arm)")
    for name, d, uses_cov, _ in arms_spec:
        npz_path = REAL_ESS_DIR / f"v2_{name}_samples.npz"
        if not score_ns_only:
            if not (d / "model__best.pt").exists():
                print(f"  [skip] {name}: checkpoint missing")
                continue
            model, scaler, _ = load_arm(d)
            ctx_np = dict(ctx_base)
            if uses_cov:
                ctx_np["oot_frames"] = oot_real
            else:
                ctx_np = {"flux": ctx_base["flux"], "wlen": ctx_base["wlen"]}
            ctx = {
                k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).expand(n_real, *np.shape(v))
                for k, v in ctx_np.items()
            }
            with torch.no_grad():
                theta_scaled, log_q = model.sample_and_log_prob_batch(context=ctx, tolerance=1e-3)
            th = scaler.inverse_array(theta_scaled.cpu().numpy())
            np.savez(
                npz_path,
                theta=th,
                log_q=log_q.cpu().numpy().reshape(-1),
                x_obs=x_obs,
                sig_obs=sig_obs,
                covered=covered,
                wlen=wl_s,
                arm=f"v2_{name}",
                planet="wasp39",
            )
        else:
            if not npz_path.exists():
                continue
            th = np.load(npz_path)["theta"]

        thetas[name] = th
        quants = {
            nm: [float(np.quantile(th[:, k], q)) for q in (0.16, 0.50, 0.84)]
            for k, nm in enumerate(names)
        }
        entry = {
            "samples_npz": str(npz_path),
            "n_draws": int(len(th)),
            "quantiles_16_50_84": quants,
            "js_to_ns": None,
            "js_to_ns_per_param": None,
        }
        if NS_ANCHOR.exists():
            rows, mean_js = compare_vs_ns(npz_path, NS_ANCHOR, quiet=True)
            entry["js_to_ns"] = float(mean_js)
            entry["js_to_ns_per_param"] = {nm: float(js) for nm, _, _, js in rows}
        out[name] = entry
        t_med = quants["T"][1]
        h2o_med = quants["log_H2O"][1]
        js_str = f"JS_NS={entry['js_to_ns']:.3f}" if entry["js_to_ns"] is not None else "JS_NS=(anchor absent)"
        print(f"  {name:<18} T={t_med:7.1f}K  logH2O={h2o_med:+6.2f}  {js_str}")

    # Pairwise JS to untranslated control (struct_rand_cov) so real-data shift is measured
    # even when the external NS anchor file is absent.
    if "struct_rand_cov" in thetas:
        ref = thetas["struct_rand_cov"]
        w_ref = np.full(len(ref), 1.0 / len(ref))
        for name, th in thetas.items():
            w_th = np.full(len(th), 1.0 / len(th))
            per_p = {
                nm: float(js_1d(ref[:, k], w_ref, th[:, k], w_th))
                for k, nm in enumerate(names)
            }
            out[name]["js_to_struct_rand"] = float(np.mean(list(per_p.values())))
            out[name]["js_to_struct_rand_per_param"] = per_p

    return out


def print_unified_table(arms_spec, mmd_map, mmd_none, synth, real) -> None:
    print(f"\n{'=' * 108}")
    print("  V2 UNIFIED TABLE — Distributional (P6-D13 held-out) + Inferential (abc_test & real WASP-39b)")
    print(f"{'=' * 108}")
    print(
        f"  {'arm':<18}{'MMD':>8}{'MMD_none':>10}{'cov@68':>8}{'cov@95':>8}"
        f"{'width68':>9}{'logdens':>10}{'JS_ctrl':>9}{'JS_NS':>9}"
    )
    print("  " + "-" * 104)
    for name, _, _, p6k in arms_spec:
        if name not in synth and name not in real:
            continue
        mmd = mmd_map.get(p6k, {}).get("mmd_mean", float("nan"))
        s = synth.get(name, {})
        r = real.get(name, {})
        js_ctrl = r.get("js_to_struct_rand")
        js_ns = r.get("js_to_ns")
        c68 = f"{s['cov68']:.3f}" if "cov68" in s else "--"
        c95 = f"{s['cov95']:.3f}" if "cov95" in s else "--"
        w68 = f"{s['width68']:.3f}" if "width68" in s else "--"
        ld = f"{s['logdens']:+.2f}" if "logdens" in s else "--"
        jsc = f"{js_ctrl:.3f}" if js_ctrl is not None else "--"
        jsn = f"{js_ns:.3f}" if js_ns is not None else "pending*"
        print(f"  {name:<18}{mmd:>8.4f}{mmd_none:>10.4f}{c68:>8}{c95:>8}{w68:>9}{ld:>10}{jsc:>9}{jsn:>9}")
    print("  " + "-" * 104)
    print(f"  * Both MMD and MMD_none (no-adaptation {mmd_none:.4f}) are printed per V2 acceptance.")
    print("  * affine in per-bin standardised space is identical to struct_rand_cov; affine_data_cov maps into")
    print("    real WASP-39b's ABC-normalised depth space (mean +2.16, median sd 0.015).")
    if not NS_ANCHOR.exists():
        print(f"  * {NS_ANCHOR} is not present locally; drop it in and run `--score-ns-only` to populate JS_NS.")
    missing = [n for n, d, _, _ in arms_spec if not (d / "model__best.pt").exists()]
    if missing:
        print(f"  * NO TRAINED CHECKPOINT for {', '.join(missing)}. The real-WASP-39b columns")
        print("    cannot be recomputed on this machine, so JS_NS is blocked on the arms, not")
        print("    the anchor. Commit the model__best.pt files or rerun where they live.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-planets", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-real", type=int, default=1000)
    ap.add_argument("--test-seed", type=int, default=1234)
    ap.add_argument("--include-p6d3-ctrl", action="store_true",
                    help="Also run CPU noisecond_cov to verify exact P6-D3 reproduction")
    ap.add_argument("--score-ns-only", action="store_true",
                    help="Skip sampling; recompute JS-to-NS from existing data/real_ess/v2_*_samples.npz")
    args = ap.parse_args()

    arms_spec = list(V2_ARMS)
    if args.include_p6d3_ctrl:
        arms_spec.append(P6D3_CTRL)

    mmd_map, mmd_none = load_p6d13_mmd()
    if args.score_ns_only and OUT_JSON.exists():
        prev = json.load(open(OUT_JSON))
        synth = prev.get("synthetic_abc_test", {})
    else:
        synth = eval_synthetic(arms_spec, args.n_planets, args.n_samples, args.test_seed)
    real = eval_real_wasp39(arms_spec, args.n_real, score_ns_only=args.score_ns_only)

    if args.score_ns_only:
        # SAFETY (P6-D33). --score-ns-only reads the saved per-arm posterior draws from
        # data/real_ess/v2_*_samples.npz, which are gitignored. On a machine without them
        # every arm is skipped, `real` comes back empty, and writing it out destroys the
        # committed table, JS_ctrl included. That happened once. Merge into the previous
        # payload instead of replacing it, and refuse to write if nothing was recovered.
        prev_real = prev.get("real_wasp39b", {}) if OUT_JSON.exists() else {}
        if not real:
            print("\n  [abort] --score-ns-only recovered no arms. The per-arm draws in\n"
                  f"          {REAL_ESS_DIR}/v2_*_samples.npz are absent, so there is nothing to\n"
                  "          rescore. Refusing to write, because doing so would erase the\n"
                  f"          existing table in {OUT_JSON}. Nothing was changed.")
            return
        merged = dict(prev_real)
        for arm, entry in real.items():
            keep = dict(merged.get(arm, {}))
            keep.update({k: v for k, v in entry.items() if v is not None})
            merged[arm] = keep
        dropped = [a for a in prev_real if a not in real]
        if dropped:
            print(f"  [note] kept previous values for {', '.join(dropped)}, not rescored this run.")
        real = merged
    print_unified_table(arms_spec, mmd_map, mmd_none, synth, real)

    payload = {
        "protocol": {
            "n_planets": args.n_planets,
            "n_samples": args.n_samples,
            "n_real_draws": args.n_real,
            "test_seed": args.test_seed,
            "ns_anchor_present": NS_ANCHOR.exists(),
        },
        "distributional_p6d13": mmd_map,
        "mmd_no_adaptation": mmd_none,
        "synthetic_abc_test": synth,
        "real_wasp39b": real,
    }
    V2_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Saved unified V2 inferential results -> {OUT_JSON}")


if __name__ == "__main__":
    main()
