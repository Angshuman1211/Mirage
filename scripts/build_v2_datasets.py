"""Build precomputed HDF5 training sets for V2's inferential half (closes G8).

Mirrors `scripts/build_translated_abc.py` and P6-D3's single-line-diff protocol:
each cheap baseline produces a deterministic HDF5 copy of `data/abc/abc_train.hdf`
so the downstream FMPE config is IDENTICAL to `configs/h9_abc_cov_s42/config.yaml`
(same architecture, seed 42, correlated-noise injection, CUDA backend) with ONLY
`dataset.file_path` changed.

Strict anti-leakage design:
  1. Real WASP-39b frames (`data/real_domain/real_domain_wasp39_abc.npz`) are
     sorted by `frame_ids` (integration order) and split contiguously in half;
     ONLY the first half (1,620 frames) is used to fit every baseline, matching
     `scripts/v2_cheap_baselines.py` (P6-D13).
  2. The train/val split of `fm4ar` (`n_train=64000, n_valid=8000`, seed 42 over
     the first 72,000 rows of `abc_train.hdf`) is replicated exactly. Every map
     and classifier is fit on `train_idx` ONLY — never on `val_idx` or
     `abc_test.hdf`.
  3. For `reweight`, only `train_idx` slots are importance-resampled (with
     replacement) from `train_idx` using density-ratio weights `w = p/(1-p)`;
     the 8,000 `val_idx` rows are left untouched so zero resampled rows leak
     into validation / early stopping.

Three HDF5 outputs under `data/abc/` (gitignored):
  - `abc_v2_quantile_train.hdf`  per-bin empirical-CDF transfer in standardised
                                 space (the exact map that scored MMD 0.106 in
                                 P6-D13, testing whether its distributional win
                                 transfers to inference).
  - `abc_v2_affine_train.hdf`    per-bin affine map from ABC standardised flux
                                 into real WASP-39b's ABC-normalised depth
                                 space (mean ~+2.16, median sd ~0.015). Note:
                                 in per-bin standardised space affine is the
                                 identity map (h9_abc_cov_s42), as P6-D13 proved.
  - `abc_v2_reweight_train.hdf`  OmniFold-style classifier density-ratio
                                 importance resampling on `train_idx` only;
                                 every (flux, theta) pair remains an unmodified
                                 TauREx simulation.

Run:
    python scripts/build_v2_datasets.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from torch.utils.data import random_split

ABC_TRAIN = Path("data/abc/abc_train.hdf")
REAL_DOMAIN = Path("data/real_domain/real_domain_wasp39_abc.npz")
SUMMARY_OUT = Path("configs/v2_baselines_wasp39/v2_dataset_build.json")

N_TRAIN = 64000
N_VALID = 8000
SPLIT_SEED = 42


def standardise(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    ref = x if ref is None else ref
    return (x - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-8)


def quantile_map(sim: np.ndarray, sim_ref: np.ndarray, real_fit: np.ndarray) -> np.ndarray:
    """Per-bin empirical-CDF transfer matching `scripts/v2_cheap_baselines.py`."""
    out = np.empty_like(sim)
    q = np.linspace(0.0, 1.0, 1001)
    for k in range(sim.shape[1]):
        sq = np.quantile(sim_ref[:, k], q)
        rq = np.quantile(real_fit[:, k], q)
        u = np.interp(sim[:, k], sq, q, left=0.0, right=1.0)
        out[:, k] = np.interp(u, q, rq)
    return out


def affine_map(sim: np.ndarray, sim_ref: np.ndarray, target_ref: np.ndarray) -> np.ndarray:
    """Per-bin affine map fit on (sim_ref -> target_ref), applied to sim."""
    a = target_ref.std(axis=0) / (sim_ref.std(axis=0) + 1e-12)
    return (sim - sim_ref.mean(axis=0)) * a + target_ref.mean(axis=0)


def fm4ar_split_indices(n_train: int = N_TRAIN, n_valid: int = N_VALID,
                        seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Exact index split produced by `fm4ar.torchutils.dataloaders.build_dataloaders`."""
    tr, va = random_split(
        range(n_train + n_valid),
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(seed),
    )
    tr_idx = np.asarray(list(tr), dtype=np.int64)
    va_idx = np.asarray(list(va), dtype=np.int64)
    assert len(set(tr_idx.tolist()) & set(va_idx.tolist())) == 0
    return tr_idx, va_idx


def main() -> None:
    if not ABC_TRAIN.exists():
        raise SystemExit(f"Missing {ABC_TRAIN}")
    if not REAL_DOMAIN.exists():
        raise SystemExit(
            f"Missing {REAL_DOMAIN} — run `python scripts/build_real_domain.py --target wasp39 --grid abc`"
        )

    # 1. Load real WASP-39b fit half (contiguous first half by integration_num)
    rd = np.load(REAL_DOMAIN, allow_pickle=True)
    real_raw = rd["samples"].astype(np.float64)
    frames = np.asarray(rd["frame_ids"])
    order = np.argsort(frames)
    half = len(order) // 2
    fit_idx = order[:half]
    real_fit_raw = real_raw[fit_idx]
    real_fit_std = standardise(real_fit_raw)

    # 2. Load ABC training set + exact fm4ar train/val indices
    with h5py.File(ABC_TRAIN, "r") as f:
        flux_all = f["flux"][:].astype(np.float64)
        theta_all = f["theta"][:].astype(np.float32)
        noise_all = f["noise"][:].astype(np.float32)
        pid_all = f["planet_id"][:]
        flux_mean = f["flux_mean"][:].astype(np.float64)
        flux_std = f["flux_std"][:].astype(np.float64)

    tr_idx, va_idx = fm4ar_split_indices()
    sim_tr = flux_all[tr_idx]
    real_fit_abc = (real_fit_raw - flux_mean) / flux_std

    summary: dict = {
        "source_hdf": str(ABC_TRAIN),
        "real_domain": str(REAL_DOMAIN),
        "real_fit_frames": int(half),
        "real_total_frames": int(len(real_raw)),
        "split_rule": "contiguous first half by integration_num",
        "fm4ar_split": {"n_train": N_TRAIN, "n_valid": N_VALID, "seed": SPLIT_SEED},
        "arms": {},
    }

    # ── Arm 1: per-bin quantile mapping (standardised space, P6-D13 winner) ──
    print("[1/3] Building abc_v2_quantile_train.hdf ...")
    q_flux = quantile_map(flux_all, sim_tr, real_fit_std).astype(np.float32)
    assert np.isfinite(q_flux).all(), "Non-finite values in quantile flux"
    dst_q = ABC_TRAIN.parent / "abc_v2_quantile_train.hdf"
    shutil.copyfile(ABC_TRAIN, dst_q)
    with h5py.File(dst_q, "r+") as f:
        f["flux"][...] = q_flux
        f.attrs["v2_arm"] = "quantile"
        f.attrs["real_domain"] = str(REAL_DOMAIN)
        f.attrs["real_fit_frames"] = int(half)
    dq = np.abs(q_flux - flux_all)
    summary["arms"]["quantile"] = {
        "hdf": str(dst_q),
        "space": "per-bin standardised (matches P6-D13 MMD=0.1056)",
        "mean_abs_delta": float(dq.mean()),
        "mean_shift": float(q_flux.mean() - flux_all.mean()),
        "flux_mean_range": [float(q_flux.mean(0).min()), float(q_flux.mean(0).max())],
        "flux_std_range": [float(q_flux.std(0).min()), float(q_flux.std(0).max())],
    }
    print(f"  done: mean|delta|={dq.mean():.4f}  flux_mean={q_flux.mean():+.4f}  flux_std={q_flux.std():.4f}")

    # ── Arm 2: per-bin affine map in ABC data space ──
    print("[2/3] Building abc_v2_affine_train.hdf ...")
    a_flux = affine_map(flux_all, sim_tr, real_fit_abc).astype(np.float32)
    assert np.isfinite(a_flux).all(), "Non-finite values in affine flux"
    dst_a = ABC_TRAIN.parent / "abc_v2_affine_train.hdf"
    shutil.copyfile(ABC_TRAIN, dst_a)
    with h5py.File(dst_a, "r+") as f:
        f["flux"][...] = a_flux
        f.attrs["v2_arm"] = "affine_data_space"
        f.attrs["real_domain"] = str(REAL_DOMAIN)
        f.attrs["real_fit_frames"] = int(half)
    da = np.abs(a_flux - flux_all)
    summary["arms"]["affine"] = {
        "hdf": str(dst_a),
        "space": "ABC-normalised depth space (standardised-space affine is identity = h9_abc_cov_s42)",
        "mean_abs_delta": float(da.mean()),
        "mean_shift": float(a_flux.mean() - flux_all.mean()),
        "flux_mean_range": [float(a_flux.mean(0).min()), float(a_flux.mean(0).max())],
        "flux_std_range": [float(a_flux.std(0).min()), float(a_flux.std(0).max())],
        "flux_std_median": float(np.median(a_flux.std(0))),
    }
    print(f"  done: mean|delta|={da.mean():.4f}  flux_mean={a_flux.mean():+.4f}  median_bin_std={np.median(a_flux.std(0)):.4f}")

    # ── Arm 3: OmniFold-style classifier density-ratio reweighting (train_idx ONLY) ──
    print("[3/3] Building abc_v2_reweight_train.hdf (fitting HistGradientBoostingClassifier on 64k train rows) ...")
    sim_tr_std = standardise(sim_tr)
    X = np.vstack([sim_tr_std, real_fit_std])
    y = np.r_[np.zeros(len(sim_tr_std)), np.ones(len(real_fit_std))]
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=SPLIT_SEED
    )
    auc = float(np.mean(cross_val_score(clf, X, y, cv=3, scoring="roc_auc")))
    clf.fit(X, y)
    p = np.clip(clf.predict_proba(sim_tr_std)[:, 1], 1e-6, 1 - 1e-6)
    w = (p / (1.0 - p)) * (len(sim_tr_std) / len(real_fit_std))
    probs = w / w.sum()
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    max_share = float(probs.max())

    rng = np.random.default_rng(SPLIT_SEED)
    drawn_local = rng.choice(len(tr_idx), size=len(tr_idx), replace=True, p=probs)
    drawn_global = tr_idx[drawn_local]

    # Verify zero overlap between resampled train slots and untouched val slots
    assert len(set(drawn_global.tolist()) & set(va_idx.tolist())) == 0, "Train/val leakage!"
    n_unique = int(len(np.unique(drawn_global)))

    rw_flux = flux_all.astype(np.float32).copy()
    rw_theta = theta_all.copy()
    rw_noise = noise_all.copy()
    rw_pid = pid_all.copy()

    rw_flux[tr_idx] = flux_all[drawn_global].astype(np.float32)
    rw_theta[tr_idx] = theta_all[drawn_global]
    rw_noise[tr_idx] = noise_all[drawn_global]
    rw_pid[tr_idx] = pid_all[drawn_global]

    dst_r = ABC_TRAIN.parent / "abc_v2_reweight_train.hdf"
    shutil.copyfile(ABC_TRAIN, dst_r)
    with h5py.File(dst_r, "r+") as f:
        f["flux"][...] = rw_flux
        f["theta"][...] = rw_theta
        f["noise"][...] = rw_noise
        f["planet_id"][...] = rw_pid
        f.attrs["v2_arm"] = "reweight"
        f.attrs["real_domain"] = str(REAL_DOMAIN)
        f.attrs["real_fit_frames"] = int(half)
        f.attrs["classifier_auc_cv3"] = auc
        f.attrs["weight_ess"] = ess
        f.attrs["max_weight_share"] = max_share
        f.attrs["n_unique_train_rows"] = n_unique

    summary["arms"]["reweight"] = {
        "hdf": str(dst_r),
        "space": "unmodified TauREx (flux, theta), train_idx resampled by classifier density ratio; val_idx untouched",
        "classifier_auc_cv3": auc,
        "weight_ess": ess,
        "weight_ess_pct": float(100.0 * ess / len(tr_idx)),
        "max_weight_share_pct": float(100.0 * max_share),
        "n_unique_train_rows": n_unique,
        "train_val_planet_overlap": int(len(set(rw_pid[tr_idx].tolist()) & set(rw_pid[va_idx].tolist()))),
        "flux_mean_after": float(rw_flux[tr_idx].mean()),
        "flux_std_after": float(rw_flux[tr_idx].std()),
    }
    print(
        f"  done: AUC={auc:.6f}  ESS={ess:.1f}/{len(tr_idx)} ({100*ess/len(tr_idx):.2f}%)  "
        f"unique={n_unique}  max_share={100*max_share:.3f}%  val_overlap=0"
    )

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved summary -> {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
