"""
Phase 3 Track 2 — CycleGAN domain translation training.

Trains a sim-to-real CycleGAN on:
  Domain A: normalised ABC synthetic spectra (52 bins, from abc_train.hdf)
  Domain B: real JWST WASP-39b spectra (52 bins, from the standardised CSV)

Writes checkpoints to configs/cyclegan/ and a training log to
configs/cyclegan/training_log.csv.

Domain B must be a real domain built from genuine frames (V7, 2026-11-15). The
single-spectrum bootstrap real side is retired and reachable only with
--legacy-bootstrap, as a negative control.

Usage:
    python scripts/train_cyclegan.py --real-domain <npz> --ckpt-dir <dir> [options]

Example (quick smoke test):
    python scripts/train_cyclegan.py --epochs 5 --batch-size 32 --n-abc 500 \
        --real-domain data/real_domain/real_domain_wasp39_abc.npz --ckpt-dir configs/_smoke

V7, all three real spectra pooled on the shared NIRISS grid:
    bash scripts/run_v7_pooled.sh
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import CycleGAN directly by file path — avoids fm4ar dependency
# (mirage/__init__.py auto-calls register() which needs fm4ar;
#  train_cyclegan.py only needs the nn module, not the full registration)
import importlib.util as _ilu
import pathlib as _pl

_cg_path = _pl.Path(__file__).resolve().parents[1] / "mirage" / "nn" / "cyclegan.py"
_spec = _ilu.spec_from_file_location("mirage.nn.cyclegan", _cg_path)
_cg_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cg_mod)
CycleGAN = _cg_mod.CycleGAN

# ── Defaults ───────────────────────────────────────────────────────────────
ABC_DIR    = Path("data/abc")
REAL_CSV   = Path("MAST_2026-05-11T1524/MAST_2026-05-11T1524/JWST/WASP39b_final_standardized.csv")
# Fallback paths using the processed corpus
REAL_CSV_FALLBACK = Path("mirage_processed_corpus_v0.1/WASP39b/WASP39b_final_standardized.csv")
FULL_CSV_FALLBACK = Path("mirage_processed_corpus_v0.1/WASP39b/WASP39b_standardized_full.csv")
CKPT_DIR   = Path("configs/cyclegan")
N_BINS     = 52
LR         = 2e-4
BETAS      = (0.5, 0.999)


# ── Data loading ───────────────────────────────────────────────────────────

def load_abc_spectra(n_samples: int | None = None,
                     cols: np.ndarray | None = None,
                     index_file: str | None = None) -> torch.Tensor:
    """
    Load simulated spectra for domain A.
    Primary: normalised ABC training spectra from abc_train.hdf.
    Fallback: use WASP39b_standardized_full.csv (per-integration spectra)
              as a proxy for domain A when ABC HDF5 is not available locally.
    Returns: (N, n_bins) float32 tensor in [-1, 1] for CycleGAN.

    `cols` selects a subset of the 52 ABC bins, so domain A sits on exactly the
    wavelengths domain B covers. Slicing happens BEFORE the percentile rescale,
    so the scaling is set by the bins actually used.
    """
    path = ABC_DIR / "abc_train.hdf"
    if path.exists():
        with h5py.File(path) as f:
            flux = f["flux"][:]
            if "flux_mean" in f and "flux_std" in f:
                mean = f["flux_mean"][:]
                std  = f["flux_std"][:]
                flux = (flux - mean) / std
        flux = flux.astype(np.float32)
    else:
        # Fallback: use per-integration WASP-39b spectra as domain A proxy
        # (Phase 3 ablation still valid — we're testing translation vs randomisation
        #  on the real instrument's own spectral manifold)
        full_csv = FULL_CSV_FALLBACK if FULL_CSV_FALLBACK.exists() else \
                   Path("MAST_2026-05-11T1524/MAST_2026-05-11T1524/JWST/WASP39b_standardized_full.csv")
        if not full_csv.exists():
            raise FileNotFoundError(
                f"Neither {path} nor {full_csv} found.\n"
                "Run scripts/prepare_abc_hdf5.py to prepare ABC data, or\n"
                "ensure WASP39b_standardized_full.csv is accessible."
            )
        print(f"  [ABC fallback] Using {full_csv.name} as domain A proxy")
        df = pd.read_csv(full_csv)
        wav_col  = next((c for c in df.columns if "wavelength" in c.lower()), None)
        flux_col = next((c for c in df.columns if "flux" in c.lower()
                         and "error" not in c.lower()), None)
        int_col  = next((c for c in df.columns if "int" in c.lower()), None)

        df = df[[c for c in [wav_col, flux_col, int_col] if c]].dropna()
        df.columns = (["wavelength", "flux", "integration"] if int_col
                      else ["wavelength", "flux"])

        # Build per-integration spectra
        if int_col and "integration" in df.columns:
            wav_edges = np.linspace(df["wavelength"].min(),
                                    df["wavelength"].max(), N_BINS + 1)
            df["bin"] = pd.cut(df["wavelength"], bins=wav_edges, labels=False)
            pivot = df.groupby(["integration", "bin"])["flux"].mean().unstack(fill_value=np.nan)
            flux = pivot.values.astype(np.float32)
            # Fill NaN bins with column median
            col_med = np.nanmedian(flux, axis=0)
            nan_mask = np.isnan(flux)
            flux[nan_mask] = np.take(col_med, np.where(nan_mask)[1])
        else:
            # No integration column — bootstrap from single spectrum
            wav_edges = np.linspace(df["wavelength"].min(),
                                    df["wavelength"].max(), N_BINS + 1)
            df["bin"] = pd.cut(df["wavelength"], bins=wav_edges, labels=False)
            single = df.groupby("bin")["flux"].mean().values.astype(np.float32)
            single = np.nan_to_num(single, nan=np.nanmedian(single))
            rng = np.random.default_rng(1)
            noise = rng.standard_normal((2000, N_BINS)) * single.std() * 0.02
            flux = (single[None, :] + noise).astype(np.float32)

    # V3: restrict domain A to rows chosen by v3_planet_neighbourhood.py. Rows
    # are selected BEFORE the percentile rescale, so the scaling is set by the
    # restricted population, exactly as `cols` does for bins.
    if index_file is not None:
        idx = np.asarray(np.load(index_file)["idx"], dtype=int)
        if not path.exists():
            raise SystemExit("[guard] --abc-index needs data/abc/abc_train.hdf; the "
                             "per-integration fallback has no ABC row indices")
        flux = flux[idx]
        print(f"  [domain A] restricted to {len(idx)} rows from {Path(index_file).name}")

    if cols is not None:
        flux = flux[:, np.asarray(cols, dtype=int)]

    # Clip extremes and rescale to [-1, 1] for Tanh output
    p1, p99 = np.percentile(flux, [1, 99])
    flux = np.clip(flux, p1, p99)
    flux = 2.0 * (flux - p1) / (p99 - p1 + 1e-8) - 1.0

    if n_samples is not None:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(flux), min(n_samples, len(flux)), replace=False)
        flux = flux[idx]

    # np.percentile returns float64 scalars, which promote the whole array, and
    # MPS refuses float64 outright. Cast back explicitly.
    return torch.from_numpy(flux.astype(np.float32))


def load_real_spectra(n_bins: int = N_BINS) -> torch.Tensor:
    """RETIRED (V7, 2026-11-15). Reachable only with --legacy-bootstrap.

    Bootstraps ONE spectrum into 500 near-copies with 1% Gaussian jitter, in
    stellar-flux units and ascending wavelength order (P5-D11). Kept only so the
    published configs/cyclegan run stays reproducible as a negative control.
    Never use it as a real domain.
    """
    # Try multiple paths
    candidates = [
        REAL_CSV,
        REAL_CSV_FALLBACK,
        Path("mirage_processed_corpus_v0.1/WASP39b/WASP39b_final_standardized.csv"),
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "WASP39b_final_standardized.csv not found in any expected location."
        )

    df = pd.read_csv(csv_path)

    # Identify wavelength and flux columns flexibly
    wav_col  = next((c for c in df.columns if "wavelength" in c.lower()), None)
    flux_col = next((c for c in df.columns if "flux" in c.lower()
                     and "error" not in c.lower()), None)

    if wav_col is None or flux_col is None:
        raise ValueError(
            f"Cannot find wavelength/flux columns in {REAL_CSV}. "
            f"Columns: {list(df.columns)}"
        )

    df = df[[wav_col, flux_col]].dropna()
    df.columns = ["wavelength", "flux"]
    df = df[df["flux"] > 0].sort_values("wavelength")

    # Bin to the model's n_bins wavelength grid
    wav_edges = np.linspace(df["wavelength"].min(), df["wavelength"].max(), n_bins + 1)
    df["bin"] = pd.cut(df["wavelength"], bins=wav_edges, labels=False)
    binned = df.groupby("bin")["flux"].mean().values.astype(np.float32)

    # Pad/trim to exactly n_bins
    if len(binned) < n_bins:
        binned = np.pad(binned, (0, n_bins - len(binned)), constant_values=np.nanmedian(binned))
    binned = binned[:n_bins]
    binned = np.nan_to_num(binned, nan=np.nanmedian(binned))

    # Expand to a small "dataset" by bootstrap sampling (real data is scarce)
    # This gives the GAN more steps on real data without information fabrication
    rng = np.random.default_rng(0)
    n_real = 500
    noise_scale = binned.std() * 0.01  # 1% photon noise jitter
    real_samples = binned[None, :] + rng.standard_normal((n_real, n_bins)) * noise_scale
    real_samples = real_samples.astype(np.float32)

    # Rescale to [-1, 1]
    p1, p99 = np.percentile(real_samples, [1, 99])
    real_samples = np.clip(real_samples, p1, p99)
    real_samples = 2.0 * (real_samples - p1) / (p99 - p1 + 1e-8) - 1.0

    return torch.from_numpy(real_samples.astype(np.float32))


def real_domain_meta(path: Path) -> tuple[int, np.ndarray]:
    """(n_bins, abc_cols) for a real-domain npz, so domain A can be matched to it."""
    d = np.load(path, allow_pickle=True)
    if "abc_cols" not in d:
        raise SystemExit(
            f"[guard] {path.name} carries no `abc_cols`, so its bins cannot be "
            f"matched to ABC wavelengths and domain A would be paired with "
            f"mismatched bins. Rebuild it with build_real_domain.py.")
    return int(d["samples"].shape[1]), np.asarray(d["abc_cols"], dtype=int)


def real_domain_targets(path: Path) -> np.ndarray | None:
    """Per-sample target labels of a pooled real domain (build_real_domain --target all)."""
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["sample_target"]).astype(str) if "sample_target" in d else None


def load_real_domain_npz(path: Path, n_bins: int) -> torch.Tensor:
    """Domain B from genuinely independent real OOT frames (build_real_domain.py).

    Replaces load_real_spectra, which had three defects (P5-D11):
      - it bootstrapped ONE spectrum into 500 near-copies with 1% Gaussian jitter,
        so the discriminator saw essentially a single example;
      - it read `normalized_flux`, i.e. continuum-normalised STELLAR FLUX, while
        domain A is standardised TRANSIT DEPTH. The two domains were different
        physical quantities, both squashed to [-1,1];
      - it left domain B on the ASCENDING wavelength grid while domain A comes
        from the hdf in STORED (descending) order, so the two domains ran in
        opposite wavelength directions.
    Here domain B is the real reduced transit-depth spectrum carrying real
    per-frame noise, in stored order, already gap-filled at build time.
    """
    d = np.load(path, allow_pickle=True)
    x = d["samples"].astype(np.float32)
    covered = np.asarray(d["covered"], dtype=bool)
    if x.shape[1] != n_bins:
        raise SystemExit(f"[guard] {path.name} has {x.shape[1]} bins, model needs {n_bins}")
    if not np.isfinite(x).all():
        raise SystemExit(f"[guard] {path.name} holds non-finite values; rebuild it")

    p1, p99 = np.percentile(x, [1, 99])          # same final scaling as both loaders
    x = np.clip(x, p1, p99)
    x = 2.0 * (x - p1) / (p99 - p1 + 1e-8) - 1.0
    print(f"  [domain B] {path.name}: {x.shape[0]} real frames, "
          f"{int(covered.sum())}/{len(covered)} bins carry real measurements")
    return torch.from_numpy(x.astype(np.float32))


# ── Training loop ──────────────────────────────────────────────────────────

def train(
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = LR,
    n_abc: int | None = None,
    n_res: int = 9,
    ngf: int = 64,
    ndf: int = 64,
    lambda_cyc: float = 10.0,
    lambda_id: float = 5.0,
    save_every: int = 10,
    device_str: str = "cpu",
    real_domain: str | None = None,
    ckpt_dir: str | None = None,
    legacy_bootstrap: bool = False,
    balance_targets: bool = False,
    abc_index: str | None = None,
    seed: int | None = None,
) -> None:

    # V7: the bootstrapped single-spectrum real side is retired. A real domain
    # built from genuine frames is now required, unless the caller explicitly
    # asks to reproduce the legacy negative control.
    if not real_domain and not legacy_bootstrap:
        raise SystemExit(
            "[guard] --real-domain is required. The bootstrap real side (one spectrum "
            "jittered 500 ways, P5-D11) is retired (V7). Build a real domain with "
            "scripts/build_real_domain.py, or pass --legacy-bootstrap to reproduce "
            "the old negative control on purpose.")
    if real_domain and legacy_bootstrap:
        raise SystemExit("[guard] --real-domain and --legacy-bootstrap are exclusive")
    if seed is not None:                 # V7 acceptance: reproducible from config
        torch.manual_seed(seed)
        np.random.seed(seed)

    global CKPT_DIR                      # P5-D11: never clobber a previous run
    if ckpt_dir:
        CKPT_DIR = Path(ckpt_dir)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    # Domain B sets the grid; domain A is sliced to match it bin for bin.
    n_bins, abc_cols = (real_domain_meta(Path(real_domain)) if real_domain
                        else (N_BINS, None))
    print(f"\n{'='*70}")
    print("  MIRAGE Phase 3 Track 2 — CycleGAN Training")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  λ_cyc:      {lambda_cyc},  λ_id: {lambda_id}")
    print(f"  Bins:       {n_bins}" + ("" if abc_cols is None else
          f" (ABC cols {abc_cols.min()}-{abc_cols.max()} of {N_BINS})"))
    print(f"  Checkpoint: {CKPT_DIR}")

    # Load data
    print("\n  Loading data ...")
    sim_data  = load_abc_spectra(n_samples=n_abc, cols=abc_cols,
                                 index_file=abc_index).to(device)
    if real_domain:
        real_data = load_real_domain_npz(Path(real_domain), n_bins).to(device)
    else:
        print("  !! LEGACY BOOTSTRAP real side: one spectrum jittered 500 ways. "
              "Negative control only (P5-D11, V7).")
        real_data = load_real_spectra(n_bins=n_bins).to(device)
    print(f"  ABC simulated spectra:  {sim_data.shape}")
    print(f"  Real JWST spectra:      {real_data.shape}")

    sim_loader  = DataLoader(TensorDataset(sim_data),  batch_size=batch_size,
                             shuffle=True, drop_last=True)
    targets = real_domain_targets(Path(real_domain)) if real_domain else None
    if balance_targets:
        if targets is None:
            raise SystemExit("[guard] --balance-targets needs a pooled real domain "
                             "(build_real_domain.py --target all)")
        # Equal expected draws per target per epoch. The small targets' frames are
        # reused, but every reused row is a genuine frame, not synthetic jitter.
        names, counts = np.unique(targets, return_counts=True)
        per = dict(zip(names, counts))
        w = torch.tensor([1.0 / per[t] for t in targets], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            w, num_samples=len(targets), replacement=True)
        real_loader = DataLoader(TensorDataset(real_data), batch_size=batch_size,
                                 sampler=sampler, drop_last=True)
        print("  balanced over targets: " +
              ", ".join(f"{n} {c} frames" for n, c in per.items()))
    else:
        real_loader = DataLoader(TensorDataset(real_data), batch_size=batch_size,
                                 shuffle=True, drop_last=True)

    # Build model
    model = CycleGAN(
        n_bins=n_bins, n_res=n_res, ngf=ngf, ndf=ndf,
        lambda_cyc=lambda_cyc, lambda_id=lambda_id,
    ).to(device)

    # Optimisers — separate G and D (Adam with β₁=0.5, standard GAN)
    opt_G = optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=lr, betas=BETAS,
    )
    opt_D = optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=lr, betas=BETAS,
    )

    # Linear LR decay from epoch 100 → 0 at epoch `epochs` (Zhu et al. 2017)
    decay_start = epochs // 2
    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (epochs - decay_start + 1e-8))

    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # Save config
    config = {
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "n_bins": n_bins, "n_res": n_res, "ngf": ngf, "ndf": ndf,
        "lambda_cyc": lambda_cyc, "lambda_id": lambda_id,
        "n_abc_samples": int(sim_data.shape[0]),
        "n_real_samples": int(real_data.shape[0]),
        "real_domain": real_domain if real_domain else "LEGACY_BOOTSTRAP",
        "abc_cols": None if abc_cols is None else abc_cols.tolist(),
        "abc_index": abc_index,
        "balance_targets": bool(balance_targets),
        "seed": seed,
        "device": device_str,
    }
    with open(CKPT_DIR / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    # Training log
    log_path = CKPT_DIR / "training_log.csv"
    log_fields = ["epoch", "loss_G", "loss_G_AB", "loss_G_BA",
                  "loss_cyc", "loss_id", "loss_D", "loss_D_A", "loss_D_B",
                  "lr_G", "lr_D"]
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    best_loss_G = float("inf")

    print(f"\n  Training for {epochs} epochs ...\n")
    for epoch in range(1, epochs + 1):

        model.train()
        epoch_metrics: dict[str, list[float]] = {k: [] for k in log_fields}

        # Zip the two loaders — iterate until the shorter one is exhausted
        for (batch_A,), (batch_B,) in zip(sim_loader, real_loader):
            batch_A = batch_A.float()
            batch_B = batch_B.float()

            # ── Train Generators ──
            opt_G.zero_grad()
            loss_G, g_metrics = model.generator_loss(batch_A, batch_B)
            loss_G.backward()
            opt_G.step()

            # ── Train Discriminators ──
            opt_D.zero_grad()
            loss_D, d_metrics = model.discriminator_loss(batch_A, batch_B)
            loss_D.backward()
            opt_D.step()

            for k, v in {**g_metrics, **d_metrics}.items():
                epoch_metrics[k].append(v)

        sched_G.step()
        sched_D.step()

        # Aggregate epoch metrics
        row = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_metrics.items()}
        row["epoch"]  = epoch
        row["lr_G"]   = opt_G.param_groups[0]["lr"]
        row["lr_D"]   = opt_D.param_groups[0]["lr"]
        log_writer.writerow(row)
        log_file.flush()

        # Console output every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>4}/{epochs} | "
                f"loss_G={row['loss_G']:.4f}  loss_D={row['loss_D']:.4f}  "
                f"cyc={row['loss_cyc']:.4f}  id={row['loss_id']:.4f}  "
                f"lr={row['lr_G']:.2e}"
            )

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = CKPT_DIR / f"cyclegan_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "G_AB": model.G_AB.state_dict(),
                "G_BA": model.G_BA.state_dict(),
                "D_A":  model.D_A.state_dict(),
                "D_B":  model.D_B.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "config": config,
            }, ckpt_path)

        # Save best generator
        if row["loss_G"] < best_loss_G:
            best_loss_G = row["loss_G"]
            torch.save({
                "epoch": epoch,
                "loss_G": best_loss_G,
                "G_AB": model.G_AB.state_dict(),
                "G_BA": model.G_BA.state_dict(),
                "config": config,
            }, CKPT_DIR / "cyclegan_best.pt")

    log_file.close()

    print(f"\n{'='*70}")
    print(f"  Training complete.")
    print(f"  Best generator loss: {best_loss_G:.4f}")
    print(f"  Checkpoints saved to: {CKPT_DIR}")
    print(f"  Training log:         {log_path}")
    print(f"{'='*70}\n")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train Phase 3 Track 2 CycleGAN — sim-to-real spectral translation"
    )
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--n-abc",       type=int,   default=None,
                   help="Limit ABC samples (default: all; use 500 for smoke test)")
    p.add_argument("--n-res",       type=int,   default=9,
                   help="Residual blocks in generator (9 full, 6 fast)")
    p.add_argument("--ngf",         type=int,   default=64)
    p.add_argument("--ndf",         type=int,   default=64)
    p.add_argument("--lambda-cyc",  type=float, default=10.0)
    p.add_argument("--lambda-id",   type=float, default=5.0)
    p.add_argument("--save-every",  type=int,   default=10)
    p.add_argument("--ckpt-dir",    type=str,   default=None,
                   help="output dir for this run (default configs/cyclegan). "
                        "Use a fresh dir so an earlier model is never overwritten.")
    p.add_argument("--real-domain", type=str,   default=None,
                   help="npz from build_real_domain.py; real OOT-noise domain B. "
                        "Required (V7) unless --legacy-bootstrap.")
    p.add_argument("--legacy-bootstrap", action="store_true",
                   help="reproduce the RETIRED single-spectrum bootstrap real side "
                        "(negative control only, P5-D11)")
    p.add_argument("--balance-targets", action="store_true",
                   help="pooled real domain: draw each target equally often (V7)")
    p.add_argument("--abc-index",   type=str,   default=None,
                   help="npz with `idx`: restrict domain A to these abc_train rows (V3)")
    p.add_argument("--seed",        type=int,   default=None,
                   help="torch/numpy seed, recorded in config.json")
    p.add_argument("--device",      type=str,   default="cpu",
                   choices=["cpu", "cuda", "mps"])
    args = p.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_abc=args.n_abc,
        n_res=args.n_res,
        ngf=args.ngf,
        ndf=args.ndf,
        lambda_cyc=args.lambda_cyc,
        lambda_id=args.lambda_id,
        save_every=args.save_every,
        device_str=args.device,
        real_domain=args.real_domain,
        ckpt_dir=args.ckpt_dir,
        legacy_bootstrap=args.legacy_bootstrap,
        balance_targets=args.balance_targets,
        abc_index=args.abc_index,
        seed=args.seed,
    )
