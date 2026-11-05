"""Precompute the CycleGAN-translated ABC dataset.

The posterior-level translation arm needs a flow TRAINED on translated spectra
(the outstanding Track-2 item in the v2 plan). Doing that through the
`TranslateCycleGAN` data transform works but runs the generator once per sample
inside the dataloader, which measured at t_data = 2.3 s per batch of 512 and
projects to roughly a day of training.

The transform is deterministic, so translating the dataset once up front is
exactly equivalent and removes the bottleneck. It also makes the comparison
cleaner to state: the translated arm then uses a config IDENTICAL to
noisecond_cov, including the same correlated-noise injection and the same seed,
and the only difference is which spectra it reads.

Everything except `flux` is copied through unchanged. `flux_mean` / `flux_std`
are copied as stored, since they describe the ORIGINAL normalisation and are
consumed by mirage/datasets/real_jwst.py to map real spectra into ABC space,
not by training. They are kept so the file stays self-describing, and they no
longer summarise the translated flux, which is why the arm must not be used as
a source of normalisation stats for real data.

    python scripts/build_translated_abc.py \
        --ckpt configs/cyclegan_t1_wasp39/cyclegan_best.pt
"""
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch

from mirage.datasets.translate import TranslateCycleGAN

ABC = Path("data/abc")
SPLITS = ("train", "valid", "test")


def translate_file(src: Path, dst: Path, t: TranslateCycleGAN, batch: int) -> None:
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        flux = f["flux"][:].astype(np.float32)
        n = len(flux)
        z = (flux - t.mean) / t.std
        u = 2.0 * (np.clip(z, t.p1, t.p99) - t.p1) / t._span - 1.0

        out = np.empty_like(u)
        with torch.no_grad():
            for i in range(0, n, batch):
                chunk = torch.from_numpy(u[i:i + batch]).float()
                out[i:i + batch] = t.generator(chunk).numpy()

        z_out = 0.5 * (out + 1.0) * t._span + t.p1
        new = (z_out * t.std + t.mean).astype(np.float32)
        if not np.isfinite(new).all():
            raise SystemExit(f"[guard] non-finite translated flux in {dst.name}")

        f["flux"][...] = new
        f.attrs["translated_from"] = str(src.name)
        f.attrs["translator_ckpt"] = t.checkpoint
        f.attrs["translator_epoch"] = t.source_epoch
        f.attrs["translator_real_domain"] = str(t.real_domain)

        d = np.abs(new - flux)
        print(f"  {dst.name:<26} {n:>6} spectra   mean|delta| {d.mean():.4f}   "
              f"shift {new.mean() - flux.mean():+.4f}")


def main(ckpt: str, suffix: str, batch: int) -> None:
    t = TranslateCycleGAN(ckpt)
    print(f"translator {Path(ckpt).parent.name}  epoch {t.source_epoch}")
    print(f"real domain {t.real_domain}")
    out_dir = ABC
    for split in SPLITS:
        src = ABC / f"abc_{split}.hdf"
        if not src.exists():
            print(f"  [skip] {src.name} not found")
            continue
        translate_file(src, out_dir / f"abc_{suffix}_{split}.hdf", t, batch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="configs/cyclegan_t1_wasp39/cyclegan_best.pt")
    ap.add_argument("--suffix", default="translated")
    ap.add_argument("--batch", type=int, default=2048)
    a = ap.parse_args()
    main(a.ckpt, a.suffix, a.batch)
