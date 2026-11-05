"""
TranslateCycleGAN: the learned-translation arm, at the posterior level.

Why this exists. `scripts/run_cyclegan_ablation_standalone.py` compares the
translated DISTRIBUTION against real instrument noise (MMD, KL, mean shift).
That is a distributional statistic, not a retrieval result, so on its own it
cannot support a claim about whether learned translation helps INFERENCE. The
v2 plan's outstanding Track-2 item needs a flow actually trained on translated
spectra, which is what this transform makes possible. `run_cyclegan_ablation.py`
reuses the existing flows and feeds them translated input at eval time instead,
which tests whether a sim-trained flow survives shifted input, a different
question.

Registered into fm4ar's `get_data_transforms` by `mirage.register` as transform
type "TranslateCycleGAN"; fm4ar itself stays unmodified (P2-D8).

Scaling is the trap, and it is the same class of bug as the five in P5-D11. The
generator was trained on ABC flux standardised by flux_mean/flux_std and then
mapped to [-1,1] by a 1/99-percentile clip taken over the whole training set.
This transform therefore applies that same map, translates, and inverts it. The
percentiles are RECOMPUTED from the training hdf rather than stored in the
checkpoint, so they cannot drift away from the generator's own preprocessing.
"""

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import torch

from fm4ar.datasets.data_transforms import DataTransform

from mirage.nn.cyclegan import CycleGAN


def _abc_scale_stats(stats_hdf: Path):
    """Everything needed to reproduce the generator's [-1,1] inputs, and invert.

    `scripts/train_cyclegan.py:load_abc_spectra` divides the stored flux by
    flux_mean/flux_std even though the hdf is already per-bin standardised
    (prepare_abc_hdf5.py:99), inflating it about 135x. That extra step is NOT
    cosmetic and must be reproduced here.

    It is tempting to argue it cancels: the [-1,1] map is built from percentiles
    of the transformed data, and percentile rescaling is affine-equivariant, so
    a *global* affine pre-map would wash out exactly. That argument fails here,
    because flux_mean and flux_std are PER-BIN (52 values each) while the
    percentile clip is GLOBAL over all bins. A per-bin affine map does not
    commute with a global percentile, so the two scalings differ by up to 0.22
    in [-1,1] units, about 11% of the generator's input range. Measured, not
    assumed. Feeding the generator the stored flux instead would hand it a
    quantity it never trained on.

    Returns (mean, std, p1, p99) with mean/std per-bin and p1/p99 scalars in the
    double-standardised space.
    """
    with h5py.File(stats_hdf) as f:
        flux = f["flux"][:].astype(np.float32)
        mean = f["flux_mean"][:].astype(np.float32)
        std = f["flux_std"][:].astype(np.float32)
    z = (flux - mean) / std
    p1, p99 = np.percentile(z, [1, 99])
    return mean, std, float(p1), float(p99)


class TranslateCycleGAN(DataTransform):
    """Map a standardised ABC spectrum through the trained generator G_AB."""

    def __init__(
        self,
        checkpoint: str,
        stats_hdf: str = "data/abc/abc_train.hdf",
    ) -> None:
        super().__init__()

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"{ckpt_path} not found. Train the translator first:\n"
                f"    python scripts/train_cyclegan.py --real-domain "
                f"data/real_domain/real_domain_wasp39_abc.npz --ckpt-dir <dir>"
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]

        self.n_bins = int(cfg["n_bins"])
        model = CycleGAN(
            n_bins=self.n_bins,
            n_res=int(cfg["n_res"]),
            ngf=int(cfg["ngf"]),
            ndf=int(cfg["ndf"]),
        )
        model.G_AB.load_state_dict(ckpt["G_AB"])
        model.G_AB.eval()
        for p in model.G_AB.parameters():
            p.requires_grad_(False)
        self.generator = model.G_AB

        self.mean, self.std, self.p1, self.p99 = _abc_scale_stats(Path(stats_hdf))
        self._span = max(self.p99 - self.p1, 1e-8)

        # Provenance, so a run can be traced to the translator that produced it.
        self.checkpoint = str(ckpt_path)
        self.source_epoch = int(ckpt.get("epoch", -1))
        self.real_domain = cfg.get("real_domain")

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        flux = np.asarray(x["flux"], dtype=np.float32)

        if flux.shape[-1] != self.n_bins:
            raise ValueError(
                f"TranslateCycleGAN: generator takes {self.n_bins} bins, got "
                f"{flux.shape[-1]}. The translator and the dataset grid differ."
            )

        # stored -> double-standardised -> [-1, 1]: the generator's own preprocessing
        z = (flux - self.mean) / self.std
        u = np.clip(z, self.p1, self.p99)
        u = 2.0 * (u - self.p1) / self._span - 1.0

        with torch.no_grad():
            t = torch.from_numpy(u).float()
            squeeze = t.ndim == 1
            if squeeze:
                t = t.unsqueeze(0)
            v = self.generator(t)
            if squeeze:
                v = v.squeeze(0)
            v = v.numpy()

        # and all the way back, so downstream sees the stored-flux space it expects
        z_out = 0.5 * (v + 1.0) * self._span + self.p1
        output["flux"] = (z_out * self.std + self.mean).astype(np.float32)
        return output
