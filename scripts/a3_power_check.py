"""Is A3's span one lucky base-noise realization?

fm4ar reseeds the global RNG from the config inside build_model (fmpe.py:79),
so every real_ess.sample() call draws the SAME theta_0. That makes the Sigma-hat
comparison perfectly paired, but it also means the measured span comes from one
realization. Here theta_0 is driven by an explicit generator instead, so the
whole sweep can be repeated under independent base noise.

Result (P6-D26): spans 0.0047, 0.0031, 0.0031 across three realizations, which
bracket the headline 0.0039, so the null is not one lucky draw. Draw noise on a
single score is sd 0.0020 at 1000 draws.

    conda run -n mirage python scripts/a3_power_check.py
"""
import sys, shutil
from pathlib import Path
sys.path.insert(0, "scripts")
import numpy as np, torch, mirage, real_ess, real_compare
import fm4ar.models.fmpe as F
import a3_noise_proxy as A3

RE = Path("data/real_ess"); A = RE / "wasp39b_ns_posterior_fitrad.npz"
SCR = Path(sys.argv[1] if len(sys.argv) > 1 else "data/a3/power"); SCR.mkdir(parents=True, exist_ok=True)
arm = "h9_rad_cov_s42"; pub = RE / f"real_ess_samples_{arm}.npz"
gen_f = RE / "real_ess_samples.npz"
shutil.copy2(pub, SCR / "keep.npz"); had_generic = gen_f.exists()

GEN = {"g": None}
def patched_theta0(self, num_samples):
    return torch.randn(num_samples, self.dim_theta, device=self.device,
                       generator=GEN["g"])
F.FMPEModel.sample_theta_0 = patched_theta0

real0 = real_ess._real_oot_frames(); sig = real0.std(axis=0)
kinds = ["real", "white", "depth_direct", "rank1"]
frames = {k: A3.make_frames(k, real0, sig, np.random.default_rng(42)) for k in kinds}
rows = {}
try:
    for ds in (101, 202, 303):
        GEN["g"] = torch.Generator().manual_seed(ds)
        js = {}
        for k in kinds:
            GEN["g"] = torch.Generator().manual_seed(ds)   # identical theta_0 per input
            real_ess._real_oot_frames = lambda n=100, _f=frames[k]: _f
            real_ess.sample(1000, arm)
            d = SCR / f"p_{ds}_{k}.npz"; shutil.move(pub, d)
            _, j = real_compare.compare(d, A, quiet=True); js[k] = j
        rows[ds] = js
        print(f"  base-noise seed {ds}: " + "  ".join(f"{k} {v:.4f}" for k, v in js.items())
              + f"   SPAN {max(js.values()) - min(js.values()):.4f}")
finally:
    shutil.copy2(SCR / "keep.npz", pub)
    if not had_generic:
        gen_f.unlink(missing_ok=True)
    print("[restore] published arm put back")

print("\nDOES THE SPAN SURVIVE A CHANGE OF BASE NOISE?")
sp = [max(v.values()) - min(v.values()) for v in rows.values()]
print(f"  spans across 3 independent base-noise realizations: "
      + ", ".join(f"{x:.4f}" for x in sp))
print(f"  largest {max(sp):.4f}  vs the paradox 0.0827 "
      f"-> at most {100 * max(sp) / 0.0827:.1f}% of it under the worst realization")
rl = [v["real"] for v in rows.values()]
print(f"  draw noise on a single score: real-input JS = "
      + ", ".join(f"{x:.4f}" for x in rl) + f"  (sd {np.std(rl, ddof=1):.4f})")
