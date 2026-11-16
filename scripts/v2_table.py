"""Summarise the V2 baseline runs across seeds into the one table the paper cites.

Reads configs/v2_baselines_wasp39/v2_baselines_seed*.json written by
scripts/v2_cheap_baselines.py and prints mean +- sd over seeds, with the
no-adaptation MMD printed beside every arm (V2 brief: report both MMD values).

    python scripts/v2_table.py
"""
import json
from pathlib import Path

import numpy as np

RUN_DIR = Path("configs/v2_baselines_wasp39")
METRICS = ("mmd_to_real", "kl_to_real", "mean_shift")


def main():
    files = sorted(RUN_DIR.glob("v2_baselines_seed*.json"))
    if not files:
        raise SystemExit(f"no runs in {RUN_DIR}; run scripts/run_v2_baselines.sh")
    R = [json.load(open(f)) for f in files]
    arms = list(R[0]["heldout"])
    base = "sim (no adaptation)"

    def stat(arm, m):
        v = np.array([r["heldout"][arm][m] for r in R])
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))

    print(f"\nV2, WASP-39b PRISM 52 bins, held-out real frames, {len(R)} seeds "
          f"({', '.join(str(r['seed']) for r in R)})")
    print(f"  {'arm':<22}{'MMD':>16}{'MMD none':>10}{'KL':>16}{'shift':>15}  vs none")
    print("  " + "-" * 88)
    b_mmd = stat(base, "mmd_to_real")[0]
    for a in arms:
        (mm, ms), (km, ks), (sm, ss) = (stat(a, m) for m in METRICS)
        rel = (mm - b_mmd) / b_mmd
        print(f"  {a:<22}{mm:>9.4f}+-{ms:<5.4f}{b_mmd:>10.4f}{km:>9.3f}+-{ks:<5.3f}"
              f"{sm:>8.3f}+-{ss:<5.3f}  {100 * rel:+.0f}% MMD")
    print("  " + "-" * 88)
    d = [r["reweight_diagnostics"] for r in R]
    print(f"  reweight: classifier AUC {np.mean([x['classifier_auc_cv3'] for x in d]):.3f}, "
          f"weight ESS {np.mean([x['weight_ess'] for x in d]):.0f} of "
          f"{d[0]['n_weighted']} (min {min(x['weight_ess'] for x in d):.0f})")
    rep = R[0]["reproduction_published_protocol"]
    pub = R[0]["published"]
    print(f"  reproduction of the published protocol: sim {rep['sim']['mmd_to_real']:.4f} "
          f"(pub {pub['sim_baseline']['mmd_to_real']:.4f}), random "
          f"{rep['domain_random']['mmd_to_real']:.4f} "
          f"(pub {pub['domain_random']['mmd_to_real']:.4f}), cycleGAN "
          f"{rep['cyclegan']['mmd_to_real']:.4f} "
          f"(pub {pub['cyclegan_trans']['mmd_to_real']:.4f})")


if __name__ == "__main__":
    main()
