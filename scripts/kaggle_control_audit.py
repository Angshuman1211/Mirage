"""Calibrate the Kaggle control gate against the MEASURED seed spread (P6-D28).

The first gate used max|deviation| < 0.05, a number chosen without measuring the
natural spread. This measures it: the same config at three training seeds, all on
the reference RTX 5070, so "does this GPU reproduce the reference" is asked
against run-to-run variation instead of against a guess.

Result on 2026-11-19, epochs 1 to 60:

    T4 s42   vs 5070 s42  (cross-device)  mean 0.0170  max 0.0503 @ ep 36
    5070 s43 vs 5070 s42  (cross-seed)    mean 0.0167  max 0.0562 @ ep 25
    5070 s44 vs 5070 s42  (cross-seed)    mean 0.0321  max 0.0807 @ ep 2

The cross-device deviation is SMALLER than both cross-seed deviations, so a
0.05 max-deviation gate fails h9_rad_cov_s43, a published 5070 arm. The gate was
miscalibrated, not the hardware.

Run from the repo root with the `mirage` env.
"""
import pathlib, torch, numpy as np, pandas as pd

ROOT = pathlib.Path(".")
SP = pathlib.Path("/private/tmp/claude-501/-Users-angshumansmac-Desktop-Actual-Projects-Exoplanet/05e9bcde-affa-4f4f-82ed-354fe5b05954/scratchpad")

def hist_from_ckpt(d: pathlib.Path):
    best = None
    for f in ("model__stage_0.pt", "model__latest.pt", "model__best.pt"):
        p = d / f
        if not p.exists():
            continue
        h = torch.load(p, map_location="cpu", weights_only=False).get("history")
        if h is None:
            continue
        h = pd.DataFrame(h)
        if best is None or len(h) > len(best):
            best = h
    return best

ref = pd.read_csv(ROOT / "configs/h9_rad_cov_s42/history_to_ep225.csv")
t4  = pd.read_csv(SP / "res/control_rad_cov_s42/history_to_ep60.csv")
N = int(t4["epoch"].max())
print(f"T4 control: {N} epochs,  reference: {int(ref['epoch'].max())} epochs\n")

r = dict(zip(ref["epoch"], ref["test_loss"]))
t = dict(zip(t4["epoch"], t4["test_loss"]))
ep = [e for e in range(1, N + 1) if e in r and e in t]

# Seed envelope: the SAME config on the SAME 5070 at seeds 43 and 44.
seeds = {}
for s in (43, 44):
    h = hist_from_ckpt(ROOT / f"configs/h9_rad_cov_s{s}")
    if h is not None:
        seeds[s] = dict(zip(h["epoch"], h["test_loss"]))
        print(f"seed {s} history from checkpoint: {len(h)} epochs")

dev_dev = np.array([abs(t[e] - r[e]) for e in ep])
print(f"\nCROSS-DEVICE  |T4(s42) - 5070(s42)| over epochs 1-{N}")
print(f"   mean {dev_dev.mean():.4f}   max {dev_dev.max():.4f} @ epoch {ep[int(dev_dev.argmax())]}")

for s, d in seeds.items():
    common = [e for e in ep if e in d]
    a = np.array([abs(d[e] - r[e]) for e in common])
    print(f"CROSS-SEED    |5070(s{s}) - 5070(s42)| over {len(common)} epochs")
    print(f"   mean {a.mean():.4f}   max {a.max():.4f} @ epoch {common[int(a.argmax())]}")

print(f"\n{'epoch':>6} {'5070 s42':>10} {'T4 s42':>10} {'delta':>9} "
      + " ".join(f"{'5070 s'+str(s):>10}" for s in seeds))
for e in ep:
    if e in (1,2,3,5,8,10,15,20,25,30,40,50,55,56,57,58,59,60) or e == ep[int(dev_dev.argmax())]:
        row = f"{e:>6} {r[e]:>10.4f} {t[e]:>10.4f} {t[e]-r[e]:>+9.4f} "
        row += " ".join(f"{seeds[s].get(e, float('nan')):>10.4f}" for s in seeds)
        print(row + ("   <- max dev" if e == ep[int(dev_dev.argmax())] else ""))

# Does the T4 curve sit inside the per-epoch min/max of the three 5070 seeds?
if seeds:
    inside = out = 0
    worst = (0.0, None)
    for e in ep:
        vals = [r[e]] + [seeds[s][e] for s in seeds if e in seeds[s]]
        if len(vals) < 3: continue
        lo, hi = min(vals), max(vals)
        if lo <= t[e] <= hi: inside += 1
        else:
            out += 1
            excess = (lo - t[e]) if t[e] < lo else (t[e] - hi)
            if excess > worst[0]: worst = (excess, e)
    tot = inside + out
    print(f"\nT4 inside the 3-seed 5070 envelope: {inside}/{tot} epochs ({100*inside/tot:.0f}%)")
    print(f"largest excursion outside it: {worst[0]:.4f} at epoch {worst[1]}")

print(f"\nFinal epochs, test loss:")
print(f"   5070 s42 @60 {r[60]:.4f}   T4 s42 @60 {t[60]:.4f}   delta {t[60]-r[60]:+.4f}")
print(f"   MPS failure value for this config: 1.3729")
print(f"   radius CUDA band at convergence  : 1.124 - 1.159")
