import argparse
from pathlib import Path
import torch
MIN_TRAIN_GAIN = 0.20
TOL = 0.02
def check(d, expect):
    f = d / "model__best.pt"
    if not f.exists():
        print(f"{d.name:28s} NO CHECKPOINT"); return False
    c = torch.load(f, map_location="cpu", weights_only=False)
    tr = [float(x) for x in c["history"]["train_loss"]]
    te = [float(x) for x in c["history"]["test_loss"]]
    dev = c["config"].get("local", {}).get("device", "?")
    gain, best = tr[0] - min(tr), min(te)
    ok = gain >= MIN_TRAIN_GAIN
    note = "ok" if ok else "*** DID NOT TRAIN ***"
    if expect is not None:
        delta = best - expect
        if abs(delta) <= TOL: note += f"  | reproduces {expect:.4f}"
        elif delta < 0:
            note += f"  | BEATS reference by {-delta:.4f} -> the REFERENCE run was degraded"; ok = False
        else:
            note += f"  | WORSE than reference by {delta:.4f} -> THIS run is degraded"; ok = False
    print(f"{d.name:28s} {dev:5s} ep={c['epoch']:>4}  train {tr[0]:.3f}->{min(tr):.3f} ({gain:+.3f})  best_val={best:.4f}  {note}")
    return ok
ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--expect", type=float, default=None)
a = ap.parse_args()
bad = [d for d in (Path(x) for x in a.dirs) if not check(d, a.expect)]
print(); print("All clear." if not bad else f"NEEDS ATTENTION: {', '.join(d.name for d in bad)}")
