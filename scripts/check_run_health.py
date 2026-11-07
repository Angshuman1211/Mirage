"""Did this run actually train? Run it after every FMPE run, before trusting a number.

Why this exists (P6-D1): published models were trained on MPS and never converged
properly, and nobody noticed for months because the runs completed and wrote
checkpoints. A run finishing is not evidence it worked.

TWO failure modes, and they need different tests.

  CATASTROPHIC: the loss barely moves. ABC no-cond went 1.9945 -> 1.9365 across
  125 epochs. Caught by the train-gain floor below, no reference needed.

  DEGRADED: the loss moves normally but settles too high. noisecond_rad_nocond
  gained +0.614 and still landed at 1.2477 where CUDA reaches 1.1328. This is
  INVISIBLE without a reference value, which is exactly how it survived review.
  Pass --expect to test for it.

    python scripts/check_run_health.py configs/h9_*
    python scripts/check_run_health.py configs/ctrl_k218_s42 --expect 1.2141

On --expect I report a BAND, not a pass/fail against one tuned number. The
measured gaps separate themselves by roughly 4x at every boundary, so no
threshold needs choosing (P6-D4):

  |gap| < 0.02   reproduces      matched seed spread is sd 0.0033 / range 0.0087
                                 across same-config nocond arms (n=6)
  0.02 - 0.05    backend shift   k218 +0.032, wasp96 -0.022. Bidirectional, so
                                 not a training failure. Different RNG streams
                                 and reduction orders across devices.
  0.05 - 0.40    DEGRADED        rad_nocond +0.115, rad_sigma +0.24
  > 0.40         CATASTROPHIC    abc_nocond +0.533

Do not widen the "reproduces" band to swallow the backend-shift one. They mean
different things, and the 0.038 spread sometimes quoted as ordinary noise is the
rad_cov group, the widest of four and the only one carrying the covariance
embedding. It is the wrong reference class for a nocond run.
"""
import argparse
from pathlib import Path

import torch

MIN_TRAIN_GAIN = 0.20   # healthy runs here gain 0.46-0.78; the catastrophic one gained 0.06
SEED = 0.02             # inside the matched same-config seed spread
BACKEND = 0.05          # device-level perturbation, acceptable but worth naming
DEGRADED = 0.40         # above this the run never really trained


def classify(delta: float) -> tuple[str, bool]:
    """-> (note, acceptable). Sign convention: delta > 0 means THIS run is worse."""
    g = abs(delta)
    if g < SEED:
        return f"reproduces (|{delta:+.4f}| < {SEED})", True
    who = "this run" if delta > 0 else "the REFERENCE"
    if g < BACKEND:
        return f"backend shift {delta:+.4f} ({who} worse, within device noise)", True
    if g < DEGRADED:
        return f"*** DEGRADED {delta:+.4f} -> {who} under-trained ***", False
    return f"*** CATASTROPHIC {delta:+.4f} -> {who} did not train ***", False


def check(d: Path, expect: float | None):
    f = d / "model__best.pt"
    if not f.exists():
        print(f"{d.name:28s} NO CHECKPOINT")
        return False
    c = torch.load(f, map_location="cpu", weights_only=False)
    tr = [float(x) for x in c["history"]["train_loss"]]
    te = [float(x) for x in c["history"]["test_loss"]]
    dev = c["config"].get("local", {}).get("device", "?")
    gain, best = tr[0] - min(tr), min(te)
    ok = gain >= MIN_TRAIN_GAIN
    note = "ok" if ok else "*** DID NOT TRAIN ***"
    if expect is not None:
        band, acceptable = classify(best - expect)
        note += f"  | {band}"
        ok = ok and acceptable
    print(f"{d.name:28s} {dev:5s} ep={c['epoch']:>4}  "
          f"train {tr[0]:.3f}->{min(tr):.3f} ({gain:+.3f})  best_val={best:.4f}  {note}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--expect", type=float, default=None,
                    help="reference best_val; classifies disagreement in EITHER direction")
    a = ap.parse_args()
    bad = [d for d in (Path(x) for x in a.dirs) if not check(d, a.expect)]
    print()
    print("All clear." if not bad else
          f"NEEDS ATTENTION: {', '.join(d.name for d in bad)}")
