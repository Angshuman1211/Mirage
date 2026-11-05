#!/usr/bin/env bash
# Pillar 2 of the v2 plan: does structured domain randomisation beat a learned
# translator at reaching the real JWST noise distribution?
#
# Tier 1  WASP-39b / NIRSpec PRISM on all 52 ABC bins, 3,241 real OOT frames.
#         The headline run, and the direct replacement for the published
#         ablation, whose real domain was one spectrum bootstrapped with 1%
#         Gaussian jitter.
#
# Tier 2  All three targets on the 26 native ABC bins inside 0.85-2.83um, the
#         range NIRISS and PRISM share. Every arm gets 56 real frames, because
#         WASP-96b only has 56 and the variable under test is the character of
#         the real noise, not how much of it there is. Per-bin fractional
#         scatter spans 28x across the three (0.0197 / 0.0011 / 0.0007), so this
#         asks whether the tier-1 conclusion holds across noise regimes and
#         instruments. Trained and scored SEPARATELY, never pooled: a
#         discriminator shown all three at once would separate them on noise
#         amplitude alone, which is not the question.
#
#   bash scripts/run_cyclegan_tiers.sh
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS=200
# mps, measured at ~10x cpu on the M1 Pro (27s vs 282s an epoch at 52 bins).
# Every run uses the SAME device, since mps and cpu numerics differ slightly and
# the point of tier 2 is to compare runs against each other.
DEVICE=mps
run () {                       # name, real-domain npz, batch size
  local name=$1 dom=$2 bs=$3
  local dir="configs/cyclegan_$name"
  echo "=============================================================="
  echo "  $name   (batch $bs, $EPOCHS epochs, $DEVICE)"
  echo "=============================================================="
  rm -rf "$dir"                # no stale checkpoints from an aborted attempt
  python3 scripts/train_cyclegan.py \
      --epochs "$EPOCHS" --batch-size "$bs" --real-domain "$dom" \
      --device "$DEVICE" --ckpt-dir "$dir" \
      || { echo "!! train failed: $name"; return 1; }
  python3 scripts/run_cyclegan_ablation_standalone.py \
      --n-samples 500 --real-domain "$dom" \
      --ckpt "$dir/cyclegan_best.pt" \
      --out "$dir/ablation.json" || { echo "!! ablation failed: $name"; return 1; }
}

D=data/real_domain
run t1_wasp39 "$D/real_domain_wasp39_abc.npz"        64
for t in wasp39 wasp96 k218; do
  run "t2_$t"  "$D/real_domain_${t}_abcniriss.npz"    8
done
echo "ALL RUNS FINISHED"
