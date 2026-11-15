#!/usr/bin/env bash
# V7: rebuild the CycleGAN real side on ALL THREE reduced spectra, retrain once,
# and report whether the translation null survives. The last CycleGAN run the
# project does (08_work_split_v2.md, "What not to do").
#
# Needs data/jwst_{wasp39b,wasp96b,k2_18b}_spectrum.csv and the three OOT files
# listed in build_real_domain.py:TARGETS.
#
# Scoring. The one pooled translator is scored against each target's own real
# frames TWICE: at 56 frames (seed 0, the same subset run_cyclegan_tiers.sh used,
# so the rows line up with the published per-target tier-2 translators), and at
# every frame (lines up with cyclegan_ctrl_wasp39_26b_full). P5-D11 showed the
# CycleGAN's MMD falls 71% from frame count alone, so only like-for-like pairs
# are compared, and the full-frame row is the one the conclusion rests on.
#
#   bash scripts/run_v7_pooled.sh            # DEVICE=cuda by default
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-0}
D=data/real_domain
RUN=configs/cyclegan_v7_pooled3

for t in wasp39 wasp96 k218; do
  "$PY" scripts/build_real_domain.py --target "$t" --grid abcniriss --suffix _full
  "$PY" scripts/build_real_domain.py --target "$t" --grid abcniriss --n 56 --suffix _n56
done
"$PY" scripts/build_real_domain.py --target all --grid abcniriss

"$PY" scripts/train_cyclegan.py --epochs 200 --batch-size 64 --seed "$SEED" \
    --real-domain "$D/real_domain_all3_abcniriss.npz" --balance-targets \
    --device "$DEVICE" --ckpt-dir "$RUN"
rm -f "$RUN"/cyclegan_epoch_*.pt     # 107 MB each, gitignored, read by nothing

for t in wasp39 wasp96 k218; do
  for n in full n56; do
    "$PY" scripts/run_cyclegan_ablation_standalone.py --n-samples 500 \
        --real-domain "$D/real_domain_${t}_abcniriss_${n}.npz" \
        --ckpt "$RUN/cyclegan_best.pt" --out "$RUN/ablation_${t}_${n}.json"
  done
done
"$PY" scripts/v7_table.py
