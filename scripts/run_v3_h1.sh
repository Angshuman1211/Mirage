#!/usr/bin/env bash
# V3 / H1, run once: does restricting domain A to the real planet's neighbourhood
# collapse the tier-1 mean shift (0.308 per-bin sigma)? Same real domain, grid,
# epochs and batch as cyclegan_t1_wasp39; the only change is domain A.
#
#   bash scripts/run_v3_h1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
DEVICE=${DEVICE:-cuda}
K=${K:-2000}
RUN=configs/cyclegan_v3_h1_k$K
DOM=data/real_domain/real_domain_wasp39_abc.npz

[ -f "$DOM" ] || "$PY" scripts/build_real_domain.py --target wasp39 --grid abc
"$PY" scripts/v3_planet_neighbourhood.py --k "$K"
"$PY" scripts/train_cyclegan.py --epochs 200 --batch-size 64 --seed 0 \
    --real-domain "$DOM" --abc-index "data/real_domain/v3_wasp39_neighbourhood_k$K.npz" \
    --device "$DEVICE" --ckpt-dir "$RUN"
"$PY" scripts/run_cyclegan_ablation_standalone.py --n-samples 500 \
    --real-domain "$DOM" --ckpt "$RUN/cyclegan_best.pt" --out "$RUN/ablation.json"
echo "compare mean_shift in $RUN/ablation.json with 0.308 (cyclegan_t1_wasp39)"
