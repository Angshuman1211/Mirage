#!/usr/bin/env bash
# SIGMA-RANGE RETRAIN. The fix P6-D7 and P6-D17 jointly point at, and the last
# optional Track 1 item (08_work_split_v2.md sequencing note 7).
#
# WHAT IT TESTS. P6-D7 measured real WASP-39b sigma at 2.16e-3 normalised, 23-139x
# BELOW the trained sigma_min of 0.05, so real data sits entirely OUTSIDE the noise
# distribution the sigma and covariance embeddings were trained on. These six arms
# retrain over a range that CONTAINS real sigma, log-uniform over [1e-3, 0.3].
#
# WHY LOG-UNIFORM AND NOT JUST A LOWER FLOOR. The generator draws sigma uniformly
# (mirage/datasets/noise.py). Uniform over [2e-3, 0.3] puts only 2.5% of samples
# below 0.01, median 0.150, so the model would still almost never see the real
# regime and the retrain would fail for a reason that has nothing to do with the
# hypothesis. Log-uniform over [1e-3, 0.3] puts 40.5% below 0.01, median 0.017, and
# 13.5% below the real 2.16e-3 against 0.4% under the uniform law.
# `sigma_log` defaults to False, so every existing config is bit-identical.
#
# PREDICTION, WRITTEN BEFORE THE RUN as P6-D8 requires. The covariance arm's JS to
# the NS anchor on real WASP-39b falls from 0.426 toward the no-conditioning 0.344,
# and the +0.083 paradox shrinks by at least half. If it does NOT shrink, P6-D26 is
# reinforced from a second direction: A3 showed the paradox is not caused by the
# inference-time Sigma-hat, and this would show it is not the training range either.
#
# HARDWARE. CUDA ONLY. P6-D1 found MPS under-trains these models, ABC best val
# 1.923 on MPS against 1.390 on CUDA at the same seed and config, which is the
# exact artifact the H9 study exists to remove. Running this on MPS produces an
# uninterpretable result. Roughly 5.5 h for all six on one GPU.
#
#   bash scripts/run_sigrange_retrain.sh            # all six
#   ARMS=cov SEEDS=42 bash scripts/run_sigrange_retrain.sh   # one arm, one seed
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
ARMS=${ARMS:-"nocond cov"}
SEEDS=${SEEDS:-"42 43 44"}

$PY - <<'CHECK'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("REFUSING TO RUN: no CUDA. See the hardware note in this script — "
             "MPS under-trains these models (P6-D1) and the result would be "
             "uninterpretable. Run this on a CUDA machine.")
print(f"[device] CUDA ok: {torch.cuda.get_device_name(0)}")
CHECK

for arm in $ARMS; do
  for s in $SEEDS; do
    d=configs/sigrange_rad_${arm}_s${s}
    echo "=== training $d ==="
    $PY scripts/train.py --experiment-dir "$d"
  done
done

echo
echo "=== evaluate on real WASP-39b, the number the prediction is about ==="
echo "  python scripts/real_ess.py  (register the arms first), then real_compare"
echo "Record the outcome as P6-D28 in Documentation/problems_and_decisions.md."
