#!/usr/bin/env bash
# V2 (closes G8): cheap translation baselines on WASP-39b PRISM, 3 seeds.
#
# Runs from the out-of-transit frames alone. In the published metric space the
# reduced spectrum's depth cancels exactly (see v2_cheap_baselines.py), and the
# run reproduces the published tier-1 MMD to four decimals, which is the check
# that this holds. Pass a real-domain npz instead of --oot to use the spectrum.
#
#   bash scripts/run_v2_baselines.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
for s in 0 1 2; do
  "$PY" scripts/v2_cheap_baselines.py --oot wasp39 --seed "$s" \
      --out "configs/v2_baselines_wasp39/v2_baselines_seed$s.json"
done
"$PY" scripts/v2_table.py
