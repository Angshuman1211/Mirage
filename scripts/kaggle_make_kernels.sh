#!/usr/bin/env bash
# Stage the Kaggle kernel directories for the sigma-range retrain (P6-D28).
#
# WHY THREE KERNELS AND NOT ONE. The Kaggle API pushes a notebook as a BATCH
# run, and a batch run that overruns the 12 h cap is killed with its output
# commit FAILED, so every checkpoint is lost. An interactive session keeps
# /kaggle/working on timeout, so the two modes are not interchangeable. Two arms
# per run keeps each one far inside the cap even at the T4 slowdown.
#
# The control is its own kernel and is pushed FIRST. It trains the published
# h9_rad_cov_s42 config from scratch and compares against the RTX 5070 curve,
# because P6-D27 found a device artifact that moved published coverage numbers by
# +0.110 and -0.128 in opposite directions.
#
#   bash scripts/kaggle_make_kernels.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=angshuman12/mirage-sigrange
USER=angshuman12

stage () {           # name  arms  control
  local name="$1" arms="$2" control="$3" dir="build/kernel_$1"
  mkdir -p "$dir"
  python scripts/make_kaggle_notebook.py \
      --arms "$arms" --control "$control" --out "$dir/kernel.ipynb" >/dev/null
  cat > "$dir/kernel-metadata.json" <<JSON
{
  "id": "$USER/mirage-sigrange-$name",
  "title": "mirage sigrange $name",
  "code_file": "kernel.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "machine_shape": "",
  "dataset_sources": ["$DATASET"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON
  printf '%-10s arms=%-55s control=%s\n' "$name" "$arms" "$control"
}

stage control "none"                                               yes

# Original split, one seed pair per kernel. Runs a and b each finished their
# nocond arm and then died on the resume path, so those two arms are already
# collected into configs/ by scripts/kaggle_salvage.py.
stage a       "sigrange_rad_nocond_s42,sigrange_rad_cov_s42"       no
stage b       "sigrange_rad_nocond_s43,sigrange_rad_cov_s43"       no
stage c       "sigrange_rad_nocond_s44,sigrange_rad_cov_s44"       no

# What is actually left after the salvage: the three cov arms. Split across two
# kernels because Kaggle runs at most 2 batch GPU sessions at once, so this is
# the shortest wall time, and a failure in one does not take the other down.
# cov is the slower arm, about 17.9 s/epoch against 10.8 for nocond.
stage cov1    "sigrange_rad_cov_s42,sigrange_rad_cov_s43"          no
stage cov2    "sigrange_rad_cov_s44"                               no

echo
echo "pushed with: kaggle kernels push -p build/kernel_<name>"
