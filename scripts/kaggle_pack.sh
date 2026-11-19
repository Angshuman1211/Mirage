#!/usr/bin/env bash
# Build the Kaggle upload bundle for the sigma-range retrain (P6-D28).
#
# WHY A BUNDLE AND NOT THE REPO. `configs/` is 4.0 GB of existing checkpoints and
# `.git` is 2.9 GB. The job itself needs about 25 MB: the two packages, the train
# entry point, seven config.yaml files and one 14 MB HDF file. Uploading the repo
# would be three orders of magnitude of waste.
#
# THE CONTROL ARM. `control_rad_cov_s42` is a copy of `h9_rad_cov_s42/config.yaml`
# with NO checkpoints beside it, deliberately. If the trained `model__latest.pt`
# were copied too, `train_local.py` would find it and RESUME instead of training,
# and the control would verify nothing. Its reference curve travels separately in
# `reference/`, so the notebook can compare without the arm being resumable.
#
#   bash scripts/kaggle_pack.sh            # -> build/kaggle_bundle/
#   bash scripts/kaggle_pack.sh --zip      # also build/mirage_sigrange_kaggle.zip
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=build/kaggle_bundle
rm -rf "$OUT"
mkdir -p "$OUT"/{fm4ar,mirage,scripts,configs,data/abc_rad,reference}

echo "[pack] python packages"
# fm4ar: the package and the training entry script. Drop caches and the test
# suite, neither of which the training path touches.
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='tests/' --exclude='.mypy_cache' --exclude='.pytest_cache' \
      fm4ar/ "$OUT/fm4ar/"
rsync -a --exclude='__pycache__' --exclude='*.pyc' mirage/ "$OUT/mirage/"

echo "[pack] entry points"
cp scripts/train.py "$OUT/scripts/"
[ -f scripts/run_sigrange_retrain.sh ] && cp scripts/run_sigrange_retrain.sh "$OUT/scripts/"

echo "[pack] configs"
for d in configs/sigrange_rad_*/; do
  n=$(basename "$d")
  mkdir -p "$OUT/configs/$n"
  cp "$d/config.yaml" "$OUT/configs/$n/config.yaml"
done

# Control arm: config only, no checkpoints (see the note at the top).
mkdir -p "$OUT/configs/control_rad_cov_s42"
cp configs/h9_rad_cov_s42/config.yaml "$OUT/configs/control_rad_cov_s42/config.yaml"
cp configs/h9_rad_cov_s42/history_to_ep225.csv \
   "$OUT/reference/h9_rad_cov_s42_history.csv"

echo "[pack] training data"
cp data/abc_rad/abc_rad_train.hdf "$OUT/data/abc_rad/"

# The theta scaler reads dataset stats from mirage/register.py, not from disk, so
# only the train file is needed. fm4ar random_splits it into train/valid itself.

cat > "$OUT/README_BUNDLE.md" <<'EOF'
# MIRAGE sigma-range retrain, Kaggle bundle (P6-D28)

Upload this whole folder as a Kaggle **Dataset**, then run the notebook
`kaggle_sigrange_retrain.ipynb` with a GPU accelerator attached.

Contents:

| path | what |
|---|---|
| `fm4ar/`, `mirage/` | the two packages, no caches or tests |
| `scripts/train.py` | training entry point, registers MIRAGE into fm4ar |
| `configs/sigrange_rad_{nocond,cov}_s{42,43,44}/` | the six arms to train |
| `configs/control_rad_cov_s42/` | control arm, config only, no checkpoints |
| `reference/h9_rad_cov_s42_history.csv` | RTX 5070 reference curve for the control |
| `data/abc_rad/abc_rad_train.hdf` | 32000 training spectra |

The control arm exists because a device mismatch silently corrupted the published
coverage numbers by +0.110 and -0.128 in opposite directions (P6-D27). Verify the
hardware reproduces the reference curve before trusting six new arms trained on it.

`README_BUNDLE.md` is also the marker the notebook searches for, so keep it here.
Kaggle mounts an attached dataset at `/kaggle/input/datasets/<owner>/<slug>/`,
three levels down and not at `/kaggle/input/<slug>/`, so the notebook searches
recursively for this file rather than guessing a depth.
EOF

echo
du -sh "$OUT"
find "$OUT" -maxdepth 2 -type d | sed "s|$OUT|  .|" | sort

if [ "${1:-}" = "--zip" ]; then
  (cd build && zip -qr mirage_sigrange_kaggle.zip kaggle_bundle)
  echo
  ls -lh build/mirage_sigrange_kaggle.zip
fi

echo
echo "[pack] done -> $OUT"
