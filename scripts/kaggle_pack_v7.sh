#!/usr/bin/env bash
# Build the Kaggle upload bundle for V7, the pooled three-spectrum CycleGAN
# rebuild (08_work_split_v2.md, item V7).
#
# WHY THE REAL DOMAINS ARE BUILT LOCALLY AND SHIPPED AS NPZ. The first step of
# run_v7_pooled.sh is build_real_domain.py, which rebins 138 MB of out-of-transit
# frame CSVs onto the 26 shared ABC bins. That is pandas work with no GPU in it,
# so running it here instead of on Kaggle keeps 138 MB out of the upload and
# removes the step most likely to fail on a remote path. The seven npz files it
# produces are 1.6 MB in total.
#
# WHY THE PUBLISHED CHECKPOINTS ARE NOT IN THE BUNDLE. v7_table.py compares the
# pooled translator against the published per-target ones, but it reads their
# stored `ablation.json`, not their weights. The weights are 29 MB each and would
# more than triple the upload to buy nothing.
#
#   bash scripts/kaggle_pack_v7.sh          # -> build/kaggle_v7_bundle/
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=build/kaggle_v7_bundle
rm -rf "$OUT"
mkdir -p "$OUT"/{mirage,scripts,data/abc,data/real_domain,configs}

echo "[pack] mirage package"
# The whole package, not just nn/cyclegan.py. Both entry points load the CycleGAN
# by file path to avoid mirage/__init__.py calling register(), which needs fm4ar,
# so fm4ar is not in this bundle at all. Shipping the package whole costs 220 KB
# and keeps that file-path import honest.
rsync -a --exclude='__pycache__' --exclude='*.pyc' mirage/ "$OUT/mirage/"

echo "[pack] entry points"
for f in train_cyclegan.py run_cyclegan_ablation_standalone.py v7_table.py; do
  cp "scripts/$f" "$OUT/scripts/"
done

echo "[pack] domain A"
cp data/abc/abc_train.hdf "$OUT/data/abc/"     # 31 MB, training spectra
cp data/abc/abc_test.hdf  "$OUT/data/abc/"     # 3.9 MB, scoring spectra

echo "[pack] domain B, prebuilt"
for f in data/real_domain/real_domain_all3_abcniriss.npz \
         data/real_domain/real_domain_wasp39_abcniriss_{full,n56}.npz \
         data/real_domain/real_domain_wasp96_abcniriss_{full,n56}.npz \
         data/real_domain/real_domain_k218_abcniriss_{full,n56}.npz; do
  [ -f "$f" ] || { echo "[pack] MISSING $f - run the build_real_domain.py calls first" >&2; exit 1; }
  cp "$f" "$OUT/data/real_domain/"
done

echo "[pack] published ablations, for the comparison column"
for d in cyclegan_t2_wasp39 cyclegan_t2_wasp96 cyclegan_t2_k218 \
         cyclegan_ctrl_wasp39_26b_full; do
  mkdir -p "$OUT/configs/$d"
  cp "configs/$d/ablation.json" "$OUT/configs/$d/ablation.json"
done

cat > "$OUT/README_BUNDLE_V7.md" <<'EOF'
# MIRAGE V7, pooled three-spectrum CycleGAN rebuild, Kaggle bundle

Upload this whole folder as a Kaggle **Dataset**, then run `kernel.ipynb` with a
GPU accelerator attached.

| path | what |
|---|---|
| `mirage/` | the package, for `mirage/nn/cyclegan.py` |
| `scripts/train_cyclegan.py` | the one training run |
| `scripts/run_cyclegan_ablation_standalone.py` | the six scorings |
| `scripts/v7_table.py` | the summary table and the decision-log sentence |
| `data/abc/abc_train.hdf` | domain A, 73,044 synthetic spectra |
| `data/abc/abc_test.hdf` | domain A held out, used by the scorings |
| `data/real_domain/real_domain_all3_abcniriss.npz` | domain B, all three targets pooled |
| `data/real_domain/real_domain_{target}_abcniriss_{full,n56}.npz` | per-target scoring sets |
| `configs/cyclegan_*/ablation.json` | published per-target numbers, comparison column only |

What V7 replaces. The published real side was ONE reduced WASP-39b spectrum
bootstrap-expanded to 500 samples with 1 percent Gaussian jitter. A discriminator
shown 500 near-copies of one spectrum learns that spectrum, so the comparison was
never a fair test. Domain B here is 3,393 genuine out-of-transit frames across
three targets, each divided by its own mean transit depth.

`README_BUNDLE_V7.md` is the marker the notebook searches for, so keep it here.
Kaggle mounts an attached dataset at `/kaggle/input/datasets/<owner>/<slug>/`,
three levels down and not at `/kaggle/input/<slug>/`, so the notebook searches
recursively for this file rather than guessing a depth.
EOF

echo
du -sh "$OUT"
find "$OUT" -maxdepth 2 -type d | sed "s|$OUT|  .|" | sort
echo
echo "[pack] done -> $OUT"
