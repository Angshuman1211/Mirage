#!/usr/bin/env bash
# Carry checkpoints from a finished Kaggle run into the next one (P6-D28).
#
# WHY THIS EXISTS. A batch kernel starts with an empty /kaggle/working, so an arm
# that hit its runtime wall cannot be resumed the way an interactive session
# resumes it. fm4ar's resume is driven by model__latest.pt sitting in the
# experiment directory, so the checkpoints have to travel back in as a dataset.
# This downloads a run's archive, rebuilds the checkpoint dataset from it, and
# attaches that dataset to the kernel for the next push.
#
# The control arm is EXCLUDED. Restoring its checkpoints would make it resume
# instead of train, and it would then verify nothing about the hardware.
#
#   bash scripts/kaggle_chain.sh a        # after mirage-sigrange-a timed out
#   kaggle kernels push -p build/kernel_a # resumes where it stopped
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: kaggle_chain.sh <kernel-name, e.g. a>}"
USER=angshuman12
KERNEL="$USER/mirage-sigrange-$NAME"
CKPT_DS="$USER/mirage-sigrange-ckpt"
OUT="build/chain/$NAME"
STAGE="build/chain/ckpt"

rm -rf "$OUT" && mkdir -p "$OUT"
echo "[chain] downloading output of $KERNEL"
kaggle kernels output "$KERNEL" -p "$OUT" >/dev/null

ZIP="$OUT/sigrange_results.zip"
[ -f "$ZIP" ] || { echo "[chain] no sigrange_results.zip in the output."; \
                   echo "        The run probably died before the collect cell."; \
                   ls -la "$OUT"; exit 1; }

mkdir -p "$STAGE"
echo "[chain] merging checkpoints into $STAGE"
TMP=$(mktemp -d)
unzip -qo "$ZIP" -d "$TMP"
for d in "$TMP"/*/; do
  arm=$(basename "$d")
  case "$arm" in control*) echo "  skip $arm (the control trains fresh)"; continue ;; esac
  mkdir -p "$STAGE/$arm"
  # Only the checkpoints. Configs and histories come from the bundle and the run.
  for f in "$d"model__latest.pt "$d"model__stage_0.pt; do
    [ -f "$f" ] && cp "$f" "$STAGE/$arm/" && echo "  $arm/$(basename "$f")"
  done
done
rm -rf "$TMP"

cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "mirage-sigrange-ckpt",
  "id": "$CKPT_DS",
  "licenses": [{"name": "unknown"}]
}
JSON

if kaggle datasets status "$CKPT_DS" >/dev/null 2>&1; then
  echo "[chain] versioning $CKPT_DS"
  # -r zip is REQUIRED. The default dir-mode is `skip`, which would silently
  # upload nothing but the metadata file and leave every per-arm directory out.
  kaggle datasets version -p "$STAGE" -r zip -d -q \
      -m "resume checkpoints from $KERNEL"
else
  echo "[chain] creating $CKPT_DS"
  # -d is a `version` flag only, not a `create` flag.
  kaggle datasets create -p "$STAGE" -r zip -q
fi

META="build/kernel_$NAME/kernel-metadata.json"
python - "$META" "$CKPT_DS" <<'PY'
import json, sys
p, ds = sys.argv[1], sys.argv[2]
m = json.load(open(p))
if ds not in m["dataset_sources"]:
    m["dataset_sources"].append(ds)
    json.dump(m, open(p, "w"), indent=2)
    print(f"[chain] attached {ds} to {p}")
else:
    print(f"[chain] {ds} already attached to {p}")
PY

echo
echo "[chain] done. Wait for the dataset to finish processing, then:"
echo "        kaggle kernels push -p build/kernel_$NAME"
