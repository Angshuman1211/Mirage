"""Pull trained arms out of a downloaded Kaggle kernel output into configs/ (P6-D28).

WHY THIS IS NEEDED. Kaggle commits /kaggle/working even when the notebook raises,
so a run that trained an arm and then died still ships that arm's checkpoints.
Runs a and b each trained their nocond arm to early stopping and then failed on
the resume path, so the training is complete and only needs collecting.

An arm counts as complete when early stopping fired, which means `stage_patience`
epochs passed with no improvement. fm4ar writes model__stage_0.pt only when the
EPOCH CAP is reached, so a finished arm usually has no stage checkpoint at all.

  python scripts/kaggle_salvage.py <downloaded_output_dir> [more dirs ...]
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def history(d: Path):
    best = None
    for f in ("model__stage_0.pt", "model__latest.pt", "model__best.pt"):
        p = d / f
        if not p.exists():
            continue
        h = torch.load(p, map_location="cpu", weights_only=False).get("history")
        if h is None:
            continue
        h = pd.DataFrame(h)
        if best is None or len(h) > len(best):
            best = h
    return best


def main(dirs: list[str]) -> int:
    found = 0
    for raw in dirs:
        base = Path(raw)
        cfg_root = base / "mirage" / "configs"
        if not cfg_root.exists():
            cfg_root = base / "configs"
        if not cfg_root.exists():
            print(f"[skip] no configs/ under {base}")
            continue
        for d in sorted(cfg_root.glob("sigrange_*")):
            if not (d / "model__latest.pt").exists():
                continue
            h = history(d)
            if h is None:
                print(f"[skip] {d.name}: checkpoints but no history")
                continue
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            patience = cfg["training"]["stage_0"]["early_stopping"]["stage_patience"]
            bi = int(h["test_loss"].values.argmin())
            after = len(h) - 1 - bi
            # Order matters. The marker is authoritative because the runner
            # wrote it from train_local.py's own verdict. The arithmetic test is
            # a fallback and gives a FALSE NEGATIVE when the longest surviving
            # history ends at the best epoch, which happens because
            # model__latest.pt is only written every backup_interval epochs.
            complete = ((d / "TRAINING_COMPLETE").exists()
                        or (d / "model__stage_0.pt").exists()
                        or after >= patience)
            dst = ROOT / "configs" / d.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("model__best.pt", "model__latest.pt", "model__stage_0.pt"):
                if (d / f).exists():
                    shutil.copy2(d / f, dst / f)
            h.to_csv(dst / f"history_to_ep{int(h['epoch'].max())}.csv", index=False)
            if complete:
                (dst / "TRAINING_COMPLETE").write_text(
                    f"early stopping fired, {after} epochs after best\n")
            found += 1
            print(f"[{'done' if complete else 'PARTIAL'}] {d.name:<26} "
                  f"{len(h):>4} epochs  best {h['test_loss'].min():.6f} "
                  f"@ ep {int(h['epoch'].iloc[bi])}  ({after} after best) -> {dst}")
    print(f"\n{found} arm(s) salvaged")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or [str(ROOT / "build/chain")]))
