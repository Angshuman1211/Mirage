"""Emit the Kaggle notebook for V7, the pooled three-spectrum CycleGAN rebuild.

V7 asks whether the translation null survives once the real domain is built from
all three reduced spectra instead of one WASP-39b spectrum jittered 500 ways
(08_work_split_v2.md, item V7). It is the last CycleGAN run the project does.

WHY THIS IS A MUCH SIMPLER NOTEBOOK THAN make_kaggle_notebook.py. That one trains
FMPE arms, which early-stop at unpredictable epochs and can overrun the 12 h batch
cap, so it needs resume, completion detection and a wall-clock budget. V7 is one
fixed 200-epoch run at roughly an hour, so none of that applies. What it keeps
from that notebook is the part that was learned the hard way: the recursive search
for the bundle marker, because Kaggle mounts a dataset three levels down.

NO fm4ar HERE. Both entry points load the CycleGAN by file path precisely to skip
mirage/__init__.py's register() call, so the bundle carries no fm4ar and the
import layout that cost a session in the sigma-range run cannot arise.

    python scripts/make_kaggle_notebook_v7.py --out build/kernel_v7/kernel.ipynb
"""
import argparse
import json
from pathlib import Path

TARGETS = ("wasp39", "wasp96", "k218")
RUN = "configs/cyclegan_v7_pooled3"

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="build/kernel_v7/kernel.ipynb")
args = p.parse_args()

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s}

cells = []

cells.append(md(f"""# MIRAGE V7, pooled three-spectrum CycleGAN rebuild

One translator trained on a real domain pooled from **all three** reduced spectra,
then scored against each target's own real frames. Replaces the published real
side, which was one WASP-39b spectrum bootstrap-expanded to 500 samples with
1 percent Gaussian jitter.

| | |
|---|---|
| train | 1 run, {args.epochs} epochs, batch {args.batch_size}, seed {args.seed} |
| domain A | 73,044 ABC synthetic spectra, 26 shared bins |
| domain B | 3,393 real out-of-transit frames, 3 targets, target-balanced |
| score | 6 cells, each target at 56 frames and at all frames |
| expect | about 1 h on a T4, far inside the 12 h batch cap |

**The verdict is internal to this run.** `v7_table.py` decides holds / weakens /
reverses from the three arms scored inside each cell, which are the simulated
baseline, the domain-randomisation baseline and this translator, all on the same
device against the same frames. The published column is informational only, and
it is NOT a controlled comparison, for two reasons worth stating in the write-up
rather than discovering later. The published translators were trained on **mps**,
and P6-D27 found a device artifact in this project that moved published numbers by
+0.110 and -0.128 in opposite directions. They also carry **no seed** in their
`config.json`, so they are not reproducible on their own device either."""))

# ------------------------------------------------------------------ environment
cells.append(md("""## 1. Environment

The training runs in a subprocess, so the probe runs in one too. A subprocess
inherits the environment but not `sys.path`, and what matters is what the trainer
will see rather than what this kernel sees.

A CPU fallback is refused rather than allowed to proceed. At batch 64 over 73,044
spectra this is 228,400 optimiser steps, which is an hour on a T4 and most of a
week on the CPUs Kaggle attaches."""))
cells.append(code('''import json, os, shutil, subprocess, sys, time
from collections import deque
from pathlib import Path

_probe = subprocess.run(
    [sys.executable, "-c",
     "import torch; print(torch.__version__); print(torch.cuda.is_available()); "
     "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"],
    capture_output=True, text=True)
print(_probe.stdout or _probe.stderr)
_lines = _probe.stdout.strip().splitlines()
assert _probe.returncode == 0 and len(_lines) == 3, _probe.stderr
assert _lines[1] == "True", (
    "No GPU visible to the subprocess that will train. Set the notebook "
    "accelerator to a GPU and re-run. Refusing to fall back to CPU.")
print("torch", _lines[0], "on", _lines[2])
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader"], check=False)'''))

# ---------------------------------------------------------------------- staging
cells.append(md("""## 2. Stage the bundle into a writable directory

`/kaggle/input` is read-only and the run writes checkpoints, a training log and
six JSON files, so the bundle is copied to `/kaggle/working` first. Every path
inside the entry points is relative to the bundle root, so the working directory
has to be that root."""))
cells.append(code('''# WHERE KAGGLE ACTUALLY MOUNTS A DATASET. An attached dataset appears at
#     /kaggle/input/datasets/<owner>/<slug>/
# THREE levels down, not at /kaggle/input/<slug>/. A fixed-depth search misses it
# and then reports the dataset as unattached when it is attached perfectly well.
# Search recursively for the bundle's own marker file instead.
roots = [q.parent
         for q in sorted(Path("/kaggle/input").rglob("README_BUNDLE_V7.md"))
         if (q.parent / "scripts" / "train_cyclegan.py").exists()]
if not roots:
    roots = [q.parent.parent
             for q in sorted(Path("/kaggle/input").rglob("scripts/train_cyclegan.py"))]
if not roots:
    print("nothing matched. /kaggle/input holds:")
    for q in sorted(Path("/kaggle/input").rglob("*"))[:40]:
        print("   ", q)
assert roots, ("Bundle not found under /kaggle/input. Upload build/kaggle_v7_bundle "
               "as a Kaggle Dataset and attach it to this notebook.")
SRC = roots[0]
print("bundle:", SRC)

WORK = Path("/kaggle/working/mirage_v7")
if WORK.exists():
    # Re-run in a resumed session: refresh code and inputs, keep any results.
    for sub in ("mirage", "scripts", "data"):
        shutil.rmtree(WORK / sub, ignore_errors=True)
        shutil.copytree(SRC / sub, WORK / sub)
    print("refreshed", WORK, "- existing results kept")
else:
    shutil.copytree(SRC, WORK)
    print("copied to", WORK)
os.chdir(WORK)

ENV = dict(os.environ)
ENV["PYTHONPATH"] = os.pathsep.join(
    [str(WORK)] + ([ENV["PYTHONPATH"]] if ENV.get("PYTHONPATH") else []))

RUN = Path("''' + RUN + '''")
print("cwd:", os.getcwd())
for q in ("data/abc/abc_train.hdf", "data/abc/abc_test.hdf",
          "data/real_domain/real_domain_all3_abcniriss.npz"):
    assert Path(q).exists(), f"missing {q}"
    print(f"  {q:<52}{Path(q).stat().st_size / 1e6:>8.1f} MB")

# The pooled domain, read back rather than trusted. Three targets have to be
# present or --balance-targets silently balances a single-target set.
import numpy as np
_d = np.load("data/real_domain/real_domain_all3_abcniriss.npz", allow_pickle=True)
_t, _c = np.unique(_d["sample_target"], return_counts=True)
print("\\npooled domain B:", dict(zip(_t.tolist(), _c.tolist())),
      "| bins", _d["samples"].shape[1], "| covered", int(_d["covered"].sum()))
assert len(_t) == 3, f"expected three targets in the pooled domain, got {_t.tolist()}"'''))

# --------------------------------------------------------------------- training
cells.append(md(f"""## 3. Train the one pooled translator

Streamed line by line rather than captured. A batch run that dies with its output
buffered gives you nothing to diagnose from, and the per-epoch lines are the only
evidence of how far it got.

`--balance-targets` is what makes the pooling meaningful. WASP-39b contributes
3,241 of the 3,393 frames, so without it the discriminator sees a WASP-39b domain
with a 4 percent contaminant and V7 reduces to the run it was meant to replace."""))
cells.append(code(f'''CMD = [sys.executable, "scripts/train_cyclegan.py",
       "--epochs", "{args.epochs}", "--batch-size", "{args.batch_size}",
       "--seed", "{args.seed}",
       "--real-domain", "data/real_domain/real_domain_all3_abcniriss.npz",
       "--balance-targets", "--device", "cuda", "--ckpt-dir", str(RUN)]
print(" ".join(CMD), "\\n")

t0 = time.time()
if (RUN / "cyclegan_best.pt").exists():
    print("cyclegan_best.pt already present, skipping the train")
    rc = 0
else:
    proc = subprocess.Popen(CMD, env=ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = deque(maxlen=60)
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    rc = proc.wait()
    if rc != 0:
        print("\\n".join(tail))
print(f"\\ntrain exit {{rc}} in {{(time.time() - t0) / 60:.1f}} min")
assert rc == 0, "training failed, see the tail above"
assert (RUN / "cyclegan_best.pt").exists(), "no cyclegan_best.pt was written"'''))

cells.append(md("""## 4. Drop the per-epoch checkpoints

29 MB each and read by nothing. `cyclegan_best.pt` is what the scorings load.
This matches `run_v7_pooled.sh`, which deletes them for the same reason."""))
cells.append(code('''_n = 0
for q in sorted(RUN.glob("cyclegan_epoch_*.pt")):
    _n += 1; q.unlink()
print(f"removed {_n} per-epoch checkpoints")
for q in sorted(RUN.iterdir()):
    print(f"  {q.name:<28}{q.stat().st_size / 1e6:>8.1f} MB")'''))

# ---------------------------------------------------------------------- scoring
cells.append(md("""## 5. Score the pooled translator, six cells

Each target twice. At 56 frames the rows line up with the published per-target
tier-2 translators, and at every frame they line up with the full-frame control.
P5-D11 measured the CycleGAN's MMD falling 71 percent from frame count alone, so
only like-for-like pairs are comparable and the verdict rests on the full-frame
rows.

WASP-96b has exactly 56 real frames, so its two rows are the same set scored
twice and should agree to the seed. That is a free consistency check rather than
a redundancy."""))
cells.append(code(f'''FRAMES = ("full", "n56")
for t in {TARGETS!r}:
    for n in FRAMES:
        out = RUN / f"ablation_{{t}}_{{n}}.json"
        if out.exists():
            print(f"[skip] {{out.name}} exists"); continue
        cmd = [sys.executable, "scripts/run_cyclegan_ablation_standalone.py",
               "--n-samples", "500",
               "--real-domain", f"data/real_domain/real_domain_{{t}}_abcniriss_{{n}}.npz",
               "--ckpt", str(RUN / "cyclegan_best.pt"), "--out", str(out)]
        r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            raise SystemExit(f"scoring failed for {{t}} {{n}}")
        c = json.load(open(out))["conditions"]
        print(f"  {{t:<8}}{{n:<6}} mmd sim {{c['sim_baseline']['mmd_to_real']:.4f}}  "
              f"random {{c['domain_random']['mmd_to_real']:.4f}}  "
              f"pooled {{c['cyclegan_trans']['mmd_to_real']:.4f}}")'''))

# ------------------------------------------------------------------------ table
cells.append(md("""## 6. The table and the decision-log sentence

`v7_table.py` prints one row per target per frame count and then the single
sentence V7's acceptance test asks for, stating whether the null holds, weakens
or reverses."""))
cells.append(code('''r = subprocess.run([sys.executable, "scripts/v7_table.py"],
                   env=ENV, capture_output=True, text=True)
print(r.stdout or r.stderr)
assert r.returncode == 0, r.stderr'''))

# ----------------------------------------------------------------------- output
cells.append(md("""## 7. Leave one archive in the output

Kaggle commits `/kaggle/working`, so everything staged there would otherwise come
home, 36 MB of inputs included. Zip the results and clear the rest."""))
cells.append(code('''import zipfile
os.chdir("/kaggle/working")
ZIP = Path("/kaggle/working/v7_results.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for q in sorted((WORK / RUN).rglob("*")):
        if q.is_file():
            z.write(q, q.relative_to(WORK))
print(f"{ZIP.name}  {ZIP.stat().st_size / 1e6:.1f} MB")
with zipfile.ZipFile(ZIP) as z:
    for i in z.infolist():
        print(f"  {i.filename:<52}{i.file_size / 1e6:>8.1f} MB")
shutil.rmtree(WORK, ignore_errors=True)
print("\\nleft in /kaggle/working:", [q.name for q in Path(".").iterdir()])'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "language": "python",
                                  "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")
