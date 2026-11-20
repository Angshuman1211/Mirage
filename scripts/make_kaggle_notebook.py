"""
Generate `build/kaggle_sigrange_retrain.ipynb`, the Kaggle notebook for the
sigma-range retrain (P6-D28).

Written as a generator rather than a checked-in .ipynb so the reference numbers
in the control cell stay sourced from the repo instead of being pasted once and
drifting. Run it, then upload the result next to the bundle from kaggle_pack.sh.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "kaggle_sigrange_retrain.ipynb"

# Reference curve for the control arm, read from the repo at generation time.
HIST = ROOT / "configs" / "h9_rad_cov_s42" / "history_to_ep225.csv"


# CLI so one generator emits both the control-only kernel and the arm kernels.
# The Kaggle API pushes a notebook as a BATCH run, and a batch run that overruns
# the 12 h cap is killed with its output commit FAILED, so splitting the six arms
# across pushes keeps every run far inside the cap.
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all",
                    help="comma-separated arm names, 'all', or 'none'")
    ap.add_argument("--control", default="yes", choices=("yes", "no"))
    ap.add_argument("--control-epochs", type=int, default=60)
    ap.add_argument("--budget-hours", type=float, default=9.5,
                    help="wall-clock budget for one run, under the 12 h cap")
    ap.add_argument("--out", default="build/kaggle_sigrange_retrain.ipynb")
    return ap.parse_args()


ALL_ARMS = ["sigrange_rad_nocond_s42", "sigrange_rad_cov_s42",
            "sigrange_rad_nocond_s43", "sigrange_rad_cov_s43",
            "sigrange_rad_nocond_s44", "sigrange_rad_cov_s44"]

args = parse_args()
arms = (ALL_ARMS if args.arms == "all"
        else [] if args.arms == "none"
        else [a.strip() for a in args.arms.split(",") if a.strip()])
unknown = [a for a in arms if a not in ALL_ARMS]
assert not unknown, f"unknown arms: {unknown}"
budget_hours = args.budget_hours
control_epochs = args.control_epochs
run_control = args.control == "yes"
OUT = ROOT / args.out
arms_literal = ("[\n    " + ",\n    ".join(f'"{a}"' for a in arms) + ",\n]"
                if arms else "[]")


def reference_rows() -> list[tuple[int, float, float]]:
    rows = []
    for line in HIST.read_text().strip().splitlines()[1:]:
        f = line.split(",")
        rows.append((int(f[0]), float(f[2]), float(f[3])))
    return rows


REF = reference_rows()
REF_BEST = min(REF, key=lambda r: r[2])
REF_SEC_PER_EPOCH = 9.41  # train_time + test_time mean, RTX 5070

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s}

cells = []

# ---------------------------------------------------------------- intro
cells.append(md(f"""# MIRAGE sigma-range retrain (P6-D28)

Trains the six `sigrange_rad_*` arms, which lower `sigma_min` to 1e-3 and switch
to log-uniform sampling so real WASP-39b noise (2.16e-3) sits **inside** the
trained range instead of 23-139x below it (P6-D7, P6-D17).

**It runs a control arm first, and that is not optional.** P6-D27 found that a
device mismatch silently corrupted the published coverage numbers by +0.110 and
-0.128 **in opposite directions**, with the artifact landing on exactly the arms
that ran on the wrong device. Training six new arms on unverified hardware and
comparing them against RTX 5070 baselines would reintroduce that confound. The
control retrains `h9_rad_cov_s42` for a few dozen epochs and compares the loss
curve against the 5070 reference shipped in `reference/`.

Reference, from `configs/h9_rad_cov_s42/history_to_ep225.csv`:

* best test loss **{REF_BEST[2]:.6f}** at epoch {REF_BEST[0]}
* **{REF_SEC_PER_EPOCH} s/epoch** on an RTX 5070, 225 epochs in 35.3 min
* radius CUDA band 1.124 to 1.159; an MPS run of the same config sat at 1.3729

**Session handling.** `local.max_runtime` is in seconds, and fm4ar saves
`model__latest.pt` on `MAX_RUNTIME_EXCEEDED`, which `train_local.py` resumes from.
`BUDGET_HOURS` is set below the Kaggle cap, so an arm that hits the wall continues
in the next run rather than restarting. A stage is finished when
`model__stage_0.pt` exists.

**Do not raise `n_workers`.** With `n_workers: 0` the noise transform draws from
the generator in the main process. Workers each get their own RNG copy, which
changes the injected noise and breaks comparability with the h9 arms."""))

# ---------------------------------------------------------------- settings
cells.append(md("## 1. Settings"))
cells.append(code(f"""# WALL-CLOCK BUDGET for this run, and why it is global rather than per-arm. A
# Kaggle BATCH run (what the API pushes) that overruns the 12 h cap is killed and
# its output commit FAILS, losing every checkpoint. An interactive session keeps
# /kaggle/working when it times out, so the two modes are NOT interchangeable.
# Everything below counts down from one deadline, and the run stops starting new
# arms while there is still time to save what finished.
BUDGET_HOURS   = {budget_hours}

# Do not start another arm with less than this left.
MIN_ARM_MIN    = 25

# Control arm length. The reference curve is monotone and smooth, so a few dozen
# epochs is enough to tell "same hardware band" from "MPS-style failure".
CONTROL_EPOCHS = {control_epochs}

# Train these.
ARMS = {arms_literal}

RUN_CONTROL = {run_control}"""))

# ---------------------------------------------------------------- env
cells.append(md("""## 2. Environment

Installs only what is missing, under a constraints file pinned to Kaggle's own
`torch` and `numpy`. Without the constraint, pip resolving a dependency can swap
out the CUDA-enabled torch build for a CPU wheel and the whole run silently
becomes the thing this experiment exists to avoid."""))
cells.append(code('''import importlib, subprocess, sys, os, json, shutil, time
from collections import deque

# Start the clock before anything slow, so the deadline covers the pip install
# and the data copy and not only the training.
T0 = time.time()
from pathlib import Path

import torch, numpy
print("torch", torch.__version__, "| numpy", numpy.__version__)
assert torch.cuda.is_available(), (
    "No CUDA. Attach a GPU accelerator (Settings -> Accelerator). "
    "Training on CPU/MPS under-trains these models (P6-D1) and the result "
    "would be uninterpretable.")
print("gpu  ", torch.cuda.get_device_name(0))
print("cpus ", os.cpu_count())

CONSTRAINTS = Path("/kaggle/working/constraints.txt")
CONSTRAINTS.write_text(
    f"torch=={torch.__version__.split('+')[0]}\\nnumpy=={numpy.__version__}\\n")

# (import name, pip name). glasflow and normflows ARE required even though this
# is an FMPE run: fm4ar/models/build_model.py imports NPEModel at module level,
# which pulls nn/flows.py.
DEPS = [("torchdiffeq", "torchdiffeq"), ("glasflow", "glasflow"),
        ("normflows", "normflows"), ("threadpoolctl", "threadpoolctl"),
        ("git", "gitpython"), ("pydantic", "pydantic"), ("wandb", "wandb"),
        ("spectres", "spectres"), ("colour", "colour-science"),
        ("deepdiff", "deepdiff"), ("dill", "dill"), ("h5py", "h5py"),
        ("yaml", "pyyaml"), ("tqdm", "tqdm"), ("pandas", "pandas")]

missing = []
for mod, pkg in DEPS:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)

if missing:
    print("installing:", " ".join(missing))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-c", str(CONSTRAINTS), *missing], check=True)
else:
    print("all dependencies already present")

# Re-verify in a SUBPROCESS, because a pip dependency can quietly swap the CUDA
# torch wheel for a CPU one and the run would silently become the very thing
# this experiment exists to avoid.
#
# It MUST be a subprocess. importlib.reload(torch) re-executes
# torch/__init__.py, which re-registers the `triton` TORCH_LIBRARY namespace and
# dies with "Only a single TORCH_LIBRARY can be used to register the namespace
# triton". A fresh interpreter also reports what the TRAINER will see, which is
# the question actually being asked.
_chk = subprocess.run(
    [sys.executable, "-c",
     "import torch; print(torch.__version__, torch.cuda.is_available())"],
    capture_output=True, text=True)
assert _chk.returncode == 0, f"torch is broken after install:\\n{_chk.stderr}"
_ver, _cuda = _chk.stdout.split()
print(f"after install, a fresh process sees torch {_ver}, cuda {_cuda}")
assert _cuda == "True", (
    f"CUDA disappeared after pip install (torch now {_ver}) - stop. "
    "Something in DEPS pulled a CPU wheel past the constraints file.")'''))

# ---------------------------------------------------------------- stage
cells.append(md("""## 3. Stage the bundle into a writable directory

`/kaggle/input` is read-only and training writes checkpoints into the experiment
directory, so the bundle is copied to `/kaggle/working` first. That also makes the
checkpoints part of the notebook output you can download."""))
cells.append(code('''# WHERE KAGGLE ACTUALLY MOUNTS A DATASET. An attached dataset appears at
#     /kaggle/input/datasets/<owner>/<slug>/
# THREE levels down, not at /kaggle/input/<slug>/. A fixed-depth search misses it
# and then reports the dataset as unattached when it is attached perfectly well,
# which is a confusing way to lose a run. Search recursively for the bundle's own
# marker file instead, so the layout stops mattering.
roots = [p.parent
         for p in sorted(Path("/kaggle/input").rglob("README_BUNDLE.md"))
         if (p.parent / "scripts" / "train.py").exists()]
if not roots:
    # Marker gone: fall back to the entry point. train.py is unique in the
    # bundle, so this cannot pick up one of fm4ar's own scripts.
    roots = [p.parent.parent
             for p in sorted(Path("/kaggle/input").rglob("scripts/train.py"))]
if not roots:
    print("nothing matched. /kaggle/input holds:")
    for q in sorted(Path("/kaggle/input").rglob("*"))[:40]:
        print("   ", q)
assert roots, ("Bundle not found under /kaggle/input. Upload build/kaggle_bundle "
               "as a Kaggle Dataset and attach it to this notebook.")
SRC = roots[0]
print("bundle:", SRC)

WORK = Path("/kaggle/working/mirage")
if not WORK.exists():
    shutil.copytree(SRC, WORK)
    print("copied to", WORK)
else:
    # Re-running the notebook in a resumed session: keep existing checkpoints,
    # refresh only code and configs.
    for sub in ("fm4ar", "mirage", "scripts", "reference"):
        if (SRC / sub).exists():
            shutil.rmtree(WORK / sub, ignore_errors=True)
            shutil.copytree(SRC / sub, WORK / sub)
    print("refreshed code in existing", WORK, "- checkpoints kept")

os.chdir(WORK)

# IMPORT LAYOUT, and getting this wrong costs a whole session. The fm4ar PACKAGE
# is at <WORK>/fm4ar/fm4ar/, so `import fm4ar` needs <WORK>/fm4ar on the path, not
# <WORK>. With only <WORK> on the path `import fm4ar` still SUCCEEDS, as an empty
# namespace package with __file__ None, and the failure surfaces much later as a
# missing submodule. The `mirage` package is at <WORK>/mirage/, so it needs <WORK>.
for p in (str(WORK / "fm4ar"), str(WORK)):
    if p not in sys.path:
        sys.path.insert(0, p)

# The trainer runs in a SUBPROCESS, which inherits the environment but not
# sys.path, so the same two entries have to go into PYTHONPATH as well.
ENV = dict(os.environ)
ENV["PYTHONPATH"] = os.pathsep.join(
    [str(WORK / "fm4ar"), str(WORK)]
    + ([ENV["PYTHONPATH"]] if ENV.get("PYTHONPATH") else []))

print("cwd:", os.getcwd())
print("data:", (WORK / "data/abc_rad/abc_rad_train.hdf").stat().st_size / 1e6, "MB")

import fm4ar, mirage  # mirage registers SpectraEncoder / InjectCorrelatedNoise
import fm4ar.models.build_model  # the import that fails under a namespace package

for name, mod in (("fm4ar", fm4ar), ("mirage", mirage)):
    assert mod.__file__ is not None, (
        f"{name} resolved to a namespace package. Check sys.path.")
    assert str(WORK) in mod.__file__, (
        f"{name} resolved to {mod.__file__}, outside the staged bundle.")
    print(f"{name:<7} -> {mod.__file__}")

# Confirm the subprocess sees them too, which is what actually trains.
_probe = subprocess.run(
    [sys.executable, "-c",
     "import fm4ar, mirage, fm4ar.models.build_model as b; print(b.__file__)"],
    env=ENV, capture_output=True, text=True)
assert _probe.returncode == 0, f"subprocess import failed:\\n{_probe.stderr}"
print("subprocess ->", _probe.stdout.strip())'''))

# ---------------------------------------------------------------- runtime
cells.append(md("""## 3b. Restore checkpoints from an earlier run

A batch kernel starts with an empty `/kaggle/working`, so an unfinished arm
cannot be picked up the way an interactive session picks it up. Checkpoints from
an earlier run arrive as a second attached dataset instead, laid out as
`<arm>/model__*.pt`, and are overlaid onto the staged configs here."""))
cells.append(code('''# The control arm is excluded on purpose. If its checkpoints were restored,
# train_local.py would RESUME it and the control would verify nothing about this
# hardware, which is the same trap kaggle_pack.sh avoids by shipping the control
# config without checkpoints.
# Recursive from /kaggle/input for the same reason as the staging cell: the real
# mount is /kaggle/input/datasets/<owner>/<slug>/.
restored = {}
for src in sorted(Path("/kaggle/input").rglob("model__*.pt")):
    arm = src.parent.name
    if arm.startswith("control"):
        print(f"skipped {arm}/{src.name} - the control trains fresh")
        continue
    dst = WORK / "configs" / arm
    if not dst.exists():
        continue
    shutil.copy2(src, dst / src.name)
    restored.setdefault(arm, []).append(src.name)

if restored:
    for arm, files in sorted(restored.items()):
        print(f"{arm:<28} restored {sorted(files)}")
else:
    print("no checkpoint dataset attached - every arm starts from scratch")'''))

cells.append(md("""## 3c. Make fm4ar's resume path work on torch 2.6 and later

`torch.load` flipped `weights_only` to `True` in torch 2.6. fm4ar stores each
run's history as a pandas DataFrame INSIDE the checkpoint, so every load on the
resume path raises `UnpicklingError` under Kaggle's torch 2.11. The repo pins
torch 2.2.2, where the default is still `False`, so this never appears locally
and only bites the moment an arm tries to resume."""))
cells.append(code('''# These are our own checkpoints, written by this same notebook minutes earlier,
# so turning the weights-only guard off here carries no trust question.
_major, _minor = (int(x) for x in torch.__version__.split(".")[:2])
if (_major, _minor) >= (2, 6):
    LOAD_PATCHES = [
        ("fm4ar/fm4ar/models/build_model.py",
         'torch.load(file_path, map_location=torch.device("cpu"))',
         'torch.load(file_path, map_location=torch.device("cpu"), '
         'weights_only=False)'),
        ("fm4ar/fm4ar/models/base.py",
         "torch.load(file_path, map_location=self.device)",
         "torch.load(file_path, map_location=self.device, weights_only=False)"),
        ("fm4ar/fm4ar/torchutils/weights.py",
         "torch.load(file_path, map_location=device)",
         "torch.load(file_path, map_location=device, weights_only=False)"),
    ]
    for rel, old, new in LOAD_PATCHES:
        p = WORK / rel
        s = p.read_text()
        if new in s:
            print(f"already patched: {rel}")
            continue
        n = s.count(old)
        assert n == 1, f"{rel}: expected 1 such load call, found {n}"
        p.write_text(s.replace(old, new))
        print(f"patched {rel}")
else:
    print(f"torch {torch.__version__}: weights_only already defaults to False")'''))

cells.append(md("""## 4. Apply the session budget

Rewrites `local.max_runtime` in every config so a run that hits the wall saves
`model__latest.pt` and exits cleanly, ready to resume."""))
cells.append(code('''import re

DEADLINE = T0 + BUDGET_HOURS * 3600
# Reserved at the end for exporting histories and zipping the output. A run
# killed before the collect cell is a run entirely wasted.
MARGIN   = 600

def remaining() -> float:
    return DEADLINE - time.time()

print(f"budget {BUDGET_HOURS} h, {remaining()/3600:.2f} h left after setup")
BUDGET = max(int(remaining() - MARGIN), 60)

def set_scalar(cfg_path: Path, key: str, value):
    """Rewrite `key: value` in a YAML file, preserving everything else.

    Deliberately textual rather than a yaml round-trip: these configs carry long
    rationale comments (the P6-D7 / P6-D28 provenance) and a load-then-dump would
    silently delete all of them.
    """
    text = cfg_path.read_text()
    pat = re.compile(rf"^(\\s*){re.escape(key)}:\\s*.*$", re.M)
    hits = pat.findall(text)
    if not hits:
        raise KeyError(f"{key} not found in {cfg_path}")
    # Refuse an ambiguous key instead of silently rewriting whichever comes
    # first. These configs really do repeat names: `sigma_min` appears twice,
    # once as the FMPE flow-matching floor under `model` and once as the
    # observational noise floor under InjectCorrelatedNoise, and the two mean
    # entirely different things. Editing the wrong one would change the
    # experiment without changing anything visible.
    if len(hits) > 1:
        raise KeyError(
            f"{key} appears {len(hits)} times in {cfg_path}, so rewriting it "
            "here would be a guess. Edit it in the config instead.")
    text = pat.sub(lambda m: f"{m.group(1)}{key}: {value}", text, count=1)
    cfg_path.write_text(text)

for d in sorted(Path("configs").glob("*")):
    cfg = d / "config.yaml"
    if not cfg.exists():
        continue
    set_scalar(cfg, "max_runtime", BUDGET)
    set_scalar(cfg, "device", '"cuda"')
    print(f"{d.name:<28} max_runtime={BUDGET}s device=cuda")

# Control arm only: cap the epoch count so it costs minutes, not half an hour.
ctrl = Path("configs/control_rad_cov_s42/config.yaml")
if ctrl.exists():
    set_scalar(ctrl, "epochs", CONTROL_EPOCHS)
    print(f"\\ncontrol capped at {CONTROL_EPOCHS} epochs")'''))

# ---------------------------------------------------------------- trainer
cells.append(md("""## 5. The runner

`train_local.py` resumes from `model__latest.pt` when it finds one, so a run that
exceeded its budget is continued simply by calling it again. The loop stops when
`model__stage_0.pt` appears, which is what `train_stages` writes on
`ExitStatus.COMPLETED`."""))
cells.append(code('''# WHY COMPLETION IS NOT "model__stage_0.pt EXISTS". fm4ar writes that file only
# on ExitStatus.COMPLETED, which means the EPOCH CAP was reached. An arm that
# ends by EARLY STOPPING writes no stage checkpoint at all and still reports
# success, and early stopping is how every real arm here ends: nocond_s42
# stopped at 267 epochs, exactly stage_patience=50 past its best at 217. So
# testing for stage_0.pt declared a finished arm unfinished and sent it into a
# resume loop. train_local.py's own closing verdict is the reliable signal.
MSG_DONE = "All training stages complete!"
MSG_WALL = "Program terminated due to runtime limit!"
MARKER   = "TRAINING_COMPLETE"

def done(name: str) -> bool:
    d = Path("configs") / name
    return (d / "model__stage_0.pt").exists() or (d / MARKER).exists()

def train(name: str, max_restarts: int = 6) -> bool:
    """Run one arm to completion, resuming across runtime-limit exits."""
    d = Path("configs") / name
    assert d.exists(), f"no such config: {d}"
    for attempt in range(1, max_restarts + 1):
        if done(name):
            print(f"[{name}] already complete")
            return True
        # Re-cut max_runtime from the time ACTUALLY left, on every attempt. A
        # budget written once during setup would let the last arm run past the
        # deadline and take the output commit down with it.
        left = remaining() - MARGIN
        if left < MIN_ARM_MIN * 60:
            print(f"[{name}] only {left/60:.0f} min left, not starting "
                  f"(needs {MIN_ARM_MIN} min). Progress is saved.")
            return False
        set_scalar(d / "config.yaml", "max_runtime", int(left))
        tag = "start" if attempt == 1 else f"resume {attempt - 1}"
        print(f"\\n=== [{name}] {tag}, max_runtime={int(left)}s "
              f"({left/3600:.2f} h left) ===", flush=True)
        t0 = time.time()
        # Tee the trainer: stream its output so a Kaggle log is still readable,
        # and keep the tail, because the verdict is printed at the very end.
        proc = subprocess.Popen(
            [sys.executable, "scripts/train.py", "--experiment-dir", str(d)],
            env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        tail = deque(maxlen=80)
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        proc.wait()
        mins = (time.time() - t0) / 60
        print(f"[{name}] exit={proc.returncode} after {mins:.1f} min", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{name} failed with exit code {proc.returncode}")
        blob = "".join(tail)
        if MSG_DONE in blob or done(name):
            (d / MARKER).write_text(
                f"{MSG_DONE}\\nattempt {attempt}, {mins:.1f} min\\n")
            print(f"[{name}] COMPLETE")
            return True
        if MSG_WALL not in blob:
            # Neither verdict appeared. Looping blindly here is how a silent
            # failure becomes six wasted hours, so stop and show the tail.
            raise RuntimeError(
                f"{name}: train_local.py printed neither verdict. Tail:\\n"
                + blob[-2000:])
        # Hit the runtime wall; model__latest.pt is on disk, loop and resume.
    print(f"[{name}] still unfinished after {max_restarts} attempts "
          f"- continue in the next session, progress is saved")
    return False

def history(name: str):
    """Per-epoch history for an arm.

    fm4ar does NOT write a history CSV. It keeps the DataFrame INSIDE the
    checkpoint (fm4ar/models/base.py saves `"history": self.history`), and the
    reference curve shipped in reference/ was extracted the same way. Reading
    `history*.csv` off disk finds nothing, so read the checkpoints and take
    whichever holds the most epochs. `weights_only=False` is required because
    the payload contains a pandas object.
    """
    import pandas as pd, torch
    d = Path("configs") / name
    best = None
    for fname in ("model__stage_0.pt", "model__latest.pt", "model__best.pt"):
        f = d / fname
        if not f.exists():
            continue
        try:
            h = torch.load(f, map_location="cpu", weights_only=False).get("history")
        except Exception as e:
            print(f"  [warn] could not read {f.name}: {e}")
            continue
        if h is None or len(h) == 0:
            continue
        if best is None or len(h) > len(best):
            best = h
    if best is None:
        files = sorted(d.glob("history*.csv"))   # fallback for exported curves
        if files:
            best = pd.concat([pd.read_csv(f) for f in files])
    if best is None:
        return None
    return best.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")'''))

# ---------------------------------------------------------------- control
ref_py = ", ".join(f"({e}, {te:.6f})" for e, _tr, te in REF[:200])
cells.append(md(f"""## 6. Control arm

Retrains `h9_rad_cov_s42` and compares against the RTX 5070 curve. Same config,
same seed, different GPU, so expect close tracking but not bit-identity: floating
point reduction order differs between devices and compounds slowly.

Judge it on the **band**, not on exact equality. The failure this is screening for
is not a third decimal place, it is the MPS-style collapse where the same config
sat at 1.3729 against a CUDA band of 1.124 to 1.159."""))
cells.append(code(f'''REF_TEST = dict([{ref_py}])
REF_BEST_LOSS, REF_BEST_EPOCH = {REF_BEST[2]:.6f}, {REF_BEST[0]}
REF_SEC_PER_EPOCH = {REF_SEC_PER_EPOCH}

if RUN_CONTROL and not done("control_rad_cov_s42"):
    train("control_rad_cov_s42")
elif RUN_CONTROL:
    print("control already trained in an earlier session")
else:
    print("RUN_CONTROL is False, skipping")'''))

cells.append(code('''if RUN_CONTROL:
    df = history("control_rad_cov_s42")
    assert df is not None, "no control history - did the control arm run?"

    obs = dict(zip(df["epoch"], df["test_loss"]))
    common = [e for e in sorted(obs) if e in REF_TEST]
    assert common, "no overlapping epochs with the reference"

    print(f"{'epoch':>6} {'reference':>11} {'this GPU':>11} {'delta':>9}")
    for e in common:
        if e in (1, 5, 10, 20, 30, 40, 50, 60) or e == common[-1]:
            print(f"{e:>6} {REF_TEST[e]:>11.6f} {obs[e]:>11.6f} {obs[e]-REF_TEST[e]:>+9.6f}")

    dev = [abs(obs[e] - REF_TEST[e]) for e in common]
    mean_dev, max_dev = sum(dev)/len(dev), max(dev)
    last = common[-1]
    end_dev = obs[last] - REF_TEST[last]

    sec_ep = float((df["train_time"] + df["test_time"]).mean())
    print(f"\\nmean |deviation| over {len(common)} epochs : {mean_dev:.6f}")
    print(f"max  |deviation|                     : {max_dev:.6f}")
    print(f"deviation at epoch {last:<18}: {end_dev:+.6f}")
    print(f"\\nthis GPU : {sec_ep:.2f} s/epoch   (RTX 5070 reference {REF_SEC_PER_EPOCH})")
    print(f"slowdown : {sec_ep/REF_SEC_PER_EPOCH:.2f}x")

    # Budgeting. 512 is the configured epoch cap; arms usually early-stop well short.
    # BUDGET_HOURS is one run's budget, so this says how many RUNS the six arms need.
    for label, eps in (("typical, ~270 epochs", 270), ("worst case, 512 epochs", 512)):
        hrs = 6 * eps * sec_ep / 3600
        runs = int(hrs / BUDGET_HOURS) + 1
        print(f"6 arms {label:<24}: {hrs:5.1f} h  "
              f"({'fits one run' if runs == 1 else f'needs {runs} runs'})")
    print(f"one arm  {'typical':<24}: {270 * sec_ep / 3600:5.1f} h")

    # THE GATE IS CALIBRATED AGAINST MEASURED RUN-TO-RUN SPREAD, not against a
    # number picked by eye. The first version used max_dev < 0.05, and the T4
    # missed it by 0.0003. scripts/kaggle_control_audit.py then measured the
    # spread of this exact config at three seeds on the reference 5070 itself,
    # over the same epochs 1 to 60:
    #
    #     5070 s43 vs s42   mean 0.0167   max 0.0562
    #     5070 s44 vs s42   mean 0.0321   max 0.0807
    #     T4   s42 vs s42   mean 0.0170   max 0.0503   <- smallest of the three
    #
    # So a 0.05 max gate rejects h9_rad_cov_s43, a PUBLISHED 5070 arm. It was a
    # test of nothing. Different GPUs differ in reduction order and cuDNN kernel
    # choice, so epoch-level agreement is not the right demand.
    #
    # What the gate is for is the MPS-style collapse: +0.25 sustained (1.3729
    # against a 1.124 to 1.159 band), not one noisy epoch. Hence a mean inside
    # the cross-seed mean spread, a max inside the cross-seed max spread, and an
    # end-of-curve that is not HIGH, which is the direction under-training moves
    # it. Cross-seed end deviation reaches 0.044, so 0.08 sits above the noise
    # and far below the 0.25 failure.
    SEED_MEAN_SPREAD = 0.0321
    SEED_MAX_SPREAD  = 0.0807
    PASS = (mean_dev <= SEED_MEAN_SPREAD
            and max_dev <= SEED_MAX_SPREAD
            and end_dev < 0.08)
    print(f"\\ncross-seed reference spread on the 5070 itself: "
          f"mean <= {SEED_MEAN_SPREAD}, max <= {SEED_MAX_SPREAD}")
    print("\\n" + ("CONTROL PASSED - this GPU sits inside the run-to-run spread "
                   "of the reference GPU."
                   if PASS else
                   "CONTROL DID NOT PASS - inspect before trusting the six arms. "
                   "A large POSITIVE end deviation is the MPS-style "
                   "under-training signature (P6-D1); check the accelerator is "
                   "really attached."))
else:
    # Arms-only run. The control passed in an earlier run on the same
    # hardware, so the check is not repeated. PASS still has to exist:
    # the gate below reads it, and a NameError here would end the run
    # before a single arm trained.
    PASS = True
    print("RUN_CONTROL is False - hardware check not repeated in this run")
'''))

cells.append(code('''if RUN_CONTROL:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ref_e = sorted(REF_TEST)
    ax.plot(ref_e, [REF_TEST[e] for e in ref_e], color="k", lw=1.6,
            label="RTX 5070 reference (h9_rad_cov_s42)")
    ax.plot(df["epoch"], df["test_loss"], color="crimson", lw=1.6, ls="--",
            label="this GPU (control)")
    ax.axhspan(1.124, 1.159, color="tab:green", alpha=0.15,
               label="radius CUDA band (P6-D12)")
    ax.axhline(1.3729, color="tab:orange", lw=1, ls=":",
               label="same config on MPS, 1.3729")
    ax.set_xlabel("epoch"); ax.set_ylabel("test loss")
    ax.set_xlim(0, max(max(ref_e), int(df["epoch"].max())))
    ax.set_title("Control: does this hardware reproduce the reference curve?")
    ax.legend(fontsize=8); fig.tight_layout(); plt.show()
else:
    print("no control curve to plot in an arms-only run")
'''))

# ---------------------------------------------------------------- main
cells.append(md("""## 7. Train the six arms

Re-run this cell in a new session to continue. Finished arms are skipped and
unfinished ones resume from `model__latest.pt`."""))
cells.append(code('''# Deliberately NOT an assert. In a batch run a raised exception ends the run, so
# the collect cell below would never write the output and whatever did train
# would be thrown away too. Skip and report instead.
GATED = RUN_CONTROL and not PASS
if GATED:
    print("!" * 70)
    print("CONTROL DID NOT PASS - not training the arms on this hardware.")
    print("Six arms trained on hardware that fails the control would repeat")
    print("exactly the error P6-D27 found. Compare the curve above against the")
    print("reference band before overriding this.")
    print("!" * 70)

results = {}
for name in ([] if GATED else ARMS):
    results[name] = train(name)

print("\\n==== status ====")
for name in ARMS:
    d = history(name)
    eps = int(d["epoch"].max()) if d is not None else 0
    best = float(d["test_loss"].min()) if d is not None else float("nan")
    print(f"{name:<28} {'done' if done(name) else 'UNFINISHED':<11} "
          f"epochs={eps:<5} best_test={best:.6f}")

if results and all(results.values()):
    print("\\nAll arms in this run complete.")
else:
    print("\\nSome arms unfinished. Re-run this notebook in a new session; "
          "progress is saved in model__latest.pt.")'''))

# ---------------------------------------------------------------- collect
cells.append(md("""## 8. Collect what to bring home

Only `model__best.pt` is needed to evaluate locally, about 4.4 MB per arm.
Evaluation does not need CUDA, so score these on the local machine the same way
the h9 pass was scored."""))
cells.append(code('''import zipfile

# Export each curve to CSV first. The history lives inside the checkpoints, so
# without this step the only way to see a loss curve locally is to load a 13 MB
# .pt file. These CSVs are a few KB and are what the write-up actually needs.
for name in ARMS + ["control_rad_cov_s42"]:
    d = Path("configs") / name
    if not d.exists():
        continue
    h = history(name)
    if h is not None:
        out = d / f"history_to_ep{int(h['epoch'].max())}.csv"
        h.to_csv(out, index=False)
        print(f"exported {out}  ({len(h)} epochs)")

bundle = Path("/kaggle/working/sigrange_results.zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ARMS + ["control_rad_cov_s42"]:
        d = Path("configs") / name
        if not d.exists():
            continue
        # model__latest.pt rides along so an unfinished arm can RESUME in the
        # next run. model__best.pt is enough to evaluate but carries no
        # optimizer state, so an arm rebuilt from it would start over.
        # TRAINING_COMPLETE must travel. Without it, completion has to be
        # re-derived downstream from "epochs after best >= patience", and that
        # test is WRONG whenever the longest surviving history ends at the best
        # epoch: model__latest.pt is only written every backup_interval (100)
        # epochs, so an arm whose best landed at 208 and stopped at 258 ships a
        # 208-row history and looks unfinished.
        for pat in ("model__best.pt", "model__latest.pt", "model__stage_0.pt",
                    "TRAINING_COMPLETE", "config.yaml", "history*.csv",
                    "requirements.txt", "git-hash.txt"):
            for f in d.glob(pat):
                z.write(f, f"{name}/{f.name}")

print(f"{bundle}  ({bundle.stat().st_size/1e6:.1f} MB)")
print("\\nDownload it from the notebook Output tab, unzip into configs/, then")
print("evaluate locally. Record the outcome as P6-D28 in")
print("Documentation/problems_and_decisions.md.")
print("\\nThe prediction to check, written before the run: the cov arm's mean 1-D")
print("JS to the NS anchor on real WASP-39b falls from 0.426 toward the no-cond")
print("0.344 and the +0.0827 gap at least halves. If it does not move, then with")
print("P6-D26 the paradox is neither the inference-time Sigma-hat nor the")
print("training sigma range.")'''))

# ---------------------------------------------------------------- trim output
cells.append(md("""## 9. Leave only the archive in the output

A batch run commits everything under `/kaggle/working` as the notebook output.
The staged tree is 250 code files plus the 14 MB training set, none of which
needs to come back, so it is removed and only the archive is kept. The archive
already holds `model__latest.pt`, which is what the next run resumes from."""))
cells.append(code('''os.chdir("/kaggle/working")
_b = globals().get("bundle")
if _b is not None and Path(_b).exists() and Path(_b).stat().st_size > 0:
    shutil.rmtree(WORK, ignore_errors=True)
    print("removed the staged tree. Output is now:")
else:
    print("archive missing or empty - keeping the staged tree so nothing is lost")
for f in sorted(Path("/kaggle/working").rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to('/kaggle/working')}  "
              f"{f.stat().st_size/1e6:.1f} MB")'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

for c in nb["cells"]:
    if isinstance(c["source"], str):
        c["source"] = c["source"].splitlines(keepends=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(cells)} cells)")
print(f"reference curve: {len(REF)} epochs, best {REF_BEST[2]:.6f} @ {REF_BEST[0]}")
