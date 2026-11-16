"""
H11 part 2 — the FUNCTIONAL half of the transfer test: importance-sampling ESS on the
recipient's real spectrum, using a calibration that never saw the recipient's anchor.

Why ESS and not just the divergences in h11_ot_transfer.py. Marginal JS saturates at
ln2 = 0.693 and several cells are already there, so it cannot rank the bad cells
against each other. ESS does not saturate, and it is the quantity the calibration was
built to fix: the raw flow proposal collapses to ESS ~5 (P3-D12) because it is
overdispersed, and OT calibration was Component 4's answer to that. So the question
"does the calibration transfer" has a functional form: does a DONOR's calibration buy
usable ESS on a RECIPIENT whose anchor was never run.

Arms per recipient:
  raw        the flow posterior itself, with its own log q. The floor.
  self_M2    donor == recipient, the published recipe. Circular, and the ceiling.
  <d>_M2     donor d's full dimensionless correction (bias + widths).
  <d>_M3     donor d's width deflation only, recipient's own centre. The only mode
             that helped anything in the moment-space pass.

inflate=1.5 on every Gaussian arm, which is the published setting in ot_calibrate.py,
so self_M2 reproduces the published number and the transferred arms are judged by the
same recipe rather than a kinder one.

    conda run -n mirage        python scripts/h11_ess.py --build
    # then one forward per arm in the TauREx env, see --build output
    conda run -n mirage        python scripts/h11_ess.py --compute
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h11_ot_transfer import TARGETS, load, calibration, transfer, corr, NAMES

H11 = Path("data/h11")
FLOOR = 0.05                      # same 5 percent error floor as ot_calibrate.compute


def arms_for(rec):
    donors = [d for d in TARGETS if d != rec]
    return ["raw", "self_M2"] + [f"{d}_M2" for d in donors] + ["pooled_M4"]


def build(M=2000, inflate=1.5, seed=0):
    H11.mkdir(parents=True, exist_ok=True)
    tg = {k: load(k) for k in TARGETS}
    cmds = []
    for rec, B in tg.items():
        meta = dict(wlen=B_meta(rec, "wlen"), x_obs=B_meta(rec, "x_obs"),
                    sig_obs=B_meta(rec, "sig_obs"), covered=B_meta(rec, "covered"))
        for arm in arms_for(rec):
            p = H11 / f"h11_{rec}_{arm}_samples.npz"
            if arm == "raw":
                f = np.load(TARGETS[rec]["fmpe"], allow_pickle=True)
                np.savez(p, theta=f["theta"], log_g=f["log_q"], planet=rec, **meta)
                n_in = len(f["theta"])
            elif arm == "pooled_M4":
                # leave-one-out pooled width correction, recipient's own centre.
                # Needs NO anchor on the recipient, which is the only form of
                # calibration that could actually amortize.
                others = [calibration(tg[a])["k"] for a in tg if a != rec]
                kp = np.exp(np.mean(np.log(np.array(others)), axis=0))
                sd = kp * B["sdf"]
                mu = B["muf"]
                S = (np.outer(sd, sd) * corr(B["Sf"])) * inflate
                th, log_g, n_in = _draw(mu, S, B["lo"], B["hi"], M, seed)
                np.savez(p, theta=th, log_g=log_g, planet=rec, **meta)
                print(f"[build] {p}  N={n_in}")
                cmds.append(f"python scripts/taurex_forward.py --forward-npz {p} "
                            f"--forward-mode radius --forward-planet {rec}")
                continue
            else:
                donor = rec if arm.startswith("self") else arm.split("_")[0]
                mode = arm.split("_")[1]
                mu, S = transfer(calibration(tg[donor]), B, mode)
                S = S * inflate
                th, log_g, n_in = _draw(mu, S, B["lo"], B["hi"], M, seed)
                np.savez(p, theta=th, log_g=log_g, planet=rec, **meta)
            print(f"[build] {p}  N={n_in}")
            cmds.append(f"python scripts/taurex_forward.py --forward-npz {p} "
                        f"--forward-mode radius --forward-planet {rec}")
    (H11 / "run_forwards.sh").write_text("set -e\n" + "\n".join(cmds) + "\n")
    print(f"\n[build] {len(cmds)} forward runs -> {H11 / 'run_forwards.sh'}")


_META_CACHE = {}


def B_meta(rec, key):
    if rec not in _META_CACHE:
        _META_CACHE[rec] = np.load(TARGETS[rec]["fmpe"], allow_pickle=True)
    return _META_CACHE[rec][key]


def _draw(mu, S, lo, hi, M, seed):
    """Draw M in-prior samples from N(mu,S) with its analytic log-density. The prior is
    uniform so its constant cancels in self-normalised IS; the truncation constant
    cancels too, so the untruncated density is the right log g here (and is what
    ot_calibrate.py uses)."""
    w, V = np.linalg.eigh(S)
    L = V * np.sqrt(np.clip(w, 1e-18, None))
    rng = np.random.default_rng(seed)
    keep_t, keep_g, tries = [], [], 0
    Sinv = np.linalg.pinv(S)
    logdet = float(np.sum(np.log(np.clip(w, 1e-18, None))))
    while sum(len(t) for t in keep_t) < M and tries < 200:
        x = mu + rng.standard_normal((4 * M, 7)) @ L.T
        ok = np.all((x >= lo) & (x <= hi), axis=1)
        x = x[ok]
        d = x - mu
        maha = np.einsum("ij,jk,ik->i", d, Sinv, d)
        keep_t.append(x)
        keep_g.append(-0.5 * (maha + 7 * np.log(2 * np.pi) + logdet))
        tries += 1
    th = np.concatenate(keep_t)[:M] if keep_t else np.zeros((0, 7))
    lg = np.concatenate(keep_g)[:M] if keep_g else np.zeros(0)
    return th, lg, len(th)


def compute():
    rows = []
    for rec in TARGETS:
        print(f"\n--- recipient {rec} {'-' * 72}")
        print(f"  {'arm':14s} {'N':>6s} {'ESS':>9s} {'68% CI':>15s} {'eps %':>7s} "
              f"{'radius':>9s} {'T':>8s} {'logH2O':>8s}")
        for arm in arms_for(rec):
            sp = H11 / f"h11_{rec}_{arm}_samples.npz"
            fp = Path(str(sp).rsplit("samples", 1)[0] + "forward.npz")
            if not fp.exists():
                print(f"  {arm:14s} forward missing")
                continue
            s = np.load(sp, allow_pickle=True)
            ms = np.load(fp)["model_spec"]
            th, log_g = s["theta"], s["log_g"]
            x, sig, cov = s["x_obs"], s["sig_obs"], s["covered"]
            m = cov & np.isfinite(sig) & (sig > 0)
            se = np.sqrt(sig ** 2 + (FLOOR * x) ** 2)
            valid = np.isfinite(ms[:, m]).all(axis=1)
            r = (x[None, m] - ms[:, m]) / se[None, m]
            ll = np.where(valid, -0.5 * np.nansum(r ** 2, axis=1), -np.inf)
            logw = ll - log_g
            logw -= np.nanmax(logw)
            w = np.exp(logw)
            w /= w.sum()
            ess = float(1.0 / np.sum(w ** 2))
            # Bootstrap CI. At ESS of order 10 the estimator is dominated by a handful
            # of draws, so individual cells MUST NOT be ranked against each other
            # without this. Resample the draws with replacement and renormalise.
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(400):
                idx = rng.integers(0, len(w), len(w))
                wb = w[idx]
                sb = wb.sum()
                if sb > 0:
                    bs.append(1.0 / np.sum((wb / sb) ** 2))
            lo_b, hi_b = np.percentile(bs, [16, 84]) if bs else (np.nan, np.nan)
            wm = np.array([np.sum(w * th[:, i]) for i in range(7)])
            chi2 = float(-2 * ll.max() / m.sum())
            print(f"  {arm:14s} {len(w):6d} {ess:9.1f} "
                  f"{f'[{lo_b:.1f}, {hi_b:.1f}]':>15s} {ess / len(w) * 100:7.2f} "
                  f"{wm[0]:9.4f} {wm[1]:8.1f} {wm[2]:8.2f}")
            rows.append(dict(recipient=rec, arm=arm, N=len(w), ess=ess,
                             eps=ess / len(w), chi2_dof=chi2,
                             ess_lo=float(lo_b), ess_hi=float(hi_b),
                             wmean=dict(zip(NAMES, wm.tolist()))))
    with open(H11 / "ess_matrix.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n[h11] wrote {H11 / 'ess_matrix.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--inflate", type=float, default=1.5)
    a = ap.parse_args()
    if a.build:
        build(a.m, a.inflate)
    elif a.compute:
        compute()
