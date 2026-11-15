"""V2 (closes G8): the cheap baselines any learned translator has to beat.

Three arms that need no new data and no adversarial training, scored with the
SAME metric code as the published tier-1 table (run_cyclegan_ablation_standalone),
beside the existing no-adaptation, structured-randomisation and CycleGAN rows:

  affine      per-bin affine map, sim -> real: match each bin's mean and sd
  quantile    per-bin quantile mapping, sim -> real: match each bin's whole
              marginal through the empirical CDF (climate bias correction,
              Cannon, Sobie & Murdock 2015)
  reweight    classifier reweighting: a classifier separating sim from real
              supplies density-ratio weights w = p / (1 - p) on the SIM samples,
              which are left unchanged (OmniFold-style, Andreassen et al. 2020).
              Its metrics are weighted, and its weight ESS is printed, because a
              classifier that separates the domains perfectly gives weights that
              collapse onto a handful of samples.

Both MMD values are printed for every arm, the no-adaptation baseline beside the
adapted one, because a single adapted MMD means nothing on its own (V2 brief).

HELD-OUT REAL FRAMES. The new arms are FIT on real data, so they are scored only on
real frames they never saw. Frames are split CONTIGUOUSLY in integration order,
first half fit and second half score, for the reason inject_real_noise.py gives:
JWST systematics are time-correlated, so an alternating split would put the same
drift on both sides. Every arm, old and new, is scored on that same held-out half.
The published protocol (random 500 of all frames) is also rerun for the three old
arms, as a reproduction check against configs/cyclegan_t1_wasp39/ablation.json.

THE METRIC SPACE IS AFFINE-INVARIANT PER BIN, and this is a finding, not a detail.
The standalone ablation standardises sim by sim's own per-bin mean/sd and real by
real's own, and only then compares. Any per-bin affine map applied before that
step is undone by it. So in the published space `affine` is no adaptation by
construction, and "affine beats no adaptation" cannot be tested there at all. It
also means the reduced spectrum's DEPTH cancels from domain B exactly: a real
sample is depth * (1 + r), and per-bin standardisation removes depth. That is why
this script can be run from the out-of-transit frames alone (`--oot`), with no
reduced spectrum, and still reproduce the published sim and randomisation rows.
The covered-bin set and the gap fill can differ slightly without the spectrum,
which is why the reproduction check exists.

    python scripts/v2_cheap_baselines.py --oot wasp39
    python scripts/v2_cheap_baselines.py --real-domain data/real_domain/real_domain_wasp39_abc.npz

What this does NOT do: the inferential half of V2 (posterior JS to the NS anchor)
needs one flow trained per arm on the mapped training set, plus the anchor, which
lives outside this repo. See P6-D13 in Documentation/problems_and_decisions.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_cyclegan_ablation_standalone as sa          # noqa: E402  (published metrics)
from build_real_covariance import rebin_frames          # noqa: E402
from build_real_domain import TARGETS, fill_uncovered, log_edges  # noqa: E402

import torch                                            # noqa: E402

N_PCA = 10           # published: 10 PCA components, RBF sigma 1.0
CKPT = Path("configs/cyclegan_t1_wasp39/cyclegan_best.pt")


# ── real domain ────────────────────────────────────────────────────────────

def abc_grid_stored(hdf: Path = sa.ABC_TEST) -> np.ndarray:
    with h5py.File(hdf) as f:
        return f["wlen"][0].astype(float)


def real_from_oot(target: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Depth-free domain B: (1 + r) per frame on the 52-bin ABC grid, stored order.

    Valid ONLY in the per-bin-standardised metric space, where depth cancels.
    Returns (samples, frame_ids, note).
    """
    wl_s = abc_grid_stored()
    order = np.argsort(wl_s)
    inv = np.argsort(order)
    edges = log_edges(wl_s[order])
    oot = pd.read_csv(TARGETS[target]["oot"],
                      usecols=["wavelength_um", "flux_jy", "dq_flag", "integration_num"])
    oot = oot[oot["dq_flag"] == 0]
    M_a, ints = rebin_frames(oot, edges)
    M = M_a[:, inv]
    covered = np.isfinite(M).all(axis=0)
    rel = np.zeros_like(M)
    rel[:, covered] = M[:, covered] / M[:, covered].mean(axis=0) - 1.0
    x = np.full_like(M, np.nan)
    x[:, covered] = 1.0 + rel[:, covered]
    x = fill_uncovered(x, wl_s, covered)
    note = (f"depth-free from OOT frames ({target}), {int(covered.sum())}/{len(covered)} "
            f"bins covered by frames")
    return x, ints, note


def real_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    d = np.load(path, allow_pickle=True)
    if "abc_cols" not in d or len(d["abc_cols"]) != 52:
        raise SystemExit("[guard] V2 runs on the full 52-bin grid; rebuild with --grid abc")
    return d["samples"].astype(np.float64), np.asarray(d["frame_ids"]), f"{path.name}"


def standardise(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    ref = x if ref is None else ref
    return (x - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-8)


# ── the three cheap arms ───────────────────────────────────────────────────

def affine_map(sim: np.ndarray, real_fit: np.ndarray) -> np.ndarray:
    a = real_fit.std(axis=0) / (sim.std(axis=0) + 1e-12)
    return (sim - sim.mean(axis=0)) * a + real_fit.mean(axis=0)


def quantile_map(sim: np.ndarray, sim_ref: np.ndarray, real_fit: np.ndarray) -> np.ndarray:
    """Per-bin empirical-CDF transfer. sim_ref sets the sim CDF, real_fit the target."""
    out = np.empty_like(sim)
    q = np.linspace(0.0, 1.0, 1001)
    for k in range(sim.shape[1]):
        sq = np.quantile(sim_ref[:, k], q)
        rq = np.quantile(real_fit[:, k], q)
        # rank of each value inside the sim reference, then the same rank in real
        u = np.interp(sim[:, k], sq, q, left=0.0, right=1.0)
        out[:, k] = np.interp(u, q, rq)
    return out


def classifier_weights(sim_train: np.ndarray, real_fit: np.ndarray, sim_eval: np.ndarray,
                       seed: int) -> tuple[np.ndarray, dict]:
    """Density-ratio weights on sim_eval from a sim-vs-real classifier.

    Fit on the full standardised 52-bin vector, not the 10 PCA features MMD sees,
    so the classifier is not blind to structure outside them. Gradient boosting,
    because the two domains differ in shape and not only in location. The ratio
    is corrected for class balance, so w is p(real)/p(sim) rather than an
    artefact of the two sample sizes.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    X = np.vstack([sim_train, real_fit])
    y = np.r_[np.zeros(len(sim_train)), np.ones(len(real_fit))]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=seed)
    auc = float(np.mean(cross_val_score(clf, X, y, cv=3, scoring="roc_auc")))
    clf.fit(X, y)
    p = np.clip(clf.predict_proba(sim_eval)[:, 1], 1e-6, 1 - 1e-6)
    w = (p / (1 - p)) * (len(sim_train) / len(real_fit))
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return w, {"classifier_auc_cv3": auc, "weight_ess": ess, "n_weighted": int(len(w)),
               "max_weight_share": float(w.max() / w.sum())}


# ── weighted versions of the published metrics ─────────────────────────────

def mmd_rbf_w(X, Y, w=None, sigma=1.0):
    if w is None:
        return sa.mmd_rbf(X, Y, sigma)
    w = w / w.sum()

    def rbf(A, B):
        d = A[:, None, :] - B[None, :, :]
        return np.exp(-np.sum(d ** 2, axis=-1) / (2 * sigma ** 2))
    return float(w @ rbf(X, X) @ w + rbf(Y, Y).mean() - 2 * (w @ rbf(X, Y)).mean())


def kl_w(P, Q, w=None, bins=50):
    if w is None:
        return sa.kl_divergence_marginals(P, Q, bins)
    kls = []
    for i in range(P.shape[1]):
        lo, hi = min(P[:, i].min(), Q[:, i].min()), max(P[:, i].max(), Q[:, i].max())
        e = np.linspace(lo, hi, bins + 1)
        ph, _ = np.histogram(P[:, i], bins=e, weights=w)
        qh, _ = np.histogram(Q[:, i], bins=e)
        ph = ph / ph.sum() + 1e-10
        qh = qh / qh.sum() + 1e-10
        ph, qh = ph / ph.sum(), qh / qh.sum()
        kls.append(float(np.sum(ph * np.log(ph / qh))))
    return float(np.mean(kls))


def shift_w(real, X, w=None):
    m = X.mean(axis=0) if w is None else (w[:, None] * X).sum(axis=0) / w.sum()
    return float(np.abs(m - real.mean(axis=0)).mean())


# ── main ───────────────────────────────────────────────────────────────────

def score(arms: dict, real: np.ndarray, V: np.ndarray, n_kl: int = 200) -> dict:
    """Published metrics for each arm against `real`. arms[name] = (X, weights|None)."""
    rp = real @ V
    out = {}
    for name, (X, w) in arms.items():
        wk = None if w is None else w[:n_kl]
        out[name] = {
            "mmd_to_real": mmd_rbf_w(X @ V, rp, w),
            "kl_to_real": kl_w(X[:n_kl], real[:n_kl], wk),
            "mean_shift": shift_w(real, X, w),
        }
    return out


def main(oot: str | None, real_domain: str | None, n: int, seed: int, out: str) -> None:
    if real_domain:
        real_raw, frames, note = real_from_npz(Path(real_domain))
    else:
        real_raw, frames, note = real_from_oot(oot)
    print(f"real domain: {note}, {len(real_raw)} frames")

    # sim exactly as the published ablation reads it (first n of abc_test)
    sim_raw, wlen = sa.load_abc(n)
    sim = standardise(sim_raw)
    model = sa.load_cyclegan(CKPT)
    with torch.no_grad():
        trans = model.G_AB(torch.from_numpy(sim.astype(np.float32))).numpy()
    dr = sa.apply_domain_randomisation(sim, wlen, seed=42)

    # ── 1. reproduction of the published protocol ──────────────────────────
    real_all = standardise(real_raw)
    rep = real_all
    if len(rep) > n:
        rep = rep[np.random.default_rng(0).choice(len(rep), n, replace=False)]
    comb = np.vstack([sim, rep, trans, dr])
    _, _, Vt = np.linalg.svd(comb - comb.mean(axis=0), full_matrices=False)
    V = Vt[:N_PCA].T
    repro = score({"sim": (sim, None), "domain_random": (dr, None),
                   "cyclegan": (trans, None)}, rep, V)

    # ── 2. held-out protocol: fit on the first half, score on the second ───
    order = np.argsort(frames)
    half = len(order) // 2
    fit_idx, test_idx = order[:half], order[half:]
    # one standardisation for real, set by the FIT half only, so nothing about the
    # held-out frames leaks into the space they are scored in
    real_fit = standardise(real_raw[fit_idx], real_raw[fit_idx])
    real_test = standardise(real_raw[test_idx], real_raw[fit_idx])
    rng = np.random.default_rng(seed)
    if len(real_test) > n:
        real_test = real_test[rng.choice(len(real_test), n, replace=False)]

    # the classifier and the quantile CDFs are fit on a DIFFERENT sim sample
    # (abc_valid) from the one being scored (abc_test), for the same reason
    with h5py.File("data/abc/abc_valid.hdf") as f:
        sv = f["flux"][:4000].astype(np.float32)
        sv = (sv - f["flux_mean"][:]) / f["flux_std"][:]
    sim_ref = standardise(sv, sim_raw)            # in the scored sim's own units

    arms = {
        "sim (no adaptation)": (sim, None),
        "domain_random": (dr, None),
        "cyclegan": (trans, None),
        "affine": (affine_map(sim, real_fit), None),
        "quantile": (quantile_map(sim, sim_ref, real_fit), None),
    }
    w, wdiag = classifier_weights(sim_ref, real_fit, sim, seed)
    arms["reweight"] = (sim, w)

    comb = np.vstack([sim, real_test, trans, dr])
    _, _, Vt = np.linalg.svd(comb - comb.mean(axis=0), full_matrices=False)
    held = score(arms, real_test, Vt[:N_PCA].T)

    # ── print ───────────────────────────────────────────────────────────────
    pub = json.load(open("configs/cyclegan_t1_wasp39/ablation.json"))["conditions"]
    pmap = {"sim": "sim_baseline", "domain_random": "domain_random",
            "cyclegan": "cyclegan_trans"}
    print("\n  1. reproduction, published protocol (random 500 of all frames)")
    print(f"  {'arm':<16}{'MMD':>9}{'published':>11}{'KL':>9}{'published':>11}")
    for k, v in repro.items():
        p = pub[pmap[k]]
        print(f"  {k:<16}{v['mmd_to_real']:>9.4f}{p['mmd_to_real']:>11.4f}"
              f"{v['kl_to_real']:>9.4f}{p['kl_to_real']:>11.4f}")

    base = held["sim (no adaptation)"]
    print(f"\n  2. V2 table, held-out real frames (fit on the first {half} by "
          f"integration, scored on {len(real_test)} of the rest)")
    print(f"  {'arm':<22}{'MMD':>9}{'MMD none':>10}{'KL':>9}{'KL none':>9}{'shift':>8}")
    print("  " + "-" * 66)
    for k, v in held.items():
        print(f"  {k:<22}{v['mmd_to_real']:>9.4f}{base['mmd_to_real']:>10.4f}"
              f"{v['kl_to_real']:>9.4f}{base['kl_to_real']:>9.4f}{v['mean_shift']:>8.3f}")
    print("  " + "-" * 66)
    print(f"  reweight: classifier AUC {wdiag['classifier_auc_cv3']:.3f} (3-fold), "
          f"weight ESS {wdiag['weight_ess']:.1f} of {wdiag['n_weighted']}, "
          f"largest weight {100 * wdiag['max_weight_share']:.1f}% of the total")
    print("  'MMD none' / 'KL none' repeat the no-adaptation value beside every arm.")
    print("  affine is undone by the per-bin standardisation of the metric space, so")
    print("  its row should equal 'sim' up to the fit/score split. If it does not, the")
    print("  space is not what this docstring says.")

    res = {"real_domain": note, "n_sim": int(n), "seed": seed,
           "split": {"fit_frames": int(half), "score_frames": int(len(real_test)),
                     "rule": "contiguous by integration_num, first half fit"},
           "reproduction_published_protocol": repro, "published": pub,
           "heldout": held, "reweight_diagnostics": wdiag,
           "metric_space": "per-domain per-bin standardised, 10 PCA comps, RBF sigma 1"}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--oot", choices=list(TARGETS), help="depth-free domain B from OOT frames")
    src.add_argument("--real-domain", help="npz from build_real_domain.py --grid abc")
    ap.add_argument("--n", type=int, default=500, help="sim and real samples per arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="configs/v2_baselines_wasp39/v2_baselines.json")
    a = ap.parse_args()
    main(a.oot, a.real_domain, a.n, a.seed, a.out)
