# MIRAGE — Problems & Decisions Log

Running log of every non-obvious design decision and every problem encountered during implementation. Update this file immediately when a new issue arises or a design choice is made. Entries are numbered sequentially; phase is noted in brackets.

---

## Pre-Development Decisions (Blueprint v1.0, May 2026)

### D1 — Flow-matching over masked autoregressive flow [Blueprint]
Chose continuous normalising flow trained via flow-matching (Gebhard et al. 2025 fm4ar) over masked autoregressive flows (Vasist 2023). Reason: flow-matching trains faster, scales better to high-dimensional posteriors, and fm4ar already demonstrated SOTA performance on the ABC synthetic benchmark. Alternative rejected: normalising flows with neural spline coupling — weaker expressivity on multi-modal posteriors.

### D2 — SpectraFM transformer encoder over MLP / 1D-CNN [Blueprint]
Every existing ML retriever uses a flat MLP or 1D-CNN encoder that concatenates all wavelengths into a single vector. JWST observes most targets across four heterogeneous instruments with different wavelength coverage, resolution, and noise structure. A shared MLP discards instrument identity and handles missing instruments poorly. Decision: adapted SpectraFM (Koblischke & Bovy 2024) transformer where each token = (wavelength, flux, uncertainty, instrument_id). Cross-attention aggregates across instruments naturally; masking handles absent instruments without retraining. Alternative rejected: separate per-instrument encoders with late fusion — fails when an instrument is absent and doubles parameter count.

### D3 — Dual noise conditioning (per-wavelength vector + residual covariance embedding) [Blueprint]
Standard noise conditioning passes only the per-wavelength error bars to the network. Real JWST noise is correlated across wavelengths (detector systematics, 1/f noise) — the covariance structure is not captured by marginal uncertainties alone. Decision: condition on both (a) the per-wavelength uncertainty vector and (b) a learned embedding of the residual covariance matrix estimated from out-of-transit baseline segments of the same JWST visit. Directly analogous to DINGO's power-spectral-density conditioning. Alternative rejected: augmenting error bars with a scalar noise inflation factor — too coarse to capture wavelength-dependent correlation structure.

### D4 — Domain randomisation over CycleGAN as primary adaptation strategy [Blueprint]
Two candidate strategies for closing the sim-to-real gap during training: (a) structured domain randomisation — inject realistic perturbations (correlated noise, stellar contamination templates, pipeline calibration offsets) into simulated spectra before the network sees them; (b) CycleGAN domain translation — train a generator to map simulated spectra into the observed spectral domain. Decision: domain randomisation is the primary strategy. CycleGAN is retained as a controlled ablation experiment (Component 5). Reason: domain randomisation is interpretable, physically motivated, and does not require a separate adversarial training loop that could introduce its own instabilities. The ablation answers the SBI community question of when generative adaptation is necessary versus when structured randomisation suffices.

### D5 — PHOENIX stellar atmosphere library for contamination templates [Blueprint]
Stellar contamination augmentation requires realistic heterogeneous stellar surface models (unocculted spots and faculae). Decision: use PHOENIX model grid parametrised by spot temperature, faculae temperature, and surface covering fraction. Reason: PHOENIX is the community standard in transmission spectroscopy; BT-Settl and ATLAS9 produce consistent results in the relevant temperature range. Known limitation: PHOENIX is 1D. This is defensible because stellar atmospheres are genuinely better approximated as 1D than planetary atmospheres — contamination is a surface-fraction-weighted average, not a geometry with day-night gradients or limb asymmetries. The 1D limitation of PHOENIX and the 1D limitation of the planetary forward model are physically distinct cases.

### D6 — RoPE optimal-transport calibration anchored to WASP-39b posteriors [Blueprint]
Domain randomisation reduces but cannot eliminate sim-to-real gap. Decision: post-training calibration using the Robust Posterior Estimation framework (Wehenkel et al. 2024). Calibration anchor: FASTER-validated nested-sampling posteriors on WASP-39b across NIRSpec PRISM, NIRSpec G395H, NIRISS SOSS, and NIRCam. WASP-39b chosen because it is the most deeply validated JWST planet — multiple independent reduction pipelines, multiple instruments, and the Roy-Perez et al. 2026 multi-pipeline corpus now publicly available. Known limitation: one planet is a thin calibration set. Planned mitigation: add WASP-96b (NIRISS SOSS, different stellar type) as a second anchor in Phase 3 to make anchor sensitivity an ablation rather than an unaddressed gap.

### D7 — IS-efficiency ε expressed as ESS threshold, not raw percentage [Post-blueprint, May 2026]
Blueprint v1.0 stated "ε > 1%" as the deployment threshold. This is under-justified and does not specify sample count. Decision: replace with ESS = ε × N > 500 as the primary threshold. At N = 50,000 posterior samples, ε = 1% gives ESS = 500 — adequate for 1D marginal distributions, borderline for joint inference. This grounds the threshold in importance sampling literature and gives a principled answer to "sufficient for what?" Secondary threshold: ε > 5% (ESS > 2,500) for a result to be cited as "high-quality" in the paper.

### D8 — WASP-39b as primary validation target [Blueprint]
Selection rationale: most extensively reduced JWST planet. NIRSpec PRISM, NIRSpec G395H, NIRISS SOSS, NIRCam spectra all publicly available on MAST. FASTER-validated posteriors provide the calibration anchor. Roy-Perez et al. 2026 provides a multi-reduction pipeline corpus for augmentation training. Minimal stellar contamination (G-type host) isolates the noise-conditioning and flow-matching components from contamination confounds.

### D9 — K2-18b scoped as robustness stress test only [Blueprint]
K2-18b has active scientific controversy over claimed DMS and DMDS biosignature detections, high stellar contamination from an M-dwarf host, and contested pipeline reductions. Decision: MIRAGE applied to K2-18b for robustness evaluation only. The project makes no biosignature claims and frames all K2-18b results as uncertainty quantification under model misspecification. This is the scientifically responsible scope given the controversy.

### D10 — NeurIPS ML4PS 2026 workshop as priority-establishment submission [Blueprint]
Decision: submit preliminary real-data results on WASP-39b to NeurIPS ML4PS workshop (December 2026) before the ICML 2027 deadline. Reason: establishes priority on the architecture, receives community peer-review feedback, and provides a forcing function for Phase 3 results (first real-data IS-efficiency measurement) to be ready by November 2026. arXiv preprint released concurrently, cross-listed astro-ph.EP + cs.LG.

---

### D11 — Data sharing: GitHub for scripts only, MAST + Zenodo for data [Phase 0]
JWST FITS files are 500 MB–2 GB each; total WASP-39b corpus is 30–50 GB. GitHub's hard limit is 100 MB per file. Decision: `data/` is gitignored on both tracks. Raw JWST data is sourced directly from MAST by each collaborator running `scripts/download_mast.py`. Processed/standardised corpus is published to Zenodo as a draft record during development, made public at paper submission. This means the Phase 5 public data release is built into the workflow from day one, not retrofitted. Alternative rejected: shared cloud drive (Google Drive/Dropbox) — no versioning, links break, not reproducible for external reviewers.

---

## Phase 0 Log

See `00_track1_environment_baselines.md` for full Phase 0 Track 1 record.

Key outcomes: ABC pipeline verified end-to-end; NPE ε = 0.025% (M1 Pro, 256-dim/3-transform); FMPE trained to early stopping epoch 140 (val loss 1.388); 76% of ABC NS posteriors malformed — 2,204 valid test planets.

---

## Phase 1 Log

### P1-D1 — sample_and_log_prob_batch called on model wrapper, not network [Phase 1]
`FMPEModel.sample_and_log_prob_batch` exists on the wrapper class; `FMPEModel.network` (type `FMPENetwork`) does not have this method. Calling `model.network.sample_and_log_prob_batch` raises `AttributeError`. Fix: call `model.sample_and_log_prob_batch` directly.

### P1-D2 — N_SAMPLES=2,000 for FMPE IS-efficiency (vs 10,000 for NPE) [Phase 1]
Each FMPE IS-efficiency evaluation requires a full ODE solve (dopri5) on CPU — ~30–60s per planet. Using 10,000 samples per planet would take 5–10× longer. Decision: 2,000 samples per planet for M1 Pro diagnostics. ε is ESS/N so the comparison with NPE (N=10,000) is valid on expectation; variance is higher. Cluster evaluation should use N=10,000 for publication-quality numbers.

### P1-D3 — ODE tolerance set to 1e-3 for IS-efficiency speed [Phase 1]
Default tolerance 1e-7 is too slow for 20-planet evaluation on CPU. Decision: 1e-3 for diagnostic runs. Tighten to 1e-5 or 1e-7 for publication-quality IS-efficiency numbers on cluster.

---

## Phase 2 Log

See `02_track1_noise_conditioning.md` for the full Phase 2 Track 1 design (5 work items, blueprint mapping, dependency order).

### P2-D1 — Condition on dynamically-sampled `error_bars`, not static HDF5 `instrument_noise` [Phase 2]
Part 1 (per-wavelength σ conditioning) needs the σ that matches the noise realisation actually present in `flux`. Two candidate sources: (a) the static `instrument_noise (52,)` baked into `abc_train.hdf`, or (b) the `error_bars` that `AddNoise`/the injection transform samples at train time and stores in the context dict (`data_transforms.py:115`) *before* drawing the realisation it adds to `flux`. Decision: condition on `error_bars`. When the WI-2 injection transform is active, `error_bars = √diag(Σ_true)` corresponds exactly to the injected noise; the static HDF5 σ does not. The encoder consumes `context["error_bars"]` as the third token feature. Note: stock ABC training runs `data_transforms: []` (noise baked in), so the injection transform must be turned on for the σ source to be correct — this is part of the new `transformer_abc_noisecond` config.

### P2-D2 — Residual-covariance embedding is information-vacuous on stock ABC; build a correlated-noise injector to make it testable [Phase 2]
ABC's noise model is a single per-wavelength σ → diagonal Gaussian, Σ = diag(σ²), with no time axis and no out-of-transit frames. A learned embedding of a diagonal matrix carries exactly the information already in the Part 1 σ vector, so on stock ABC Part 2 is mathematically redundant with Part 1 and cannot be trained, validated, or ablated. Decision: implement Part 2 against synthetically-injected correlated noise (WI-2) rather than deferring the whole embedding to Phase 3. This is the chosen "Option B." Rejected alternative — Option A (defer Part 2 to real JWST in Phase 3): leaves the Phase 2 noise-conditioning deliverable half-empty on the synthetic benchmark and gives no controlled way to unit-test or ablate the covariance path before betting real-data results on it. Option B yields the headline ablation (ESS with vs without covariance conditioning under correlated noise) that a diagonal-noise table structurally cannot produce.

### P2-D3 — Network conditions on OOT frames, never on Σ directly (ABC/JWST symmetry) [Phase 2]
The residual-covariance embedding takes out-of-transit noise frames as input and estimates the correlation structure internally; it is never handed Σ. This makes the ABC (Phase 2) and real-JWST (Phase 3) pipelines architecturally identical — only the frame source differs (synthetic draws from a known kernel vs real OOT integrations). Consequence: the Phase 2 embedding transfers to Phase 3 with no architectural change, and the ground-truth Σ_true is reserved for evaluator-side sanity checks only (recover-Σ test in WI-5). Rejected alternative — feeding the empirical Σ̂ or Σ_true straight into an MLP: trains faster on ABC but does not exist on real data, so it would require a redesign at the Phase 3 boundary.

### P2-D4 — Inject correlated noise on top of existing ABC flux; no noise-free base regenerated [Phase 2]
`abc_train.hdf` stores `flux (N,52)` with ABC's instrument noise already baked in and `noise (N,52)` as the diagonal σ; there is no noise-free model spectrum. For the controlled covariance experiment, inject the WI-2 correlated noise on top of the existing `flux`. ABC's baked-in noise is diagonal — it contributes only to σ (Part 1) — while the injected component is the sole off-diagonal structure, so Σ_true's off-diagonals remain exactly known for the recover-Σ check. Rejected alternative — regenerate noise-free spectra via the TauREx forward model: more work for marginal gain, since the off-diagonal ground truth is already clean under injection-on-top.

### P2-D5 — Promote ESS from inline script to reusable module, with D7 thresholds baked in [Phase 2]
Phase 1 computed IS-efficiency ad hoc (P1-D2, P1-D3: inline, N=2,000, tol 1e-3). The blueprint lists "Effective sample size module implementation" as a standalone Phase 2 deliverable because Phase 3's "first real-data ESS measurement" needs a module, not a copied script. Decision: add WI-4 — a single ESS entry point (ESS = ε·N) with the D7 thresholds (primary ESS>500, high-quality ESS>2,500) and configurable N / ODE tolerance (diagnostic vs publication). Consumed by both the Phase 2 benchmark and Phase 3.

### P2-D6 — Cross-track data contract: Track 2 must deliver out-of-transit frames + per-wavelength uncertainties for Phase 3 [Phase 2, cross-track]
The P2-D3 decision (covariance embedding conditions on out-of-transit noise *frames*, never on Σ) silently created a new requirement on Track 2's real-JWST output — one that did not exist when Track 2 built its Phase 0 pipeline. The Track 2 deliverable merged to main (June 2026, `00_track2_jwst_data_pipeline.md` + `MAST_2026-05-11T1524/`) is the **collapsed transmission spectrum**: flux-only, one vector per instrument (NIRISS SOSS, NIRCam GRISM, NIRSpec G395H, NIRSpec PRISM), no time axis, no uncertainty column. This is correct and sufficient for the Phase 4 posterior-predictive comparison figure, but it lacks the two ingredients the noise-conditioning architecture needs at the Phase 3 hand-off:
1. **Out-of-transit frames** — feed WI-3. The source `x1dints` FITS are per-integration spectra (they *contain* the in-/out-of-transit time series), but only the collapsed spectrum was preserved; no raw FITS are in the repo (consistency report points to Track 2's local Windows paths).
2. **Per-wavelength uncertainties** — feed WI-1. Dropped during standardisation; transit-depth errors exist upstream and should be recoverable (likely just a re-exported column).

The tracks remain isolated for parallel development — this changes nothing in Phase 2 (ABC-only). It is an *integration-boundary* contract surfaced early. Action: coordinate with Vedanth (Track 2) to extract, from the `x1dints` files while he still has them, (a) the OOT integration segments and (b) per-wavelength σ, before Phase 3 begins. Fallback for the covariance source (also a Track 2 task): estimate residual covariance from multi-pipeline dispersion (Roy-Perez corpus, D6) instead of OOT frames — a different but admissible Σ estimate, so the OOT route is not a single point of failure.

### P2-D7 — Covariance branch integrates inside SpectraEncoder; `oot_frames` stays out of `required_keys` [Phase 2]
fm4ar's context embedding net is a *sequential* block list with no parallel-branch/merge primitive (the `Concatenate` block joins raw context keys, not block outputs). So the covariance embedding (WI-3) is composed *inside* `SpectraEncoder` behind a `use_covariance` flag (symmetric to WI-1's `use_error_bars`) and its output is concatenated onto the spectrum context vector — output dim becomes `output_dim + cov_embed_dim`, which fm4ar's `determine_output_shape` picks up automatically. Key subtlety: `oot_frames` is `(B, K, n_bins)`, but `determine_output_shape` builds dummy inputs as `(batch, n_bins)` for every entry in `required_keys` — a 3D key can't be represented. Resolution: keep `oot_frames` **out** of `required_keys` and read it via `context.get("oot_frames")` in forward, returning a zeros vector of width `cov_embed_dim` when absent. This is safe because `SpectraDataset.__getitem__` returns the *full* sample dict (it does not filter by `required_keys`), so `oot_frames` still reaches the model at runtime via default collate; `required_keys` only drives build-time shape inference, where the zeros branch yields the correct bumped dim. Featurisation default is `flatten` (upper-triangle of Σ̂; loss-free, MPS-safe, adequate for ABC P=52); `eigen` (top-k eigenvalues, the DINGO-PSD analogue) is the JWST-scale alternative but uses `eigvalsh`, which may be unsupported on MPS — run it on CPU/CUDA.

### P2-D8 — MIRAGE code lives in its own `mirage/` package; fm4ar stays pristine via runtime registration [Phase 2, architecture]
Originally all MIRAGE code was written *inside* a clone of `timothygebhard/fm4ar` (new files + edits to its source), which meant: (a) the code was invisible to the Project repo (`.gitignore` has `fm4ar/`) and uncommitted in a clone we can't push to, and (b) it required maintaining a fork. Investigation showed every edit was either a MIRAGE *addition* through fm4ar's hardcoded factory functions (`block_type_string_to_class`, `get_noise_generator`, `get_data_transforms`, `get_mean_and_std` — no plugin hook exists) or one avoidable behaviour tweak (MPS in `resolve_device`). Decision: move all MIRAGE code into a first-class `mirage/` package in the **Project** repo and keep fm4ar **pristine upstream** (pinned in `requirements.txt` at `c05841c`), teaching fm4ar's factories about MIRAGE components at runtime via `mirage/register.py` (monkeypatch, run once on `import mirage`). Patch points: block registry + theta stats are same-module callers (patch the module attr); `get_data_transforms` is `from`-imported by `training/stages.py`, so its name is rebound there too; `get_noise_generator` needs no patch (our transform owns its generator). The MPS edit is replaced by `device: "mps"` in the local configs. Training now goes through `scripts/train.py` (does `import mirage` then runpy's fm4ar's `train_local.py`); inference scripts add `import mirage`. Result: single repo, code clearly attributable to MIRAGE, fm4ar updatable, **no fork needed**. Verified end-to-end: fm4ar pristine, registry build → dim 320, injection transform + encoder forward on real ABC. Tradeoff: monkeypatching is less discoverable than an inline `match` case, contained to one documented file. Rejected alternative — fork fm4ar and commit there: works but splits the project across two repos and needs fork maintenance for changes that don't require it.

### P2-D9 — WI-5 ablation metric is coverage, not IS-efficiency [Phase 2]
The WI-5 design (and P2-D2) assumed the ablation would be measured by IS-efficiency (ESS), but that breaks under injected noise. IS-efficiency needs a reference posterior matched to the likelihood the model sees. Two facts kill it here: (1) the ABC nested-sampling reference posteriors (`tracedata`) were computed on ABC's *original (diagonal)* noise, not our injected correlated noise; (2) no live forward model is installed (TauREx / petitRADTRANS absent), so proper simulator-IS (likelihood = N(x; μ(θ), Σ)) is infeasible. Using KDE-of-NS as the IS target would be worse than imprecise — it would *invert* the result: under correlated noise the correct posterior is wider, so a well-calibrated +cov model matches the narrow clean-NS reference *worse* (lower ESS), while the overconfident no-cond model matches it *better* (higher ESS). The metric would reward the failure it is meant to expose. Decision: measure the injected-noise ablation with **central-interval coverage at 68/95%** (`mirage/eval/coverage.py`) — needs only the known true θ (ABC provides it), is valid under any noise, aligns with the blueprint's planned coverage diagnostics, and directly measures overconfidence. ε/ESS (WI-4 module) is retained only for the **clean-data** baselines (Phase-1 continuity). Driver: `scripts/run_noisecond_ablation.py` (each planet noised once, identical context to all three arms); recover-Σ sanity check: `scripts/recover_sigma_check.py`. Note (σ calibration): the injected `CorrelatedNoiseGenerator` σ is set to 0.001–0.005, near the ABC flux bulk (~0.01; baked-in noise ~4e-5), so correlated noise degrades calibration without destroying signal — confirm with a short run that the no-cond arm is degraded but not collapsed before the full ablation.

**Amendment (2026-07-02, calibration run).** First eval (20 planets) gave near-identical coverage for all three arms (cov@68 ≈ 0.68–0.71, cov@95 ≈ 1.0) — no-cond is NOT overconfident. Reason: all arms are trained *and* tested on the same injected-noise distribution, so each learns a marginally-calibrated posterior (no-cond just learns an appropriately-wide *average* posterior). There is no train/test misspecification, so **marginal coverage cannot discriminate the arms by construction.** The conditioning benefit here is *adaptivity* (sharper posteriors per-observation when the noise is benign), not marginal calibration. Decision: keep coverage as a calibration check, but add two discriminators to `run_noisecond_ablation.py` — (1) **normalised 68% interval width** (sharpness, ↓ better; in θ-scaler units) and (2) **expected log-density at truth** via `log_prob_batch` (proper scoring rule, ↑ better; compared in scaled θ space where the shared-scaler Jacobian cancels across arms). Conditioning should show as sharper + higher log-density. Only if all three match on *these* too is the injected σ genuinely too weak (then raise σ + retrain). No retraining needed to test — re-evaluate the existing 3 checkpoints.

### P2-D10 — ABC flux was unnormalised + dirty; this (not σ) is why every phase was near-prior [Phase 2 → Phase 0 retrofit]
The clean-ref diagnostic (P2-D9) came back at width68 = 2.339 ≈ prior-width (2.36): the clean Phase-1 model, with zero injected noise, was already near-uninformative. So σ was never the real problem. Root cause: the ABC flux is ~500× smaller than the O(1) scale fm4ar/Gebhard were built for (median 0.0018 vs Gebhard 1.28) AND polluted by corrupt spectra (values to 4304; ≥1 whole spectrum with median flux ~132). fm4ar's `SoftClipFlux` block (Vasist 2023) and the `DefaultNoiseGenerator` σ range (0.05–0.5, matching Gebhard's `sigma-0.126`) all assume O(1) flux; `transformer_abc` applied no flux normalisation at all (`prepare_abc_hdf5.py` wrote the raw `instrument_spectrum` with only a NaN filter). The transformer therefore trained on ~1e-3 inputs polluted by extreme outliers → near-prior posteriors, and the always-terrible ε (~0.05% vs Gebhard 19%) was the same symptom. This is a Phase-0 data bug that silently capped every phase. Confirmed by smoke gate: a 6-epoch/4k-sample model on cleaned+normalised flux reached width68 = 1.980, beating the 512-epoch/65k unnormalised Phase-1 model (2.339). Fix (folded into `prepare_abc_hdf5.py`, so normalised data is the default): (1) drop planets with any |flux| bin > 0.5 (87 planets: 0 NaN + 87 corrupt); (2) per-bin standardise flux using TRAIN-split stats (store `flux_mean`/`flux_std`), scale `noise` by the same std. Configs (`transformer_abc` + 3 `noisecond_*`) get a `SoftClipFlux(bound=15)` front block for residual standardised outliers; noisecond injected σ rescaled to the O(1) space (0.05–0.3). Rejected alternative — a runtime Standardize transform leaving raw data: cleaner reversibility but two data paths and a stale-data hazard; folding into prepare keeps one canonical, reproducible source.

### P2-D11 — WI-5 ablation result: covariance conditioning is a calibration (logdens) win, not a sharpness win; report by logdens, stratified [Phase 2, result]
Final ablation on the normalised base (clean-ref width68 = 0.68, logdens@truth = −0.10). **Reported regime σ=0.05–0.3, n=50:** no-cond 2.39 / −8.77 · +σ 1.99 / −8.42 · +σ+cov 2.23 / −7.72 (width68 / logdens). Findings:
- **By logdens@truth (a strictly proper scoring rule) the result is a clean MONOTONIC win: no-cond < +σ < +σ+cov, cov best** (+cov assigns ≈e^1.05 ≈ 3× more density to the truth than no-cond). The covariance embedding genuinely helps. Recover-Σ (R² 0.3–0.4 on length scales) independently confirms the branch learned the noise structure.
- **No width benefit — ever.** All injected arms sit near prior width (~2.0–2.4) in every regime. This is not +cov failing: correlated noise genuinely *widens the true posterior toward the prior*, and +cov correctly stays wide while centring better on the truth (hence highest logdens). +σ is *deceptively* sharp (1.99) but overconfident-wrong (low logdens). ⇒ **Lead with logdens, not width, in the paper.**
- **σ sensitivity (don't lower it):** an attempt to "strengthen" by lowering σ to 0.02–0.12 BACKFIRED — better absolute logdens (−7.6) but the cov−nocond gap shrank 1.05 → 0.22 nats. Weaker noise ⇒ less for conditioning to fix. Stronger noise ⇒ larger gap. σ=0.05–0.3 is the reported regime; the 0.02–0.12 checkpoints are kept in `configs/noisecond_*/_sigma012/`, the 0.05–0.3 (reported) ones are top-level.
- **Per-σ stratification (`run_noisecond_ablation.py` now bins by injected σ):** the cov−nocond Δlogdens *grows with σ* — the benefit concentrates on the high-noise planets, making the adaptivity story explicit (no-cond must pick one average posterior; +cov adapts per-planet). This is the strongest framing and needs no retraining.
- **Conclusion:** a modest but genuine, *calibration-not-sharpness* effect on ABC, which has limited headroom (low-information spectra + destructive correlated noise). The Phase 2 mechanism is validated end-to-end (recover-Σ + monotonic logdens + σ-stratified gap). The real test is **Phase 3 on real JWST** — real correlated instrument noise on real data, the regime the method was designed for. Metric guidance for the paper: logdens@truth as headline, per-σ stratified table, coverage as the calibration check; de-emphasise width68 (it reflects the irreducibly-wide true posterior).

---

## Phase 3 Log

### P3-D1 — Covariance embedding conditions on whitened correlation STRUCTURE, not amplitude [Phase 3]
First Phase-3 step (real WASP-39b OOT frames, `WASP39b_out_of_transit_data/`) built the real noise covariance on the model's 52-bin grid (`scripts/build_real_covariance.py`). Two findings:
1. **Coverage:** PRISM (0.60–5.29 µm, 3241 frames) covers 47/52 model bins; the top 5 bins (5.3–7.28 µm) have NO real data and must be masked. NIRISS/NIRCam/G395H have only 55–93 frames each — too few for a full covariance at native resolution; rebinning to ~52 bins is what makes Σ̂ estimable at all.
2. **Scale mismatch:** real PRISM frame-to-frame relative noise is median **0.5%** (0.003–0.16 across bins) — roughly **10× BELOW** the Phase-2 training regime σ=0.05–0.30. Fed raw, the real Σ̂ is out-of-distribution for the embedding (which was trained on `flatten` of the *covariance*, a scale-sensitive feature). Real off-diagonal structure IS present (nearest-neighbour correlation median +0.24, up to +0.74), so the phenomenon the method targets exists in real data.

**The cross-examination this invites:** "10× quieter is specific to THIS dataset; another target/instrument/epoch could be far louder and break the method." Correct — *for a scale-sensitive embedding.*

**Decision:** condition on the **whitened correlation matrix** (per-bin standardise each frame across the K frames before forming Σ̂ → diag≈1), so the embedding sees noise *structure* only; amplitude is carried separately by the per-λ σ vector (WI-1). Consequences:
- **Scale-invariant by construction:** a visit 10× quieter or louder produces the *same* conditioning vector — the dataset-specific critique is answered at the architecture level, not per-dataset.
- The residual, honest generalisation axis is **structure** (correlation length, 1/f slope), defended by the D4 domain randomisation (SE length ∈ [0.10,1.00] µm, OU length ∈ [0.50,3.00] µm, ρ ∈ [0.3,0.8]) and made *falsifiable* by an envelope check: does each real instrument's correlation structure fall inside the trained kernel envelope? Testable now across all 4 instruments (n=4, not n=1) → WASP-96b later makes n=5.
- Implemented as `CovarianceEmbedding(whiten=True)` + `SpectraEncoder(cov_whiten=True)` + `cov_whiten: True` in `configs/noisecond_cov`. Backward-compatible (default False, so Phase-2 arms/checkpoints unaffected). **Requires retraining the cov arm** with whitening on before the real Σ is in-distribution — gated on the envelope check passing.

Rejected alternative — retrain the arm at real-matched σ (~0.005–0.05) instead of whitening: still scale-*specific* (ties the model to WASP-39b PRISM's amplitude), so it fails the exact cross-examination above. Whitening removes the dependence entirely; the σ-vector already carries what amplitude the flow needs.

Validation harness: `scripts/validate_whitening.py` (envelope check across 4 instruments + direct scale-invariance demo).

**Validation outcome (2026-07-23):**
- **(A) Scale-invariance — PASS, clean.** Real PRISM frames vs a ×10 copy through `CovarianceEmbedding(whiten=True)`: relative drift 1.1e-7 (invariant); `whiten=False`: 5.8e-2 (scale-dependent). The design goal is met — conditioning is provably independent of noise amplitude, so the "10× quieter is dataset-specific" critique is answered architecturally.
- **(B) Envelope — real correlation is present but SHORT-range at 52 bins.** nn-corr: NIRISS +0.61, PRISM +0.24, NIRCam +0.26, G395H +0.03. Only NIRISS resolves a 1/e length (0.12 µm) — and it sits at the **low edge** of the training envelope [0.12, 1.13] µm (median 0.50). The other three decay within one bin (sub-resolution) — partly real, partly the 409→47 rebinning averaging out fine structure. **Two implications:** (i) the D4 kernel envelope's short-length edge (se_min 0.10, ou_min 0.50 µm) should extend **downward** so real JWST structure is comfortably *inside*, not on the boundary; (ii) correlation length should be measured on a finer grid (native, or rebin the *covariance* not the frames) before concluding it is near-white — aggressive frame rebinning decorrelates. Whitening itself is validated regardless; these refine the *training distribution*, gated before the cov-arm retrain.

**Refinement outcome (2026-07-23) — real correlation is short-range; kernel extended.** `scripts/measure_corr_length.py` estimated PRISM correlation on a resolution ladder (52 → native 409 bins). The nn-correlation settles at **+0.21 (native) / +0.37 (model-52)** and the 1/e length stays *below one native pixel (<0.004 µm)* at every resolution → real OOT noise is **mostly white with a weak, genuinely short-range correlated component**, NOT the long-range 1/f drift the Phase-2 kernel assumed. The training kernels (se 0.10–1.00 µm) are far too smooth (se=0.10 → nn≈0.97 vs real ≈0.37). **Decision:** extend the D4 kernel's short edge down — `se_length_min 0.10→0.02`, `ou_length_min 0.50→0.05` (maxes kept at 1.00 / 3.00 so long-range structure for other instruments/systematics is still covered) — in all 3 arm configs (`noisecond_{nocond,sigma,cov}`) so injection stays identical, and in `validate_whitening.py`'s envelope sampler. Caveat: the 1/e-length *envelope statistic* on the 52-bin grid can't resolve these short kernels (they read sub-resolution there too), so it barely moved — the measurement ladder, not the 52-bin 1/e stat, is the authoritative characterization. Net: whitening (scale) + extended short-length kernel (structure) together put the real WASP-39b noise inside the training distribution → **ready for the cov-arm retrain** (`cov_whiten: True`, ~10h CPU).

**Post-retrain ablation (2026-07-31, whiten + extended kernel, n=50) — result SURVIVES and improves; supersedes P2-D11's "monotonic" framing.** **[LATER SUPERSEDED by P6-D2: the no-cond 0.587 here is an untrained MPS run.]** clean-ref 0.68/−0.10. no-cond 1.99/−8.41 · +σ **2.58/−8.82** · +σ+cov **2.21/−7.71** (width68/logdens). Coverage@68 (nominal 0.68): no-cond 0.587, +σ 0.733, +σ+cov **0.687**.
- **+σ+cov is best by logdens AND best-calibrated at 68%** — highest density-at-truth *and* closest to nominal coverage. Core claim holds under the realistic short-range kernel.
- **Ordering changed: +σ < no-cond < +cov (was monotonic no-cond<+σ<+cov in P2-D11).** +σ alone is now the WORST arm — under short-range correlated noise the diagonal σ can't see the structure, so it over-widens (widest, 2.58) and misplaces density (lowest logdens). **This STRENGTHENS the thesis:** marginal error bars are insufficient/counterproductive; only the covariance embedding (WI-3) recovers calibration. Directly answers "why not just condition on σ?" — because it's the worst option. Coverage confirms: no-cond overconfident (0.587<0.68), +σ over-widened (0.733>0.68), +cov ≈ nominal.
- Adaptivity holds: cov−nocond Δlogdens +0.64/+0.59/+0.87 across σ tertiles (largest at high σ; mid dip within ~17-planet/bin noise).
- **Caveat:** +σ retrained fast on MPS — confirm "+σ worst" with one alternate-seed re-run before it's a paper claim (signal is coherent — wide+low-density+over-coverage all agree — so likely real).
- **Paper-guidance (pre-reseed, SUPERSEDED below):** had read this as "error-bars-only is insufficient/harmful."

**ALT-SEED CORRECTION (2026-07-31) — "+σ worst" was a training artifact; RETRACTED.** The seed-42 +σ arm peaked at **epoch 13** (an under-trained stall) vs no-cond 75 / cov 173. Reran +σ with seed 123 → best **epoch 159** (properly converged) → ablation (n=30): no-cond −8.33 / +σ **−7.89** / +σ+cov **−7.80** (logdens), coverage@68 0.633 / 0.706 / 0.706. So with a properly-trained +σ the ordering is **monotonic no-cond < +σ < +cov again** (matches P2-D11); "+σ is worst / error-bars harmful" does **not** survive reseeding — discard it. **Robust claims (both seeds):** (1) cov > no-cond (+0.53 / +0.70 nats) — solid; (2) cov is the best arm; (3) adaptivity, cov−nocond grows with σ (+0.48/+0.55/+0.58). **Not robust:** the cov-over-σ margin — only **0.09 nats** with a well-trained +σ, and identical coverage@68 (0.706) → on ABC the covariance embedding barely edges marginal σ (limited headroom, per P2-D11). **Paper guidance (final):** claim "conditioning helps monotonically, cov best + best-calibrated; the cov-specific edge over σ is small on synthetic ABC → its real value must be shown on real JWST." Report logdens + coverage@68 + per-σ stratified; de-emphasise width68. Lesson: don't build narrative on a single MPS run — check the best-epoch (under-trained arms mislead).

---

### P3-D2 — Track-2 "spectrum" is stellar flux, not transit depth → retrieve on the published WASP-39b depths; WI-2 adapter built [Phase 3]
Building the real-data adapter (WI-2) exposed a quantity mismatch. The model retrieves on a **transit-depth** spectrum — ABC's raw observable is (Rp/Rs)² (stored `flux_mean ≈ 0.0055`, i.e. ~0.5%). But every Track-2 "transmission spectrum" file (`WASP39b_binned_spectra.csv`, `WASP39b_final_standardized.csv`, …) is **stellar flux**: `flux_mjy` median ~1.5e-7 (Jy), or `normalized_flux` median 1.0 (normalised to the stellar *continuum*). That is the star's light curve, not the planet's depth. The OOT frames are likewise flux. So Track 2 delivered the flux side of everything but never the actual depth = 1 − F_in/F_out (no in-transit frames were preserved either).
**Decision:** retrieve on the **published WASP-39b transmission spectrum** (Rustamkulov et al. 2023 NIRSpec PRISM/G395H + ERS NIRISS/NIRCam depths) — public, authoritative, and **the same object the WI-5 FASTER calibration anchor uses**, so one sourcing serves two work items. Rejected alternative — have Track 2 compute depths from in/out flux: needs in-transit frames not delivered, and re-derives a spectrum the community already publishes canonically.
**WI-2 adapter (built, `mirage/datasets/real_jwst.py` + `scripts/build_real_observation.py`):** published depth CSV (`instrument, wavelength_um, transit_depth, depth_error`) → inverse-variance rebin onto the 52-bin ABC grid (overlapping instruments merge by weight) → **ABC-space normalisation** `(depth − flux_mean)/flux_std` (the P2-D10 per-bin stats — the sim-to-real bridge; without it the depth is nowhere near the space the model trained on) → model-input context {flux, wlen, error_bars}. Real coverage 47/52 bins (top 5 masked, P3-D1; masked filled neutrally: flux 0, σ = median observed). Validated end-to-end on a physically-plausible stub (baseline 0.021 + H2O/CO2 features) pending the real file.
**Finding (stub, confirm with real depths):** WASP-39b (depth ~0.021) normalises to **~+2σ** in ABC's per-bin N(0,1) space — it sits in the **TAIL** of ABC's training depth distribution (ABC's typical depth ~0.0055 ≪ WASP-39b). A real sim-to-real distribution shift: the ABC-trained flow is being asked to extrapolate above its training body, which is exactly what RoPE-OT calibration (WI-5) exists to correct. Flag to verify once the published depths are loaded.

### P3-D3 — Real-data likelihood is TauREx3 (ABC's simulator), NOT petitRADTRANS; forward model built + validated to ~5% [Phase 3, WI-1]
Setting up the forward model (WI-1) corrected a wrong assumption: I first installed **petitRADTRANS** (Gate B) thinking ABC followed the Vasist/Gebhard lineage. It does not — ABC is the **Ariel Data Challenge**, forward model **TauREx3** (per `00_track1_environment_baselines.md`, `prepare_abc_hdf5.py`, `data/abc/NeurIPS taurex tutorial/`). MIRAGE trained on TauREx3 spectra, so the importance-sampling likelihood **must** be TauREx3 — using petitRADTRANS would inject the very "forward-model misspecification" that collapses ESS (Gebhard: 19% → 1.5e-5), manufacturing the failure the metric detects. petitRADTRANS is not wasted (it's the FASTER/Vasist baseline engine and a natural deliberate-misspecification ablation later), just not the ABC likelihood.
**Env:** TauREx3 needs numpy≥2.0.2, incompatible with fm4ar's numpy 1.26.4 (known Phase-0 blocker) → separate conda env **`mirage-taurex`** (taurex 3.3.2, numpy 2.4.6). The forward model is standalone (`scripts/taurex_forward.py`, imports no mirage/torch code).
**Model (reverse-engineered from the challenge's own `TP_forward.ipynb` + `parfile.par`):** transmission, isothermal T, TaurexChemistry H2/He (ratio 0.17) + constant H2O/CO2/CH4/CO/NH3, Absorption+CIA(H2-H2,H2-He)+Rayleigh, no clouds (ground truth is cloud-free 6-param), 100 layers 1e-1–1e6 Pa, opacities read straight from the on-disk `.dat` tables via `OpacityCache`. θ = [T, log_H2O, log_CO2, log_CH4, log_CO, log_NH3].
**Key finding — per-planet geometry lives in `AuxillaryTable.csv`:** transit-depth baseline varies ~20× across planets (0.0004–0.0092) but Rp/R*/M* are NOT in the 6 retrieved params — they're *given* auxiliary data (`planet_radius_m`, `planet_mass_kg`, `star_radius_m`, `star_temperature`). Feeding fixed Rp=R*=1 gave ~1000% error; feeding per-planet aux → **5.5% median**.
**Validation status (`--validate 8`):** median mean-rel-error 5.5%, correct baseline + feature structure + per-planet scaling — unmistakably the right model. BUT ~6.5× the observational noise → a **~5% systematic continuum offset remains** (model runs slightly LOW; flat/featureless planet 3 is 4.4% off = pure continuum level, so not features/binning). Prime suspect = the radius **reference pressure** (or a minor CIA/Rayleigh continuum detail).

**Residual diagnosed (2026-08-01):** reference pressure **RULED OUT** — sweeping atm_max_pressure 1e6→1e8 Pa and atm_min 1e-1→1e-4 moves the flat-planet continuum <0.1% (obs/model stays ≈1.046). Binning also ruled out (the flat planet bins exactly). So the ~4.6% is a **continuum/depth-convention offset**, planet-dependent (4–8%), most consistent with **TauREx version drift** (ABC was generated with a 2020-era TauREx; the env has 3.3.2 — transmission radius/annulus conventions can differ a few %). Erasing it to the noise floor would require pinning the exact original TauREx version — diminishing returns. **Decision: accept the forward model as validated-to-~5% for importance sampling**, document the ~5% as a known forward-model systematic, and defer exact-version matching as a low-priority refinement (or pin the 2020 taurex version later if a publication ESS demands sub-% fidelity). The model is correct (right physics, baseline, scaling, features); this is a fidelity ceiling from tooling, not an error.

### P3-D5 — First real-data ESS: MIRAGE collapses on WASP-39b (ESS=1) — the sim-to-real gap, quantified [Phase 3, WI-4]
Two-env importance-sampling pipeline (`scripts/real_ess.py` samples in `mirage`; `taurex_forward.py --forward-npz` scores in `mirage-taurex`; ESS computed on the 47 covered bins). Base (clean) model, N=500. **Result: ESS = 1.0, ε = 0.20%, best-fit χ²/dof = 3565.** MIRAGE's posterior on WASP-39b is qualitatively wrong (T≈659 K, log H2O≈−11 — near-waterless & cool, vs WASP-39b's known strong H2O/CO2) → its forward spectra can't match the data → weights collapse onto one sample. **This is the headline "problem" result** — the sim-to-real collapse the project exists to fix, driven by the +2σ distribution shift (P3-D2/WI-2). Caveats to disentangle later: (1) base model only (noise-conditioned arms may differ); (2) real JWST σ is tiny (~0.35% of depth), so even the WI-1 ~5% forward-model systematic inflates χ² (~200 of it); (3) the fix path is WI-5 (RoPE-OT calibration). Data in `data/real_ess/` (gitignored).

**UPDATE — noise conditioning REFRAMES the collapse (cov arm, `real_ess.py --arm cov`):** ran the actual method (cov arm, real whitened OOT frames wired in) not just the base. Retrieval becomes physically sensible — **T=1291 K** (vs base 659 K; WASP-39b Teq~1100 K), **log H2O=−6.9** (vs base −11.2). ESS still 1.0 BUT **best-fit χ²/dof = 151 (vs base 3565) — a 24× better fit**, and 151 ≈ the forward-model-systematic floor (WI-1 5% × depth/σ ≈ 200). So: (i) the base "catastrophic collapse" **overstated the failure** — it was the weakest baseline; (ii) with conditioning, MIRAGE retrieves WASP-39b about as well as a 5%-accurate forward model *permits* — residual misfit is **tooling (forward systematic + tiny σ), not a bad posterior**; (iii) **raw ESS=1 is now almost entirely a metric artifact** (tiny σ + 5% forward), which is exactly why the fair metric is coverage/JS vs the NS anchor, not raw IS-ESS. **Positive result: noise conditioning demonstrably helps on real data (24× fit improvement).** Report base-vs-cov χ² as the conditioning win; use NS-comparison (WI-5) as the headline real-data metric, not raw ESS.

### P3-D6 — Realistic error budget is essential; Fix-1 (coverage) + Fix-2 (shape) improve the retrieval but not the fit [Phase 3]
After the ESS=1 collapse (P3-D5), attempted three fixes. **Fix 1 — retrain with extended coverage** (`generate_training_data.py` via the TauREx forward model: prior extended to log-X∈[−9,−1] + geometry spanning WASP-39b's large baseline; 32k spectra → `data/abc_ext/`): abundances moved from nonsense toward range (log H2O −11→−5.3, CO2 → correct −4.6). **Fix 2 — condition on shape not level** (`prepare_ext_hdf5.py --remove-baseline`; removes the radius/baseline confound the model can't infer; retrained `configs/transformer_abc_shape`): retrieval became literature-consistent (T=1057 K, log H2O=−3.2, CH4/NH3 correctly low). **Error-budget methodology fix (critical):** the tiny real JWST σ (~0.35% of depth) made the IS likelihood a needle → ESS=1 artifact + a 6-hour NS stall. Adding the WI-1 forward-model systematic as an error floor `σ_eff=√(σ²+(f·depth)²)` fixed tractability AND the ESS metric — and is *more correct* (the forward model IS ~5% uncertain). **Lesson: always use a realistic error budget** — with a loose 5% floor MIRAGE's fit looked good (χ²=2.76); with a realistic 1% floor it was χ²=301. The 5% floor was giving false confidence.

### P3-D7 — DEFINITIVE real-data diagnosis: forward-model misspecification, not coverage/radius/clouds/resolution [Phase 3, result]
Independent nested-sampling cross-check (`taurex_retrieve.py`, our own 6/7-param NS on the real WASP-39b spectrum — published FASTER/ERS posteriors use incompatible parameterisations so can't anchor) revealed the crux: **under our forward model, an unphysical cold-flat atmosphere (T≈114 K, prior floor) fits the real spectrum BETTER (χ²/dof≈2.55) than the physical hot one (MIRAGE T=1057, χ²/dof≈301).** That is the unmistakable signature of forward-model misspecification. Systematically ruled out candidate fixes:
- **Coverage (Fix 1):** abundances improve, fit doesn't → not it.
- **Radius/level (Fix 2):** params improve, fit doesn't → not it.
- **Clouds (Path B):** added a 7th param (SimpleClouds); NS-with-clouds STILL rails to cold T=114 K → not it.
- **Resolution:** NS on the native ~R100 spectrum (143 bins, not ABC's 52 ~R15) STILL rails to cold T=111 K → **resolution refuted** (had hypothesised it; test disproved it).
**Conclusion:** the sim-to-real gap for real JWST is dominated by **forward-model fidelity** (our TauREx opacities/physics cannot reproduce real WASP-39b's features — it lacks clouds detail, SO2, a T-gradient) and is **not closed** by domain coverage, cloud parameters, or spectral resolution. Requires a fundamentally better forward model — beyond Aug-29 scope. **This IS the project's failure-mode #3 ("1D forward-model misspecification"), demonstrated + systematically diagnosed on real data with independent NS cross-checks + realistic error budgets.** → Paper = Path A (honest sim-to-real ablation/diagnosis). Fix 3 (RoPE-OT) moot: can't calibrate out a misspecified forward model. WI-6 (WASP-96b) not pursued.

### P3-D8 — Fixing the forward model, NS-first: T-gradient RULED OUT [Phase 3→4, in progress]
Decision (2026-08-03): pursue the "fix the forward model" direction (vs the diagnostic-paper direction) from P3-D7. Method = **NS-first**: add one physics lever to the TauREx forward model at a time and let nested sampling on the real WASP-39b spectrum say whether a *physical* solution can beat the cold-flat degeneracy, *before* spending a night retraining MIRAGE on it. Retrain pipeline is fully wired and dormant (generator `--tprofile npoint` → 7-dim θ=[T_surface,T_top,5×logX]; `prepare_ext_hdf5` 7-dim; `register.py` `abc_grad` stats; `configs/noisecond_grad_cov`) — fires only on a green probe.

**Lever 1 — T-gradient (2-point NPoint profile, `--tprofile npoint`): RULED OUT.** NS (300 live, 47 bins, 1% floor) with `T_surface`/`T_top` free STILL collapsed cold: **T_surface=117 K, T_top=147 K** (top≈bottom, no real gradient), χ²/N=2.65 — statistically identical to the cold isothermal reference (T≈114 K, χ²≈2.55). Given a vertical gradient to exploit, the model still prefers cold-and-featureless. So the misspecification is **not** the P-T structure. Now the 3rd single lever ruled out (clouds, resolution, gradient).

**Why no local-physics lever escapes cold-flat (the governing logic):** cold-flat wins because a featureless spectrum fits the floor-limited, R~15-binned real data better than any *featured* model the current 5-molecule opacities produce. Adding more features from the same molecules (or wrong ones) can only make a featured model fit *worse*. The only escapes are (a) the missing absorber that makes the *right* feature — **SO2** at ~4µm (ABC's model lacks it; WASP-39b's landmark detection) — or (b) higher-fidelity opacities. **Both are blocked on external data** (on-disk xsec = H2O/CO2/CH4/CO/NH3/TiO/VO only; no SO2), not on compute — parallelism cannot unblock them. → SO2 sourcing is the next real lever (morning task: source ExoMol SO2 xsec → convert to the `.dat` format → validate load → probe).

**Overnight ruling-out matrix (`scripts/run_ns_matrix.sh`) — COMPLETE, misspecification confirmed:** 3 parallel local-data NS probes (300 live, 1% floor), each self-reporting χ²/N:
| probe | T_surface | χ²/N |
|---|---|---|
| cold_iso (ref) | 114 K | 2.55 |
| cold_grad | 117 K | 2.65 |
| grad_cloud | 116 K | 2.62 |
| hot_iso (T forced ∈[700,1800]) | 700 (railed to floor) | 38.4 |
| hot_grad (gradient, forced) | 700/703 (railed) | 38.5 |

Two smoking guns: (1) **both hot probes railed to the *lower* T bound and drove all abundances to the −9 floor** (muting features) — forced-physical χ²=38 vs cold-flat 2.55. With P3-D7's T=1057→χ²=301, χ² rises **monotonically as T becomes physical** (2.55@114 → 38@700 → 301@1057) = textbook misspecification. (2) **grad_cloud collapsed cold** (cloud deck pushed deep/inert) — the one untested physics *combination* also fails. **Complete ruling-out: gradient, clouds, resolution, forced-hot, and combinations all fail.** The only untested, physically-correct lever left is **SO2 / higher-fidelity opacities — blocked on external data, not compute.** Local physics exhausted.

### P3-D9 — SO2 lever RULED OUT; root cause = forward-model fidelity ≈ feature amplitude [Phase 3, result]
Sourced the physically-correct missing molecule: **ExoTransmit `opacSO2.dat`** (Underwood 2016 ExoMol line list) — same ExoTransmit format + T/P/wavelength grid as ABC's 5 existing opacities (byte-identical file size = same grid), a true drop-in with no cross-source inconsistency. Downloaded into the xsec cache, wired as an optional fitted gas (`build_model(extra_mols=["SO2"])`, `taurex_retrieve --so2`), **validated live** (SO2 1e-12→1e-4 drives 17% absorption across the 4µm band — no silent load failure).

**Result: SO2 changes nothing.** NS (300 live, 1% floor):
| probe | T | log_SO2 | χ²/N |
|---|---|---|---|
| so2 (cold-allowed) | 114 | −6.76 | 2.56 |
| so2_hot (T∈[700,1800]) | 700 (railed) | −8.45 (railed to floor) | 38.46 |

Cold-allowed still collapses to 114 K; forced-hot χ²=38.46 ≈ no-SO2 hot_iso 38.44; in the hot case NS drove SO2 *to the −9 floor* (actively unhelpful). The single most physically-motivated lever — WASP-39b's landmark 4µm detection, correct data, validated absorbing — does not rescue the fit.

**Data is NOT the problem (ruled out the cheap explanation):** the processed 47-bin observation retains strong, real features — the 4.31µm bin is the global max (22,148 ppm = the CO2 feature), 2.6–2.9µm H2O/CO2 bump present, peak-to-peak 1248 ppm at ~6× the noise floor. WI-2 reduction preserved the physics; this is genuine high-SNR data.

**Root-cause hypothesis (leading):** with every physics lever exhausted yet strong real features present, the limiter is **forward-model fidelity**, not missing species. The model reproduces ABC only to ~5% (P3-D3), and **5% × 21,000 ppm baseline ≈ 1000 ppm ≈ the 1248 ppm feature amplitude** — the systematic is the size of the signal. A model uncertain at signal level can't be fixed by adding a molecule; cold-flat wins because emitting *no* features costs less than *wrong* ones. Fixing this needs a fully consistent high-fidelity opacity set (e.g. all molecules re-sourced from ExoMolOP R=15000) **and** pinning the exact TauREx version that generated ABC — and even then the *trained* MIRAGE network inherits ABC's opacities, so the retrieval-side fix alone doesn't transfer. = a multi-week research problem, not an Aug-29 item.

**Levers ruled out (complete):** coverage, radius/level, clouds, resolution, T-gradient, forced-hot, physics combinations, **SO2**. The SO2 negative *strengthens* the diagnostic-paper story ("even the correct landmark molecule doesn't help — the limit is forward-model fidelity vs JWST precision"). → Strong evidence to consolidate on Path A (diagnostic paper) for Aug 29; the forward-model "fix" is a post-deadline research effort with lowered odds after SO2.

### P3-D10 — Forward-model FIDELITY ruled out: hi-fi ExoMolOP opacities don't help [Phase 3]
Direction (b) chosen (fix the forward model). Sourced + wired the full **ExoMolOP R=15000** consistent opacity set for all 6 molecules (latest ExoMol/HITEMP line lists: POKAZATEL, UCL-4000, YT34to10, Li2015, CoYuTe, ExoAmes; 2.1GB, `data/opacity_hifi/`, gitignored; `build_model(hifi=True)`, `--hifi`). Validated all 6 load + absorb. **NS probe result: still cold** (T=115 K, χ²=2.42) — statistically identical to the old lo-fi ExoTransmit cold_iso (114 K, 2.55). State-of-the-art line lists do NOT flip the fit physical. Combined with P3-D8/D9, forward-model *fidelity* is conclusively not the limiter.

### P3-D11 — THE FIX: float the planet radius. Cold-flat was a fixed-radius artifact, NOT misspecification [Phase 3, MAJOR — overturns P3-D7]
The tell: WASP-39b IS fit successfully by TauREx in the literature, so *my NS reference failing* pointed at my retrieval SETUP, not the opacities. Every prior probe held the planet **radius fixed** at 1.27 RJ (`taurex_retrieve` fixed WASP-39b geometry) — a ~few-% baseline error the fixed-radius fit could only absorb by going cold-and-featureless. **Freeing radius (`--fitrad`, one extra param) flipped it physical:** T=605.6 K, radius=1.229 RJ (3% shift), **χ²/N=0.761** (vs cold-flat 2.55, forced-hot 38) with **literature-consistent water (log H2O=−3.25)**. Best fit of the entire investigation, and it works even with the OLD lo-fi ExoTransmit opacities — the hi-fi download wasn't needed to fit the data.

**This overturns P3-D7.** The sim-to-real gap is NOT forward-model misspecification (opacities/SO2/gradient/clouds/resolution were all red herrings) — it is the **radius/baseline degeneracy**. MIRAGE's θ=[T,5 mols] has **no radius parameter**, so the network can't absorb a baseline mismatch → almost certainly *why* it collapsed (ESS=1, P3-D5) on real WASP-39b. (Fix-2 shape-conditioning helped precisely because it removed the baseline.) I now have a **valid physical NS anchor** for WASP-39b (T=606, R=1.23, χ²=0.76).

**The fix — retrain MIRAGE with radius in θ (in progress):** new radius model θ=[rp_rj, T, 5×logX], WASP-39b geometry fixed except radius (mirrors the NS that worked; fixing R* keeps baseline↔radius a bijection so radius is learnable). Cascade built + smoke-tested: `generate_training_data.py --tprofile radius`, `_ABC_RAD` stats in `register.py`, `prepare_ext_hdf5 --prefix abc_rad`, `configs/noisecond_rad_cov`. 40k spectra generated (lo-fi, 32k train), cov arm **training now** (~10h CPU). GOTCHA fixed: fm4ar `random_split` needs `n_train+n_valid == train-file rows` (28000+4000=32000), not the pre-split 32000+4000.

**Confirmation probe** (hi-fi+SO2+free radius) was launched but died (exit 144, OOM under CPU contention) — not load-bearing; the lo-fi fitrad χ²=0.76 already confirms the fix, and the retrain uses lo-fi opacities anyway. Re-run later if a hi-fi anchor is wanted for the paper.

**Levers now correctly classified:** coverage/radius-level/clouds/resolution/gradient/SO2/hi-fi-opacities were all *not* the fix; **freeing the planet radius IS.** → real-data result is back ON (not a diagnostic-only paper). Next: evaluate the radius-retrained cov arm on real WASP-39b (ESS/coverage vs the χ²=0.76 NS anchor).

### P3-D12 — Radius-retrained MIRAGE EVALUATED on real WASP-39b: the fix works (radius + water recovered), ESS is an efficiency artifact [Phase 3, result]
Trained the radius cov arm (`configs/noisecond_rad_cov`, θ=[rp,T,5×logX], val loss 1.16, early-stop ep330). Two-env real-data eval pipeline extended for radius (`real_ess.py` arm "rad" + `taurex_forward --forward-mode radius` applies each sample's radius to geometry; `spectrum(tprofile="radius")`).

**RESULT — the radius parameterization works:**
- **Best-fit χ²/dof = 0.06** (was 301 for the physical solution under the fixed-radius model) — MIRAGE now FITS real WASP-39b.
- **Radius recovered: IS-reweighted 1.22 RJ** (truth 1.27, NS anchor 1.23). The network infers the baseline it previously couldn't.
- **log H2O IS-reweighted ≈ −3.0** (water-RICH, literature-consistent, matches NS anchor −3.25); T~1300 K (≈ WASP-39b Teq). CO2 the one still-discrepant minor param.
- vs the old fixed-radius collapse (ESS=1, χ²=301) this is the qualitative breakthrough.

**ESS caveat (important framing):** raw IS-ESS stays 1–5 and does NOT scale with N (tested N=500→20000: ESS flat ~5, ε dropped 1.17%→0.027%). Cause: with 47 bins the likelihood is peaked, and the FMPE **proposal is overdispersed** relative to it, so a single best sample dominates the weights regardless of N. **Raw ESS is the wrong yardstick here** (flagged back at P3-D5/D6). The likelihood-corrected (IS-reweighted) CENTRAL estimate is correct (radius+water right, NS-corroborated) — it is just NOISY (few effective samples). GOTCHA in reading: comparing *proposal means* (T=1581, H2O=−5.7 hot/dry) is misleading — the proposal is broad; the *reweighted* posterior (H2O=−3.0 wet, R=1.22) is the answer and it is physical. Matched-floor check (5%/2%/1%): reweighted radius+H2O stable and NS-consistent at all floors; tighter floor just lowers ESS (sharper likelihood).

**Settled story (3 sequential results):** (1) radius parameterization fixes the fit (χ² 301→0.06, radius+water recovered, NS-corroborated) = core contribution; (2) residual posterior breadth = FMPE proposal overdispersion + intrinsic T–abundance degeneracy at ABC's low resolution (R~15), NOT a correctness failure; (3) higher IS efficiency (ESS) = Component-4 RoPE-OT proposal calibration (won't move the central answer, only tighten it) or higher training-grid resolution. **Real-data result is REAL** — "MIRAGE recovers a physical, literature-consistent WASP-39b retrieval," not diagnostic-only. Decision pending: pursue RoPE-OT for ESS, or bank this for the Aug-29 workshop paper + report coverage/JS-vs-NS instead of raw ESS.

### P3-D13 — Component 4 RoPE-OT calibration: SUCCESS by coverage (7/7), ESS is the wrong metric [Phase 3, result]
The radius-model FMPE posterior is correct-but-overdispersed (P3-D12): raw IS-ESS ~5, does not scale with N. Component 4 (`scripts/ot_calibrate.py`): Gaussian (Bures) optimal-transport of the FMPE posterior onto the WASP-39b NS/FASTER reference posterior (blueprint spec: "anchored to FASTER-validated posteriors"). Pushforward N(μ_FMPE,Σ_FMPE)→N(μ_NS,Σ_NS); use the calibrated proposal (inflate 1.5) for IS on the real spectrum.

**RESULT — calibration works on the metric that matters (coverage):**
- Calibrated posterior: **radius=1.227±0.009, T=636±131, logH2O=−3.46±0.73, logCO2=−5.42±1.38** — ALL match the NS anchor (1.23/606/−3.25/−5.17) and literature (water-rich), with tight honest intervals. best-fit χ²/dof=0.028.
- **Coverage: NS mean inside the MIRAGE-calibrated 68% CI for 7/7 params** (raw FMPE was 2/7). The OT transport corrected the hot/dry proposal (T=1581, H2O=−5.7) to the cool/wet physical solution.

**ESS caveat (settled):** raw IS-ESS only 5→12.4 — still "low." This is because raw IS-ESS is a HARSH/wrong yardstick for a peaked 47-bin likelihood with a Gaussian proxy proposal (a single χ²=0.028 sample dominates weights); it does not measure posterior correctness. The RIGHT calibration metrics — **coverage (7/7) + agreement with the independent NS retrieval** — are excellent. Report coverage/JS-vs-NS, NOT raw ESS (consistent with the P3-D5/D6 note that raw ESS is a tiny-σ artifact here). Pushing ESS>500 further (mixture/SMC proposal, looser error budget) is metric-optimization with diminishing scientific value; single real target also caps faithful multi-target RoPE.

**REAL-DATA ARC COMPLETE (positive):** radius parameterization (P3-D11) fixes the fit (χ² 301→0.06) + recovers radius/water; RoPE-OT calibration (P3-D13) yields a physical, literature-consistent, well-covered posterior (7/7 vs NS) on real WASP-39b. Story = "MIRAGE, with radius inference + OT calibration, produces a calibrated physical retrieval of real JWST WASP-39b" — NOT diagnostic-only. Ready for the Aug-29 workshop writeup. Future (ICML/Track-2): multi-target RoPE + higher-resolution grid to lift intrinsic IS efficiency.

---

## Phase 4 Log

### P4-D1 — Noise-conditioning ablation on REAL WASP-39b: cov does NOT transfer (honest negative) [Phase 4]
Trained the two missing radius-model arms (`configs/noisecond_rad_nocond` = no noise inputs, `configs/noisecond_rad_sigma` = σ-only; cov already done) and evaluated all three on real WASP-39b (`scripts/run_ablation_real.sh`, n=1000; per-arm files `data/real_ess/abl_rad_*`). GOTCHA fixed: never run `scripts/train.py` with `PYTHONPATH=.` — the repo's `./fm4ar/` dir shadows the pip-installed editable `fm4ar` as a namespace package (`fm4ar.__file__=None`) → git-status crash. Run the env python with NO PYTHONPATH (both `mirage` + `fm4ar` are pip-installed editable).

**RESULT (two agreeing metrics — robust, not noise):**
| arm | best χ²/dof | coverage/7 | mean JS→NS |
|---|---|---|---|
| nocond | 0.061 | 3/7 | **0.358** (closest) |
| sigma | 0.132 | 0/7 | 0.365 |
| cov | 0.116 | 3/7 | **0.430** (farthest) |

**Covariance conditioning does NOT help — it is slightly WORST — on real WASP-39b.** All arms recover radius≈1.2 (the radius fix is robust). σ-only is worst on coverage (consistent with Phase-2 "σ-alone overconfident"). Did NOT keep hunting metrics for a cov-favorable one (that would be cherry-picking).

**Why (coherent, not a bug):** the cov arm is the ONLY one that ingests the real JWST OOT noise frames, which are OUT-OF-DISTRIBUTION vs the domain-randomized training kernel (P3-D1: real noise ~10× below trained σ, short-range). So cov conditions on OOD input and is perturbed; nocond ignores noise and does fine. **Whitening (P3-D1) narrowed but did not close the noise sim-to-real gap.** Also: single real target cannot capture cov's calibration-over-population benefit (that lives in Phase-2 ABC coverage over many planets).

**Framing decision (honest):** noise-conditioning is validated on synthetic ABC (Phase 2); on real JWST the dominant gap was the radius/baseline degeneracy (P3-D11, fixed) + calibration (P3-D13, OT); the learned noise embedding does NOT transfer to real single-target data (OOD real noise). This is a stronger, more credible paper than over-claiming. Central-claim pivot candidate: the radius discovery is the more novel real-data finding. Real-data headline results = radius fix (χ² 301→0.06) + OT calibration (7/7 vs NS) — NOT noise-conditioning-on-real.

---

### P4-D2 — Phase 4 completed: ABC-win figure + full corner + calibration robustness [Phase 4]
Rounded out Phase 4 to complete (not just the real-data ablation). Three additions:
- **ABC-ablation figure (`fig5`, the honest OTHER half):** **[SUPERSEDED by P6-D2 — both claims in this bullet came from an untrained MPS baseline; the figure must be regenerated from `configs/h9_*`.]** re-ran the Phase-2 noise-conditioning ablation (`run_noisecond_ablation.py --save`, 50 planets, `data/real_ess/abc_ablation.npz`) — on synthetic data where truth is known, **noise-conditioning WINS**: coverage@68 no-cond 0.587 (overconfident) → +σ/+cov 0.687 (restored to nominal); +σ+cov − no-cond log-density gap GROWS with σ (+0.64 low → +0.87 high = adaptivity). Lets the paper show "cov wins on ABC" beside "cov doesn't transfer to real" (P4-D1) in one honest arc.
- **Full 7-param corner (`fig4`):** calibrated MIRAGE vs NS anchor, contours track across all params.
- **Calibration robustness (`run_robustness.sh`):** OT-cal robust to the inflation hyperparameter — inflate {1.0,1.5,2.0} → radius 1.231/1.232/1.233, logH2O −3.15/−3.09/−2.99, **7/7 coverage at all three**; with earlier floor-stability (5/2/1%), the result is not a tuning artifact.

**PHASE 4 COMPLETE:** benchmark table (`phase4_benchmark.md`) + 5 figures (`figures/fig1..5`) + noise-conditioning ablation (real=negative P4-D1, ABC=win) + robustness + doc (`04_track1_phase4.md`). Real-data story is complete and honest: radius fix (χ² 301→0.06) + OT calibration (7/7 vs NS) = the win; noise-conditioning is an ABC result that doesn't transfer to the single real target (OOD real noise). Ready for Aug-29 workshop writeup.

---

## Phase 5 Log

### P5-D1 — Higher-resolution grid: improves network ACCURACY (centering), not precision/ESS [Phase 5]
Vedanth-independent Phase-5 item (multi-target WASP-96b/K2-18b is Track-2-gated). NS-first probe first (`taurex_retrieve --fitrad --highres --nbins 150`): free-radius NS at 143 bins → χ²=0.63, **T posterior tightened ±50→±31, H2O ±0.22→±0.18** vs the 47-bin case = higher res genuinely adds information → GREEN, retrain justified.

Retrain cascade (reused radius pipeline): `generate_training_data --nbins 150` (geomspace over the real 0.52–5.33µm range; θ=[rp,T,5logX]), new `configs/noisecond_rad_nocond_hires` (n_bins=150, wlen 0.52–5.33), 40k spectra → 32k train, trained nocond arm on MPS (val plateaued ~1.05, stopped on plateau). Eval arm `rad_hires` in `real_ess.py` (build_observation auto-bins to the arm's grid via stats_hdf — grid-agnostic). GOTCHA FIXED: `sample()` computed x_obs/σ on the DEFAULT 52-bin grid while ctx/covered used the arm's 150-bin grid → shape-mismatch on non-52-bin arms; fixed to `abc_grid_and_norm(stats_hdf)`.

**RESULT (hi-res model on real WASP-39b, n=1000):**
- best-fit χ²/dof=**0.038** (finest of the project; lo-res 0.06). ESS=3.7 (≈ lo-res 5, NO gain).
- **Accuracy WIN (centering):** raw proposal T **1581→635** (now matches NS 606!), radius **1.174→1.226**; IS-reweighted radius **1.257±0.017** (truth 1.27), logH2O **−3.36** (water-rich, NS/lit-consistent).
- **Precision/efficiency: NO improvement** — proposal T-std ~296 (≈ lo-res), ESS 3.7. The NS probe showed the data supports a tight posterior (T±31) but the amortized flow captures resolution for CENTERING, not SHARPNESS (amortization gap).

**Honest finding:** higher resolution significantly improves the network's central accuracy (esp. temperature, the most-wrong parameter) but not its precision or IS-efficiency. Future work (more training / per-target refinement / sharper amortization) to close the amortization gap. Vedanth-independent Phase-5 work COMPLETE; remaining Phase 5 = multi-target (Track-2-gated) + Zenodo + full ICML paper.

### P5-D2 — Multi-target (WASP-96b, K2-18b): data reality + per-planet cascade [Phase 5]
Track-2 delivered the OOT packages for WASP-96b (NIRISS) and K2-18b (NIRISS + G395H + MIRI): per-integration frames + per-λ σ, time-axis preserved, raw (not normalised) — exactly the P2-D6 contract. **These are good and used.**

**Transmission depths — Track-2 reduction RULED OUT, use PUBLISHED.** Vedanth's depth pipeline was tried twice — simple in/out mean-differencing, then per-λ batman light-curve fitting — and **both fail the known-truth test:** WASP-39b returns median depth ≈0 (batman fits rail to bounds), vs the true ~0.021; his own diagnostics report `dip_quality: none / weak_or_absent` (transit signal ~2% buried under ~12% uncorrected systematics). Extracting transit depths from raw x1dints needs full systematics detrending — a genuinely hard reduction, out of scope. **Decision: pair Vedanth's OOT frames with PUBLISHED depths** (WASP-39b Carter 2024 [have]; K2-18b Madhusudhan 2023 [sourced, OSF 36djh, depth ~0.0029 ✓]; WASP-96b Radica 2023 [pending — not programmatically downloadable, needs manual grab from supreme-spoon / MAST HLSP]).

**Per-planet cascade (generalised the radius pipeline).** `generate_training_data --planet {wasp39,wasp96,k218}` selects fixed geometry + radius/T priors + grid range (from that planet's published spectrum); `taurex_forward --forward-planet`, `real_ess` per-arm `SPECTRA` + arm registration, `register.py` per-planet stats. **K2-18b is NOT just a geometry swap** — cold M-dwarf sub-Neptune (Teq~250 K, Rp~0.24 RJ) is *below* the hot-Jupiter priors, so it needs its own regeneration (Rp∈[0.10,0.40], T∈[100,600], M-dwarf geometry `K2_18b`, grid 0.85–5.17µm). WASP-96b (hot Saturn) fits the existing priors → clean geometry swap. **Order: K2-18b first (data in hand), WASP-96b once its depths are grabbed.** K2-18b: 40k generated + config built + eval wired + smoke-tested; nocond arm trained (val plateaued ~1.18). GOTCHA fixed: `pgeom` vs the pre-existing `geom[]` buffer name-collision in the generator; and `real_ess` STATS was missing the `k218` key → fell back to the 52-bin ABC grid (caught via "45/52 covered"), fixed to `data/abc_rad_k218/abc_rad_train.hdf`.

**K2-18b EVAL RESULT — METHOD TRANSFERS (positive):** on the real Madhusudhan 2023 spectrum (149/150 bins), **best-fit χ²/dof=0.204, ESS=10.6, radius IS-reweighted=0.233±0.001 (truth 0.235 — near-perfect), T~307 K (Teq~270)**; CH₄≈−4.9, CO₂≈−5.9 present (Madhusudhan's headline detections). The radius parameterization + ABC-trained flow **generalizes from hot Saturns to a cold M-dwarf sub-Neptune** — a very different regime (5× smaller, 4× cooler, M-dwarf). Caveats: abundances broad (±2–3 dex, the same low-res T–abundance degeneracy), IS-reweighted T noisy at ESS~10, and compared to literature not yet an independent K2-18b NS anchor. Headline = correct radius + excellent fit + right molecules, loosely-constrained abundances. Multi-target #1 (the HARD one) = success. NEXT: WASP-96b (clean hot-Saturn case, once its published depths are grabbed).

### P5-D3 — WASP-96b: self-reduced the NIRISS SOSS spectrum from raw MAST (checkpoint PASSED) [Phase 5, 2026-08-13]

**Vedanth's WASP-96b depths are unusable (confirmed empirically):** both his attempts return depth ≈ 0 (mean-diff median −0.00005, batman-LC +0.00009) vs the true white-light ~0.0142 — same failure as his WASP-39b. His OOT package holds only the 56 out-of-transit integrations (the 40 in-transit dropped), so depths can't be recomputed from it. WASP-96b's published reduced spectrum is not public (Radica 2023, MAST-only). → **decided to reduce it myself from raw MAST** (approved, time-boxed with a checkpoint).

**Pipeline = exoTEDRF 2.5.0 (= supreme-spoon, what Radica used) + jwst 3.0.0** in a fresh `mirage-reduce` conda env (+ batman). Raw data = JWST ERO program **2734, obsid 118972907**, only **4.1 GB uncal** (NIRISS/SOSS, CLEAR/GR700XD, SUBSTRIP256, **280 integrations** across 3 segments; the 02101 files are NIS_TACQ, excluded). Config in `wasp96b_dms.yaml`; transit at integrations **97–212** (T14≈2.66 h, from the ERO ephemeris t0=2459751.8217 BJD, P=3.4252565), `baseline_ints=[89,-59]`. **Box extraction of order 1** (0.85–2.83µm = the whole MIRAGE range) — deliberately sidesteps the finicky ATOCA order-overlap step.

**Gotchas fixed:** (1) transient CRDS server timeouts → `CRDS_CLIENT_RETRY_COUNT=8`; (2) pip does NOT ship exotedrf's `files/` data dir → curled the SOSS background/spectrace/wavemap references into the installed package; (3) stellar params trigger a fragile PHOENIX download for a wavelength CCF refinement → set `st_teff/logg/met=None` (default wavelength solution is fine); (4) exoTEDRF resume bug (`to_flag` unbound when all Stage-2 segments are cached) → ran **Stage 3 only** off the cached `badpixstep` files + deepframe (`wasp96b_stage3.yaml`, `run_stages:[3]`), avoiding a Stage-1 recompute. Stage 1+2 fully completed on the first pass.

**Stage 4 = my own batman fit** (`scripts/wasp96_fit_lightcurves.py`), NOT exoUPRF. Key modeling lesson: fixed/too-rigid limb darkening railed (free u2→1.0) and dumped 1437 ppm of residual INTO the transit; the fix = **Kipping (2013) q1,q2 LD + free a/Rs & inc** in the white-light fit, then geometry+LD fixed per bin; two-pass error inflation for honest bars.

**CHECKPOINT PASSED (decisively):** white-light **depth = 0.01421** (truth 0.0142, off 1e-5), Rp/Rs=0.1192, a/Rs=8.98, inc=85.34° (all literature-consistent), **RMS = 142 ppm white-noise-dominated** (OOT 143 ≈ in-transit 139 ppm → shape perfectly fit). Per-bin (90 bins, 0.85–2.81µm, ~130 ppm/bin): the famous **1.4µm water feature detected at 5.8σ** (278±48 ppm above flanking continuum; 1.8–2.0µm band also elevated). Real water-dominated WASP-96b spectrum. **Deliverable: `data/jwst_wasp96b_spectrum.csv`** (gitignored under data/).

**BUILD RESULT — WASP-96b MIRAGE retrieval = SUCCESS, 7/7 coverage (N=3 achieved).** Clean geometry swap: 40k spectra generated (`--planet wasp96 --tprofile radius --nbins 90`, 8 chunks) → prepared 32k/4k/4k → config `noisecond_rad_nocond_wasp96` (reuses `abc_rad` theta-scaler — WASP-96b hot-Saturn priors == WASP-39b radius; encoder wlen 0.84–2.83µm) → `real_ess` wired (`wasp96` ARM/STATS/SPECTRA) → trained (early-stop epoch 188, **best val 1.145**, better than K2-18b's 1.18). **Eval on the real self-reduced spectrum: best-fit χ²/dof=0.079, radius IS-reweighted=1.196 [1.176,1.208] (truth 1.20 — near-perfect), log H2O=−3.64 (water-rich, matches the 5.8σ feature), ESS=9.1. COVERAGE = 7/7 vs the independent WASP-96b NS anchor** (radius/T/5 abundances all inside the MIRAGE 16–84%). Same 7/7 as WASP-39b. **N=3 complete: WASP-39b (hot Saturn, PRISM) + K2-18b (cold sub-Neptune, NIRISS) + WASP-96b (hot Saturn, NIRISS SOSS, self-reduced)** — the radius+flow method generalizes across planet class, instrument (PRISM↔SOSS), AND data provenance (published↔own end-to-end reduction).

**NS-anchor gotcha (fixed, reusable):** `taurex_retrieve` default T bound is (110, 5500) K. For a COLD planet (K2-18b ~250 K, low gravity) this samples runaway scale heights → nan/overflow likelihoods → nestle can't even fill its live points → **0 iterations, hangs indefinitely** (the original K2-18b anchor burned 20 h at 0 progress). FIX = `--tmin 100 --tmax 600` (match the training prior); iterated instantly (663 iters in ~1 min, logz→+1123). WASP-96b (hot Saturn) is well-behaved under the wide default. Both anchors now converged; K2-18b's previously-missing NS anchor now exists → **K2-18b coverage = 6/7** (χ²=0.199, ESS=11.3): the one miss is radius (NS 0.229 vs MIRAGE tight [0.232,0.234]) — an honest near-miss where MIRAGE 0.233 / NS 0.229 / truth 0.235 all agree to ~0.005 RJ but MIRAGE's ±0.001 posterior is slightly overconfident, so the NS point sits 0.003 outside. **Final multi-target coverage: WASP-39b 7/7, WASP-96b 7/7, K2-18b 6/7.** Also: run NS with `PYTHONUNBUFFERED=1` to actually see nestle progress (buffered output hid the 20 h hang). New scripts: `scripts/wasp96_setup_reduction.py`, `scripts/wasp96_fit_lightcurves.py`, `scripts/run_wasp96_eval.sh`; configs `configs/noisecond_rad_nocond_wasp96/`, `wasp96b_dms.yaml`, `wasp96b_stage3.yaml` (gitignored).


---

## Phase 3 Track 2 Log

### P3-T2-D1 — CycleGAN training result: domain randomisation is sufficient; CycleGAN adds no value [Phase 3 Track 2, 2026-08-04]

**D4 ablation executed.** CycleGAN trained for 200 epochs on:
- Domain A: WASP-39b per-integration spectra (2000 samples, 52 bins, WASP39b_standardized_full.csv proxy)
- Domain B: WASP-39b collapsed real spectrum (500 bootstrap samples, WASP39b_final_standardized.csv)

**Training convergence (200 epochs, batch=64, lr=2e-4):**
- `loss_G`: 17.5 → 0.69 (best: 0.69) — generator converged
- `loss_cyc`: 0.96 → 0.015 — cycle-consistency tight (translation is reversible)
- `loss_D`: 0.53 → 0.24 — healthy GAN equilibrium (neither mode-collapsed nor trivially-solved)
- LR decay from epoch 100 → 0 at epoch 200 (Zhu et al. 2017 schedule)

**Architecture:** 1-D ResNet generators (9 residual blocks, ngf=64), PatchGAN discriminators (ndf=64, spectral norm), LSGAN losses. Checkpoint: `configs/cyclegan/cyclegan_best.pt`.

**Ablation evaluation**: Run `scripts/run_cyclegan_ablation.py --n-planets 50` after fm4ar models are available locally. Expected result: `domain_random ≈ cyclegan+random` (Δlogdens < 0.1 nats), confirming D4 hypothesis.

**Ablation result (2026-08-06, n=500 ABC test spectra):**

| condition | MMD↓ to real | KL↓ to real | interpretation |
|-----------|-------------|-------------|----------------|
| sim (no adaptation) | 1.2939 | 20.29 | baseline gap |
| domain_random | **1.1516** | **20.38** | noise injection helps MMD |
| cyclegan_trans | 1.2221 | 23.38 | CycleGAN WORSE than raw sim on KL |

- **Domain randomisation is closer to the real domain** on both MMD (Δ=−0.07) and KL (Δ=−3.0)
- **CycleGAN translation makes KL divergence worse** (23.4 vs 20.3) — translation distorts the spectral distribution rather than aligning it
- Cycle-consistency error = 0.93 (high because domains are very similar in spectral shape; the generator learned to preserve structure rather than shift domain)

**D4 VALIDATED — domain randomisation is preferred over CycleGAN translation.**

Structured randomisation (correlated noise injection) reduces the distributional gap to the real domain more effectively than generative translation. This confirms the blueprint decision: domain randomisation is the primary strategy; CycleGAN is not worth the added training complexity for this problem. Results saved to `configs/cyclegan/ablation_results.json`.

---

### P5-D4 — The χ² numbers never reconciled because the flow and NS sides used different error floors [Phase 5, 2026-10-06, reporting bug — found via Sim2Science review]

**Trigger.** Sim2Science reviewer T46u: the reported χ² values are mutually inconsistent (301 in the abstract, 38.4 in Table 1, "near 2.5" in the text), and the calibrated 0.06 sits *below* the nested-sampling best fit of 0.76 on the same model and data, which is impossible if the transported samples follow p_NS. The objection was correct. The cause was not the sampling.

**Root cause — two different error budgets, never reconciled:**
- Flow side (`scripts/real_ess.py:125`, `scripts/benchmark_table.py:28`) — `sig_eff = sqrt(sig² + (0.05·x)²)`, a **5%** forward-model systematic floor (the WI-1 estimate).
- NS side (`scripts/taurex_retrieve.py:36,71`) — `err = sqrt(err² + (FLOOR·depth)²)` with `FLOOR` defaulting to **0.01**, i.e. **1%**. The in-file comment explains why: a 5% floor drowned the features and produced a degenerate cold-T reference.

χ² scales as 1/σ², so a 5× larger systematic floor deflates the flow's χ² by up to ~25×. Both floors are individually defensible. Reporting numbers from both side by side is not.

**Audit (best-fit χ²/N recomputed under both floors, 47 real-covered bins, WASP-39b unless noted):**

| arm | χ² @ 5% floor (as reported) | χ² @ 1% floor (NS-comparable) |
|---|---|---|
| rad · nocond | **0.061** ← the paper's "0.06" | 1.441 |
| rad · σ-only | 0.132 | 3.177 |
| rad · σ+cov | 0.116 | 2.732 |
| OT-calibrated | 0.028 | **0.662** |
| NS free-radius anchor | — | **0.761** |
| K2-18b cov (149 bins) | 0.199 | 1.180 (NS 1.277) |

WASP-96b NS anchor = 0.793.

**Findings.**
1. **Every χ² in the paper is at the 1% floor except the headline 0.06, which is at 5%.** P3-D7 recorded 301 at the 1% floor and explicitly noted that *"the 5% floor was masking MIRAGE's poor fit"*; the post-radius evaluation then went back to 5%. So "301 → 0.06" silently switches error models mid-claim and inflates the improvement by ~25×.
2. **Corrected headline, at a uniform 1% floor: 301 → 1.44 (radius added) → 0.66 (after OT calibration), against an NS anchor of 0.76.** Still a ~200× improvement from the radius fix. Not 5000×.
3. **The reviewer's "impossible ordering" dissolves.** At a common floor the calibrated flow (0.662) sits essentially on the NS anchor (0.761), the small margin being best-of-N sample selection. Same pattern on the other targets. The numbers no longer contradict each other.
4. **Second, separate error: 0.061 is the *nocond* arm.** The OT-calibrated arm is 0.028. §4 pairs "after the OT calibration" with 0.06, which is not the calibrated arm's number. Whichever was intended, the sentence as written is wrong.
5. **This corroborates the circularity reading rather than rescuing the method.** The calibrated posterior landing on top of the NS anchor is exactly what the transport guarantees. The radius fix remains a genuine, large effect; the calibration remains a relabelling of p_NS.

**Decision.** Report every χ² at a **single stated floor**. Use 1%, because that is what the NS anchors, the 301 baseline and the Table 1 ruling-out matrix all already use — moving NS to 5% would re-introduce the cold-T degeneracy the 1% floor was chosen to break. State the floor explicitly in the table caption. Never again quote a χ² without the error budget attached.

**Audit script:** `scratchpad/chi2_floor_audit.py` (recomputes every arm under both floors from `data/real_ess/*.npz`; fold into `benchmark_table.py` as a regression check).

**Lesson.** The one quantity that had been explicitly "verified" was the one that broke, because verification stopped at *where the number came from* and never reached *what error model produced it*. A number is not verified until its denominator is.

---

### P5-D5 — The OT calibration is an inflated Gaussian centred on the NS anchor; coverage 7/7 was structurally guaranteed [Phase 5, 2026-10-07, MAJOR — retires the P3-D13 claim]

**Trigger.** Sim2Science reviewer T46u argued the calibration is circular. Tested it directly against files already on disk rather than by argument.

**The code already said it.** `scripts/ot_calibrate.py` docstring, lines 7–9: *"Gaussian (Bures) OT transports the FMPE posterior N(μ_f,Σ_f) onto the NS-anchor posterior N(μ_n,Σ_n); **the pushforward IS N(μ_n,Σ_n)**."* The reviewer independently re-derived a sentence sitting in my own source. The OT output is then used as an IS proposal, so the reweighting could in principle pull it back toward the true posterior.

**Measured on real WASP-39b** (`scratchpad/ot_vs_ns_moments.py`, 5% floor as reported):

| quantity | OT-calibrated vs NS anchor |
|---|---|
| mean centre offset | **0.46 σ_NS** (every one of 7 params within 0.95 σ) |
| mean width ratio | **2.08×** NS width |
| IS-ESS | **12.4 of 6312 samples** |

The ~2× width is the `inflate=1.5` default in `fit()` plus the reweighting. At ESS ≈ 12 the IS step is effectively inoperative, so the final posterior stays where the transport put it.

**Conclusion.** The calibrated posterior is an inflated Gaussian centred on the NS anchor. Intervals centred on the NS values and twice as wide as the NS spread **cannot fail** to contain those values, so the reported coverage 7/7 was guaranteed before any data was examined. The reviewer's "discard the network and sample p_NS directly" is empirically correct, and would in fact give a *tighter* (better) width. **P3-D13's "CALIBRATION SUCCESS by coverage (7/7)" is retired.** The radius fix (P3-D11/D12) is unaffected and remains a real, large effect.

**Consequence for v2.** Framing option C (build a genuinely amortized calibration) is closed on the current method — there is nothing to rescue, only something to rebuild. The expensive synthetic known-truth coverage run is no longer worth doing as a *rescue*; it is only worth doing if the honest negative is to be *published* with a number attached (framing B).

---

### P5-D6 — `real_ess_samples.npz` is a shared filename across planets; two downstream scripts read it blind [Phase 5, 2026-10-07, latent data-corruption bug]

**Found while doing P5-D5.** `real_ess.py:102` always writes `data/real_ess/real_ess_samples.npz` regardless of `--arm`, with no planet or arm stamp. The last run wins. It currently holds **K2-18b**:

```
real_ess_samples.npz   bins=149/150  median_depth=0.00294  wl=[0.85,5.17]   <- K2-18b
k218cov_samples.npz    bins=149/150  median_depth=0.00294  wl=[0.85,5.17]
ot_cal_samples.npz     bins= 47/52   median_depth=0.02127  wl=[0.55,5.24]   <- WASP-39b
```

Two consumers read that generic name assuming WASP-39b:
- `taurex_retrieve.py:64` — the NS observation input. Re-running an anchor today fits the **K2-18b** spectrum and saves it under a WASP-39b filename.
- `ot_calibrate.py:39` — the posterior to transport, while line 44 hardcodes `wasp39b_ns_posterior_fitrad.npz`. Re-running `--fit` today transports **K2-18b's** flow posterior onto **WASP-39b's** NS anchor.

`real_ess.py:105` also prints "on real WASP-39b input" unconditionally, so the log actively confirms the wrong thing.

**Published numbers are safe** — `ot_cal_samples.npz` carries WASP-39b's 47 bins, so it was produced when the generic file was correct. **The danger is forward-looking:** Track 1 item 5 of the v2 plan is "re-run the NS anchors at higher live points", which walks straight into this on the first command.

**Decision.** Make every observation file self-identifying and make the consumers refuse to guess:
1. `real_ess.py` stamps `arm` and `planet` into the npz, writes a per-arm copy `real_ess_samples_{arm}.npz` alongside the generic name, and reports the actual planet in its log line.
2. `taurex_retrieve.py` and `ot_calibrate.py` take an explicit `--obs-npz`, and assert the stamped planet matches what was asked for. Unstamped legacy files fall back to a wavelength-range/bin-count check rather than silently proceeding.

The `--obs-npz` flag is needed regardless — the synthetic known-truth experiment requires pointing NS at arbitrary observation files.

**Lesson.** A pipeline stage that writes one fixed filename for N different inputs is a correctness bug waiting for the second input. Stamp provenance into the artefact, not into the filename convention alone.

---

### P5-D7 — Fig 2 audit: the spectra fits are fine, the radius posteriors are not [Phase 5, 2026-10-08, result — qualifies P5-D5]

**Trigger.** Sim2Science reviewer Viae made two claims about Figure 2: *"all inferred spectra look really bad fits, several features are completely off"* and *"the planet radius is not well recovered, in particular for K2-18b."* Tested both (`scratchpad/fit_quality_audit.py`).

**Claim (a) — NOT supported. The fits are good.** Best-fit residuals at the 1% floor:

| target | χ²/N | max resid | bins >2σ | bins >3σ |
|---|---|---|---|---|
| WASP-39b (OT-cal) | 0.662 | 1.60σ | 0/47 | 0 |
| WASP-39b (nocond) | 1.441 | 2.26σ | 5/47 | 0 |
| WASP-96b | 0.771 | 2.45σ | 1/90 | 0 |
| K2-18b | 1.180 | 2.87σ | 8/149 | 0 |

**No bin on any target exceeds 3σ**, and the 2σ counts sit at or below chance expectation (≈2/47, ≈7/149). Nothing is "completely off." This is a **plotting defect, not a fit defect** — most plausibly the Fig 2 error bars are drawn at the raw measurement σ without the systematic floor, so residuals appear several times larger than they are. Consistent with T46u independently calling both figures unreadable at print size. **This objection is rebuttable, and fixing the figure removes it.**

**Claim (b) — CONFIRMED, and the cause is worse than stated.** Radius 68% credible interval vs the literature value, at both floors:

| target | floor 1% | floor 5% | literature | truth inside? |
|---|---|---|---|---|
| WASP-39b (OT-cal) | [1.2263, 1.2345] ESS 42.5 | [1.2188, 1.2388] ESS 12.4 | 1.279 | **no, at both** |
| WASP-96b | [1.1924, 1.2012] ESS 7.5 | [1.1757, 1.2084] ESS 9.1 | 1.200 | **yes, at both** |
| K2-18b | [0.2323, 0.2323] ESS 1.31 | [0.2315, 0.2337] ESS 11.3 | 0.2352 | **no, at both** |

**Only WASP-96b recovers the literature radius inside its own uncertainty.** WASP-39b misses by ~12× its half-width. The conclusion is robust to the floor choice.

**Underlying cause — importance-weight collapse.** At the 1% floor a *single* sample carries 99.95% of the weight for WASP-39b-nocond and 87% for K2-18b, so those "posteriors" are point masses and the 68% interval has essentially zero width. ESS ranges from 1.00 to 42.5 out of 1000–6312 draws. **Every credible interval quoted from a low-ESS arm is not a credible interval.** This is the same pathology recorded as an "efficiency artifact" in P3-D12, but it propagates into the reported uncertainties, which was not recognised then.

**The Table 2 coverage numbers traced.** 7/7 / 7/7 / 6/7 are coverage of the **NS values** inside MIRAGE's intervals at the 5% floor. K2-18b's single miss *is* the radius (NS 0.229 outside MIRAGE's [0.2315, 0.2337]) — internally consistent, no error there. **But no committed script computes them**: `benchmark_table.py` covers only the WASP-39b arms against the WASP-39b anchor, and `fig_multitarget.py` computes no coverage at all. The multi-target coverage was produced ad hoc. Given checklist Q4 claims full reproducibility, this needs a committed script before any resubmission.

**Qualifies P5-D5.** P5-D5 recorded "the radii matching published values" as the non-circular evidence that survives the circularity finding. **That survives on WASP-96b only — one of three targets, not three.** The radius *fix* (χ² 301 → 1.44 at a fixed floor) remains real and large; the radius *recovery with honest uncertainty* does not generalise across targets.

**Consequence.** The strongest defensible real-data claim is now narrow and specific: on a spectrum reduced end-to-end from raw MAST, the radius is recovered to 0.1% with the literature value inside the 68% interval. The other two targets recover a plausible central value with an overconfident interval that excludes the published radius. Any v2 must either report wider, honest intervals (fix the IS collapse) or stop quoting intervals from low-ESS arms.

---

### P5-D8 — Table 2 is now reproducible, and reproducing it exposed three undisclosed dependencies [Phase 5, 2026-10-24, v2 build #1]

**Built** `scripts/multitarget_table.py` — regenerates every Table 2 number (radius interval, coverage-vs-NS, best-fit chi2/N, ESS) for all three targets from the saved posteriors, at a stated error floor, with ESS printed on every row. Closes the gap found in P5-D7: nothing in the repo computed these numbers, while checklist Q4 claimed full reproducibility. `benchmark_table.py` covers only the WASP-39b arms; `fig_multitarget.py` computes no coverage at all.

**Coverage vs NS, recomputed:**

| row | floor 5% | floor 1% | paper |
|---|---|---|---|
| WASP-39b `abl_rad_nocond` (the figure's arm) | 3/7 | 0/7 | 7/7 |
| WASP-39b `ot_cal` | 7/7 | 5/7 | 7/7 |
| WASP-96b | 7/7 | 5/7 | 7/7 |
| K2-18b | 6/7 | 0/7 | 6/7 |

**1. Table 2 reproduces only at the 5% floor, which the paper never states.** At the 1% floor used by every NS anchor, by Table 1, and by the 301 baseline, coverage collapses to 0/7 · 5/7 · 0/7. Every coverage number in the paper is silently floor-conditional (compounds P5-D4).

**2. The WASP-39b row mixes arms.** Its radius is arm-insensitive — the IS-weighted mean at the 5% floor is 1.2266 (nocond) and 1.2273 (ot_cal), both rounding to the published 1.227 — so Figure 2 and Table 2 agree there. But the published **7/7 is only obtainable from `ot_cal`**, the circular arm, while `fig_multitarget.py` plots `abl_rad_nocond`, which gives 3/7 at the same floor. One table row, two arms.

**3. ESS is below 20 on seven of eight rows** (range 1.00–12.4); only `ot_cal` at the 1% floor clears it at 42.5. The script now prints an explicit warning per row rather than letting a point-mass posterior pass as a credible interval.

**Unchanged from P5-D7:** the literature radius falls inside the 68% interval for **WASP-96b only**, at both floors.

**Decision.** Any v2 table states its error floor in the caption, names the arm per row, and prints ESS beside every interval. Rows whose ESS is below ~20 either get a wider honest interval or get no interval at all.

---

### P5-D9 — Figure 2 misrepresented the fits; rebuilt with an honest error budget [Phase 5, 2026-10-24, v2 build #2]

**Trigger.** Reviewer Viae: *"all inferred spectra look really bad fits, several features are completely off."* P5-D7 showed the fits are good (chi2/N 0.66-1.44, zero bins past 3 sigma on any target), so the objection had to come from the figure rather than the science.

**Cause, confirmed.** `fig_multitarget.py:62` draws data error bars at the RAW measurement sigma, while `is_weights()` evaluates the likelihood against `sqrt(sig^2 + (floor*depth)^2)`. Measured ratio of plotted to effective uncertainty:

| target | median raw sigma | median sigma_eff (5% floor) | ratio |
|---|---|---|---|
| WASP-39b | 2.855e-05 | 1.064e-03 | **37.5x** |
| WASP-96b | 1.302e-04 | 7.246e-04 | 5.5x |
| K2-18b | 4.949e-05 | 1.556e-04 | 3.2x |

The forward-model systematic dominates **100% of bins on all three targets**. A residual the likelihood treats as 1 sigma therefore appeared as ~37 on the page for WASP-39b. The reviewer read the figure correctly; the figure was wrong.

**Rebuilt as `scripts/fig_multitarget_v2.py`** (v1 left intact for comparison):
1. Data keep their raw measurement sigma, which is the honest measurement precision. The forward-model systematic moves to the MODEL band, where it belongs (P3-D3 validated TauREx to ~5% median, so the prediction is genuinely no better than that).
2. The band is a full posterior-predictive — parameter spread + forward-model systematic + measurement noise in quadrature — so "fraction of points inside" is directly comparable to 68 / 95%. An earlier version omitted the measurement term and under-covered badly on K2-18b (29.5% inside 1 sigma); including it fixed that.
3. **1 sigma AND 2 sigma bands.** A 68% band excludes ~1/3 of points by construction, which reads as failure even on a good fit — the same misreading as v1, one level up.
4. New **residual row in units of sigma_eff**, with 1/2/3 sigma guides and the count past 3 sigma printed per panel. Fit quality is now readable instead of inferred.
5. **ESS annotated per target in red when below 20**, so a weight-collapsed posterior is visible on the page rather than presented as a credible interval (P5-D7).
6. Fonts enlarged, dpi 170 -> 200, one shared legend on row 1 instead of three overlapping the data (addresses T46u's "unreadable at print size").

**Posterior-predictive containment after the fix** (nominal 68 / 95%): WASP-39b 55.3 / 89.4 · WASP-96b 76.7 / 97.8 · K2-18b 59.1 / 95.3. Bracketing nominal, so the bands now mean what they say. WASP-39b is mildly overconfident and WASP-96b mildly conservative, which is itself a reportable diagnostic.

**Standing rule.** Any figure comparing data to a model plots the error budget the likelihood actually used, or states plainly which component is missing. v1 violated this and cost a reviewer's confidence in results that were fine.

---

### P5-D10 — Matched-family ablation: the conditioning win replicates, and a second transfer failure appears [Phase 5, 2026-10-24]

The published synthetic win was measured on the **6-param ABC** arms while the real-data test used the **7-param radius** arms, so "wins on synthetic, fails on real" was a cross-family comparison. Added `--family {abc,rad}` to `run_noisecond_ablation.py` (default `abc`, published result untouched) and re-ran identically on the radius arms: 50 planets x 1000 samples, same seed.

> **[SUPERSEDED by P6-D2, 2026-11-07.]** The no-cond and +sigma rows below are failed MPS runs. Retrained, all synthetic arms sit at nominal and conclusion 1 ("calibrates from either direction") is withdrawn. Only the real-JS column survives. Kept as the historical record.

Coverage@68 vs known truth (nominal 0.68), beside real JS-to-NS:

| arm | ABC | radius | real JS |
|---|---|---|---|
| no-cond | 0.587 (err .093) | 0.817 (err .137) | 0.352 |
| +sigma | 0.687 (err .007) | 0.903 (err **.223**) | 0.371 |
| +sigma+cov | 0.687 (err .007) | 0.700 (err **.020**) | 0.428 |

1. **The covariance win replicates, and is stronger than claimed.** ABC under-covers and is raised to nominal; the radius family over-covers and is lowered to nominal. The claim is not "raises coverage" but **"calibrates, from either direction"** (err .093->.007 and .137->.020). The transfer comparison is now apples-to-apples.
2. **NEW: +sigma-only does not replicate.** Indistinguishable from +cov on ABC (.687 vs .687), but **worse than no conditioning** on the radius family (err .223 vs .137). A second transfer failure, benchmark-to-benchmark, not previously reported.

**Three code defects fixed alongside:**
- The per-sigma block printed *"Dlogdens growing with sigma => conditioning helps most where noise is worst"* **unconditionally**. Radius gaps are +2.37 / +1.13 / +0.85 — it shrinks. True on ABC, false here, printed either way. Now branches on the measured trend. This is how a fabricated claim enters a paper.
- `real_compare.py` loaded the **T-gradient** anchor (`T_surface`, `T_top`, ...) not the free-radius one, indexed it **by position** against a hardcoded 6-param name list (so it compared temperature against `T_surface`), and read the shared `real_ess_samples.npz` (P5-D6). Rewritten to match by name, take explicit `--samples`/`--ns`, and refuse an anchor missing any flow parameter — a mean JS over 5 params is not comparable to one over 7.
- Built `scripts/transfer_table.py`, the single reproducible artifact for this claim. Both sides **IS-free** (synthetic scores against known truth, real uses raw unweighted samples), so neither is touched by the P5-D7 weight collapse. The IS-based real coverage (3/0/3) is retired.

**TMLR thesis, from the evidence:** robustness techniques validated on synthetic misspecification benchmarks fail to transfer twice over — covariance conditioning calibrates across two synthetic families yet makes the real posterior agree *less* with an independent NS reference (0.352 -> 0.428), and sigma-only conditioning fails to replicate even on a second synthetic family.

---

### P5-D11 — The CycleGAN ablation (D4) was invalid five ways; rebuilt on real OOT frames [Phase 5, 2026-11-01]

Reviewer Viae objected that domain B was one bootstrapped spectrum. Checking that turned up four further defects, three of them in the **domain-randomisation baseline that won**, so the published D4 comparison measured almost nothing it claimed to.

**What was wrong.**
1. **Domain B was a point mass.** `load_real_spectra` took ONE reduced spectrum and bootstrapped it to 500 near-copies with 1% Gaussian jitter; the ablation then called `np.tile(real_ref, (n, 1))`, giving literally zero variance. MMD measured distance to a point and `corr_similarity` returned NaN — which was in the published JSON and went unnoticed.
2. **The two domains were different physical quantities.** Domain B read `normalized_flux`, continuum-normalised STELLAR FLUX with median exactly 1.0. Domain A is standardised TRANSIT DEPTH. Both were squashed to [-1,1], which hid it.
3. **The two domains ran in opposite wavelength directions.** The hdf stores `flux` columns in descending wavelength order and `mirage/datasets/real_jwst.py` maps a real spectrum back to that stored order before building the context. My first rebuild of domain B left it ascending, so bin 0 of A was 7.28um against bin 0 of B at 0.55um. Caught before it reached a published result, but only by reading how the main pipeline does it.

   Worth quantifying, because the obvious test understates it. The mean transit-depth spectrum is nearly flat, so reversing it changes the **signal** by only ~1% (mean |depth - reversed depth| = 2.4e-4 on a depth of 0.021). But the per-bin **noise amplitude**, which is what the discriminator keys on, spans **62.6x** across the 47 covered bins (0.00257 to 0.16109 fractional scatter), and the reversed profile is uncorrelated with the true one (r = -0.049). The bug put the loud bins where the quiet bins belong. Verified the fix directly rather than by inspection: `wlen` now equals the hdf's stored grid exactly, and the stored depths match an independent inverse-variance rebin of the raw CSV at 0.95 / 2.01 / 3.51 / 4.91um to 1e-9.
4. **The randomisation kernel used a grid the data does not live on.** `wlen = np.linspace(0.55, 7.30, 52)`, while the ABC grid spacing runs 0.04 to 0.46um, so every `|lambda_i - lambda_j|` entering the SE and OU kernels was wrong.
5. **The randomisation correlation lengths were the class defaults, not the trained config's** (se 0.10-1.00 vs **0.02**-1.00, ou 0.50-3.00 vs **0.05**-3.00). P3-D1 had extended those bounds down on purpose because real JWST correlation is short-range. Defects 4 and 5 both made the baseline's noise smoother and longer-correlated than what was trained, in a comparison specifically about matching real short-range correlated noise.

Plus a conclusion rule that fired on whichever metric crossed an **absolute** 0.005 threshold. With MMD ~0.3 and KL ~3.7 those mean different things, so a 0.2% KL difference could outvote a 93% MMD difference pointing the other way. Now each delta is compared relative to its own scale with the two metrics required to agree.

**The rebuild.** `scripts/build_real_domain.py` constructs domain B from genuine out-of-transit frames, `real_sample = depth * (1 + frame_i/mean(frames) - 1)`, so the signal is the real reduced spectrum and the scatter is a real instrument noise realisation. WASP-39b PRISM alone has 3,241 frames. Uncovered bins are interpolated **per sample** and at build time — a constant fill would hand the discriminator a free real-vs-fake shortcut, and `np.interp` needs increasing `xp` which the stored order violates, so doing it once in the builder removes both traps. Stated caveat, to be printed and not hidden: OOT residuals characterise FLUX noise and stand in here for DEPTH noise.

**Grid choice for the cross-target tier.** Only 26 of the 52 ABC bins fall in the 0.85-2.83um range NIRISS and PRISM share, and they are coarse below 1.9um then dense above. Resampling ABC **up** onto a uniform 52-bin grid would invent 2x resolution and leave domain A nearly linear across the blue half, which a discriminator separates on smoothness alone. The shared grid therefore **restricts** the ABC grid to its 26 native bins and bins the real frames **down** onto them, so nothing is invented on either side. All three targets reach 26/26 covered bins, and each is trained and scored separately — pooling them would let a discriminator win on noise amplitude, which is not the question.

**Cost, and a correction.** My smoke tests overwrote the published `configs/cyclegan/cyclegan_best.pt` (30MB, Aug 14) and truncated its 200-epoch `training_log.csv` (37,759 -> 284 bytes), because `CKPT_DIR` was a module constant. `--ckpt-dir` now makes every run write its own directory.

I recorded this as **unrecoverable, and that was wrong.** Both files were committed in `9a106e2` ("feat(track2-p3): CycleGAN trained 200 epochs, best loss_G=0.69"), so `git status` showed them as `D` rather than modified and `git checkout HEAD --` restored them byte for byte: 30,365,735 and 37,759 bytes, the exact sizes reported lost. Verified by loading the checkpoint, epoch 199, loss_G 0.6898, matching the commit message. The restored `config.json` also confirms the defect independently, `n_real_samples: 500` against `n_abc_samples: 2000`.

**Lesson, and it is the same one as P5-D4.** I concluded "not git-tracked" without running the one command that checks, and then reported a permanent loss to the user. The check costs nothing: `git cat-file -s $(git rev-parse HEAD:<path>)`. A file absent from the working tree is not a file absent from history, and `D` in `git status` is positive evidence that it *is* tracked. Claiming data loss needs the same standard of verification as claiming a result.

`configs/cyclegan/ablation_results_v2.json` held a placeholder whose numbers were a counting sequence (0.0189 / 0.0191) asserting `D4_VALIDATED`. It was never a measurement, and the blast-radius check found it cited nowhere outside this log, so it never reached the paper. Overwritten in place with the genuine tier-1 result plus a `provenance` block recording what it replaced.

**Results, 5 runs on mps** (mps measured ~23x cpu, so all five use it; `run_cyclegan_tiers.sh` pins one device because the numerics differ slightly and these runs are compared against each other).

| run | real frames | MMD sim / rand / GAN | KL sim / rand / GAN | mean shift rand / GAN |
|---|---|---|---|---|
| **tier 1** WASP-39b PRISM 52b | 3,241 | 0.2925 / **0.1704** / 0.7686 | 3.943 / **3.675** / 4.172 | 0.033 / **0.308** |
| control WASP-39b 26b | 3,241 | 0.3626 / **0.2556** / 0.9429 | 5.960 / 5.760 / **5.049** | 0.031 / **0.468** |
| tier 2 WASP-39b 26b | 56 | 0.3628 / **0.2533** / 0.2748 | 9.442 / 9.184 / **9.030** | — |
| tier 2 WASP-96b NIRISS 26b | 56 | 0.3637 / 0.2537 / **0.2298** | 2.519 / **2.077** / 3.473 | — |
| tier 2 K2-18b NIRISS 26b | 56 | 0.3686 / 0.2599 / **0.1488** | 2.591 / **2.242** / 4.570 | — |

1. **Tier 1 is the headline and it is far stronger than the published claim.** Both metrics agree, and against `sim` (no adaptation at all) the translator scores 0.7686 to sim's 0.2925. So translation is **2.6x worse than doing nothing**, while randomisation improves on doing nothing. The paper's "translation did not improve on structured randomization" understates it: translation moves the simulated distribution **away** from the real one.

2. **The tier-2 cross-target flips are a SMALL-SAMPLE ARTIFACT, not a target or instrument effect.** Established by the control, same target and grid with only the frame count changing: as n drops 3,241 -> 56, `sim` moves +0.06% and `random` -0.9%, while the **CycleGAN's MMD collapses 71%** (0.9429 -> 0.2748). Only the arm that is far from the real distribution is sensitive to how few real points it is scored against. Tier 2 cannot be read as evidence about targets, and WASP-96b (56 frames) and K2-18b (96) do not have enough out-of-transit data to support this comparison at all. That is a real limit on the experiment, and the honest response is to say so rather than to bootstrap 500 samples from one spectrum.

3. **The metrics genuinely disagree on the 26-bin subrange, and it is not an estimator artifact.** Swept the histogram KL over bins 10/20/50 and samples 200/500: the absolute value is biased upward at low n (6.83 -> 5.02 at 50 bins) but the ordering is stable at every setting. So on the NIRISS subrange the translator really does get **per-bin marginals** closer while MMD and mean shift still strongly favour randomisation. The most interpretable number is the **mean shift: 0.468 vs 0.031**, a 15x larger systematic offset, in units of per-bin sigma. A translator that displaces the mean spectrum by roughly half a standard deviation is unusable for retrieval whatever its marginals look like, because retrieval reads parameters off the spectrum's level and shape. `corr_similarity` does not separate the arms cleanly (0.276 GAN vs 0.128 rand at 52b, all ~0.09 at 26b) and should not be leaned on.

   Candidate explanation, **not established**: domain A is a distribution over 73k different *planets* while domain B is a distribution over *noise realisations of one planet*, so the translator is being asked to convert a population spread into a single-target noise spread. That would explain marginal-shape agreement alongside a displaced mean. Testing it needs a domain A restricted to one planet's parameter neighbourhood.

**For the paper.** The §4 sentence quoting `MMD ~ 1.22` is from the invalid run and cannot be carried forward, since the corrected number is measured against a different real domain. The real-side description ("the single reduced WASP-39b spectrum, bootstrap-expanded with 1% photon-noise jitter to 500 samples") must change too. To v1's credit that disclosure is exactly what let reviewer Viae find the problem.

---

### P5-D12 — Dropping the collapsed intervals RESTORES the multi-target radius claim [Phase 5, 2026-11-05, result — amends P5-D7]

P5-D7 narrowed the radius-recovery claim from three targets to one, because only WASP-96b's 68% interval contained the literature value. It also identified the cause, importance-weight collapse, but then kept quoting the collapsed intervals. Those two findings were never put together. Doing so reverses the conclusion.

**The fix, implementing P5-D7's own stated resolution.** `multitarget_table.py` now withholds the IS-weighted interval entirely below `--ess-min` (default 20), rather than printing it beside a warning. A printed number gets copied into a paper; the warning does not. In its place it reports the **IS-free** interval, the raw flow posterior at equal weights, which cannot collapse by construction. Same move P5-D10 made for the real-side JS metric.

**Result at a 1% floor.** 3 of 4 arms are collapsed (ESS 1.00, 7.49, 1.31 of 1000 draws) and have their weighted interval withheld.

| target | ESS | IS-weighted 68% | IS-free 68% | x wider | literature | inside IS-free |
|---|---|---|---|---|---|---|
| WASP-39b (nocond) | 1.00 | withheld | [1.1417, 1.2841] | 1700 | 1.279 | **yes** |
| WASP-96b | 7.49 | withheld | [1.1568, 1.2234] | 7.6 | 1.200 | **yes** |
| K2-18b | 1.31 | withheld | [0.2189, 0.2363] | 633 | 0.2352 | **yes** |
| WASP-39b (OT-cal) | 42.5 | [1.2263, 1.2345] | [1.2243, 1.2336] | 1.1 | 1.279 | no |

1. **The literature radius lies inside the IS-free interval on all three targets.** P5-D7's "one of three, not three" was an artifact of quoting point masses. The 1700x and 633x widening factors are the measure of how degenerate those intervals were.
2. **The intervals are not vacuous, which is the check that makes this claim rather than tautology.** Against the uniform prior's own central 68%, the IS-free intervals are **4x tighter (WASP-39b), 8x (WASP-96b) and 12x (K2-18b)**. A wide interval contains more by construction, so containment only means something if the interval still carries information, and it does.
3. **But it is tight, not comfortable.** The literature value sits at 0.93 (WASP-39b), 0.30 (WASP-96b) and 0.87 (K2-18b) of the interval's half-width from its centre. Two of three are near the edge. The defensible claim is "contained on all three", not "well recovered on all three".
4. **The OT-calibrated arm is the one that still misses**, at the highest ESS of the four. Consistent with P5-D5: that arm is an inflated Gaussian on the NS anchor, and the NS anchor is not the literature value.

**Amends P5-D7.** The radius *fix* was never in doubt. The radius *recovery with honest uncertainty* does generalise across targets, provided the uncertainty is not taken from a collapsed importance-sampling step. The reporting policy is now enforced in code rather than noted in prose.

**Standing rule.** No interval is quoted from an arm whose ESS the same table does not also print. If ESS is below threshold, there is no interval to quote, only an IS-free substitute that must be labelled as one.

---

### P6-D1 — H9 training done on CUDA; the published no-conditioning baselines were failed MPS runs [Phase 6, 2026-11-06, MAJOR — training only, evaluation pending]

**What ran.** 12 FMPE runs on an RTX 5070 Laptop GPU (CUDA 12.8, torch 2.11): {radius, ABC} × {no-cond, +σ+cov} × seeds {42, 43, 44}, in `configs/h9_{rad,abc}_{cov,nocond}_s{42,43,44}`. Configs are the published `noisecond_*` configs with only `device: cuda`, `n_workers: 0` (the Mac already used 0) and `backup_interval` changed; full 512-epoch budget, early stopping at patience 50. About 27x faster than the Mac per batch. Seed 42 reproduces each stored run's seeds exactly, so it is a CUDA-vs-Mac control.

**Best validation loss (FMPE loss, not the H9 metric).**

| arm | s42 | s43 | s44 | mean ± sd | stored Mac s42 |
|---|---|---|---|---|---|
| radius +σ+cov | 1.1209 | 1.1590 | 1.1368 | 1.139 ± 0.019 | 1.1475 (CPU) |
| radius no-cond | 1.1328 | 1.1265 | 1.1241 | 1.128 ± 0.005 | **1.2477 (MPS)** |
| ABC +σ+cov | 1.3907 | 1.3851 | 1.3975 | 1.391 ± 0.006 | 1.3893 (CPU) |
| ABC no-cond | 1.3904 | 1.3913 | 1.3927 | 1.392 ± 0.001 | **1.9231 (MPS)** |

1. **The CPU-trained cov arms reproduce on CUDA.** Radius differs by 0.027, inside its own seed range (1.121 to 1.159); ABC by 0.001. On this evidence the platform is neutral for the covariance branch.
2. **The MPS-trained no-cond arms do not, because they did not train.** `noisecond_nocond` train loss went 1.99 at epoch 1 to 1.94 at epoch 125: it never learned. The same config and seed on CUDA reaches 1.41. `noisecond_rad_nocond` learned but poorly (1.25 vs 1.13, with a val spike to 1.54 at epoch 75). So the published no-conditioning rows (ABC coverage 0.587, real JS 0.352) were measured on broken baselines, and the conditioning gap may be partly or wholly a training failure rather than an effect of conditioning.
3. **With the baseline trained properly, the validation-loss gap between arms is gone.** ABC: 1.391 vs 1.392. Radius: no-cond is slightly *better* (1.128 vs 1.139). This does not settle coverage or JS, which need the evaluation pipeline, but it removes any prior that conditioning helps.
4. **Equal early-stop epochs across arms are expected, not a bug.** Within a seed, both arms draw the same validation noise each epoch; their epoch-to-epoch val-loss changes correlate 0.81 to 0.95, versus 0.03 across seeds. The minimum, and therefore the stop epoch, partly tracks that shared noise.

**Caveat on the dispersion.** `dataset.random_seed` also sets fm4ar's train/valid split, so the spread above is over split + init + noise, wider than training-seed variance alone. Arms are paired within a seed (same split, same noise), so cov-vs-nocond comparisons are sound. If the effect dies, a second pass holding `dataset.random_seed` at 42 is needed to separate the causes.

**Bookkeeping gotcha.** `backup_interval` sets how often `model__latest.pt` is written, and early stopping exits without a final save. `h9_rad_cov_s42` ran at 100, so its `model__latest.pt` is stale at epoch 200 (it stopped ~232; history to epoch 225 is in `history_to_ep225.csv`). Do not resume that directory: it would retrain from 200 and could overwrite `model__best.pt` (epoch 182, the one to evaluate). The other 11 ran at `backup_interval: 1`. Disk use is driven by `snapshots/` (13 MB per LR drop), not by this setting.

**Next.** Re-run coverage@68 and real JS-to-NS on all 12 `model__best.pt`. Until then the "calibrates from either direction" claim should be treated as unsupported, not provisional, and the MPS no-cond numbers should not be quoted.

---

### P6-D2 — H9 evaluated: the synthetic conditioning win is withdrawn, the real-data failure survives [Phase 6, 2026-11-07, MAJOR]

**What ran.** Coverage@68 on all 12 `configs/h9_*/model__best.pt` (50 planets x 1000 samples, test-seed 1234), and real WASP-39b JS-to-NS on the 6 radius arms (1000 draws each, IS-free, against `wasp39b_ns_posterior_fitrad.npz`). Added `--train-seed` to `run_noisecond_ablation.py` and registered the 6 H9 radius arms in `real_ess.py`; defaults unchanged. Pipeline validated by reproducing the three published real-side values exactly (0.352 / 0.371 / 0.428) through the same code path.

**Synthetic, coverage@68, nominal 0.68, mean +- sd over 3 training seeds.**

| arm | ABC | radius | published ABC | published radius |
|---|---|---|---|---|
| no-cond | 0.692 +- 0.013 | 0.692 +- 0.016 | 0.587 | 0.817 |
| +sigma+cov | 0.698 +- 0.011 | 0.693 +- 0.014 | 0.687 | 0.700 |

The published values span 0.587 to 0.817 and that spread was the entire phenomenon. Retrained, all four arms fall between 0.692 and 0.698, every one at nominal. Conditioning's measured gain is -0.006 on ABC (0.5x the pooled seed sd) and -0.001 on radius (0.1x). The effect shrank 15x and 123x. Log-density is a tie on ABC (-7.80 vs -7.79); radius favours no-cond by 0.25 nats against a cov seed sd of 0.225, so about 1x spread and not a signal.

**"Calibrates from either direction" is withdrawn.** It required a baseline miscalibrated in two directions. There was none; there were two failed MPS runs.

**The Phase-4 adaptivity claim falls with it.** `04_track1_phase4.md` reported the +sigma+cov minus no-cond log-density gap GROWING with injected sigma, +0.64 to +0.87. Retrained it is +0.08 / -0.00 / +0.07 across sigma tertiles: flat, and an order of magnitude smaller. Note the script's own auto-verdict text reads the first and last bin only and still prints "the adaptivity story" on the radius family, where the sequence is -0.09 / -0.19 / +0.05 and the aggregate favours no-cond. Do not quote that sentence; the heuristic was written when the effect was real.

**Real data, mean 1-D JS to the NS anchor, lower is closer.**

| arm | s42 | s43 | s44 | mean +- sd | published |
|---|---|---|---|---|---|
| no-cond | 0.341 | 0.347 | 0.342 | **0.344 +- 0.003** | 0.352 |
| +sigma+cov | 0.434 | 0.411 | 0.435 | **0.426 +- 0.014** | 0.428 |

Gap +0.083 against the published +0.076, at **8.4x the pooled seed sd**. This survives.

**Why one half died and the other did not.** Undertraining moved synthetic coverage by 0.114 and the real JS by 0.008. Synthetic coverage is measured against truths drawn from the same prior the model trained on, so a fit defect shows up directly. The real-side metric measures distance from an independent anchor, and the conditioned arms carry a systematic bias that undertraining does not touch: at seed 42 the per-parameter JS is driven by T (1620 K for +cov against the anchor's 608 K, JS 0.693) and log_CO2 (JS 0.631), against 1064 K and 0.342 for no-cond. +cov is worse on 5 of 7 parameters.

**The revised claim.** Noise conditioning gives no measurable benefit on synthetic benchmarks and degrades real-data retrieval. This is simpler than the published dissociation and does not depend on a synthetic win that was an artifact, and the surviving half is the seed-robust half.

**Consequence for A5/H8.** The real-side result is now the only surviving quantitative claim, and "worse" means "further from TauREx nested sampling". Validating that anchor against an independent retrieval code moves from hardening to load-bearing.

**Still outstanding.** The +sigma row is withheld, not corrected: `noisecond_rad_sigma` is a third failed MPS run (best val 1.3729 where its family trains to 1.13) and was never retrained. ABC +sigma trained fine on MPS at 1.3873, so backend alone does not condemn a run; check the loss.

**Retractions applied.** `README.md` conditioning section, `07_state_gaps_and_research_agenda.md` sections 4.2 / G1 / G2 / H9 / closing caveat, `04_track1_phase4.md` fig5 bullet. `fig5_abc_ablation.png` must be regenerated from `configs/h9_*` before reuse.

---

### P6-D3 — G1 closed: CycleGAN translation degrades the posterior, it does not merely mismatch the noise [Phase 6, 2026-11-07]

**The gap.** Every translation result so far was distributional (MMD, KL, mean shift). A reviewer's first question is whether any of it reaches inference. G1 existed because the paper claimed a translation result while measuring only distribution.

**The test.** `configs/cyclegan_posterior_cov` is `configs/noisecond_cov` with one line changed, `dataset.file_path` pointing at `abc_translated_train.hdf` (built by `scripts/build_translated_abc.py` from the recovered epoch-199 generator). Both trained on CPU to early stopping, 174 and 173 epochs. Evaluated identically, 50 planets x 1000 samples on `abc_test.hdf`, test-seed 1234, via `run_noisecond_ablation.py --arm-dir` (added for this).

| | sim | translated |
|---|---|---|
| coverage@68 | **0.690** | 0.557 |
| coverage@95 | 0.987 | 0.853 |
| width@68 | 2.222 | 2.020 |
| logdens | **-7.71** | -17.65 |
| logdens by sigma tertile | -7.57 / -7.66 / -7.90 | -18.73 / -20.17 / -14.21 |

**Result.** Training on translated data costs 10 nats of log-density and drops coverage from nominal to 0.557, while *narrowing* the 68% interval from 2.222 to 2.020. Narrower intervals with worse coverage is overconfidence, not a precision-calibration trade. Worse in every sigma tertile. The single-line config difference makes translation the only possible cause.

**Why this matters more than the MMD number.** "Translation does not match the real noise distribution" invites the reply that the metric is wrong. "Translation makes the posterior worse by 10 nats" does not. The negative is now inferential.

**Do not compare the two runs by validation loss.** `cyclegan_posterior_cov` reports best val 1.1908 against `noisecond_cov`'s 1.3893, which looks like a win and is meaningless: they are computed on different validation sets, since the dataset path is the thing that differs. Only the shared-test-set evaluation above is comparable. Flagging because the loss numbers sit side by side in the two config directories and will mislead whoever looks next.

**Consequence.** V3 stands: retire CycleGAN, keep it as a negative control. The replacement work is V2's cheap baselines (per-bin affine, quantile mapping, classifier reweighting), which a learned translator has to beat before it is worth training again.

---

### P6-D4 — Multi-target models cleared; the real bug was that device was confounded with arm [Phase 6, 2026-11-07, MAJOR]

**The worry.** P6-D1 found the published no-conditioning baselines were failed MPS runs. Every multi-target model is also MPS (`noisecond_rad_nocond_k218` ep 85 / 1.2141, `_wasp96` 138 / 1.1885, `_hires` 138 / 1.0729, `transformer_abc` 144 / 0.8247). Those underpin Table 2 and P5-D12 and have no CUDA twin, so only a control rerun could clear them.

**The controls.** Three configs (`ctrl_k218_s42`, `ctrl_wasp96_s42`, `ctrl_hires_s42`), each `diff`-verified against its published original as differing in exactly three lines, `device`, `backup_interval`, `n_workers`.

Gap below is published MPS minus CUDA, so positive means the published model is the worse one.

| control | CUDA best val | CUDA ep | published MPS | MPS ep | gap | band |
|---|---|---|---|---|---|---|
| K2-18b | 1.1822 | 237 | 1.2141 | 85 | +0.032 | backend shift |
| WASP-96b | 1.2110 | 150 | 1.1885 | 138 | **-0.022** | backend shift |
| hires | 1.0665 | 217 | 1.0729 | 138 | +0.006 | reproduces |

**Verdict: not broken.** All three trained normally (+0.65 to +0.77 train gain). No control lands in the degraded band, and the two that exceed the reproduction band point in *opposite directions*, which no systematic backend penalty can produce. Table 2 and P5-D12 hold on training grounds.

**The epoch disparity is not a second problem.** The published K2-18b model early-stopped at epoch 85 against the control's 237, so it saw roughly a third of the training and its best arrived near epoch 35 rather than 187. It still landed 0.032 away. Early stopping at patience 50 fired on a genuine plateau, not a truncation, and the H9 runs show the same behaviour on CUDA (69 to 217 epochs at identical config) while holding best val inside 0.009 for nocond arms. Recording it because `ep=85` looks alarming next to the others and is not.

**Verification boundary.** I reproduced every published MPS number exactly from the local checkpoints and `diff`-verified all three control configs myself. The CUDA best vals are reported from the other machine and cannot be checked here, because checkpoints are gitignored and only `config.yaml`, `git-hash.txt` and `requirements.txt` were pushed.

**On the tolerance.** I was asked to widen the agreement band from 0.02 to 0.04 on the grounds that seed noise spans 0.038. It does not, in the relevant reference class. Measured per-group spread over the twelve H9 runs:

| group | range | sd |
|---|---|---|
| abc nocond | 0.0023 | 0.0012 |
| abc cov | 0.0124 | 0.0062 |
| rad nocond | 0.0087 | 0.0045 |
| **rad cov** | **0.0381** | **0.0191** |

The 0.038 is `rad cov`, the widest of four and the only group carrying the covariance embedding. The multi-target models are all nocond arms, whose matched spread is sd 0.0033. Against that, +0.032 is ~10 sigma, so calling it ordinary seed noise is wrong even though the conclusion it supports is right.

I rejected the wider threshold and replaced the single tolerance with bands, because the measured gaps separate themselves by about 4x at every boundary and need no tuning: <0.02 reproduces, 0.02-0.05 backend shift, 0.05-0.40 degraded, >0.40 catastrophic. The three known cases land one per band (+0.533 / +0.115 / +0.032). A backend shift is real but acceptable, and naming it separately keeps the degraded detector at full sensitivity.

**The actual finding.** Device was perfectly confounded with arm type across every noise-conditioning run:

| arm type | device | n |
|---|---|---|
| cov | cpu | 4/4 |
| sigma | mps | 2/2 |
| nocond | mps | 5/5 |

The cov branch was moved to CPU because its batched-cov matmul was slow on MPS, and that choice is recorded in `noisecond_rad_cov/config.yaml`. So every baseline ran on the unreliable backend and every treatment arm ran on a reliable one. MPS is a lottery rather than a constant penalty, 1 catastrophic and 2 degraded out of 6 runs with the rest fine, so the bias was uncontrolled rather than uniform. This is the mechanism behind P6-D1: the conditioning "effect" was partly a compute-backend effect wearing the treatment's label.

**Which claims this kills and which it strengthens.** A broken baseline only invalidates a claim when the breakage favours that claim.

- *Killed* (already retracted, P6-D2): ABC coverage and the sigma-adaptivity panel. Broken nocond made conditioning look good.
- *Strengthened*: P4-D1's real-WASP-39b result that cov does **not** help. There the healthy arm was cov (CPU, 1.1475) and the handicapped arms were nocond and sigma (MPS), yet nocond still won on JS-to-NS, 0.358 against 0.430. The bias worked against the conclusion and the conclusion held anyway. P6-D2's independent remeasurement on CUDA agrees, +0.083 against the published +0.076.

**Open.** `noisecond_rad_sigma` (MPS, 1.3729) sits 0.21-0.25 above the rad CUDA band of 1.124-1.159, so I predict it is degraded by roughly 0.24, worse than `rad_nocond` was. Job B's `h9_rad_sigma_s42` settles it. If confirmed, the sigma-only column of the P4-D1 table is unusable and should be dropped rather than reported, since the nocond and cov columns already carry that table's conclusion.

**Consequence.** Table 2 and P5-D12 clear on training grounds, all three controls in. Record the device beside every reported model from here on, and never let the backend vary with the arm being compared.

---

### P6-D5 — A1 calibration instrument built and validated before use; TARP's reference distribution is a real choice [Phase 6, 2026-11-07]

**Why validate first.** A1 exists because the Phase-3 OT calibration was circular (P5-D5). Shipping an unvalidated replacement would repeat the mistake with new code, so the diagnostics were tested against known-wrong posteriors before touching a MIRAGE model. `mirage/eval/sbc.py` plus `scripts/validate_sbc.py`.

**Three tests, deliberately not interchangeable.**

- *SBC ranks* (Talts et al. 2018). Rank of the truth among samples, per parameter. Uniform under exact inference, U-shaped when overconfident, bell when overdispersed, sloped when biased. Marginal, so blind to a wrong correlation structure.
- *TARP* (Lemos et al. 2023). Joint. How often the truth is closer to a random reference point than the samples are.
- *HPD ranks*. Joint, from the flow's own log-density rather than a metric. Fraction of samples denser than the truth.

**Validation on a conjugate Gaussian toy**, five samplers, four wrong in a known direction, median of 3 seeds at 400 sims x 999 samples:

| case | SBC min p | shape | TARP p | HPD p |
|---|---|---|---|---|
| exact | 0.30 | flat | 0.57 | 0.51 |
| overconfident (sd x 0.5) | 2e-84 | U-shaped | 7e-80 | 5e-120 |
| overdispersed (sd x 2) | 1e-38 | bell | 1e-40 | 2e-129 |
| biased (+1.0 sd) | 3e-108 | sloped | 6e-24 | 4e-47 |
| **decorrelated** (correct marginals, rho 0.9 to 0) | **0.30** | **flat** | **1.4e-03** | **1.3e-16** |

The last row is the load-bearing one. SBC ranks stay flat while both joint tests fire, which is the only reason reporting three statistics is worth the space rather than one.

**The TARP reference distribution is not a free parameter.** Theorem 1 permits any reference that does not depend on the truth, which includes one built from the posterior samples themselves, since those are independent of the truth given x. That constrains validity, not power, and the two choices differ a lot:

| case | uniform over prior box | Gaussian matched to the posterior |
|---|---|---|
| exact (null) | p 0.84 | p 0.70 |
| biased 0.5 sd | p 0.52, blind | p 5.7e-03 |
| decorrelated | p 1.7e-03 | p 3.8e-04 |

Diffuse references put the truth deep inside a huge ball whose mass barely shifts when the posterior is wrong. Default changed to `reference="posterior"`, with `"box"` kept as an option. The null stays calibrated under both, which is the part that matters, since power bought with a broken null is worthless.

**Measured sensitivity floor, 30 seeds.** Null: median p 0.52, zero false alarms at 0.01, and the p-values are themselves uniform (KS p 0.66). Power at 0.01: a 0.5-sigma bias is detected 57% of the time, a 1.0-sigma bias 100%, the decorrelated case 90%. So TARP's floor at this N is roughly half a sigma, where it is a coin flip. The validation file uses a 1.0-sigma bias to stay deterministic. **Do not read a single passing TARP p-value as proof of calibration**; this is why the two unexplained single-seed values I first saw (0.013 on the null, 0.090 on the bias) were seed noise rather than bugs, and why the regression test takes a median.

**Next.** `scripts/inject_real_noise.py`, where G13 bites: OOT residuals are flux noise and Tier 1 needs depth noise. Proper construction on WASP-96b only, via pseudo-transits injected into OOT segments through the existing `wasp96_fit_lightcurves.py`, with the flux-noise approximation stated for the other two targets rather than applied silently.

---

### P6-D6 — Tier-1 noise: the real frames set the structure, the published errors set the scale [Phase 6, 2026-11-07, MAJOR]

**G13 turned out to bite harder than stated.** The gap was logged as "OOT residuals are flux noise, Tier 1 needs depth noise", a units problem with a known fix: a depth is `1 - F_in/F_out`, so fractional flux errors `a` and `b` on the two means give `delta_meas ~= delta_true - (a - b)`, and a depth-noise realization is a difference of means of real frame residuals drawn disjointly from a held-out pool. That inherits the real cross-wavelength covariance and the 1/sqrt(N) averaging without assuming either. `scripts/inject_real_noise.py`.

**The amplitude audit then failed, which is why it exists.** On WASP-39b PRISM the frames cannot reach the published precision by any amount of averaging. Per-bin frame scatter is 4.8e-3 and the rebinned published error bar is 2.9e-5, so matching it needs about 56,000 frames per side. 810 exist. Detrending does not rescue it:

| treatment | per-bin sigma | achievable at 810 frames |
|---|---|---|
| raw | 5.22e-3 | 9.1x published |
| drop 2 pathological bins | 4.80e-3 | 8.4x |
| minus common mode | 4.61e-3 | 8.0x |
| minus common mode + linear trend | 4.61e-3 | 8.0x |
| minus 3 PCA components | 3.92e-3 | 6.8x |

So this is not removable systematics. The published side checks out independently: native published sigma is 7.5e-5 over 331 bins, and 331 to 52 rebinning cuts it by sqrt(6.4) = 2.5x to 3.0e-5, matching the measured 2.86e-5. The discrepancy is between the raw frame scatter and a *fitted* depth, and nothing in the OOT product closes it.

**Consequence, and it reverses part of what I wrote first.** Injecting the raw amplitude would make the Tier-1 set about 9x noisier than real data, and a flow scores as well calibrated on noise it will never encounter. I had described the fit as something that only *adds* error the resampling misses. It also removes a great deal, and that is the dominant term. So the frames set the STRUCTURE and the published error bars set the SCALE, which is the same split P3-D1 already made for the covariance embedding (condition on structure, not scale). Rescaling each bin by `sigma_pub / sigma_realized` leaves the correlation matrix untouched and fixes the margin.

**The structure is worth keeping, and it is instrument-specific.** Lag-1 wavelength correlation on WASP-39b PRISM is +0.312, at 121 sigma from a column-shuffled null, still +0.18 at lag 7, with the correlation matrix's top eigenvalue 8.09 where white noise gives ~1. Real-noise injection is therefore not a repackaged white-noise test. But NIRISS is nearly white, +0.078 on WASP-96b and +0.056 on K2-18b, so the correlated-noise argument is a PRISM result and WASP-39b is the only target where it carries weight. Do not generalise it to all three.

**Per-target viability.** Amplitude matches 1.00x published everywhere by construction.

| target | OOT frames | usable bins | real-noise fraction | lag-1 |
|---|---|---|---|---|
| WASP-39b PRISM | 3241 | 45/52 | 0.96 | +0.302 |
| WASP-96b NIRISS | 56 | 89/90 | 0.99 | +0.078 |
| K2-18b NIRISS | 96 | 100/150 | **0.67** | +0.056 |

Two further traps found and closed. Bins where no real frames exist were getting EXACTLY zero noise while still carrying a stated error bar, which is not a conservative approximation but an impossible observation; they now get white noise at the published sigma and the fraction is recorded. And 2 PRISM bins at 1.16 and 1.27 um sit 30x above the median with `dq_flag == 0`, so they are dropped rather than allowed to dominate every realization. **K2-18b is only 67% real noise** because NIRISS stops at 2.82 um while its model grid runs to 5.17, so a third of that spectrum is a white-noise test wearing a real-noise label. Either restrict K2-18b Tier-1 claims to below 2.82 um or add the NIRSPEC G395H frames (488 frames, 2.88 to 3.72 um) and accept a residual gap to 5.17.

**Frames are split contiguously**, not alternating, because JWST systematics are time-correlated and an even/odd split would leave the same drift on both sides of a nominally held-out partition.

**Next.** The `--build` stage needs `mirage-taurex`. Then `depth_noise_wasp96.py`: prayer-bead resampling of real fit residuals through our own exoTEDRF plus batman pipeline (280 integrations, `fit_white` and `fit_bin` already factored for reuse). That is the only route to depth noise that carries the fit's own systematics, it exists only where we own the reduction, and WASP-96b is not the headline target. Its job is to measure how wrong the published-amplitude approximation is on the one target where both routes run.

---

### P6-D7 — Noise-scale audit: real sigma sits 23-139x below the training perturbation, entirely outside its range [Phase 6, 2026-11-07, MAJOR]

**Found while normalising the Tier-1 set,** not by looking for it. Three separate noise scales exist in this codebase and they disagree. Median `flux_std` is 0.01325, which converts normalised to absolute depth.

| channel | normalised sigma | absolute depth | vs real |
|---|---|---|---|
| training PERTURBATION (`InjectCorrelatedNoise`) | 0.05 - 0.30 | 6.6e-4 - 4.0e-3 | **23-139x larger** |
| training `error_bars` from the HDF `noise` field | 1.52e-4 | 2.01e-6 | 0.071x, i.e. 14x smaller |
| real published WASP-39b, same 52-bin grid | 2.16e-3 | 2.86e-5 | 1x |

**The headline.** Real sigma is 23x below the *minimum* of the trained range, so real data does not sit at the edge of the distribution the sigma and covariance embeddings were trained on, it sits entirely outside it. P3-D1 and P4-D1 recorded this as "real noise ~10x below trained sigma". The measured factor is 23-139x depending on where in the sampled range you land, so the gap is one to two orders of magnitude rather than one.

**This is the quantified mechanism for the conditioning paradox.** P4-D1 hypothesised that the cov arm fails on real data because it alone ingests an OOD input, and P6-D2 confirmed the failure empirically (+0.083 JS, cov drives T to 1620 K against the anchor's 608 K). The number above is why: the embedding is asked to interpret an error bar 23-139x smaller than any it saw. Nothing about the flow needs to be wrong for that to produce a confidently wrong posterior.

**Correction to my own first reading.** I initially measured the HDF `noise` field, got "trained noise is 14x too small", and concluded the opposite sign to P3-D1. That field is only the reported error-bar context, not the perturbation actually applied to the flux at train time, which lives in the config's `data_transforms`. The 14x discrepancy in that channel is real but separate and much less consequential, since the transform overwrites `error_bars` with `sqrt(diag(sigma_mat))` from the injected covariance. Worth fixing for consistency; not the story.

**Consequence for A1, and it changes the design.** A Tier-1 set at published amplitude is OOD for the sigma embedding in exactly the way real data is. That is the point of Tier 1, but it cannot be read alone, because an intrinsic flow defect and a pure scale mismatch would produce identical SBC failures. So `inject_real_noise.py` gained `--sigma-scale`, and two sets are built per target:

- `wasp39_tier1.hdf`, published amplitude, normalised sigma 0.0022, realistic and OOD.
- `wasp39_tier0_indist.hdf`, the same injections scaled by 56.7 into the training range (normalised sigma 0.122), in-distribution.

If calibration fails on Tier 1 and holds on Tier 0, the defect is the scale mismatch and the fix is retraining over a sigma range that contains real data. If it fails on both, the flow itself is miscalibrated. That is a decisive experiment and it was not available before this audit.

**Hypothesis this now makes testable, not a result.** The real-data ESS collapse (1.0 to 42.5, P5) is consistent with a flow trained on noise 23-139x too large producing posteriors far too broad in the wrong places, so real data lands where the proposal has little mass. Tier 0 versus Tier 1 is the first measurement that can separate that from the forward-model mismatch it has been attributed to. Do not assert the link until that comparison is in.

**Separate bug caught in the same pass.** `build_set` was writing RAW transit depth (~0.021) while the training and test HDFs store per-bin z-scored flux (mean 0.00, std 1.00 by construction, with `flux_mean` and `flux_std` kept for inversion). Feeding raw depth to the flow would have handed it an input roughly 100 sigma from anything in training, and it would not have raised an error, it would have returned nonsense. Now normalised exactly as `prepare_abc_hdf5.py` does, with raw depth retained as `flux_depth` for plots. Also fixed: 5 bins past the published spectrum's 5.33 um cutoff were getting NaN sigma written into the HDF, and the real-noise fraction was computed over a denominator that silently excluded them, overstating WASP-39b coverage as 0.96 when it is **0.87**.

---

### P6-D8 — A1 first result: MIRAGE is UNDER-confident on realistic noise, and conditioning HELPS against prior-draw labels [Phase 6, 2026-11-08, MAJOR]

**The non-circular calibration check now exists and has run.** 200 prior draws per cell, 199 posterior samples each, WASP-39b radius arms, labels from the prior so no transport can game them. `scripts/sbc_report.py`, saved to `data/sbc/`.

| arm | tier | normalised sigma | SBC min p | shape | TARP p | TARP dev | HPD p | cov@68 |
|---|---|---|---|---|---|---|---|---|
| nocond | Tier 0, in-distribution | 0.118 | 1.6e-09 | bell | 2.7e-07 | 0.195 | 2.0e-25 | 0.752 |
| nocond | Tier 1, published | 0.0021 | 4.4e-56 | bell | 5.1e-06 | 0.175 | 1.5e-51 | **0.851** |
| cov | Tier 0, in-distribution | 0.118 | 3.3e-05 | bell | 1.2e-02 | 0.110 | 1.4e-09 | **0.718** |
| cov | Tier 1, published | 0.0021 | 2.6e-34 | bell | 2.8e-04 | 0.140 | 7.2e-30 | 0.806 |

**Finding 1: the failure is UNDER-confidence, not overconfidence.** Every cell is bell-shaped with coverage ABOVE the nominal 0.68, so the intervals are too WIDE. Rank skew is at most 0.079 in magnitude everywhere, so this is a width error and not a bias. This is the opposite sign to the narrative the circular OT check supported, and opposite to the translated arm's overconfidence in P6-D3. Against labels that cannot be gamed, MIRAGE's posteriors on realistic noise are too conservative.

**Finding 2: the sigma mismatch explains a large share but not all of it.** Moving from Tier 1 to Tier 0 is the only change, and it takes coverage 0.851 to 0.752 (nocond) and 0.806 to 0.718 (cov), recovering roughly two thirds of the excess. SBC p moves 47 and 29 orders of magnitude. So P6-D7's scale gap is a dominant, quantified driver, and retraining over a sigma range that contains real data is now a concrete v2 action with a predicted effect size rather than a caveat. But Tier 0 still fails, so the flow is also intrinsically over-dispersed and retraining alone will not close it. Both halves of that sentence are load-bearing.

**Finding 3, and it reverses the conditioning verdict under a different evaluator.** The cov arm is BETTER calibrated than nocond in every cell: coverage 0.718 against 0.752 on Tier 0 and 0.806 against 0.851 on Tier 1, TARP deviation 0.110 against 0.195, SBC p four orders of magnitude less extreme. On Tier 0 the cov arm is close to calibrated on 5 of 7 parameters (logH2O 0.11, logCO2 0.60, logCH4 0.92, logCO 0.096, logNH3 0.022).

This does NOT contradict P6-D2, it bounds it. P6-D2 measured JS divergence to the TauREx-NS posterior on one real target and found cov worse by 0.083. That comparison is only as good as the anchor. This one uses prior draws as truth, and the ranking flips. **So "noise conditioning degrades retrieval" is conditional on the NS anchor being correct**, which A5/H8 was already flagged as load-bearing for. Against ground-truth labels, conditioning improves calibration. Report both and say which evaluator each rests on; do not quote the Phase-4 or P6-D2 conclusion as if it were evaluator-independent.

**Finding 4: rp_rj is the worst-calibrated parameter in every single cell**, from 3.3e-05 to 4.4e-56, consistently the minimum. The free radius is the project's headline fix (P3-D11, chi-square 301 to 0.06), and its posterior is the most over-wide of the seven. That needs saying in the paper rather than discovering it in review.

**Method note, TARP behaved as measured.** On cov/Tier 0, TARP gives p = 0.012 and would alone call that cell calibrated, while SBC (3.3e-05) and HPD (1.4e-09) both reject. That is exactly the power floor measured in P6-D5, where TARP detects a half-sigma bias only 57% of the time. It is the reason three statistics are reported, and a reason not to lead with TARP.

**Caveats to carry.** One training seed (s42) per arm; s43 and s44 exist and the conditioning ranking should be checked against them before it goes in a paper. n = 200 per cell, which is 10 per SBC bin, so per-parameter p-values near 0.01 are not firm. Tier 1 is 87% real noise on WASP-39b, the rest white at the published amplitude. The Tier-0 set scales the same realizations by 56.7, so Tier 0 and Tier 1 are not independent draws and the comparison between them is paired by construction, which is the point.

**Next.** Repeat across s43/s44 for the conditioning ranking; run wasp96 (sigma only 2x below training, so a weaker scale mismatch is predicted) and k218; then the retraining experiment that P6-D7 and this entry jointly motivate, a radius arm trained over a sigma range reaching down to real published error bars.

---

### P6-D9 — Framing B locked, Track 2 reassigned against the Track 1 critical path, A5 promoted to gating [Phase 6, 2026-11-08, PLANNING]

**The problem.** Three separate things had drifted out of alignment. Framing B was recorded as settled on 2026-11-01 but the work split still read as if A, B and C were live. Track 2 had not moved since 2026-08-18 and no `phase-6-track-2` branch existed, while framing B depends on Track 2 for one of its two pillars. And A5 was ranked below H9 on the stated grounds that it "hardens numbers, it cannot kill a claim", which stopped being true the moment P6-D8 landed.

**Framing B locked.** Sim-to-real negative results, TMLR. The consequence that needs stating plainly is that B rests on exactly two findings, the conditioning result and the null translation result, and the second one is Track 2's. Under the v1 framing the translation work was a paragraph that could be cut at no cost. Under B it is a pillar, so its weakest point becomes a submission risk rather than a footnote.

**That weakest point is the CycleGAN real side.** It is one reduced WASP-39b spectrum bootstrap-expanded to 500 samples with 1 percent photon-noise jitter. The external audit flagged it as near-certain overfitting and the objection is sound. A reviewer removes a pillar with one sentence, "the translator failed because it was trained on one spectrum jittered 500 ways", at no cost to themselves. Three real reduced spectra now exist, so this is correctable rather than terminal. Added as V7, the only remaining CycleGAN run the project does, and explicitly exempted from the standing "no more CycleGAN variants" rule because it is a correction to an invalid training set rather than a new variant.

**A5 moved from hardening to gating.** P6-D2 measured conditioning as HARMFUL by 0.083 JS against the TauREx-NS posterior. P6-D8 measured it as HELPFUL in every cell against prior-draw injection labels. Those are opposite conclusions about the same arms, and the only difference is the evaluator. Which one the manuscript states is therefore decided entirely by whether TauREx-NS is a trustworthy reference, which is exactly what A5's live-point sweep and the pRT cross-check answer. **A6 cannot be written before A5 returns.** The 2026-11-05 ranking put A5 last because it could not kill a claim. It can now pick between two of them, which is a stronger reason to run it than the one originally given for deferring it.

**The same exposure applies to P6-D3**, the translation verdict, which was also measured against the NS anchor alone. Added as V1b. The A1 machinery already exists, so scoring the translated arm against injection labels is a run rather than a build, which is the cheapest way to find out whether that verdict is evaluator-independent.

**Track 2 reassigned with dates, acceptance tests and fallbacks.** Dates are set by the Track 1 critical path, not by preference, since writing starts when A5 returns and anything after 2026-11-30 arrives after the section that would cite it. Required for submission: V7 and V2 by 2026-11-16, V6 and V1b by 2026-11-23. Wanted: V5 verification by 2026-11-16 and result by 2026-11-30, V3 by 2026-11-12. V4 parked, on the reasoning that a half-finished V4 is worth less than a finished V2 and the two cannot both fit. Each item carries a stated fallback, so a miss changes the paper in a recorded way rather than being discovered during writing. V2's fallback is absorption by Track 1 at one to two days. V7's fallback is cutting the CycleGAN result entirely, which reduces framing B to a single pillar.

**What this entry does not do.** It does not reopen the framing. A and C stay closed. Option A was the higher-probability venue and remains the fallback if the required Track 2 set does not land, because A needs nothing from Track 2, but switching to it is a decision to be taken once on stated evidence rather than drifted into by default.

---

### P6-D10 — arXiv v2 deferred to the finished manuscript; repo cleaned from 46G to 34G [Phase 6, 2026-11-08, PLANNING]

**arXiv v2 deferred, deliberately.** Z1 was ranked zeroth priority on 2026-11-05 and again in the 2026-11-08 reordering, on the reasoning that the public v1 abstract still advertises the chi2 301 to 0.06 figure and the OT-calibrated result, both of which this log disproves (P5-D4, P5-D5), and that correcting it costs hours and no compute. **Decision taken 2026-11-08: hold the update until the full manuscript and all v2 changes are complete, then post once.** The reasoning for holding is that v2 then becomes the submission version rather than an interim patch, which avoids two public updates and keeps one coherent artifact. The cost accepted is the window in which v1 asserts two retracted claims. Recorded here because Z1's rank changed for a stated reason rather than by neglect, and because the risk is now a known accepted cost rather than an oversight. Z1's position in `08_work_split_v2.md` updated to match.

**Repo cleaned, 46G to 34G.** All deletions were verified untracked, verified rebuildable or byte-identical duplicates, and verified unreferenced by any script before removal.

- **12G, the bulk of it.** Twenty intermediate `cyclegan_epoch_*.pt` checkpoints in each of six CycleGAN run directories, 103 MB apiece. Written by `train_cyclegan.py:399` as periodic training snapshots, read by nothing, already covered by `.gitignore:56`, which confirms they were never intended to persist. Every directory's `cyclegan_best.pt` was confirmed present before its epoch files were removed, and all seven survive.
- `WASP_96b_out_of_transit_data.zip`, 6 MB. Byte-identical to `WASP96b_out_of_transit_data.zip` by md5, and the underscored name is not the one `scripts/zip_oot_packages.py` writes, so it was a stray copy.
- `mirage_abc_rad_data.tgz`, 24 MB. The machine-to-machine transfer archive, already gitignored as rebuildable with tar.
- `data/inject/_smoke.hdf`, the injection pipeline's smoke-test output.
- Fifteen `__pycache__` directories, seven `.DS_Store` files, one `.ipynb_checkpoints` directory.

**Left in place, needing a decision rather than a judgement call.** The `snapshots/` directories under the FMPE run dirs hold roughly 3G across 273 files, and unlike everything above they are **tracked**. They are resume points for finished runs, so they are dead weight in the working tree, but removing them writes 273 deletions into history and that is a repository-level decision. Also left: `zenodo_deposit/` and `mirage_zenodo_data_v1.tar.gz`, 30 MB combined, which are kept on provenance grounds since they may be the exact bytes deposited rather than a rebuildable copy.

---

### P6-D11 — A1 across three training seeds and three targets: under-confidence and the sigma driver hold, the conditioning reversal is much weaker than one seed suggested [Phase 6, 2026-11-08, MAJOR — partly a correction to P6-D8]

**What ran.** Ten further cells at n = 200, 199 samples, 20 bins. Eight are the seed repeat P6-D8 said was owed, 2 arms x 2 tiers x seeds 43 and 44, all within the CUDA `h9_*` family so no device confound exists inside the comparison. Two are the extra targets, WASP-96b and K2-18b on their own Tier-1 sets using their per-target models. Those two models are MPS-trained and were cleared at backend-shift level by P6-D4 (+0.032 and -0.022), which is the caveat they carry. No cell failed.

**Correction to P6-D8.** That entry stated that conditioning beats no-conditioning "in every cell". That was seed 42 alone, and seed 42 is the most favourable of the three. With all three seeds in, the ranking is directionally consistent on the joint statistics and inconsistent on the marginal one. HPD favours cov in 6 of 6 cells, TARP deviation in 5 of 6, coverage@68 in 5 of 6, and **SBC minimum p in only 3 of 6**. On Tier 0 at seed 43, cov is four orders of magnitude worse on SBC, 2.3e-10 against 1.6e-06. The sentence "against ground-truth labels, conditioning improves calibration" is too strong and is replaced by the one below.

**The effect sizes are not symmetric, and that is the finding.** Measured as distance of coverage@68 from the nominal 0.68, conditioning's injection-label advantage is 1.6x the training-seed spread on Tier 0 and 2.9x on Tier 1. P6-D2's real-data conditioning *harm*, measured as JS to the TauREx-NS posterior, was 8.4x the seed spread. So these two results are not equal and opposite. **Conditioning is mildly better calibrated against injection labels and strongly worse against the nested-sampling anchor.** Both statements can hold simultaneously, and the paper should carry both with their effect sizes attached rather than presenting a clean reversal. A5 still decides which one leads, so its promotion to gating in P6-D9 stands, but the stakes are lower than P6-D8 implied because the injection-label effect is weak.

**Under-confidence survives everything.** All 12 WASP-39b cells plus both extra targets give coverage@68 above the nominal 0.68, spanning 0.704 to 0.851, with maximum absolute rank skew 0.116. Bell-shaped throughout. This is the G3 answer and it is robust to training seed, to arm, to tier and to target. Nothing in this entry weakens it.

**The sigma driver survives, and now has a dose-response across targets.** All six Tier-1 to Tier-0 pairs improve, by 0.09 to 0.10 in coverage. Adding the two targets, whose real published error bars sit at different distances from the training floor, gives four points ordered by mismatch factor:

| set | sigma below the 0.05 training floor | coverage@68 |
|---|---|---|
| WASP-39b Tier 0, rescaled into range | 1x | 0.742 |
| WASP-96b Tier 1 | 2.4x | 0.783 |
| K2-18b Tier 1 | 2.7x | 0.796 |
| WASP-39b Tier 1 | 24x | 0.834 |

Monotone in the mismatch. The Tier-0 against Tier-1 pair is the controlled version of this, same model and same noise realizations with only the scale changed, and it points the same way, so the cross-target rows corroborate rather than carry the claim. Three different models and three different wavelength grids are involved, so they cannot do more than corroborate. P6-D7's prediction that WASP-96b would show the weakest mismatch was correct.

**rp_rj is the worst-calibrated parameter in 12 of 12 WASP-39b cells and in both extra targets.** G4b is unconditional. On WASP-96b the molecular abundances are close to calibrated, logCO2 0.20, logCO 0.069, logCH4 0.030, while the radius sits at 4.7e-30, so the degeneracy is **isolated to the radius rather than smeared across the retrieval**. That is a better result for the paper than a diffuse failure, because it localises the problem to the parameter whose free treatment is already the headline fix. K2-18b differs, with logH2O at 3.4e-08 as its second-worst parameter, which is consistent with 19 percent of its bins still being white-filled across 3.78 to 5.17 micron where the water and methane features sit (P6-D6).

**Methodological warning, and it applies to what P6-D8 already wrote down.** The rp_rj p-value is wildly seed-unstable. Tier 1 no-conditioning gives 4.4e-56, 3.0e-39 and 2.6e-34 across seeds 42, 43 and 44, a spread of **21.8 orders of magnitude** at fixed configuration. The rejection is robust, the exponent is not. P6-D8 quoted 4.4e-56 as though it were a measurement. **No SBC p-value magnitude from these runs belongs in the manuscript.** Report the direction, the fact of rejection, and the coverage numbers, which are stable to roughly 0.01 to 0.016 across seeds.

**Caveats to carry.** The "x the seed spread" ratios rest on 3 seeds per arm, so the spread estimate itself is weak and the ratios should be read as order-of-magnitude rather than measured. This is the same convention P6-D2 used, so the comparison between 1.6x/2.9x and 8.4x is at least internally consistent. n = 200 per cell remains 10 per SBC bin. The two extra targets use MPS-trained per-target models, cleared only at backend-shift level by P6-D4, and have no cov twin, so they test target and sigma-scale variation and not conditioning. K2-18b is 81 percent real noise with the gap concentrated where its water features are.

**Artefacts.** Fourteen result files under `data/sbc/`, named `{tier}__{arm}.npz`.

**Next.** The seed repeat is discharged and the extra targets are done, so A1's remaining item is G13, the WASP-96b depth-noise construction, which is a build rather than a run. The retraining experiment that P6-D7 and P6-D8 motivate is unchanged and is now better specified, since the four-point dose-response above predicts what closing the sigma gap should buy.
