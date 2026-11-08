# MIRAGE: full state, every gap, and the research agenda

Written 2026-11-05, revised the same day after an external literature audit
("Making MIRAGE Publishable: A Real-Noise Audit of Amortized Exoplanet Retrieval").
This is the single document to read before doing anything else on this project.
It is long on purpose: the project's recurring liability has been claims drifting
away from what the code supports, and the fix is writing down exactly what is
supported and what is not.

Companions: `problems_and_decisions.md` (chronological log, through P5-D12),
`06_v2_post_rejection_plan.md` (review response), `08_work_split_v2.md` (who does what).

---

## 0. Summary

Amortized SBI trained on TauREx spectra works on synthetic JWST data and collapses on
real reduced spectra. On WASP-39b the cause was an omitted latent, the planet radius,
which the simulator encodes but the inference did not infer. Restoring it took best-fit
chi2/N from 301 to 1.44 at a fixed error floor. That holds. A workshop review then found
the calibration validation was circular, and a self-audit found a silently switched error
floor, a radius claim that held on one target rather than three (since reversed, P5-D12),
and five independent defects in the domain-translation comparison. All are fixed or
reversed, and those corrections are the most interesting part of the work.

**The strategic position changed on 2026-11-05.** An external audit established that
running SBI on real JWST spectra is no longer new, and that the radius finding cannot
carry novelty. What remains open is narrow and real: nobody has tested whether SBI
robustness techniques transfer to real instrument noise under a calibration check that
cannot be circular. That is the paper.

---

## 1. What has been built

### 1.1 Inference stack

| component | where | state |
|---|---|---|
| multi-instrument transformer encoder | `mirage/nn/` | complete |
| covariance embedding (noise conditioning) | `mirage/nn/`, `datasets/noise.py` | complete |
| flow-matching posterior (FMPE) | `fm4ar`, pinned, unmodified | complete |
| registration without forking fm4ar | `mirage/register.py` | complete, P2-D8 |
| correlated-noise injection | `datasets/transforms.py` | complete |
| CycleGAN translator | `nn/cyclegan.py`, `train_cyclegan.py` | complete, rebuilt P5-D11 |
| learned-translation transform | `datasets/translate.py` | new 2026-11-05 |

Noise model: white floor plus squared-exponential and Ornstein-Uhlenbeck kernels, drawn
per sample, one covariance shared between the injected realisation, the error bars and
the OOT frames. Sharing it is the point (P2-D3).

### 1.2 Real-data pipeline

Three targets: WASP-39b (NIRSpec PRISM, published), WASP-96b (NIRISS SOSS, **self-reduced
end to end from raw MAST**, ERO 2734, exoTEDRF plus batman), K2-18b (NIRISS, published).
Independent nested-sampling anchor per target, TauREx plus Nestle, 300 live points. Real
OOT frame stacks for all three: **3,241 / 56 / 96**.

### 1.3 Evidence artifacts, committed and runnable

`multitarget_table.py` (Table 2, stated floor, ESS policy) · `transfer_table.py`
(synthetic coverage both families beside real JS, both IS-free) · `fig_multitarget_v2.py`
· `run_cyclegan_tiers.sh` and `cyclegan_tier_table.py` · `build_real_domain.py` ·
`build_translated_abc.py` · `seed_dispersion.py` · `run_noisecond_ablation.py`.

---

## 2. The rejection, and what it found

Sim2Science #60, ratings 4/2/2. The single accept declared confidence 2 and unfamiliarity
with JWST retrieval; both rejects declared confidence 3 and converged independently.

**1. The calibration validation was circular.** Gaussian OT maps the flow posterior onto
the NS posterior; the paper then checked whether NS values fell inside intervals built
from transported samples. `ot_calibrate.py` lines 7-9 already said "the pushforward IS
N(mu_n, Sigma_n)". Measured (P5-D5): offset 0.46 sigma_NS, width 2.08x, IS-ESS 12.4 of
6312. Coverage 7/7 was guaranteed before any data was examined.

**2. The chi-squared numbers did not reconcile.** Cause (P5-D4): flow side used a 5%
systematic floor, NS side 1%. chi2 scales as 1/sigma^2, inflating the improvement ~25x.

**3. The central finding is not novel.** `fm4ar/datasets/vasist_2023/prior.py:29` has
`[0.9, 2.0, "R_P", ...]`: radius is parameter two in the backbone's own prior. The external
audit adds that Welbanks and Madhusudhan (2019) already characterised the
radius/reference-pressure degeneracy. A nuisance parameter fixed wrong *is* a form of
misspecification, so "identifiability, not fidelity" does not survive as written.

---

## 3. Where the field is (external audit, 2026-11-05)

**Running SBI on real JWST spectra is not new.** Gebhard et al. (A&A 2025) introduced FMPE
with importance sampling and noted IS "identifies model failures via low efficiencies",
which is essentially our ESS observation. Exoformer (arXiv 2603.27623) ran on real WASP-39b
PRISM and WASP-17b SOSS. FASTER (arXiv 2502.18045) used neural ratio estimation on real
WASP-39b PRISM and reproduced the NS model ranking.

**Robust-SBI is crowded with methods and thin on real-noise tests.** RNPE (Ward 2022),
RSNL (Kelly, TMLR 2024), MMD-regularised robust statistics (Huang, NeurIPS 2023),
summary-space misspecification detection (Schmitt), RoPE (Wehenkel, ICML 2025), RVNP,
flow-matching robust SBI, preconditioned RNPE. Most validate on toy simulators.

**Pipeline shifts are studied, but with nested sampling, not SBI.** "Knobs and dials II"
(Schleich et al., A&A 704 A223, 2025) ran TauREx on three WASP-39b reductions. Roy-Perez
et al. (A&A 707 A297, 2026) compared six PRISM reductions and found abundances deviate up
to 2 dex. Schmidt et al. ran 250+ retrievals on K2-18b.

**The niche actually open**, and it is narrow:
1. noise-model sim-to-real for transmission spectra, evaluated **inferentially** as well as
   distributionally;
2. calibration checked against labels that **do not come from the reference retrieval**;
3. robustness measured across **real pipeline shifts**.

**RoPE explains the retired OT step**, and offers a route back. RoPE uses OT between learned
representations plus "a small real-world calibration set of ground-truth parameter
measurements". Ours failed because the "ground truth" *was* the NS posterior we then checked
coverage against. Rebuild the same machinery with labels from injections into real noise and
it stops being circular. That is a better story than retiring it.

---

## 4. Current verified numbers

Reproducible from a committed script. **Quote nothing not in this table or regenerable.**

### 4.1 Real data, 1% error floor

| target | bins | chi2/N | max resid | >3sig | ESS | IS-free radius 68% | literature | inside |
|---|---|---|---|---|---|---|---|---|
| WASP-39b (nocond) | 47 | 1.441 | 2.26 | 0 | 1.00 | [1.1417, 1.2841] | 1.279 | yes |
| WASP-39b (OT-cal) | 47 | 0.662 | 1.60 | 0 | 42.5 | [1.2243, 1.2336] | 1.279 | no |
| WASP-96b | 90 | 0.771 | 2.45 | 0 | 7.49 | [1.1568, 1.2234] | 1.200 | yes |
| K2-18b | 149 | 1.180 | 2.87 | 0 | 1.31 | [0.2189, 0.2363] | 0.2352 | yes |

NS anchors 0.761 / 0.793 / 1.277. Pre-radius baseline 301. IS-weighted intervals are
**withheld** below ESS 20 (P5-D12); the IS-free intervals above are 4x, 8x and 12x tighter
than the prior's own central 68%, so containment is informative rather than vacuous.

### 4.2 Noise conditioning, coverage@68, nominal 0.68

**H9 ANSWERED. The synthetic half is withdrawn, the real half survives (P6-D1, P6-D2).**

The published no-conditioning arms were trained on MPS and never converged. Retrained on CUDA at
identical configs and seeds, 3 seeds per arm:

| arm | ABC (6-param) | radius (7-param) | real JS to NS |
|---|---|---|---|
| no conditioning | 0.692 ± 0.013 | 0.692 ± 0.016 | **0.344 ± 0.003** |
| + sigma + covariance | 0.698 ± 0.011 | 0.693 ± 0.014 | 0.426 ± 0.014 |

Superseded, do not quote: 0.587, 0.817, 0.687, 0.700, 0.352, 0.428. The +sigma row (0.687 / 0.903
/ 0.371) is withheld, not corrected, because `noisecond_rad_sigma` is a third failed MPS run and
has not been retrained. It sits at 1.3729 against a radius CUDA band of 1.124 to 1.159, so the
degradation is about 0.24, worse than `rad_nocond`'s 0.115. Job B's `h9_rad_sigma_s42` settles it
(P6-D4).

Every synthetic arm is now at nominal. The conditioning gap is 0.5x the seed spread on ABC and
0.1x on radius, so "calibrates from either direction" is withdrawn; it required a baseline
miscalibrated in two directions, and that was the training failure.

The real-side gap is +0.083 nats at **8.4x the seed spread** and replicates the published +0.076,
so the transfer failure stands on its own. Undertraining moved synthetic coverage by 0.114 and the
real JS by 0.008, which is why one half died and the other did not. Mechanism is a systematic
bias, not miscalibration: conditioning drives retrieved T to 1620 K against the anchor's 608 K.

**Why the baselines broke and the treatment arms did not (P6-D4).** Device was perfectly
confounded with arm. All four cov arms ran on CPU, because the batched-cov matmul was slow on MPS
and the choice is recorded in `noisecond_rad_cov/config.yaml`. All five nocond arms and both sigma
arms ran on MPS. MPS is a lottery rather than a constant penalty, 1 catastrophic and 2 degraded out
of 6 runs, so every baseline carried an uncontrolled risk that the treatment arms did not. The
conditioning "effect" was partly a compute-backend effect wearing the treatment's label. Record the
device beside every reported model, and never let the backend vary with the arm being compared.

**The multi-target models are clear (P6-D4).** They are also all MPS, so I reran all three as CUDA
controls at `diff`-verified identical configs. K2-18b 1.1822 against the published 1.2141, WASP-96b
1.2110 against 1.1885, hires 1.0665 against 1.0729. No control lands in the degraded band, and the
two outside the reproduction band point in opposite directions, which no systematic backend penalty
produces. **Table 2 and P5-D12 clear on training grounds.**

**The revised claim:** noise conditioning gives no measurable benefit on synthetic benchmarks and
degrades real-data retrieval. This now rests entirely on TauREx-NS being a valid anchor, which
makes A5/H8 load-bearing rather than a hardening task.

### 4.3 Translation, 3,241 real OOT frames

| | sim, none | structured randomization | CycleGAN |
|---|---|---|---|
| MMD to real | 0.2925 | **0.1704** | 0.7686 |
| KL to real | 3.943 | **3.675** | 4.172 |
| mean shift (per-bin sigma) | 0.030 | **0.033** | 0.308 |

Translation is **2.6x worse than no adaptation**. A control varying only frame count showed
the CycleGAN's MMD collapsing 71% while sim and random moved under 1%, so the cross-target
version is a small-sample artifact and is not evidence about targets.

**Posterior-level, G1 closed (P6-D3, 2026-11-07.)** The distributional result above says
translation does not match real noise. This says it does not help inference either, which is
the claim the paper actually needs. `cyclegan_posterior_cov` differs from `noisecond_cov` in
exactly one line, the dataset path, so the difference is attributable to translation alone.
50 planets x 1000 samples on `abc_test.hdf`:

| | sim | translated |
|---|---|---|
| coverage@68 (nominal 0.68) | **0.690** | 0.557 |
| coverage@95 | 0.987 | 0.853 |
| width@68 | 2.222 | 2.020 |
| logdens | **-7.71** | -17.65 |

Training on translated data costs **10 nats** of log-density and drops coverage to 0.557 while
*narrowing* the intervals. Narrower plus worse coverage is overconfidence, not a trade-off.
Worse in all three sigma tertiles (-18.73 / -20.17 / -14.21 against -7.57 / -7.66 / -7.90).

So the CycleGAN negative is now inferential, not just distributional: translation actively
degrades the posterior. Retire CycleGAN and keep it as a negative control (V3).

### 4.4 Never repeat

`chi2/N = 0.06` (floor-mismatched; use 1.44 or 0.66) · `coverage 7/7` (circular) ·
`MMD ~ 1.22` (invalid comparison) · any interval from an arm with ESS below ~20 ·
anything from `ablation_results_v2.json` before 2026-11-01.

---

## 5. Every known gap

**G1. The translation claim is distributional, not inferential.** CLOSED.
`configs/cyclegan_posterior_cov` differs from `noisecond_cov` in exactly one line, the dataset
path, so any difference is attributable to translation alone. Result in section 4.3.

**G1b. That result was measured against the nested-sampling anchor only, and P6-D8 showed the
conditioning ranking flips between the NS anchor and injection labels.** So P6-D3 carries the
same exposure and is not yet known to be evaluator-independent. **Assigned as V1b, due
2026-11-23 (P6-D9).** The A1 ladder already exists in `scripts/sbc_report.py`, so this is a run
and not a build.

**G2. Every headline is a single run.** CLOSED for the conditioning arms: 3 training
seeds per arm, 12 runs, in `configs/h9_*` (P6-D1, P6-D2). Still open for every other headline,
and the H9 runs vary `dataset.random_seed`, so their spread covers split plus init plus noise,
which is wider than training-seed variance alone. Arms stay paired within a seed.

**G3. No non-circular calibration validation. CLOSED (P6-D5, P6-D6, P6-D8).** Tier-1
injections now exist and have been scored: prior-draw labels, real OOT noise structure at the
published amplitude, SBC ranks plus TARP plus HPD ranks, with the diagnostics themselves
validated against five known-wrong samplers before use. The answer is that MIRAGE is
**under-confident** on realistic noise, coverage@68 of 0.72 to 0.85 against a nominal 0.68,
bell-shaped ranks, negligible skew. That is the opposite sign to what the circular check
implied. Two thirds of the excess width traces to the sigma scale gap of P6-D7; the remainder
is intrinsic. Also reverses the conditioning verdict under this evaluator: cov beats nocond in
every cell, so P6-D2's "conditioning degrades retrieval" is conditional on the NS anchor being
right. `scripts/sbc_report.py`, `data/sbc/`.

**G4b. rp_rj is the worst-calibrated parameter in every A1 cell** (3.3e-05 to 4.4e-56), and
the free radius is the headline fix. Say so in the paper rather than meeting it in review.

**G4. NS anchors never shown stable** at 300 live points, and everything cites them.

**G5. N=3, and the NIRISS pair is data-starved** (56 and 96 frames). Addressed by the
three-tier data strategy.

**G6. One simulator.** Sim and reference share TauREx, so part of the "transfer failure"
could be shared misspecification. The Ariel Big Challenge cannot fix this: it is TauREx-based
too. Only a petitRADTRANS or POSEIDON reference settles it (H8).

**G7. CycleGAN domain mismatch** is a hypothesis, untested (H1). **Assigned as V3, due
2026-11-12 (P6-D9).** One retrain, about 90 minutes, then the hypothesis is either given a
mechanism or struck from this list rather than left standing.

**G8. One translation architecture.** Cheap principled baselines exist and are missing:
per-bin affine, quantile mapping, classifier reweighting. **Assigned as V2, due 2026-11-16
(P6-D9).** Under the locked framing B this is no longer an optional sharpening. The null
translation result is one of two pillars, and a negative with no comparison class is the first
thing a reviewer asks about.

**G8b. The CycleGAN real side is one spectrum bootstrap-expanded to 500 samples** with 1
percent jitter, which the external audit flagged as near-certain overfitting. Under framing B
that single objection removes a pillar at no cost to the reviewer. Three real reduced spectra
now exist, so it is correctable. **Assigned as V7, due 2026-11-16 (P6-D9), and it is the last
CycleGAN run the project does.**

**G9. Amortization is dead and nothing replaces it.** Revivable: a calibrated amortized
posterior mapping reduction-induced spread across targets is something NS cannot afford
(TauREx needs ~498 single-core hours for WASP-39b).

**G10. Radius containment is tight, not comfortable** (0.93 and 0.87 of half-width).

**G11. `corr_similarity` does not separate the arms** and points the other way from MMD.

**G12. KL estimator biased upward at low n.** Ordering stable, absolute value not. Replace
with MMD plus a permutation test, or a classifier two-sample test.

**G13. OOT residuals are flux noise standing in for depth noise.** Now load-bearing, because
Tier 1 depends on it. Construct depth noise by running the light-curve fit on pseudo-transits
injected into OOT segments.

**G14. Prose and presentation.** `R_p` vs `r_p`; Table 2's 0.24 vs Fig 2's 0.235; four
unparseable sentences; reads as a report on what was explored; editorial notes in checklist.

**G15. `run_cyclegan_ablation.py` answers the wrong question.** Relabel as a robustness probe.

---

## 6. Research agenda

### 6.1 The data strategy: three tiers

**Tier 1, semi-synthetic injection into real noise. Highest payoff, laptop-only.** Draw
theta from the prior, simulate with TauREx, inject into real OOT-derived noise held out by
frame index, run the flow, record coverage. This gives thousands of labelled cases with real
correlated noise, and it is simultaneously the non-circular calibration check (G3) and the
large benchmark (G5). It enables SBC, expected coverage and TARP. Precedent: DINGO trains on
"simulated signals injected into real noise realizations taken from detectors".

Two cautions, both real. **G13 stands**: build depth noise through the actual light-curve fit,
not from flux residuals. Scope that properly to **WASP-96b**, where we own the exoTEDRF plus
batman pipeline end to end; treat the other two as flux-noise approximations with the caveat
stated. **Frame counts bite**: 56 SOSS frames give very few independent blocks at a 0.02-0.05
micron correlation length, so frame-resampling uncertainty is available on WASP-39b and not on
the other two. Say so rather than bootstrapping past it.

**Tier 2, multiple reductions of the same visit** as natural domain shifts, since each pair
shares the same astrophysical truth and isolates pipeline shift from atmosphere. Sources:
WASP-39b ERS 1366 four-mode uniform reduction (Carter et al. 2024, Zenodo 10.5281/zenodo.10161743)
and six PRISM reductions compared in A&A 2026; K2-18b (Schmidt et al. 2025 Zenodo 14735689;
Luque et al. 2025 Zenodo 15685551); WASP-107b, WASP-17b, TOI-270 d, GJ 1214b, COMPASS seven.
**Verify each Zenodo record actually contains per-reduction spectra in usable formats before
planning around it.**

**Tier 3, bulk published spectra** from the NASA Exoplanet Archive spectroscopy table, 15-25
planets, supporting posterior-predictive tests but not noise-bank tests. Gives a
population-level statistic that cannot be dismissed as N=3.

### 6.2 Hypotheses, in priority order

**H9. Seed variance swamps the conditioning effect. ANSWERED, split verdict (P6-D1, P6-D2).**
Tested with 12 CUDA runs, 2 families x 2 arms x 3 seeds, in `configs/h9_*`. Synthetic: yes, the
effect is 0.1x to 0.5x the seed spread and is withdrawn, though the deeper cause is that the
published baselines never trained rather than seed noise. Real: no, the 0.344 to 0.426 gap is
8.4x the seed spread and stands. Pillar 1 is reshaped, not withdrawn. See section 4.2.

**H5. The conditioning paradox comes from the noise proxy.** Real JS may rise under covariance
conditioning because Sigma-hat from OOT *flux* misrepresents *depth*-noise covariance. Condition
on three inputs: Sigma-hat from OOT, Sigma-hat from pseudo-transit depth noise, and published
diagonal errors. If the paradox disappears with depth-noise Sigma-hat, there is a mechanism; if
it persists, the paradox is real and is the headline.

**H2. Fit kernel hyperparameters to the real OOT covariance**, then randomize around the fit
rather than over a broad hand-chosen range. Direct GW precedent: Wildberger et al. fitted a
parametric PSD and got accurate inference across 37 real events. **Predict in writing, before
running, that it improves MMD.** Then test whether it also improves JS to NS. If MMD improves
and inference does not, that strengthens the dissociation thesis.

**H10. Offset latents generalize the radius fix.** The "missing latent" is really a free
transit-depth baseline offset, and multi-instrument JWST work already fits per-instrument
offsets (Carter et al.: 138 ppm NIRSpec/NIRISS shift; Luque et al.: uniform +/-200 ppm priors).
Adding per-instrument offset latents should remove the chi2 excess in joint multi-instrument
retrievals; leaving them out should reproduce a radius-style collapse. **This is the live
methodological question the dead radius claim turns into.**

**H3. ESS collapse is driven by the systematic floor.** Sweep 0.5, 1, 2, 3, 5, 7, 10 percent on
all three targets, plot ESS, chi2/N and JS against floor with seed error bars. Turns a pathology
into a stated property of the error budget.

**H6. Summary-space distance predicts real error.** Rank-correlate a Schmitt-style MMD detector
score against JS to NS across every arm. A null is a notable negative about amortized
misspecification detectors.

**H7. Pipeline shift dominates noise-model shift.** If posterior JS between two reductions of
one visit exceeds the JS change from any noise-model intervention, the field is tuning the wrong
thing.

**H8. The reference shares the misspecification.** If TauREx-NS and pRT-NS disagree by more than
the flow-to-NS gap, the reference is not a reference, and that is a finding (closes G6).

**H1. CycleGAN domain mismatch.** Run **once** as a mechanism check for G7, then retire CycleGAN
as a method and keep it as a negative control. A single bootstrapped spectrum was never a domain.

**H4, restated.** The original framing was too weak, since CycleGAN is already worse than no
adaptation. Correct version: "affine and quantile mapping beat no adaptation, which beats
CycleGAN, on both distributional and inferential metrics."

### 6.3 Methods worth importing

- **SBC, TARP, HPD coverage** on Tier 1 injections. TARP (Lemos et al., ICML 2023) gives
  necessary and sufficient coverage conditions from samples alone.
- **Classifier reweighting** (OmniFold-style density-ratio weights) as a principled third
  translation arm that needs no unpaired image translation. About a day.
- **Per-bin quantile mapping**, the standard bias-correction baseline in climate downscaling.
- **Moving-block bootstrap** over OOT frames for MMD and covariance estimates; blocks of several
  bins given the correlation length.
- **Posterior predictive checks** with residual autocorrelation and chi2 per instrument segment,
  used as detectors, not certificates.
- **Skip conformal prediction.** It needs an exchangeable calibration set; on Tier 1 it would
  only restate coverage.

### 6.4 A possible astrophysical result

A measured **reduction-induced posterior spread** for K2-18b and WASP-39b, computed with a
calibrated amortized posterior across every released reduction, is reachable and revives G9,
because NS cannot afford it at scale. Do **not** attempt a new DMS verdict: Schmidt et al. report
Bayes factors under 3 for CO2 and DMS, and an independent AJ reanalysis concluded the apparent
MIRI features "are most likely the result of red noise". Show instead that a calibrated amortized
retrieval reproduces the CH4 constraint and the absence of robust CO2 and DMS across every
reduction, and quantify the spread due to pipeline alone. Leave stellar contamination out of
scope and say so.

---

## 7. Honest assessment

**Strengths.** A real end-to-end system with a self-reduced spectrum from raw telescope data. A
large, well-established central effect. An unusually complete audit trail where every retired
claim carries the measurement that retired it. Under a venue that rewards correctness that is the
contribution, not an embarrassment.

**Weaknesses.** The headline framing was wrong and is retired. The method contribution went with
the OT step, though it is revivable on injection labels. N=3, one simulator, one translator, single
runs. The remaining positive result is a debugging finding: correct, useful, not novel.

**Target.** TMLR under framing B. Its published criteria ask only whether claims are supported by
convincing evidence and whether some audience would be interested, and state explicitly that
novelty "is not a necessary criterion for acceptance". It has published robust-SBI work (Kelly et
al. 2024). The risk is the evidence criterion: single runs, circular calibration and three targets
are exactly what TMLR rejects on. Title around the dissociation, for example "Robustness to
synthetic misspecification does not transfer to real JWST noise". Keep the radius story as a short
motivating case study, not the contribution.

If an astronomy paper is also wanted, the reduction-multiverse result belongs in AJ or RASTI as a
separate, smaller paper. Do not merge them.

**What will not move the needle:** more CycleGAN variants, more prose about the radius degeneracy,
more synthetic-only coverage tables.

**The one caveat on all of this:** H9 has now run, and the dissociation thesis did not survive
in the form written here. The synthetic win was an artifact of untrained baselines, so the
dissociation is not "wins on synthetic, fails on real" but "no synthetic benefit, real harm".
The surviving half is the seed-robust one at 8.4x spread. The Tier 1 and Tier 2 benchmarks still
stand on their own.
