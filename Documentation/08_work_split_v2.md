# v2 work split

Written 2026-11-05, revised the same day after the external literature audit.
Read `07_state_gaps_and_research_agenda.md` first; G and H identifiers refer to it.

Branch convention `phase-6-track-N`. Merge at phase boundaries.

---

## The split in one line

**Angshuman owns the validation ladder**, which is what decides whether v2 exists:
training-seed variance, Tier 1 injections into real noise, and the calibration rebuilt
on labels that do not come from nested sampling. **Vedanth owns the translation and
pipeline-shift side**, which under framing B is a load-bearing pillar rather than a
paragraph, and which now means cheap principled baselines rather than more CycleGAN.

---

## Zeroth priority, before anything else

### Z1. Post arXiv v2. Hours, no compute. (Angshuman)

The public v1 abstract still advertises "chi2/N from 301 to 0.06" and the OT-calibrated
result. Both are disproved by our own later work (P5-D4, P5-D5). Every day it stands is a
day the project's most visible artifact asserts something we know is false. Replace the
abstract's headline with the corrected 301 to 1.44 at a stated floor, and cut the
calibration claim. This is the highest credibility per minute available anywhere on the
project.

### Z2. Test H9. Everything downstream depends on the answer. (Angshuman)

The conditioning result (0.352 to 0.428) may sit inside training-seed dispersion. If it
does, pillar 1 of framing B must be withdrawn and the paper's headline changes. Do not
build on top of a result that might not exist.

Scope, given measured cost of about 14 h per FMPE arm on the laptop: **2 arms (no-cond and
+sigma+cov) x 3 training seeds = 6 runs.** See the compute note below. The currently
running evaluation-seed sweep is the *less* important variance and should be rescoped or
killed rather than allowed to block the machine.

---

## Angshuman, Track 1: the validation ladder

### A1. Tier 1 injections into real noise, with SBC, TARP and HPD coverage.

The core build, and the thing that makes v2 publishable. Draw theta from the prior,
simulate with TauREx, inject into real OOT-derived noise held out by frame index, run the
flow, record rank statistics. Labels come from the prior draw, so no transport can game
them. This is the non-circular calibration check (G3) and the large benchmark (G5) in one.

Build `scripts/sbc.py` (rank histograms per parameter plus a uniformity test) and
`scripts/inject_real_noise.py`. Read Talts et al. on SBC and Lemos et al. on TARP.

**Scope the depth-noise construction honestly.** G13 is now load-bearing: OOT residuals are
flux noise, and Tier 1 needs depth noise. Do it properly on **WASP-96b**, where we own the
exoTEDRF plus batman pipeline end to end, by running the light-curve fit on pseudo-transits
injected into OOT segments. For the other two, use the flux-noise approximation and state
it. Do not quietly treat all three alike.

**Frame counts bite.** WASP-39b has 3,241 frames, the others 56 and 96. Frame-resampling
uncertainty is available on WASP-39b only. Say so.

### A2. Rebuild the OT calibration on injection labels. (revives G3 and part of G9)

RoPE works because it calibrates against "a small real-world calibration set of ground-truth
parameter measurements". Ours failed because the ground truth was the NS posterior we then
checked coverage against. With A1 in hand the labels are independent, and the same machinery
becomes legitimate. "Our validation was circular, here is the non-circular rebuild" is a far
better paper than "we removed the contribution".

### A3. H5, the noise-proxy test. (3 to 5 days)

Condition on three inputs: Sigma-hat from OOT flux, Sigma-hat from pseudo-transit depth
noise, and published diagonal errors. If the conditioning paradox disappears with depth-noise
Sigma-hat, there is a mechanism. If it persists, the paradox is real and it is the headline.
Pairs naturally with A1, since A1 produces the depth-noise machinery.

### A4. H3, the floor sweep. (3 to 5 days)

Sweep 0.5, 1, 2, 3, 5, 7, 10 percent on all three targets. Plot ESS, best-fit chi2/N and JS
to NS against floor, with seed error bars. Cheap, mostly a loop over existing work, and it
converts the weight collapse into a stated property of the error budget.

### A5. NS anchor stability and H8. (1 to 2 weeks wall clock, heaviest item)

Live-point sweep 300 / 600 / 1000 on WASP-96b, the cheapest target, reporting whether
posterior widths and the radius interval move (G4). Then one petitRADTRANS or POSEIDON
reference at matched physics (H8). If TauREx-NS and pRT-NS disagree by more than the
flow-to-NS gap, the reference is not a reference, and that is a finding. **Deprioritised
below H9**: it hardens numbers, it cannot kill a claim.

### A6. Manuscript, inference side.

Drop "exhaustively" and "unchanged across targets". Rewrite the identifiability framing to
concede that the backbone's prior already infers R_P, that Welbanks and Madhusudhan (2019)
characterised the degeneracy, and that a nuisance parameter fixed wrong is misspecification.
State per-target NS cost so the lost amortization is visible (G9).

---

## Vedanth, Track 2: translation, baselines, pipeline shift

CycleGAN is finished as a method. The work is now about what replaces it and about pipeline
shift, which is a bigger effect than anything in the noise model.

### V1. Evaluate the translated-arm run when it lands. (finishes G1)

Training now. Compare against `noisecond_cov` on coverage@68, interval width and log-density
at truth. The configs differ in exactly one line, the dataset path, so any difference is
attributable to translation alone. This upgrades "translation does not match the real noise
distribution" to "translation does not help inference", which is what the claim must be.

### V2. The cheap baselines any learned translator must beat. (closes G8)

Three arms, all inexpensive: **per-bin affine**, **per-bin quantile mapping** (the standard
bias-correction baseline in climate downscaling), and **classifier reweighting** (train a
classifier to separate simulated from real-OOT-derived noise, use its output odds as
density-ratio weights, OmniFold-style; about a day).

Report as H4 restated: "affine and quantile mapping beat no adaptation, which beats CycleGAN,
on both distributional and inferential metrics." That is a much sharper negative than the one
we have, and it is the first thing a sceptical reviewer asks for.

### V3. H1 once, as a mechanism check, then stop. (closes G7)

Rebuild domain A restricted to one planet's parameter neighbourhood, retrain once, see whether
the 0.308 per-bin-sigma mean shift collapses. If it does, the negative gains a mechanism. If
not, remove the hypothesis from the decision log rather than leaving it standing. About 90
minutes on MPS. Use `--ckpt-dir`. **Then retire CycleGAN and keep it as a negative control.**

### V4. H2, system-identified randomization. (strongest translation-side hypothesis)

Fit the kernel hyperparameters to the real OOT covariance, then randomize around the fit
instead of over a broad hand-chosen range. Direct precedent in gravitational waves, where
Wildberger et al. fitted a parametric PSD and got accurate inference across 37 real events.
Three arms scored by both MMD to real and posterior JS to NS: broad randomization,
system-identified randomization, raw noise-bank resampling.

**Write the prediction down before running it.** Predict that it improves MMD. Then test
whether it also improves inference. If MMD improves and inference does not, that is the
dissociation thesis demonstrated a second time.

### V5. Tier 2, multiple reductions of the same visit. (G5, H7, and the astronomy angle)

Each pair of reductions shares the same astrophysical truth, so pipeline shift is isolated
from atmosphere. Start with K2-18b (Schmidt et al. Zenodo 14735689; Luque et al. Zenodo
15685551) and WASP-39b's four-mode Carter et al. reduction (Zenodo 10.5281/zenodo.10161743).

**Verify first that each record contains per-reduction spectra in a usable format.** The audit
flagged that it did not open them. Do that before planning around the dataset.

H7 is the question: does posterior JS between two reductions of one visit exceed the JS change
from any noise-model intervention we make? If yes, the field is tuning the wrong thing.

### V6. Related work, and relabel the old script. (G14, G15)

Own the related-work section: Swierc et al. (2410.16347) on unsupervised domain adaptation for
NPE in strong lensing, which is the right comparison class and aligns representations rather
than translating data; diffusion and Schrodinger bridges; "Knobs and dials II" and Roy-Perez et
al. on pipeline-level effects. Then relabel `run_cyclegan_ablation.py` as a robustness probe, so
it is never cited as the posterior-level translation result.

---

## Shared

- **One robust-SBI baseline, done correctly, not two done shakily.** Huang's MMD regulariser has
  public code and is the natural first choice. This project's history of comparisons that were
  invalid five ways is the reason to prefer depth over breadth here.
- **Verify every citation against the primary source.** The audit's own caveat says several facts
  came from secondary summaries, including the K2-18b Bayes factors. Treat that as binding.
- **G6, simulator diversity.** Note that the Ariel Big Challenge cannot serve as the independent
  test: it is TauREx-based too. Only A5's pRT reference settles it.
- **G14 prose.** `R_p` vs `r_p`; Table 2's 0.24 vs Fig 2's 0.235; the four unparseable sentences;
  editorial notes in the checklist; restructure toward a claim and its evidence.

---

## Order of operations

1. **Z1** (arXiv v2) immediately. Hours, no compute, removes a false public claim.
2. **Z2 / H9** next, on rented GPU. Nothing else is worth building until the conditioning result
   is known to exist.
3. **A1** (Tier 1) once H9 lands. Vedanth runs **V2** and **V3** in parallel; neither depends on
   the running jobs.
4. **V1** the moment the translated arm finishes.
5. **A2** once A1 produces labels. **A3** and **V4** next, since both are mechanism results.
6. **A5**, **V5** and the hardening passes while writing.
7. Writing starts only after H9 and A1, because both can change what the paper claims.

## What not to do

- Do not rebuild coverage against the reference the map transports onto.
- Do not restore the amortization claim while an NS anchor is paid per target.
- Do not quote an interval from an arm whose ESS the same table does not print.
- Do not build more CycleGAN variants, more radius prose, or more synthetic-only coverage tables.
- Do not attempt a new DMS verdict on K2-18b.
- Do not aim at ICML. TMLR under framing B is the supportable target.

---

## Compute note

Measured on the M1 Pro: about **5 min per epoch**, roughly 173 epochs with early stopping, so
**about 14 h per FMPE arm**. The audit's rule is that above 12 h per run you reduce arms rather
than seeds, so the laptop caps H9 at 2 arms x 3 seeds and cannot do the full matrix.

On a rented GPU the constraint effectively disappears. The model is tiny (3 to 4 MB checkpoints),
so the per-batch cost should fall from about 2.5 s to tens of milliseconds and the data pipeline
becomes the limit. Estimated **1.5 to 2.5 h per arm**, a 6 to 10x speedup.

| plan | runs | GPU-hours | cost at $0.40/h | at $0.69/h |
|---|---|---|---|---|
| H9 minimum, 2 arms x 3 seeds | 6 | ~12 | ~$5 | ~$8 |
| H9 full, 2 arms x 5 seeds | 10 | ~20 | ~$8 | ~$14 |
| full matrix, 3 arms x 5 seeds | 15 | ~30 | ~$12 | ~$21 |

Indicative rates as of late 2026: RTX 4090 around $0.34 to $0.74/h on RunPod and roughly $0.39 to
$0.59/h on-demand on Vast.ai, with interruptible near half that; A40 about $0.44/h. An A100 is not
needed for a model this size. **Check live prices before renting.**

Two practical notes. `train.py` writes snapshots and resumes, so **interruptible instances are
viable** and halve the cost again. And because the model is small, several arms can share one 24 GB
card, bounded by CPU dataloading rather than VRAM, which cuts wall clock further.

**Before renting anything, try MPS on the no-cond arm.** The covariance arm was put on CPU because
100 OOT frames per sample destabilised MPS, but the no-cond arm sets `n_oot_frames: 0` and may run
fine. The CycleGAN runs went 23x faster on MPS. If that holds, part of H9 is free.

**Separately, A5 is CPU-bound, not GPU-bound.** TauREx nested sampling needs roughly 498 single-core
hours for WASP-39b (Exoformer, A&A 2026). A many-core CPU instance is the right rental there, not a
GPU, and a 32 to 64 core box should bring it inside a day for a few tens of dollars.
