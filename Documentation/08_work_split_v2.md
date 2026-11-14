# v2 work split

Written 2026-11-05, revised the same day after the external literature audit.
**Revised 2026-11-08: framing B locked, Track 2 reassigned with dates and acceptance tests.**
Read `07_state_gaps_and_research_agenda.md` first; G and H identifiers refer to it.

Branch convention `phase-6-track-N`. Merge at phase boundaries.

---

## The split in one line

**Angshuman owns the validation ladder**, which is what decides whether v2 exists:
training-seed variance, Tier 1 injections into real noise, and the calibration rebuilt
on labels that do not come from nested sampling. **Vedanth owns the translation and
pipeline-shift side**, which under framing B is a load-bearing pillar rather than a
paragraph, and which now means cheap principled baselines rather than more CycleGAN.

**Framing B is locked as of 2026-11-08.** That makes the Track 2 split a dependency rather
than a preference, because framing B rests on two findings and the second one is Track 2's.

---

## Zeroth priority, before anything else

### Z1. Post arXiv v2. DEFERRED to the finished manuscript, 2026-11-08 (P6-D10). (Angshuman)

> **No longer zeroth priority, by decision rather than neglect.** v2 goes up once the
> manuscript and every v2 change are complete, so that it is the submission version rather
> than an interim patch and there is one public update instead of two. The accepted cost is
> the window in which v1 asserts two retracted claims. The corrections it must carry are
> listed below and have grown to four.

The public v1 abstract still advertises "chi2/N from 301 to 0.06" and the OT-calibrated
result. Both are disproved by our own later work (P5-D4, P5-D5). Every day it stands is a
day the project's most visible artifact asserts something we know is false. Replace the
abstract's headline with the corrected 301 to 1.44 at a stated floor, and cut the
calibration claim. This is the highest credibility per minute available anywhere on the
project.

### Z2. Test H9. DONE, 2026-11-07. Split verdict (P6-D1, P6-D2). (Angshuman)

Ran as **12** runs, not 6: 2 families x 2 arms x 3 seeds, on an RTX 5070 at about 27x the
Mac's per-batch speed, so the 14 h/arm constraint that capped the original scope no longer
binds. Seed 42 reproduces each stored run's seeds exactly, which made it a backend control.

**The answer was not the one the hypothesis anticipated.** The published no-conditioning arms
were never a baseline: both were trained on MPS and failed to converge (ABC best val 1.923
against 1.390 on CUDA at identical config and seed).

- **Synthetic: withdrawn.** All four retrained arms sit at nominal, 0.692 to 0.698. The gap is
  0.1x to 0.5x the seed spread. "Calibrates from either direction" is gone, and the Phase-4
  adaptivity claim (+0.64 to +0.87 growing with sigma) goes with it, now flat at +0.08/+0.07.
- **Real: survives, 8.4x the seed spread.** 0.344 +- 0.003 no-cond against 0.426 +- 0.014 cov,
  replicating the published +0.076 gap as +0.083.

**Pillar 1 is reshaped, not withdrawn.** The claim becomes "no measurable synthetic benefit,
real-data harm", which no longer rests on an artifact. Because that is now the only surviving
quantitative claim and it is measured as distance from TauREx-NS, **A5 is promoted**: it is no
longer a hardening task, it is the foundation the result stands on.

Outstanding: the +sigma row is withheld. `noisecond_rad_sigma` is a third failed MPS run
(1.3729 where its family trains to 1.13) and needs 2 configs x 3 seeds, about 5.5 h, if the
three-arm table is kept.

---

## Angshuman, Track 1: the validation ladder

### A1. Tier 1 injections into real noise, with SBC, TARP and HPD coverage. MOSTLY DONE.

**Status 2026-11-08.** Built and run. `mirage/eval/sbc.py` validated against five known-wrong
samplers before use (P6-D5), `scripts/inject_real_noise.py` builds the sets (P6-D6),
`scripts/sbc_report.py` scores them. **14 cells complete** across 3 training seeds, 2 arms, 2
tiers and 3 targets (P6-D8, P6-D11). G3 is closed. The seed repeat P6-D8 owed is discharged,
and it corrected P6-D8's conditioning claim, see G3b. **The one remaining item is G13 below**,
which is a build rather than a run.

The core build, and the thing that makes v2 publishable. Draw theta from the prior,
simulate with TauREx, inject into real OOT-derived noise held out by frame index, run the
flow, record rank statistics. Labels come from the prior draw, so no transport can game
them. This is the non-circular calibration check (G3) and the large benchmark (G5) in one.

Read Talts et al. on SBC and Lemos et al. on TARP. The module landed as `mirage/eval/sbc.py`
rather than `scripts/sbc.py`, to sit beside `mirage/eval/coverage.py`.

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

**Reassigned 2026-11-08. Framing B is locked**, sim-to-real negative results at TMLR
(`06_v2_post_rejection_plan.md`, option B). That decision is what makes this section
load-bearing. Under framing B the paper rests on exactly two findings, the conditioning
result and the null translation result, and the second one is Track 2's. It is no longer a
paragraph that can be cut without consequence.

**Status going in.** Nothing on Track 2 has moved since 2026-08-18, and no
`phase-6-track-2` branch exists. **V1 is finished and Track 1 did it** (P6-D3, 2026-11-07,
translation degrades the posterior rather than merely mismatching the noise). The items
below are what framing B still needs from this track.

**Dates are set by the Track 1 critical path, not by preference.** Writing starts when A5
returns. Anything landing after 2026-11-30 lands after the section that would have cited it.
Each item below carries an acceptance test, so "done" is checkable rather than claimed, and a
fallback stating what happens to the paper if it does not arrive.

---

### Required. Framing B cannot be published without these.

#### V7. Rebuild the CycleGAN real side on all three real spectra. Due 2026-11-16. (NEW)

The current real side is **one** reduced WASP-39b spectrum, bootstrap-expanded to 500 samples
with 1 percent photon-noise jitter. Under v1's framing that was a weak spot in a minor
section. Under framing B it is fatal, because the single sentence "your translator failed
because you trained it on one spectrum jittered 500 ways" removes a pillar at no cost to the
reviewer. Three real reduced spectra now exist, WASP-39b, WASP-96b and K2-18b.

Rebuild the real domain from all three, retrain, and report whether the null survives.

*Acceptance.* The bootstrap expansion is retired from the real side, the run is reproducible
from a committed config, and the decision log carries one sentence stating whether the null
holds, weakens or reverses.

*Fallback if late.* The CycleGAN result is cut from the paper rather than defended, and
framing B drops to a single pillar. Say so before the writing starts, not during it.

#### V2. The cheap baselines any learned translator must beat. Due 2026-11-16. (closes G8)

Three arms, all inexpensive, none needing new data. **Per-bin affine**. **Per-bin quantile
mapping**, the standard bias-correction baseline in climate downscaling. **Classifier
reweighting**, where a classifier separating simulated from real-OOT-derived noise supplies
density-ratio weights, OmniFold-style, about a day.

This is the first thing a sceptical reviewer asks for, and without it the negative has no
comparison class. "CycleGAN did not help" is a weak claim. "Affine and quantile mapping beat
no adaptation, which beats CycleGAN, on both distributional and inferential metrics" is a
sharp one, and it is the same compute.

**Report both MMD values, not one.** v1 quotes only the translated 1.22. It means nothing
without the 1.15 domain-randomized baseline beside it. Fix that here.

*Acceptance.* A table with four rows, no adaptation, affine, quantile mapping, classifier
reweighting, plus CycleGAN, scored on MMD to real **and** posterior JS to NS. Both MMD values
printed.

*Fallback if late.* Track 1 absorbs it, one to two days, and Track 2's contribution to the
paper reduces to V6.

#### V6. Related work, and relabel the old script. Due 2026-11-23. (G14, G15)

Own the related-work section. Swierc et al. (2410.16347) on unsupervised domain adaptation for
NPE in strong lensing is the right comparison class, and it aligns representations rather than
translating data, which is the distinction the paper turns on. Then diffusion and Schrodinger
bridges, "Knobs and dials II", and Roy-Perez et al. on pipeline-level effects.

Then relabel `run_cyclegan_ablation.py` as a robustness probe, so it is never cited as the
posterior-level translation result.

*Acceptance.* Draft prose in the shared document, every citation checked against the primary
source rather than a secondary summary, and the script renamed with its docstring corrected.

#### V1b. Score the translated arm on the A1 injection ladder. Due 2026-11-23. (NEW)

**This requirement did not exist before 2026-11-08 and it comes from P6-D8.** A1 found that
conditioning HELPS against prior-draw labels while P6-D2 found it HURTS against the
TauREx-NS anchor. The ranking flips with the evaluator. P6-D3's translation verdict was
measured against the NS anchor only, so it now carries the same exposure, and a reviewer who
reads both results will ask.

Run the translated arm through the same ladder, SBC plus TARP plus HPD against injection
labels. **The machinery already exists**, `scripts/sbc_report.py` with
`--tier data/inject/wasp39_tier1.hdf`. This is a run, not a build.

*Acceptance.* The translated arm added as a row to the A1 table, and one sentence in the
decision log saying whether P6-D3 is evaluator-independent or not.

---

### Wanted. Real upside, not required for submission.

#### V5. Tier 2, multiple reductions of the same visit. Verification due 2026-11-16, result due 2026-11-30. (G5, H7)

Each pair of reductions shares the same astrophysical truth, so pipeline shift is isolated
from atmosphere. K2-18b (Schmidt et al. Zenodo 14735689, Luque et al. Zenodo 15685551) and
WASP-39b's four-mode Carter et al. reduction (Zenodo 10.5281/zenodo.10161743).

H7 is the question. Does posterior JS between two reductions of one visit exceed the JS change
from any noise-model intervention made anywhere in this project? If yes, the field is tuning
the wrong thing, and that is the largest single claim available to either track.

**Verify the data first.** The literature audit flagged that it never opened these records.
The 2026-11-16 deliverable is a one-line yes or no on whether each record contains
per-reduction spectra in a usable format. Do not plan around the dataset before that answer
exists.

#### V3. H1 once, as a mechanism check, then stop. Due 2026-11-12. (closes G7)

Rebuild domain A restricted to one planet's parameter neighbourhood, retrain once, see whether
the 0.308 per-bin-sigma mean shift collapses. If it does, the negative gains a mechanism. If
not, remove the hypothesis from the decision log rather than leaving it standing. About 90
minutes on MPS, use `--ckpt-dir`. **Then retire CycleGAN and keep it as a negative control.**

---

### Cut unless the required set is in.

#### V4. H2, system-identified randomization.

Fit the kernel hyperparameters to the real OOT covariance, then randomize around the fit
instead of over a broad hand-chosen range. Direct precedent in gravitational waves, where
Wildberger et al. fitted a parametric PSD and got accurate inference across 37 real events.
Three arms scored by both MMD to real and posterior JS to NS, broad randomization,
system-identified randomization, raw noise-bank resampling.

The strongest translation-side hypothesis on the list, and the one most likely to turn a
negative into a positive. It is parked rather than dropped because it cannot be done well in
the time the required items take, and a half-run V4 is worth less than a finished V2.

**If it is attempted, write the prediction down before running it.** Predict that it improves
MMD. Then test whether it also improves inference. If MMD improves and inference does not,
that is the dissociation thesis demonstrated a second time.

---

### What Track 2 owns in the manuscript

Under framing B, one of the two results sections, the related-work section, and the baseline
table. That is a real share of a real paper. It is contingent on V7, V2 and V6 landing, since
a section with no finished experiment behind it cannot be written by anyone.


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

Revised 2026-11-08.

1. **Z1** (arXiv v2) **last, not first** (P6-D10). It carries **four** corrections: the chi2
   floor mismatch, the OT calibration, the conditioning result, and the P6-D8 reversal of that
   conditioning result under injection labels. Posted once, with the finished manuscript.
   P6-D12 adds no fifth correction. It supplies the third one's mechanism, which is what
   makes it safe to state in print rather than merely withdraw.
2. ~~**Z2 / H9**~~ **FULLY DONE 2026-11-08 (P6-D12).** The six sigma retrains finished on
   CUDA, 3 seeds per family, so all eighteen H9 arms are device-matched and the published
   conditioning ordering is now shown to be a device artifact in both families, spread
   collapsing 20x on radius and 114x on abc and reversing sign. No GPU work remains on H9.
   The remainder is an evaluation pass, coverage and JS-to-NS on the six sigma checkpoints,
   to repopulate the Phase-4 sigma row, plus regenerating fig5 from `configs/h9_*`.
3. ~~**A1** first result~~ DONE 2026-11-08 (P6-D8). Continuing with the s43 and s44 seed
   repeat, which gates the conditioning ranking, and the two extra targets.
4. ~~**V1**~~ DONE 2026-11-07 (P6-D3), by Track 1.
5. **A5 is now gating, not hardening.** P6-D2 and P6-D8 reach opposite conclusions about
   conditioning, and which one the paper states is decided entirely by whether TauREx-NS is a
   trustworthy reference. **A6 cannot be written before A5 returns.** This is a change from the
   2026-11-05 ordering, where A5 sat below H9 on the grounds that it could not kill a claim.
   It can now pick between two of them.
6. **V7**, **V2** by 2026-11-16. **V6**, **V1b** by 2026-11-23. **V5** verification by
   2026-11-16, result by 2026-11-30.
7. **A2** once the A1 labels are consolidated. **A3**, **A4** and the sigma-range retrain as
   time allows, in that order, since the retrain is the actual fix that P6-D7 and P6-D8
   jointly point at.
8. Writing starts after A5 and the required Track 2 set. Either one missing changes what the
   paper is allowed to claim.

## What not to do

- Do not rebuild coverage against the reference the map transports onto.
- Do not restore the amortization claim while an NS anchor is paid per target.
- Do not quote an interval from an arm whose ESS the same table does not print.
- Do not build more CycleGAN variants, more radius prose, or more synthetic-only coverage
  tables. **V7 is not an exception to this.** Rebuilding the real domain from three real
  spectra is a correction to an invalid training set, not a new variant, and it is the
  last CycleGAN run the project does.
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
