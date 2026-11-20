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
and it corrected P6-D8's conditioning claim, see G3b. **G13, the one remaining item, is DONE
2026-11-16 (P6-D25), so A1 IS COMPLETE.**

The core build, and the thing that makes v2 publishable. Draw theta from the prior,
simulate with TauREx, inject into real OOT-derived noise held out by frame index, run the
flow, record rank statistics. Labels come from the prior draw, so no transport can game
them. This is the non-circular calibration check (G3) and the large benchmark (G5) in one.

Read Talts et al. on SBC and Lemos et al. on TARP. The module landed as `mirage/eval/sbc.py`
rather than `scripts/sbc.py`, to sit beside `mirage/eval/coverage.py`.

**Scope the depth-noise construction honestly.** G13 was load-bearing: OOT residuals are
flux noise, and Tier 1 needs depth noise. Done properly on **WASP-96b**, where the exoTEDRF
plus batman pipeline is mine end to end. For the other two, use the flux-noise approximation
and state it. Do not quietly treat all three alike.

**DONE 2026-11-16 (P6-D25), and the answer is split.** `scripts/depth_noise_wasp96.py`, 280
integrations, 90 bins, three arms including a white-noise control through the identical fit.
**The recipe as specified does not run**, because the transit is 2.66 h and the longest OOT
block is 2.27 h, so prayer-bead injection on the real time axis replaces injection into an
OOT segment. **Amplitude is fine.** The control recovers the published error bar to 0.99x,
and the depth arms' 0.67x and 0.83x are a prayer-bead bias predicted in closed form to 1.5
percent, not a property of the data. **Structure is not fine.** Depth noise correlates across
wavelength at 0.22-0.48 mean absolute off-diagonal against the flux route's 0.159, with a
31-37 ppm common-mode depth offset that is H10's free baseline offset showing up as a
measurement. Tier 1 on WASP-39b and K2-18b is unchanged and still stated as an approximation.

**Frame counts bite.** WASP-39b has 3,241 frames, the others 56 and 96. Frame-resampling
uncertainty is available on WASP-39b only. Say so.

### A2. Rebuild the OT calibration on injection labels. (revives G3 and part of G9)

RoPE works because it calibrates against "a small real-world calibration set of ground-truth
parameter measurements". Ours failed because the ground truth was the NS posterior we then
checked coverage against. With A1 in hand the labels are independent, and the same machinery
becomes legitimate. "Our validation was circular, here is the non-circular rebuild" is a far
better paper than "we removed the contribution".

**A2 SURVIVES H11, and H11 is the argument for it. Noted 2026-11-15 (P6-D21).** Read quickly,
P6-D21 looks like it kills A2 as well. It does not, and the distinction is worth stating
because it decides whether A2 is still worth building. H11 killed **"fit the calibration on
target A's anchor and apply it to target B"**. A2 proposes **"fit the calibration on target
B's own injection labels and apply it to target B"**, which needs no nested sampling at all,
only forward simulations and flow evaluations. H11 found the calibration is specific to both
the target and the individual training run, so a calibration has to be derived per target and
per trained model. **Injections are the only way to do that without an anchor**, so H11 makes
A2 the right next move rather than a dead one.

**The two measurements agree in sign, which is the encouraging part.** A1 found
under-confidence on injections, intervals too WIDE (P6-D8, coverage 0.851 and 0.806 against a
nominal 0.68). H11 finds the flow over-dispersed against the real-data anchor, deflation
k < 1 on every parameter of every target. Same direction, measured two independent ways.

**And one quantitative warning to carry into A2, so it is not a surprise.** Reading the A1
coverage as a Gaussian width inflation, 0.851 at the 68 percent interval implies roughly
1.4x too wide. H11's real-data deflations imply 1.5x to 13x, with **radius the worst at about
13x**. So an injection-fitted calibration looks likely to recover the generic inflation and to
**badly under-correct radius**, which is the one parameter the P3-D11 radius story depends on.
Treat "does the injection-fitted k match the real-data k, per parameter" as A2's first
measurement rather than its last. The Gaussian reading of a multi-parameter coverage number is
an estimate, not a measurement, so do not quote 1.4x as a result.

**DONE 2026-11-16 (P6-D22). All three targets.** `scripts/a2_inject_calibrate.py`, 250
injections per target at 256 draws each, Tier-1 noise, no nested sampling in the estimator.
Figure `figures/fig_a2_injection.png`, scores `data/a2/a2_scores.json`.

**The warning above was right, and it was written before the measurement.** Injections demand
far less deflation than the real anchor on every parameter of every target, median ratio
3.06x / 1.85x / 2.39x, and **radius is the worst or near-worst parameter on all three** at
3.83x, 5.02x and 5.41x. The predicted under-correction of the one parameter the P3-D11 story
depends on is exactly what happened.

**What A2 delivered, in one line each.**
- **It is the only anchor-free route that beats doing nothing on all three targets**, recovering
  9.5% / 18.4% / 16.6% of the circular arm's apparent JS gain. The H11 pooled transfer recovers
  -16.2% / 0.8% / 12.4%, so it is worse than nothing on WASP-39b.
- **It is a better importance proposal than the uncalibrated flow on all three** (disjoint 68%
  CIs) and better than the circular arm on two of three. WASP-96b at ESS 65.8 is the only arm
  anywhere in this project to clear `ot_calibrate.py`'s own usable threshold of 50.
- **It shows the anchor-fitted calibration over-narrows.** On K2-18b that arm has the best JS of
  any route (0.0283) and an ESS of 5.3 against 11.3 for no calibration, disjoint intervals. That
  corroborates P5-D5 functionally, independently of the structural pushforward argument.
- **It attributes the gap.** Tier 0 to Tier 1 to real anchor is 0.916 to 0.712 to 0.225, so
  noise amplitude accounts for under a fifth. With H8 at 0.07 sigma the forward model is out too.

**A2 does NOT repair the calibration and must not be written as if it does.** It recovers about
a seventh of the circular gain; the rest was never calibration. **G3 and G9 close by measurement,
not by repair.** Carry the four honest limits in P6-D22 into the manuscript, in particular that
A2's ESS edge is partly mechanical because its proposals are simply wider.

### A3. H5, the noise-proxy test. DONE 2026-11-17 (P6-D26). NEGATIVE, and it is the headline.

Condition on three inputs: Sigma-hat from OOT flux, Sigma-hat from pseudo-transit depth
noise, and published diagonal errors. If the conditioning paradox disappears with depth-noise
Sigma-hat, there is a mechanism. If it persists, the paradox is real and it is the headline.
Pairs naturally with A1, since A1 produces the depth-noise machinery.

**The third input now exists**, `data/inject/wasp96_depth_noise*.npz` from P6-D25, on the
model grid with no regridding step because `abc_rad_wasp96` was built from this very spectrum.
**This is the last open experiment on Track 1.**

**A PREDICTION ON THE RECORD, WRITTEN BEFORE RUNNING, per the H2 precedent.** The depth-noise
Sigma-hat differs from the flux Sigma-hat exactly and only in the direction H5 names, more
correlated across wavelength and with a much stronger leading mode, top eigenvalue 15.0 to
45.0 against 7.4. So if the paradox is a noise-proxy artifact it should move under this input.
**If it does not move, the paradox is real and it is the headline**, and that outcome is the
more likely one given P6-D17 already found the conditioning effect survives on calibration.

**Two cautions for whoever runs it.** The two depth pools bracket rather than agree, so run
both and report both; a result that holds on one pool only is not a result. And WASP-96b is
the only target with a depth-noise Sigma-hat, so A3 is a single-target test and cannot be
written as if it settles H5 on the other two.

**DONE 2026-11-17 (P6-D26), and THE PREDICTION ABOVE WAS RIGHT.** It did not move.
`scripts/a3_noise_proxy.py`, three seeds, seven Sigma-hat inputs. **The specified form could not
run**: no covariance-conditioned model exists on WASP-96b's grid, both its configs being
`n_oot_frames: 0`, so A3 became the sensitivity half, legitimate because the embedding whitens
and therefore carries correlation structure alone, which is exactly H5's axis. **Span of real
mean JS across every input 0.0039, seed sd 0.0137, paradox +0.0827, so the proxy explains at
most 4.7 percent, and 2.2 percent across the six realistic inputs.** The `real` input reproduces
P6-D2 exactly on all three seeds, `scale10` differs by 0.0000, and a forced re-run reproduced
every digit, so the harness is the published pipeline with one input swapped. **The
deflationary reading is refuted by measurement**: the embedding moves 96.9 percent of its own
norm between inputs but is 3.2 to 5.9 percent of the context norm, 15.1 percent even under the
impossible rank-1 input. The channel works and cannot matter.

**The caution above about running both pools is now genuinely satisfied, but only after a
repair.** An audit found the first version's Toeplitz depth probes retained 46 and 29 percent of
P6-D25's measured structure, and the OOT-block probe at 0.064 sat below the K=100 sampling floor
of 0.080, so it was a second white arm and the two pools were never really both tested. Carrying
the measured matrices over as principal submatrices fixes it, at 0.504 and 0.247 fed structure,
and **both pools now agree**, 0.425 and 0.427 against the real input's 0.426.
**This was the last open experiment on Track 1.**

### A4. H3, the floor sweep. (3 to 5 days)

Sweep 0.5, 1, 2, 3, 5, 7, 10 percent on all three targets. Plot ESS, best-fit chi2/N and JS
to NS against floor, with seed error bars. Cheap, mostly a loop over existing work, and it
converts the weight collapse into a stated property of the error budget.

**DONE 2026-11-16 (P6-D23). It was cheap as predicted, and it returned more than expected.**
63 cells, all by re-weighting stored H11 and A2 forwards, no TauREx call. Figure
`figures/fig_a4_floor.png`, cells `data/a4/a4_floor_sweep.json`.

**The weight collapse is now a stated property of the error budget, which is exactly what this
item promised.** Best-fit chi2/N reaches 1 at a floor of 0.67 to 1.37 percent depending on
target, so **the published 5 percent floor is 3.6 to 7.5 times what the data supports**, and at
5 percent chi2/N is 0.03 to 0.20. At the supported floor every arm on every target falls to
epsilon 0.06 to 1.52 percent, so **nothing in this project samples usefully at an honest error
budget**. Error bars are bootstrap over draws rather than over seeds, since WASP-96b and K2-18b
have one trained arm each.

**It also qualifies A2 above.** A2's efficiency advantage over no calibration holds only at
floors of 3 percent and up, and WASP-96b's ESS 65.8 is a peak at exactly the published floor.
A2's MEASUREMENT half has no floor in it and is untouched. See P6-D23 Results 3 and 4.

### A5. NS anchor stability and H8. (1 to 2 weeks wall clock, heaviest item)

**Part 1 DONE 2026-11-15 (P6-D18). G4 closes, and it took 50 minutes, not weeks.** Swept
150 / 300 / 600 / 1000 live points on WASP-96b (the published anchor was at 150, not the 300
this line assumed). Radius 68% width ratio 1.04x, radius mean drift 0.0002 RJ, logZ drift
0.213, no parameter width moving over 8 percent. The anchor is stable. The "1 to 2 weeks"
estimate below belongs entirely to part 2.

**Part 2 DONE 2026-11-15 (P6-D19). A5 IS COMPLETE AND A6 IS UNBLOCKED.** petitRADTRANS at
matched physics, matched line lists (matched on the DOI inside each opacity file, not the
filename) and matched binning, integrating TauREx's own atmosphere so only the radiative
transfer differs. The two codes agree to a 12.9 ppm offset and an 8.9 ppm shape residual
against a 192.6 ppm error bar, 0.067 and 0.046 of one error bar, and 0.00055 RJ in radius
against a 0.0057 RJ anchor width. A line-list-only control moves the spectrum 4.9x more on
shape, so the null is a measurement and not a blind spot. The full pRT nested-sampling retrieval
then confirmed it directly, in 23.7 minutes: mean JS between the two codes' posteriors 0.0116 nats
against a flow-to-NS JS of 0.352 to 0.428, every parameter inside 0.29 sigma. The reference is a
reference.

Both halves of A5 are therefore negative results, and the "1 to 2 weeks wall clock, heaviest
item" in the heading was wrong by about two orders of magnitude: part 1 took 50 minutes, part 2
about a working day including rebuilding petitRADTRANS from source because the installed copy
could not run (see P6-D19, x86_64 gfortran on an arm64 Mac).

Historical note, kept because it explains why this was promoted. **Promoted to gating,
2026-11-08 (P6-D9).** The line that stood here, "deprioritised below H9, it hardens numbers, it
cannot kill a claim", was withdrawn then. P6-D2 and P6-D8 reach opposite conclusions about
conditioning, and only the trustworthiness of the NS anchor decides which one the paper states,
so A5 picked between two claims rather than hardening either. That is now settled in favour of
the anchor being trustworthy, so the conditioning question is decided on the evidence rather
than left open.

### A6. Manuscript, inference side.

Drop "exhaustively" and "unchanged across targets". Rewrite the identifiability framing to
concede that the backbone's prior already infers R_P, that Welbanks and Madhusudhan (2019)
characterised the degeneracy, and that a nuisance parameter fixed wrong is misspecification.
State per-target NS cost so the lost amortization is visible (G9).

**DONE 2026-11-15 (P6-D20). Blueprint at `Documentation/A6_inference_blueprint.md`.** All four
asks covered. The two deletions are mechanical. The identifiability rewrite got **sharper**,
not weaker: Welbanks and Madhusudhan's "the degeneracy is harmless" result is conditional on
one of radius/reference-pressure being free, and the published MIRAGE setup freed **neither**
(verified in code at `taurex_forward.py:78` and `taurex_retrieve.py:115-127`), so the
contribution is that amortization turns a parameterisation choice into a structural commitment.

**The fourth ask returned a number that runs against the paper's own motivation.** A serial
per-target anchor costs **5 min 24 s**, so "NS is unaffordable" cannot motivate amortization
here, and with training at 0.59 h plus a still-required per-target anchor, **at three targets
amortization is strictly more expensive than nested sampling**. Own it as a negative result.

**H11 then closed the one conditional that was left. DONE 2026-11-15 (P6-D21), negative.** The
calibration does not transfer to a new target: 2.5 percent of the benefit it shows on the
target it was fitted to, against 73 percent across a mere change of random seed. See A2 above,
which H11 does **not** kill and in fact motivates.

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

> **Status 2026-11-18 (P6-D27): RUN AND CLOSED.** Trained 200 epochs on CUDA with
> `data/real_domain/real_domain_all3_abcniriss.npz` (`configs/cyclegan_v7_pooled3/cyclegan_best.pt`).
> Full-frame scoring yields a mixed verdict: WASP-39b PRISM split (MMD worse 0.952 vs 0.256, KL
> better 4.31 vs 5.76), WASP-96b holds (MMD 1.009 vs 0.258, KL 5.99 vs 2.08), K2-18b holds (MMD
> 1.001 vs 0.253, KL 3.14 vs 1.58). On MMD, CycleGAN is 3.7x to 4.0x worse than domain randomisation
> across all three targets. Translation actively degrades the noise distribution. Closes G8b.

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

> **Status 2026-11-19 (P6-D29): BOTH HALVES DONE AND CLOSED (CLOSES G8).** Unified evaluation completed
> (`scripts/v2_inferential_eval.py --n-planets 50 --n-samples 1000 --n-real 1000 --include-p6d3-ctrl`,
> persisted to `configs/v2_baselines_wasp39/v2_inferential.json`). All 3 cheap baselines trained to convergence on CUDA
> and verified leak-free. Exact P6-D3 reproduction verified (CPU `p6d3_ctrl_cov` matches CUDA `struct_rand_cov`
> within 0.003 coverage and JS 0.009 on real WASP-39b). Quantile mapping achieves nominal coverage
> (cov@68 = 0.677, cov@95 = 0.950), narrowest 68% credible intervals (width68 = 2.073, 6.3% sharper than
> domain randomisation 2.212), and lowest MMD (0.1056 vs 0.1682). Classifier reweighting achieves highest
> truth log-density (-7.61 nats). CycleGAN undercovers (cov@68 = 0.550, cov@95 = 0.843), collapses log-density
> (-17.65 vs -7.70), and has 4.6x higher MMD (0.782 vs 0.168). Both MMD and MMD_none (0.2883) reported.
> G8 is definitively closed.

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

> **Status 2026-11-15: DONE.** Draft prose and verified references in
> `Documentation/09_track2_v6_related_work.md`, with five corrections to our own docs found during
> the check. Script renamed to `scripts/probe_cyclegan_input_shift.py`, docstring rewritten.

Own the related-work section. Swierc et al. (2410.16347) on unsupervised domain adaptation for
NPE in strong lensing is the right comparison class, and it aligns representations rather than
translating data, which is the distinction the paper turns on. Then diffusion and Schrodinger
bridges, "Knobs and dials II", and Roy-Perez et al. on pipeline-level effects.

Then relabel `run_cyclegan_ablation.py` as a robustness probe, so it is never cited as the
posterior-level translation result.

*Acceptance.* Draft prose in the shared document, every citation checked against the primary
source rather than a secondary summary, and the script renamed with its docstring corrected.

#### V1b. Score the translated arm on the A1 injection ladder. Due 2026-11-23. (NEW)

> **Status 2026-11-15 (P6-D15): BLOCKED, AND IT IS A BUILD, NOT A RUN.** The translated arm is the
> 6-parameter ABC model while every Tier-1 set is 7-parameter radius, and its `model__best.pt` is
> not committed. Recommended route: a 6-parameter Tier-1 set via `inject_real_noise.py --build` in
> `mirage-taurex`, which needs Track 1's machine.

> **UNBLOCKED AND FIRST RESULT IN, 2026-11-16 (P6-D24).** Track 1 built the ladder. P6-D15 was
> right that it is a build, and it was bigger than either of us thought: ABC is a population of
> 73,044 distinct planets whose geometry sits outside theta, so the set resamples ABC's own joint
> theta-geometry rather than fixing a radius. `sbc_report.py` also had to stop hardcoding 7
> parameters. Set at `data/inject/wasp39_tier1_abc.hdf`.
>
> **The answer is that P6-D3 holds.** Translated arm cov@68 **0.483** against nominal 0.68, TARP
> dev 0.335. Untranslated ABC controls 0.668 and 0.686, dev 0.090 and 0.110, same injections.
> Translation damages calibration against ground-truth labels, not only against the NS anchor, so
> the P6-D8 evaluator-flip exposure is discharged and **G1b closes**.
>
> **Status 2026-11-18: WRITE-UP COMPLETE.** Full interpretation documented in
> `Documentation/10_track2_v1b_interpretation.md`. Unified 7-param radius vs 6-param ABC table,
> evaluator-independence proof, and manuscript draft text contrasting translation with noise
> conditioning completed and accepted.

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

> **Verification 2026-11-15 (P6-D14):** Schmidt K2-18b **yes** (up to 4 G395H reductions, NIRISS
> FIREFLy + exoTEDRF). Luque K2-18b **yes for NIRSpec and MIRI** (exoTEDRF / Eureka! / SPARTA),
> **no for NIRISS** (exoTEDRF only). Carter WASP-39b **no**: one uniform analysis of four visits,
> with only limb-darkening and dilution variants per visit. The H7 test should use K2-18b G395H.
>
> **Spectral Result 2026-11-18 (P6-D30): DELIVERED.** Extracted 7 independent G395H reductions
> from Zenodo (`scripts/v5_h7_pipeline_shift.py --spectral`, `configs/v5_k218_h7/spectral_shift.json`).
> Pairwise offsets reach +36 to +46 ppm (0.75-0.91σ) and shape RMS reaches 57 to 84 ppm (1.06-1.66σ).
> Pipeline-to-pipeline variation exceeds forward-model simulator discrepancy (TauREx vs petitRADTRANS,
> 8.9 ppm) by 6.4x to 9.5x. Canonical CSVs formatted for flow evaluation (`--run-flow`).

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

> **Status 2026-11-19 (P6-D28): RUN AND CLOSED.** `scripts/run_v3_h1.sh`: domain A restricted to
> the 2,000 ABC spectra nearest the real WASP-39b spectrum, retrained 200 epochs on CUDA
> (`configs/cyclegan_v3_h1_k2000`). Mean shift is 0.240 per-bin sigma (vs 0.308 previously),
> remaining 7.4x to 8.1x higher than unadapted sim (0.030) and domain randomisation (0.033).
> MMD is 0.318 (worse than unadapted sim 0.293, 1.87x worse than domain randomisation 0.170).
> The translation artifact does NOT collapse. H1 is refuted and struck from the research agenda;
> CycleGAN is retired as an active intervention (closes G7).

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
   The evaluation pass is also **DONE 2026-11-14 (P6-D17)**: six A1 cells on the radius sigma
   arms populate the Phase-4 sigma row, and against injection labels the three arms fall in
   conditioning order on every calibration statistic, which P6-D12's in-distribution null
   cannot see. **Z2 is fully closed.** Only regenerating fig5 from `configs/h9_*` remains.
3. ~~**A1**~~ **14 of 15 cells DONE 2026-11-08 (P6-D8, P6-D11).** The s43 and s44 seed repeat
   and both extra targets are discharged, and the repeat corrected P6-D8's conditioning claim
   (G3b). **COMPLETE 2026-11-16**: the sole remainder, **G13**, landed as P6-D25.
4. ~~**V1**~~ DONE 2026-11-07 (P6-D3), by Track 1.
5. **A5 is now gating, not hardening.** P6-D2 and P6-D8 reach opposite conclusions about
   conditioning, and which one the paper states is decided entirely by whether TauREx-NS is a
   trustworthy reference. **A6 UNBLOCKED 2026-11-15 (P6-D19): A5 returned and the anchor is
   trustworthy, on both the sampler side (part 1) and the forward-model side (part 2).** The
   line that follows is the history of why it was blocked, not a live constraint. This is a
   change from the
   2026-11-05 ordering, where A5 sat below H9 on the grounds that it could not kill a claim.
   It can now pick between two of them. **Part 1 landed 2026-11-15 (P6-D18) and does not open
   the gate**: it rules out "the anchor is under-converged noise" but not "the forward model is
   misspecified", and only H8 tests the latter. A sampler can converge tightly onto a wrong
   model. The gate is H8 alone now.
6. **V7**, **V2** by 2026-11-16. **V6**, **V1b** by 2026-11-23. **V5** verification by
   2026-11-16, result by 2026-11-30.
7. ~~**A2**~~ DONE 2026-11-16 (P6-D22). ~~**A4**~~ DONE 2026-11-16 (P6-D23). ~~**A3**~~ DONE
   2026-11-17 (P6-D26). **Every Track 1 experiment is now closed.** What remains is the
   sigma-range retrain, which is optional and is the fix P6-D7 and P6-D8 jointly point at,
   regenerating fig5, and the manuscript.
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

**Separately, A5 is CPU-bound, not GPU-bound. The rental advice below is WITHDRAWN (2026-11-15).**
The 498 single-core hours is a real published number (Exoformer, A&A 709 A17, 2026, Sect. 5.2 and
Table 3) but it is the cost of THEIR uniform-prior WASP-39b retrieval on one core, and it was wrong
to carry it over as ours. A5 part 1 measured the real figure: four WASP-96b nested-sampling runs at
150 to 1000 live points, in parallel on this laptop, finished in 50 minutes wall clock. H8's
petitRADTRANS retrieval is the slower one at roughly 2.4 hours, still on the laptop. Nothing here
needs renting.
