# v2 plan — post Sim2Science rejection

Status as of 2026-09-30. Supersedes the Sim2Science draft. Read this before touching the paper again.

## Outcome

Rejected at NeurIPS 2026 Workshop Sim2Sci (submission #60). Ratings 4 / 2 / 2, average 2.67.
The single accept (dQk2) declared confidence 2 and unfamiliarity with JWST retrieval. Both
rejects declared confidence 3 and converged independently on the same defect.

v1 is on arXiv with a stated caveat in Section 5. It is an honest preliminary preprint, not a
correction target. v2 is a rebuild, not a patch.

## What the reviews actually found

Three things, in order of severity. Everything else in the reviews is downstream of these.

### 1. The calibration validation is circular

Equation 4 constrains the pushforward of the flow proposal to match the nested-sampling
posterior. The paper then validates by checking whether nested-sampling values fall inside
intervals built from those transported samples. They do, because the map was constructed to
put them there.

The 7/7 coverage measures how well the optimal-transport problem was solved. It says nothing
about whether the flow learned anything about the real spectrum. As T46u put it, the network
could be discarded and p_NS sampled directly for the same table.

This is correct and it is fatal to the method claim as written. It also kills two secondary
claims:

- **Amortization.** A nested-sampling run is paid per target, so nothing is amortized end to
  end.
- **The RoPE comparison.** RoPE learns a transport on a calibration set and reuses it on new
  observations. Here the map is refit per target against that target's own reference. Not the
  same operation, and "inspired by" does not cover the gap.

### 2. The chi-squared numbers do not reconcile

Reported for the same collapse: 301 (abstract, flow), 38.44 (Table 1, NS physical with radius
fixed), and "near 2.5" (Section 3 text, unphysical cold-flat corner). Three numbers, one
phenomenon, no stated relationship between them.

Worse, the calibrated 0.06 sits **below** the nested-sampling best fit of 0.76 on the same
model and data. If the transported samples followed p_NS this is impossible. Either they do
not follow p_NS, which breaks the coverage claim, or 0.06 reflects per-bin uncertainties
inflated relative to the residuals. A reduced chi-squared of 0.06 over 47 bins puts residuals
near 0.24 sigma, which is a red flag rather than a result.

Resolve this before anything else is written. One of the reported numbers is wrong.

### 3. The central finding is not novel

Verified directly: `Project/fm4ar/fm4ar/datasets/vasist_2023/prior.py:29` contains
`[0.9, 2.0, "R_P", r"$R_P$"]`. Planet radius is parameter two in the standard Vasist 2023 /
Gebhard 2025 prior, beside `log_g`. The backbone this work builds on already infers it.

The transit-side version of the same point, the radius / reference-pressure degeneracy in
transmission spectroscopy, is textbook in the retrieval literature.

Stated plainly, the finding is that a standard parameter was omitted from theta, the fit
broke, and restoring it fixed the fit. That is a real debugging result. It is not a discovery,
and "identifiability, not misspecification" is a framing layered on top of it. T46u is right
that a nuisance parameter fixed to the wrong value is a misspecified forward model.

Do not carry this framing into v2 unchanged.

## The framing decision — settle this first

No code should be written until this is chosen. Each option implies a different venue and a
different set of experiments.

**A. Real-data application and failure characterisation.** Venue: A&A, AJ, MNRAS. Centre the
three real JWST targets, two instruments, the end-to-end self-reduction from raw MAST, and the
nested-sampling cross-checks. The calibration is reported honestly as a per-target refit, not
as a method contribution. Highest probability of publication, and the work largely supports it
today once the circular claim is cut.

**B. Sim-to-real negative results.** Venue: TMLR, which evaluates correctness and rigour rather
than novelty. Thesis: SBI robustness techniques that win on synthetic misspecification
benchmarks fail to transfer to real instrument noise. Evidence already in hand is covariance
conditioning winning on synthetic ABC (coverage 0.587 to 0.687) and not transferring to real,
plus the null CycleGAN result. This is the strongest genuinely ML-shaped claim left standing.

**C. Build a real amortized calibration.** Venue: ICML, NeurIPS. Learn one transport on a
calibration set and apply it to new targets with no per-target nested sampling, which is what
RoPE actually does. This is new research, not a rewrite, and it is the only path that restores
a method contribution.

Recommendation: A or B. C only if there is appetite for a further research cycle rather than a
writeup cycle.

---

## Track 1 (Angshuman)

Core method, real data, nested sampling, training, calibration.

### Blocking — do these before any writing

1. **Reconcile the chi-squared numbers.** Trace 301, 38.44, 2.5, 0.76 and 0.06 back to source
   (`scripts/benchmark_table.py`, `scripts/real_ess.py`, `data/real_ess/*.npz`). Establish which
   error model each uses. Determine whether 0.06 comes from inflated per-bin uncertainties. Write
   down one coherent narrative in which every number has a stated meaning, or drop the numbers
   that cannot be defended.

2. **Replace the circular validation.** Coverage against the transport target is dead. Options,
   roughly in order of strength:
   - Simulation-based calibration on synthetic data with known ground truth, where the map has
     not seen the truth.
   - Hold out a subset of parameters from the transport and validate on those.
   - Report recovery against published literature values, which the map does not target. The
     radii (1.227 vs 1.28, 1.196 vs 1.20, 0.232 vs 0.235) are already non-circular evidence and
     are the strongest real-data result in the paper.

3. **Decide the honest role of the OT step.** Either drop it, or keep it and describe it as a
   per-target refit with no amortization claim, or build the amortized version under option C.
   No middle position survives review.

### Required if the paper keeps a method claim

4. **Describe the OT map.** Solver, sample sizes, out-of-sample behaviour. T46u noted it is
   never specified.

5. **Strengthen the nested-sampling reference.** 300 live points (`scripts/taurex_retrieve.py:150`)
   was called light. Rerun the anchors at higher live-point counts and confirm the reference is
   stable.

6. **Repeated runs and dispersion.** dQk2 W2. Multiple seeds, report mean and spread rather than
   point values. Currently every headline number is a single run.

7. **Compute accounting for calibration.** dQk2 W1. State the per-target cost of the
   nested-sampling anchor explicitly so the overhead is visible rather than implied.

### Claim repairs

8. Drop "exhaustively" (three levers is not exhaustive) and state the misspecification
   conclusion as ruling out the levers tested, per dQk2 W3.

9. Drop "unchanged" across targets. Appendix A already shows different priors for K2-18b and
   per-target wavelength grids.

10. Rewrite the identifiability framing to acknowledge that the reference implementations infer
    R_P, and that a nuisance parameter fixed to the wrong value is a form of misspecification.
    The defensible version is narrower and about what happens when it is omitted in a transit
    setup, not a general claim about identifiability versus fidelity.

### Figures and data

11. **Fig 2 fit quality.** Viae reported the inferred spectra look like poor fits with several
    features off, and that the K2-18b radius is not well recovered. Determine whether the fits
    are genuinely poor or the plotting is misleading. If genuinely poor, that is a result and
    must be reported, not restyled.

12. Both figures are unreadable at print size. Rebuild at publication resolution with larger
    axis labels.

## Track 2 (Vedanth)

CycleGAN and learned domain translation. Scope is smaller than Track 1, but under framing B the
null translation result becomes one of two load-bearing findings rather than a single paragraph.

1. **Rebuild the CycleGAN real side.** The current unpaired real set is one reduced WASP-39b
   spectrum bootstrap-expanded with 1% photon-noise jitter to 500 samples. Viae flagged this as
   near-certain overfitting, and the objection is sound. Three real reduced spectra now exist
   (WASP-39b, WASP-96b, K2-18b). Rebuild the real side using all of them and report whether the
   null result survives.

2. **Run the posterior ablation that was never completed.** The four-condition posterior
   comparison has been a placeholder since Component 5. Until it exists, "translation did not
   improve on structured randomization" rests on MMD alone (1.22 translated versus 1.15
   domain-randomized), which is a distributional statistic and not a retrieval result.

3. **Report both MMD values, not one.** v1 quotes only 1.22. The comparison only means something
   alongside the 1.15 baseline.

4. **Own the negative honestly.** If the rebuilt ablation still shows no benefit, that is a
   publishable negative under framing B and should be written as a designed experiment rather
   than a footnote. If it shows a benefit, that changes the paper and Track 1 needs to know
   early.

5. **Related work on domain translation for spectra.** Refresh and own this section. v1 cites
   CycleGAN and MMD only.

## Shared

- Notation: `R_p` in Section 2.1 versus `r_p` in Equation 2. Pick one.
- Table 2 reports K2-18b radius as 0.24, Fig 2 reports 0.235. Reconcile.
- Sentences flagged as unparseable: the abstract "ruling levers out nested-sampling-first", the
  same phrasing in the intro, l.85 "integrated by drawing theta from Gaussian sample to t = 1",
  l.136 "making the provenance a real axis of the breadth claim".
- Viae found the paper reads as a report on what was explored rather than a focused presentation
  of method and results. Restructure toward a claim and its evidence.
- Remove the editorial notes still present in the checklist.

## Do not rebuild

- Coverage measured against the nested-sampling reference the map transports onto. It cannot be
  rescued by better presentation, and it is the reason for the rejection.
- The claim that the method is amortized while a nested-sampling anchor is required per target.
- The RoPE comparison, unless option C is actually built.
