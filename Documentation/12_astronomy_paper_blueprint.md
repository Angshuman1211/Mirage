# Astronomy paper blueprint, reduction systematics against a forward-model yardstick

Blueprint only, cover-in-this-order points rather than prose, because Overleaf is canonical
and the prose is yours.

Built on V5 / H7 (P6-D32), A5 / H8 (P6-D19) and A4 (P6-D23), with the prior-work gate in
`11_prior_work_reduction_audit.md` passed on 2026-11-22.

**Target.** RASTI primary. AJ secondary and harder, because Schmidt et al. is accepted there
on the same planet and AJ referees will know it cold. Do not aim at MNRAS main, where
single-visit scope invites "show me this generalises".

**Separate from the v2 manuscript. Do not merge them** (`07_state_gaps_and_research_agenda.md`
line 658, standing instruction).

---

## 0. READ THIS FIRST. One defect in the headline, and the build that fixes it

The claim as currently written in P6-D32 is that reduction choice moves the spectrum
**6.4x to 9.5x** more than the difference between the two standard radiative transfer codes.

**That ratio divides a number measured on K2-18b by a number measured on WASP-96b.**

| piece | target | instrument | value | median single-bin error |
|---|---|---|---|---|
| reduction shape RMS, V5 / H7 | K2-18b | NIRSpec G395H | 57.3 to 84.3 ppm | 51.8 ppm |
| code shape residual, A5 / H8 | WASP-96b | NIRISS SOSS | 8.9 ppm | 192.6 ppm |

Different planet, different instrument, different error bar, different wavelength range. A
referee kills the ratio in one sentence, and they should.

**In error-bar units the gap is actually larger**, 1.10 to 1.66 sigma against 0.046 sigma, so
about 24x to 36x. The conclusion survives either way. The *comparison* does not.

**THE ONE BUILD THIS PAPER NEEDS: run A5 / H8 on K2-18b.** The machinery exists and is
target-agnostic in shape, `scripts/h8_dump_atmosphere.py` then `scripts/h8_prt_forward.py`
then `scripts/h8_compare.py`, with `h8_control.py` for the line-list-only control. Dump
TauREx's own atmosphere for a K2-18b posterior draw, integrate petitRADTRANS through those
exact arrays, bin with TauREx's own `FluxBinner`, and report the offset and shape residual on
the same 60-bin 2.80 to 5.10 um grid V5 / H7 uses.

Then the ratio is same-planet, same-instrument, same grid, same error bar, and the paper has a
spine. Until that run exists, this paper should not be drafted.

Second-order but worth fixing while in there: `configs/v5_k218_h7/spectral_shift.json` records
`canonical_dir` as `data\v5_k218\canonical` with Windows separators, which will not resolve on
anyone else's machine.

---

## 1. Title and abstract

Cover in this order.

- The measurement, in spectrum space and in ppm, not in retrieved abundances.
- The yardstick it is divided by, named as a matched-physics code comparison.
- The ratio, same-target.
- The error-floor consequence.
- One sentence conceding this quantifies what the literature asserts qualitatively, with
  Roy-Perez and Schmidt named. Put this *in the abstract*, not buried in discussion. It is
  what stops a referee feeling misled.

Title should lead on the yardstick, since that is the unclaimed part. Something along the
lines of reduction systematics measured against forward-model difference. Avoid "multiverse",
which is Schmidt's territory and invites the comparison you will lose on scale.

## 2. Introduction

- JWST transmission spectroscopy now precision-limited by systematics rather than photon noise.
- Reduction choice is known to matter. Cite Schleich 2025 (three WASP-39b spectra, posterior
  classes), Roy-Perez 2026 (six spectra, parameters shifting beyond one order of magnitude),
  Schmidt 2025 (60 combinations, 256 retrievals on K2-18b). Be generous here. These papers
  establish your premise and you are not competing with them.
- The gap: every one of those is measured in retrieved-parameter space, which folds the
  reduction difference together with the retrieval setup that reads it. State that a
  parameter-space shift cannot be compared across studies with different forward models,
  priors and chemistry, and that a spectrum-space ppm figure can.
- The second gap: "comparable to modelling uncertainty" has been asserted (Roy-Perez, against
  cloud parametrisation) but never measured against a *radiative transfer code* difference at
  matched physics.
- What this paper does, three bullets, matching sections 4, 5, 6.
- State plainly that this is one visit of one planet. Own the scope in the intro rather than
  conceding it under pressure in the discussion.

## 3. Data

- K2-18b NIRSpec G395H, the single transit. Seven independent reductions from two teams,
  Schmidt et al. four (exoTEDRF, Eureka! Ahrer, Eureka! Bell, FIREFLy) and Luque et al. three
  (exoTEDRF, Eureka!, SPARTA). Cite the Zenodo records with DOIs, 14735689 and 15685551, and
  resolve both before submission.
- Say that both teams reduced the same photon stream. This is the whole design.
- The shared 60-bin grid, 2.80 to 5.10 um, and the NIRISS baseline treatment below 2.75 um.
- Resolution differences between the R100 and R200 sets and how they are reconciled onto the
  shared grid. A referee will ask whether the shared-grid interpolation itself creates shift.
  **Pre-empt it with a null test**: interpolate one reduction onto the grid by two different
  routes and show the induced difference is small against 57 ppm.

## 4. Result 1, reduction shift in spectrum space

- The 21 pairwise comparisons. Offsets and shape RMS, in ppm and in units of the median
  single-bin error (51.8 ppm), both.
- Lead with the structure, not the offset. A constant offset is partly absorbable by a fitted
  nuisance offset, which Schmidt already fits with a -100 to +100 ppm prior. **Shape RMS is
  the part that cannot be absorbed**, and it is 57.3 to 84.3 ppm, at or above one error bar.
  Say this explicitly, because it is what makes the result matter rather than being a
  calibration offset anyone already handles.
- The cross-team pair (Schmidt exoTEDRF vs Luque Eureka!, 42.3 ppm, 60.0 ppm) carries more
  weight than the within-team pairs. Two independent groups, same data. Promote it.
- The exoTEDRF-sits-above-Eureka! pattern holding across both teams, since a reproducible
  sign is stronger than a scatter.
- Table of all 21 pairs goes in an appendix. Main text gets the eight-row summary already in
  P6-D32 plus the max and median.

## 5. Result 2, the forward-model yardstick

- The design, and lead with the design because it is the contribution. The atmosphere is not
  recomputed. TauREx's own pressure, temperature, per-gas VMR and mean-molecular-weight
  profiles plus the reference radius, gravity and pressure triple are dumped and
  petitRADTRANS integrates through those exact arrays. Mean molecular weights agree to 1.1e-7
  relative. Binning is TauREx's own `FluxBinner` on both sides. Line lists matched on the DOI
  inside each opacity file, not the filename.
- State why each of those matters in one clause. The reader must understand that the only
  remaining difference is the radiative transfer and the opacity treatment.
- The line-list-only control moving the spectrum 4.9x more on shape. **This is what makes the
  null a measurement rather than a blind spot** and it is the most referee-proof sentence in
  the paper. Do not bury it.
- The K2-18b numbers from the new run. Quote the WASP-96b numbers (12.9 ppm offset, 8.9 ppm
  shape, 0.067 and 0.046 error bars) as a second target, which strengthens rather than
  weakens once the K2-18b run exists.
- Concede that Schmidt reports code-choice robustness qualitatively and say that this
  quantifies it. Cite their "robust to the choice of retrieval code" verbatim.

## 6. Result 3, the error floor the data supports

- A4's sweep, 0.5 to 10 percent across three targets, 63 cells, all by reweighting stored
  forwards with no new TauREx call. Say that, because cheapness is a virtue here.
- Best-fit chi2/N reaches 1 at a floor of 0.67 to 1.37 percent by target, so a 5 percent floor
  is 3.6 to 7.5 times what the data supports, and at 5 percent chi2/N is 0.03 to 0.20.
- Connect it to section 4 as the payoff, not as a separate topic. A floor set loosely enough
  absorbs a 57 ppm shape shift and makes reductions look consistent when they are not. That
  sentence is the paper's argument for why anyone should care.
- **Cite Schleich's Table D.2 here.** Three reductions of one spectrum, and they disagree
  about whether to inflate error bars at all, one not inflating, one inflating to reduced
  chi2 of 1, one unspecified. A published paper recorded the inconsistency and left it. That
  is your motivation handed over.
- Note Schmidt's three different ad hoc schemes (Allan variance, jitter term, white-noise
  multiplier) with no stated supported floor.
- Keep the ESS material OUT of this paper. It belongs to the v2 manuscript and it is the
  weakest flank in the whole project.

## 7. Discussion, and the three attacks to answer in the text

Answer each in a named paragraph rather than waiting for review.

1. **"Schmidt already did this."** Answer: in parameter space, on four reductions, with no ppm
   figure and no code yardstick. You add three reductions, a spectrum-space metric and a
   denominator. Concede their scale openly, 256 retrievals and 50 CPU-years, and say your
   contribution is orthogonal rather than larger.
2. **"Roy-Perez already said reduction is comparable to modelling choice."** Answer: against
   cloud-opacity parametrisation, which is an assumption inside one code, not a different
   radiative transfer implementation. Quote their sentence and draw the distinction.
3. **"Your own numbers can be read the other way."** The Schmidt CH4-significance arithmetic
   in `11_prior_work_reduction_audit.md` section 3 gives comparable code and reduction gaps.
   Answer: detection significance is not spectral distance, their two codes differ in
   chemistry, clouds, stellar model, priors and opacity sampling simultaneously, and a
   matched-physics comparison is the only way to isolate the code. **Write this paragraph
   before a referee writes it for you.**

- Limitation, stated not hedged: one visit, one planet, one instrument mode. P6-D14 found the
  WASP-39b record does not hold multiple reductions of a single visit, so the obvious
  extension is blocked by data availability, not by effort. Say that.

## 8. Figures

- Fig 1, the seven reductions on the shared grid with the reference residuals. Exists in
  substance, needs rebuilding at publication resolution.
- Fig 2, offset and shape RMS for all 21 pairs, in error-bar units, with the cross-team pairs
  marked distinctly.
- Fig 3, the code comparison. TauREx against petitRADTRANS residual with the line-list-only
  control overplotted, same axes as Fig 2 so the reader sees the ratio without arithmetic.
  **This is the paper's one figure.** Build it to carry the argument alone.
- Fig 4, the floor sweep. `figures/fig_a4_floor.png` exists, 2026-11-16.
- Carry forward the v1 figure lesson from P5-D7 and the T46u note. Fonts enlarged, dpi at
  least 200, one shared legend rather than per-panel legends overlapping data.

## 9. Authorship and sequencing

- V5 / H7 is Vedanth's run (Track 2, filed under *Wanted*). A5 and A4 are yours. Settle the
  author order before drafting, not after.
- Pulling V5 / H7 out does **not** damage the v2 manuscript. Track 2's share there rests on
  V7, V2 and V6 (`08_work_split_v2.md`, "What Track 2 owns in the manuscript").
- Everything except the K2-18b pRT run is already computed, so this is a writing project with
  one bounded build in front of it.

## 10. Order of operations

1. Run A5 / H8 on K2-18b. Nothing else starts until the ratio is same-target.
2. Re-verify the two Zenodo DOIs and the Schmidt citation once AJ assigns a volume.
3. Re-read Roy-Perez Sections 4.2 to 5 properly. Only the front half is verified, and the
   back half is where their cloud-parametrisation comparison lives, which is the claim this
   paper positions against.
4. Build Fig 3, then write sections 5 and 4, then 6, then the intro last.
