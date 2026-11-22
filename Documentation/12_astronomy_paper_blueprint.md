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

## 0. The gating run is DONE (P6-D37, 2026-11-22). Headline numbers to write to

A5 / H8 was rerun on K2-18b, so the ratio is now same planet, same instrument, same grid,
same error bar. **Drafting is unblocked.** The defect this section used to describe, a K2-18b
reduction shift divided by a WASP-96b code difference, is gone.

| | shape RMS (ppm) | vs the 50.0 ppm single-bin error |
|---|---|---|
| reduction choice, 21 pairs | 32.2 / **57.3** / 83.5 (min/median/max) | 0.71 / **1.11** / 1.66 |
| opacity tables, control | **29.2** | 0.58 |
| radiative transfer code | **2.5** (1.2 to 4.3 across 20 draws) | **0.050** |

**The headline is 22.9x at the median, 12.9x to 33.4x across the pairs.** Conservative worst
case, smallest reduction pair against largest code draw, still 7.5x. **Never quote 57.3 as a
floor**, it is the median, and that error is what P6-D36 was about.

**The three-level ordering is the paper's best sentence and it is new.** A deliberately crude
opacity downgrade moves the spectrum only half as much as changing which team reduced the same
photons. Reduction is 2.0x the line-list swap and 22.9x the code choice. Lead section 5 on it.

Supporting facts worth stating in the text. Matched-physics check, worst mean-molecular-weight
disagreement 1.66e-07 relative. Grid robustness, the same comparison on the observation's own
149-bin 0.852 to 5.174 um grid gives 2.3 ppm against 2.5 ppm, so the result is not a binning
artifact. Code-swap offset -3.3 ppm, equivalent radius shift -0.00013 RJ.

Still worth fixing while in there, `configs/v5_k218_h7/spectral_shift.json` records
`canonical_dir` as `data\v5_k218\canonical` with Windows separators.

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
- The shared grid, 55 log-spaced bins over 2.80 to 5.10 um with 53 populated per pair, and the
  NIRISS baseline treatment below 2.75 um.
- Resolution differences between the R100 and R200 sets and how they are reconciled onto the
  shared grid. A referee will ask whether the shared-grid interpolation itself creates shift.
  **Pre-empt it with a null test**: interpolate one reduction onto the grid by two different
  routes and show the induced difference is small against 57 ppm.

## 4. Result 1, reduction shift in spectrum space

- The 21 pairwise comparisons. Offsets and shape RMS, in ppm and in units of the median
  single-bin error (50.0 ppm across pairs), both. Quote the full spread, 32.2 to 83.5 ppm with
  a median of 57.3, not the median as if it were a floor.
- Lead with the structure, not the offset. A constant offset is partly absorbable by a fitted
  nuisance offset, which Schmidt already fits with a -100 to +100 ppm prior. **Shape RMS is
  the part that cannot be absorbed**, and its median across the 21 pairs is 57.3 ppm, 1.11
  times the single-bin error, with **16 of 21** pairs at or above one error bar. Say it that way
  rather than as a bare range, because a referee who sees a 32.2 ppm minimum quoted as 57.3
  will check the rest of the paper twice.
- **The cross-team pairs carry more weight than the within-team pairs, and they now carry the
  headline.** The single largest shape disagreement in the whole set is Schmidt exoTEDRF
  against Luque SPARTA at 83.5 ppm and 1.66 sigma, and Schmidt exoTEDRF against Luque Eureka!
  is 45.9 ppm offset and 74.4 ppm shape. Two independent groups, same photon stream, worse
  than anything within a team. This is a stronger result than what P6-D32 originally published.
- The exoTEDRF-sits-deeper-than-Eureka! pattern, which holds in **all six** such pairs,
  +46.3, +41.5, +45.9, +36.3, and -36.8 and -31.9 with the ordering reversed. A reproducible
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
- **Lead on the three-level ordering**, reduction 57.3 ppm against line list 29.2 ppm against
  code 2.5 ppm, all on one grid with one error bar. It isolates one factor at a time, which no
  prior paper does, and it is the cleanest form of the thesis.
- The line-list-only control moving the spectrum **11.7x** more on shape than the code swap.
  **This is what makes the null a measurement rather than a blind spot.** Do not bury it. Note
  the equivalent figure on WASP-96b was 4.9x, so the sensitivity check is stronger here.
- The K2-18b numbers, offset -3.3 ppm and shape 2.5 ppm at the posterior mean, 1.2 to 4.3 ppm
  across 20 draws. Quote the WASP-96b numbers (12.9 ppm offset, 8.9 ppm shape) as a second
  target. The two differ by roughly 3x on shape, which is expected from the scale-height
  difference and is worth one clause rather than hiding.
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

**Three of four are built (P6-D38). Every number in them is read from the committed
artifacts at plot time, never typed into the script.** That rule caught two wrong numbers
already, so keep it.

- **Fig 1, BLOCKED and the only one.** The seven reductions on the shared grid with reference
  residuals. Needs `data/v5_k218/canonical`, which is not on this machine. It is on Vedanth's.
  **Add the canonical CSVs to the P6-D35 file exchange.**
- **Fig 2, built** → `figures/fig_v5h7_pairs.png`, script `scripts/fig_v5h7_pairs.py`. All 21
  pairs, shape and offset in error-bar units, cross-team pairs coloured distinctly so the
  worst pair being cross-team reads without the caption. All 21 rather than a selected 8,
  which is the direct remedy for P6-D36.
- **Fig 3, built, and it is the paper's one figure** → `figures/fig_h8_three_level.png`,
  script `scripts/fig_h8_three_level.py`. Left, offset-removed residual against wavelength
  with the reduction envelope behind it, so the code trace sits visibly flat inside a band it
  is 23x smaller than. Right, the three levels as one ordering, 1.105 against 0.58 against
  0.050 sigma. It carries the argument alone, which was the requirement.
- **Fig 4, rebuilt** → `figures/fig_a4_floor.png`. Its title had read "4 to 7 times what the
  data supports" since 2026-11-16. The crossings are 0.67 to 1.37 percent, so it is
  **3.7 to 7.5**, and rounding 3.66 up to 4 moved the lower bound in our own favour. Now
  computed from the sweep.

**One consistency trap, not an error.** The reduction sigma must come from the JSON's own
per-pair `shape_rms_over_err`, median **1.105**, never from median shape divided by median
error, which gives 1.15. Each pair carries its own error bar, so the median of the ratios is
not the ratio of the medians.

**Layout, from P5-D7 and the T46u note.** dpi 200, one shared legend rather than per-panel
legends over data, top and right spines off, annotations in opaque boxes. Render and look at
every figure rather than assuming, which is how the clipped titles and a legend covering two
bars were caught.

## 9. Authorship and sequencing

- V5 / H7 is Vedanth's run (Track 2, filed under *Wanted*). A5 and A4 are yours. Settle the
  author order before drafting, not after.
- Pulling V5 / H7 out does **not** damage the v2 manuscript. Track 2's share there rests on
  V7, V2 and V6 (`08_work_split_v2.md`, "What Track 2 owns in the manuscript").
- **Everything is computed.** The K2-18b pRT run closed 2026-11-22 (P6-D37). This is now a
  pure writing project with no build in front of it.

## 10. Order of operations

1. ~~Run A5 / H8 on K2-18b.~~ **Done 2026-11-22, P6-D37.**
2. Re-verify the two Zenodo DOIs and the Schmidt citation once AJ assigns a volume.
3. Re-read Roy-Perez Sections 4.2 to 5 properly. Only the front half is verified, and the
   back half is where their cloud-parametrisation comparison lives, which is the claim this
   paper positions against.
4. Build Fig 3, then write sections 5 and 4, then 6, then the intro last.
