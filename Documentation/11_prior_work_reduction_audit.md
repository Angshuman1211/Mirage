# Prior work on JWST reduction systematics, audited

Written 2026-11-22 to answer one gating question before any astronomy paper is drafted.

**The question.** Does anyone already benchmark the size of a reduction-induced spectral shift
against the size of a forward-model or radiative-transfer-code difference, in common units?

**The answer is no, for all three papers.** Detail below, with what each paper does instead,
and with the three lanes that are left open.

**Sourcing discipline.** Metadata resolved from the arXiv API and the publisher record, not
recalled. Where a paper body was only partly readable, that is marked, and nothing below is
filled in from memory. Verbatim quotes are marked as such.

---

## 1. Schleich et al. 2025, "Knobs and dials of retrieving JWST transmission spectra. II"

**Citation.** S. Schleich, S. Boro Saikia, Q. Changeat, M. Güdel, A. Voigt, I. Waldmann,
*Astronomy & Astrophysics* **704**, A223 (2025). arXiv:2511.05652, submitted 7 Nov 2025,
20 pages, 12 figures.

**Confidence: HIGH.** Full text extracted locally from the publisher PDF with `pdftotext`
and read directly, not via a summariser.

**Data.** WASP-39b, the single NIRSpec PRISM transit of 10 July 2022, programme PID 1366
(PI N. Batalha). Three transmission spectra from the same raw data.

| label | reduction |
|---|---|
| SP-TW | theirs, Eureka! |
| RU-23 | Rustamkulov et al. 2023, Eureka! + ExoTEP |
| CA-24 | Carter & May et al. 2024, FIREFLy |

**What they measure.** Retrieved-parameter (posterior) space. Three posterior classes.
Stable and near-Gaussian for species constrained across the whole spectrum (H2O, CO2, H2S).
Upper-limit posteriors for weak constraints (CO, HCN). Skewed heavy-tailed and unstable for
species resting on minor features (SO2, C2H2, CH4). Planetary radius and the p-T profile are
stable. They also perturb their own spectrum randomly to separate noise from reduction choice.

**Forward-model comparison: ABSENT, and deliberately so.** One retrieval code, TauREx.
Verbatim from Section 4.4: they "circumvent the problem of disagreeing results by applying
the same model setup to data derived from the same underlying observation". They hold the
model fixed on purpose. Verbatim limitation: "using homogenised atmospheric retrievals on all
three spectra overlooks the model-tuning possibility", and "We leave the solution to
addressing this tension in results to future work". Their only cross-model number is a
citation to others, discrepancies "as large as 5 orders of magnitude" (Tsiaras 2018,
Wakeford 2018, Lueber 2024), not their own measurement.

**Reduction-to-reduction ppm: ABSENT.** The one ppm quantity in the paper is Fig. D.2,
the median absolute deviation of *their own* spectroscopic light curves against a JexoSim
photon-noise budget. It is a data-quality check, not a between-reduction difference.

**Useful to us, and a gift.** Table D.2 tabulates that the three reductions *disagree about
whether to inflate error bars at all*. SP-TW "No", RU-23 "Yes (χ̄²ν to 1)", CA-24
"unspecified". They record the inconsistency and do not resolve it. That is a motivating hook
for A4, handed over by a published paper.

---

## 2. Roy-Perez et al. 2026, "The effect of JWST/NIRSpec data reduction on the retrieval of WASP-39b atmospheric properties"

**Citation.** Roy-Perez, Pérez-Hoyos, Barrado-Izagirre, Chen-Chen, *Astronomy & Astrophysics*
**707**, A297 (2026). arXiv:2602.06722, v2 17 Jun 2026.

**Confidence: MEDIUM-HIGH for Sections 1 to 4.1, LOWER after that.** About 40k of 128k
characters were read verbatim; Sections 4.2 onward, the Table 2 Bayes factors and the
conclusions came through a summariser and are unverified. Re-check before citing anything
from the back half.

**Data.** WASP-39b, the same ERS PRISM transit, 8.23 h, 10 July 2022. BOTS mode, NRSRAPID,
S1600A1 slit, SUB512, 21,500 integrations. **Six** independently reduced spectra: FIREFLy,
Tshirt, Tiberius, Eureka!+ExoTEP, plus Carter and "Carter Modified" (the latter applying a
wavelength-dependent treatment of the saturated region). Verbatim: "the present work is not a
comparison of the pipelines themselves".

**Forward model and sampler.** Planetary Spectrum Generator (PSG), with PUMAS as the
radiative transfer core, correlated-k at R=200. MultiNest. **Not TauREx**, which an earlier
version of our own notes had wrong and which `07_state_gaps_and_research_agenda.md` already
corrected on 2026-11-15.

**What they measure.** Retrieved-parameter space. Temperature spans "some 300K from the
hottest (Tiberius) to the coldest (Carter modified)". Abundances: "In some cases deviations
are even bigger than two orders of magnitude". Radius and stellar radius show "high
compatibility". Abstract, verbatim: differences "often at a level exceeding one order of
magnitude, comparable to uncertainties from modelling assumptions".

**Forward-model comparison: ABSENT as a code comparison, PRESENT as an assumption
comparison, and this is the single most important distinction in this document.** They
benchmark reduction choice against **cloud-opacity parametrisation** (flat, Ångström,
MOPSMAP Mie), not against a different radiative transfer code. Abstract, verbatim:
"Differences introduced by data reduction are comparable in magnitude to those produced by
distinct cloud-opacity models."

So the *shape* of the claim "reduction matters as much as a modelling choice" is already
published. Any paper of ours must cite this sentence and position against it rather than
appear to discover it.

**Reduction-to-reduction ppm: ABSENT.** Spectra are interpolated to a common R=200 grid with
FIREFLy as reference, and Fig. 2 residuals are described only qualitatively. Bin counts are
not given.

**Error floor: ABSENT.** No inflation method is described. Verbatim intro: the data are used
"without adding any data from other sources or random noise, as done in previous works".

---

## 3. Schmidt et al. 2025, "A Comprehensive Reanalysis of K2-18 b's JWST NIRISS+NIRSpec Transmission Spectrum"

**Citation.** S. P. Schmidt, R. J. MacDonald, S.-M. Tsai, M. Radica, L.-C. Wang,
E.-M. Ahrer, T. J. Bell, C. Fisher, D. P. Thorngren, N. Wogan, E. M. May, P. Ferrari,
K. A. Bennett, Z. Rustamkulov, M. López-Morales, D. K. Sing. arXiv:2501.18477,
**accepted to AJ**, 42 pages, 21 figures, v2 29 Aug 2025. Volume, pages and DOI not yet
resolved, so cite as accepted rather than inventing a volume. Note arXiv spells it
"Reanalysis" while the Zenodo record spells it "Re-analysis".

Data DOI 10.17909/3ds1-8z15 (MAST). Supplementary Zenodo 10.5281/zenodo.14735689, and the
paper itself cites 10.5281/zenodo.14735688, a *different* record. Check which one is wanted.

**Confidence: HIGH for the body.** About 199.7k of 213.4k characters verbatim. The last
~13.6k (Appendix B, Appendix C text, the Appendix A tail) is summariser-only, which includes
the "typically within 1 to 1.5-σ" and the 44 ppm offset figures. Treat those two as unverified.

**This is the heavyweight in the field and the direct competitor on K2-18b.**

**Data.** NIRISS SOSS, one transit 1 June 2023, GO 2722 (PI Madhusudhan), two reductions
(FIREFLy, exoTEDRF). NIRSpec G395H, one transit 20 to 21 January 2023, four reductions
(FIREFLy, exoTEDRF, Eureka! A, Eureka! B). 6 NIRISS variants x 10 NIRSpec variants = **60
data combinations**.

**Retrievals.** "In total, our analysis comprises over 250 retrievals", being POSEIDON on 60
combinations x 4 model variants = 240, plus BeAR on 4 combinations x 4 = 16. PyMultiNest,
1,000 live points. "Each retrieval took 2–5 days on 24 CPUs", total ">~50 CPU years".

**Headline.** log10 CH4 = -1.15 (+0.40/-0.52), CH4 at 4.2 (+0.7/-0.9) sigma ensemble.
log10 CO2 < -1.70 and log10 DMS < -3.70 at 95%. K2-18b "unlikely to be a hycean planet".
They introduce an ensemble posterior that "folds uncertainties from data processing into
atmospheric inferences", which they call novel, plus a detection-classification scheme
reserving "robust detection" for species detected across the grid.

**Two codes, but no ranking: the comparison is ABSENT in quantitative form.** They do run
POSEIDON and BeAR on the same reduction, so a code-vs-code difference is isolable on 4 of 60
combinations. But they never state which factor dominates, and give no ratio or common-unit
effect size. What exists is qualitative:

- Verbatim: "indicating that our conclusions are robust to the choice of retrieval code".
- Verbatim: "Finally, we demonstrate the excellent agreement between POSEIDON and BeAR".
- Appendix A, the closest they come: the CO2 and DMS non-detections are "primarily driven by
  differences in the data reduction and/or light curve fitting approaches" rather than
  "differences in the atmospheric modelling and/or retrieval approaches".
- Fig. 8 caption: composition "is insensitive to whether or not stellar contamination is
  included", again with no number.
- Table 3 shows the two codes use *different CH4 line lists* (MM vs YT10to10) and do not
  quantify what that costs.

**Reduction-to-reduction ppm: ABSENT.** Figures 3 and 4 are visual, with "The four reductions
demonstrate broad agreement". The retrieved NIRISS-to-NIRSpec offset is a fitted nuisance
parameter with a -100 to +100 ppm prior, not a measured reduction difference.

**Error floor: ad hoc and per-pipeline, with no supported value stated.** Table 2: FIREFLy
"Red noise via Allan variance", inflated "typically by a factor of no more than two";
exoTEDRF "Jitter term"; both Eureka! arms a "White noise term" or multiplier. No numerical
noise floor for the data and no statement of what floor the spectra support.

**A RISK TO CARRY, not a result.** A reader can assemble a code-vs-reduction comparison from
their printed CH4 significances even though the authors decline to. Doing so gives a
code gap of roughly 0.6 to 1.0 sigma at fixed reduction, against a reduction gap of 0.1 to
1.2 sigma at fixed code, so the two look *comparable* rather than reduction dominating. That
arithmetic is ours, it is in detection-significance units rather than ppm, and it is badly
confounded because POSEIDON and BeAR differ in chemistry, clouds, stellar model, priors and
opacity sampling (their Table 3). **It must not be quoted as a finding, and it must be
answered**, because a referee can build it in ten minutes and ask why it points the other way
from ours. The answer is that significance-of-detection is not a measure of spectral
distance, but that answer has to be written down.

---

## 4. What is left open

Three lanes, each matched to work that is already run.

1. **Reduction shift in spectrum space, in ppm, on a shared grid.** All three papers measure
   in posterior space. None publishes a between-reduction ppm figure. V5 / H7 (P6-D32) has
   21 pairwise comparisons across 7 G395H reductions on a shared 60-bin grid.

2. **A matched-physics radiative-transfer yardstick to divide by.** Schleich uses one code by
   design. Roy-Perez uses one code and compares against a cloud *parametrisation*. Schmidt
   uses two codes and declines to rank them. A5 / H8 (P6-D19) is a matched-atmosphere,
   matched-line-list, matched-binning TauREx against petitRADTRANS comparison, which is the
   only clean denominator any of this has.

3. **What error floor the data actually supports.** Schleich tabulates that reductions
   disagree about inflation and leaves it. Roy-Perez applies none. Schmidt applies three
   different ad hoc schemes. A4 (P6-D23) measures the floor at which best-fit chi2/N reaches
   1, at 0.67 to 1.37 percent by target.

**What is pre-empted and must be cited, never claimed.** That reduction choice matters
(all three). That it is comparable to a modelling choice (Roy-Perez, verbatim, against cloud
parametrisation). That conclusions are robust to retrieval-code choice (Schmidt,
qualitatively, which our A5 null *corroborates with a number* rather than discovers). That
K2-18b's reductions have been swept comprehensively (Schmidt, 60 combinations, 256 retrievals,
AJ).

**The honest position.** Our contribution is quantification and calibration of claims the
literature currently states qualitatively, plus the error-floor consequence nobody states at
all. That is a methods contribution, not a discovery, and the paper should say so in the
abstract.
