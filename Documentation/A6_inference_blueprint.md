# A6 blueprint, inference side

Working document for writing A6 into the manuscript. Blueprint only, cover-in-this-order
points rather than prose, because Overleaf is canonical and the prose is yours.

Source of the four asks: `08_work_split_v2.md`, section A6. Everything below is checked
against the current `Paper/neurips_2026 (7).tex` and against the code, with line numbers.

---

## 0. One finding that changes what A6 should say

A6 says to concede "that Welbanks and Madhusudhan (2019) characterised the degeneracy". That
is right, but the paper says something more specific than the concession assumes, and it
makes the surviving claim **sharper**, not weaker.

**Welbanks and Madhusudhan 2019, The Astronomical Journal, 157, 206, DOI
10.3847/1538-3881/ab14de.** Verified by resolving the DOI, not from memory. Note the arXiv
comment field and the ar5iv header both say "ApJ" and **both are wrong**, so do not copy the
venue from arXiv.

Verbatim from their abstract: the degeneracy between planetary radius and its reference
pressure is "well characterised" and "has little effect on abundance estimates, contrary to
previous claims using semi-analytic models". From their introduction: "the choice of a R_p
versus P_ref as a free parameter is inconsequential to constraining molecular abundances".

**Their result is conditional on one of the pair being free.** In their own retrievals both
are free, and they conclude it is redundant to free both.

**The published MIRAGE setup freed neither.** Verified in code, not assumed:
`scripts/taurex_forward.py:78` hardcodes `atm_max_pressure=1e6`, and
`scripts/taurex_retrieve.py:115-127` only ever enables `T`, the five molecules, optionally
`clouds_pressure`, and optionally `planet_radius`. A reference pressure is never a fit
parameter in any arm.

So the configuration that failed is not one the retrieval literature uses or endorses. That
is the whole contribution and it should be stated that plainly.

---

## 1. Delete "exhaustively" (line 145)

Current: "We test the first exhaustively."

- Four levers were tested, not all of them. A referee lists five more in one sentence without
  effort, for example He/H2 ratio, photochemical haze, stellar contamination, patchy cloud,
  refraction, more P-T nodes. The word loses the argument the moment someone does that.
- Replace with the **count and the names**. Say four and name them, then let the reader judge.
- Cover, in this order: what was varied, that each left the fit unchanged, and that an
  unchanged fit across four independent physics levers is evidence of a *shared* cause rather
  than of a missing molecule.
- **New material available since v1.** Do not only add physics, say you swapped the
  radiative transfer code. H8 reran the same atmosphere through petitRADTRANS and agreed with
  TauREx to 0.067 of an error bar on the constant offset and 0.046 on shape. That is a much
  stronger statement than "we added SO2 and nothing happened", and it is the one that closes
  the fidelity question rather than chipping at it.

## 2. Delete "unchanged across targets" (lines 61, 74, 76)

Current, three places: "transfers unchanged across two instruments and three real JWST
targets", "the same method, unchanged, across three real targets", "transfers unchanged
across three real JWST targets".

- The phrase conflates two different claims. **We did not retune** is true. **It worked
  equally well everywhere** is not, and that is what a reader hears.
- P5-D7 is the specific problem. The radius interval contains the literature value on
  **WASP-96b only**. K2-18b also needed its own radius bounds, `rb=(0.10, 0.40)` against
  `(0.8, 1.6)`, in `taurex_retrieve.py:31-34`, which is itself a per-target choice.
- Replace with the narrow true claim, something along the lines of "with no per-target
  retuning of the architecture or the training set", then state per-target outcomes in the
  table and let them differ.
- Say the quiet part in Section 5. One target out of three reproduces the literature radius
  inside 68 percent. That is the honest transfer result and hiding it is what a reviewer
  punishes.

## 3. The identifiability rewrite (lines 72-76, 139-167, 219)

Four beats, in this order.

1. **Concede the degeneracy is old.** R_p against P_ref, characterised by Welbanks and
   Madhusudhan 2019, who showed it does not harm abundances and that freeing both is
   redundant. Cite as AJ 157, 206. Nothing here is a new degeneracy.
2. **Concede the backbone already infers R_P.** Freeing the radius is the textbook
   configuration, one free parameter of the pair. MIRAGE's fix is standard practice, not a
   discovery, and claiming otherwise invites a reviewer who does retrievals to stop reading.
3. **Concede that a nuisance parameter fixed wrong is misspecification.** The current
   sentence at line 74, "The failure looks like a model error but is non-identifiability", is
   a false dichotomy as written. Fixing R_P at a wrong value with nothing free to absorb it
   *is* a wrong model. Drop the opposition and keep the mechanism.
4. **State the claim that survives, which is about amortization and not about physics.**
   Conventional retrieval frees one of the pair per target, after seeing the data. Amortized
   SBI chooses theta once, at training time, before any target exists. That is how a
   configuration nobody would choose deliberately got frozen into the pipeline and then
   failed **silently and confidently** rather than loudly. The transferable lesson is not
   "radius matters". It is that amortization converts a parameterisation choice into a
   structural commitment, and the usual retrieval safeguard of freeing one nuisance parameter
   is not available at inference time.

Line 219's closing already says the simulator-agnostic version well. Keep it, but make it
follow from beat 4 rather than from "identifiability, not fidelity".

## 4. Per-target NS cost, so the lost amortization is visible (G9)

The structural point to make, before any number: **MIRAGE's optimal-transport calibration
needs a nested-sampling anchor per target.** So the NS cost is paid per target anyway, and
the amortization that motivates SBI is not delivered end to end. G9 calls this "amortization
is dead and nothing replaces it", and the manuscript should say it rather than let a reviewer
discover it.

Numbers, all measured on the project laptop unless stated:

- WASP-96b, 90 bins, 7 parameters, 150 live points, run alone on the laptop CPU:
  **5 min 24 s**. Measured 2026-11-15, `data/real_ess/ns_wasp96_timing.wall`. It independently
  reproduced the anchor, radius 1.1970 against 1.1972 and T 688.0 K against 682.9 K, so it
  doubles as a reproducibility check.
- Same target and settings through petitRADTRANS instead: **23.7 min**, 1992 iterations.
- The A5 live-point sweep, 150 / 300 / 600 / 1000, four runs in parallel on one laptop:
  **50 min total wall clock**. Per-run times there are contended, about 19.5 min for the
  150-live run against 5.4 min when it runs alone, so do not quote them as single-run costs.

**Do not reuse the 498 single-core hours figure as ours.** It is real, Exoformer, A&A 709
A17, 2026, Sect. 5.2 and Table 3, but it is their uniform-prior WASP-39b retrieval on one
core. Quoting it as MIRAGE's cost overstates ours by about 5500 times and is checkable.

### The uncomfortable consequence, and it should be in the paper

**At this problem size the per-target reference is cheap, so "nested sampling is
unaffordable" cannot carry the motivation for amortization.** G9's revival argument rests on
NS being something the field cannot afford. Ours costs 5.4 minutes.

Set that against what the amortized side costs. One H9 arm trained for **0.59 h**, 225 epochs
at 9.4 s/epoch on the CUDA box, from `configs/h9_rad_cov_s42/history*.csv`, and that excludes
simulating the training set, which is the larger hidden cost. So training alone is roughly
6.5 target-equivalents of nested sampling **before** the training data exists.

And the calibration still needs an NS anchor per target. The amortized route therefore does
not replace the per-target cost, it **adds** training on top of it. At three targets it is
strictly more expensive than just running nested sampling.

Do not hide this and do not let a reviewer find it. The defensible position has two parts.
First, the per-target cost argument for amortization does not hold at this scale, and the
motivation has to come from catalogue scale, Ariel's thousand-target class, or from
resolutions where NS cost grows, not from today's three targets. Second, amortization
survives only if the calibration transfers to a new target without its own anchor.

### That second part is no longer untested. It was tested and it failed (H11, P6-D21)

This paragraph previously said to write "untested, not unlikely". **Do not write that any
more.** H11 ran the leave-one-out transfer across all three targets with the recipient's
anchor held out, and the calibration does not transfer.

Numbers to use, mean marginal JS to the held-out anchor:

| donor relationship | mean JS |
|---|---|
| same target, same training run (circular, the published setting) | 0.0254 |
| same target, different random seed | 0.1073 |
| different target | 0.2632 |
| no calibration at all | 0.2693 |

**Transfer to a new target recovers 2.5 percent of the benefit the calibration shows on the
target it was fitted to.** Across a change of random seed alone it recovers 73 percent, so
the calibration is largely a property of the individual training run.

The attribution, in 2-Wasserstein so nothing saturates, in units of the anchor's own sd: an
anchor-free pooled correction removes **85 to 97 percent of the shape error and none of the
location error**, leaving the posterior 5 to 13 anchor sigmas out against 0.7 to 0.8 with its
own anchor. **The transferable half of the calibration is the half that does not matter.**

The one sentence worth building the paragraph around. Whether an anchor-free correction helps
is decided by the flow's mean error against the width being shrunk, and that quantity needs
the anchor to evaluate, so **an anchor-free calibration cannot be validated anchor-free**. That
is a structural obstacle and not a sample-size complaint, which matters because a referee will
otherwise read "three targets" and assume more targets would fix it.

Figure available at `data/h11/fig_h11_transfer.png` if a panel is wanted. Left panel is the
location/shape split, right panel is the four-class decay.

### One more thing H11 turned up, and it is a claim-level warning for the whole paper

**Distributional agreement and importance-sampling efficiency are different claims. Do not
let one stand in for the other anywhere in the manuscript.**

Re-running the four stored published OT calibration artefacts through their own `--compute`
gives ESS 10.9 to 14.7, and `ot_calibrate.py` prints its own lowest verdict, **"ESS still
low"**, against thresholds it sets at 500 and 50. Component 4's stated purpose is efficient
importance sampling on the real spectrum. **That was never achieved**, circularly, on the
target it was fitted to, with no transfer involved.

The same object therefore scores JS 0.025, which reads as the calibration working, and ESS
18, which is unusable. A reader shown only the divergence concludes the calibration succeeded.
Across all 15 H11 arms epsilon runs 0.06 to 1.56 percent and **nothing is usable**, so this is
not a transfer-specific problem.

How to use it: this is a second, independent negative result and it is cheap to state. It
also protects the paper, because the ESS numbers are in the repo and a referee who recomputes
them will otherwise find the gap first. Report ESS alongside every divergence.

This is a negative result about amortized retrieval, which is the paper framing B wants. It
reads as rigour when you state it and as a gap when a referee states it for you.

## 5. Two things to fold into Section 5 while you are here

- **The H8 scope limit.** The cross-code check holds both codes to the same isothermal,
  cloud-free, constant-abundance column, so it closes "the radiative transfer is wrong" and
  not "the atmospheric model is wrong". A shared cloud or temperature-gradient error is
  invisible to it by construction. State this, because a referee who knows retrieval will
  think of it immediately and it costs nothing to own.
- **Radius agreement in sigma, never as containment** (G4c). The literature 1.20 RJ sits
  inside the TauREx 68 percent interval and outside the petitRADTRANS one, and the flip is
  driven by 0.0004 RJ. In sigma the two codes agree, 1.0 against 1.1. Any sentence in the
  paper phrased as "the interval contains the literature value" is one estimator-noise away
  from being false, so phrase radius agreement in sigma throughout.

## 6. What NOT to change

- Line 145's "in published retrievals the same class of forward model fits WASP-39b" is a
  good, cheap argument. Keep it.
- The chi2 301 to 0.06 and coverage 2/7 to 7/7 numbers are the paper's spine and A6 does not
  touch them.
- Do not soften the collapse itself. ESS = 1 and chi2/N = 301 are real and are the hook.
