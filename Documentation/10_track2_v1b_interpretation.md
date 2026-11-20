# V1b: Interpretation and Manuscript Write-Up — Evaluator-Independent Harm of Data-Space Translation

Written 2026-11-18 for framing B (sim-to-real negative results, TMLR). Owner: Vedanth (Track 2).
Work item **V1b** in `08_work_split_v2.md`, gap **G1b** in `07_state_gaps_and_research_agenda.md`,
interpreting Track 1's 6-parameter ABC injection build and first measurement in **P6-D24**.

---

## 1. Decision-log verdict (V1b acceptance sentence)

> **P6-D3's negative translation verdict is evaluator-independent**: on the 6-parameter ABC Tier-1
> real-noise injection ladder (`data/inject/wasp39_tier1_abc.hdf`, 200 draws × 199 samples),
> `cyclegan_posterior_cov` collapses to **cov@68 = 0.483** (nominal 0.68) and **TARP dev = 0.335**,
> against matched untranslated ABC controls at **0.668 / 0.090** (`h9_abc_cov_s42`) and
> **0.686 / 0.110** (`h9_abc_nocond_s42`), driven by systematic upward bias on temperature and
> log CH₄ rather than width inflation — discharging the P6-D8 evaluator-flip exposure and closing G1b.

---

## 2. Unified A1 Calibration Table (7-Parameter Radius Ladder + 6-Parameter V1b Translated Row)

The 7-parameter radius rows (P6-D17, 3 seeds per group on `wasp39_tier1.hdf`) and the 6-parameter
ABC rows (P6-D24, single seed 42 on `wasp39_tier1_abc.hdf`) live in **different parameter spaces**
and must be presented in the same table with an explicit block separator so absolute coverage
values are never compared across families without their matched controls.

| family & ladder | arm | tier | cov@68 (nom. 0.68) | TARP dev (↓) | SBC rank shape | max \|skew\| | diagnosis |
|---|---|---|---|---|---|---|---|
| 7-param radius (`wasp39_tier0_indist`, 3 seeds) | `h9_rad_cov` | Tier 0 | 0.719 ± 0.016 | 0.117 | bell | ≤ 0.116 | mildly under-confident |
| 7-param radius (`wasp39_tier0_indist`, 3 seeds) | `h9_rad_sigma` | Tier 0 | 0.728 ± 0.010 | 0.125 | bell | ≤ 0.116 | mildly under-confident |
| 7-param radius (`wasp39_tier0_indist`, 3 seeds) | `h9_rad_nocond` | Tier 0 | 0.742 ± 0.011 | 0.172 | bell | ≤ 0.116 | mildly under-confident |
| 7-param radius (`wasp39_tier1.hdf`, 3 seeds) | `h9_rad_cov` | Tier 1 | 0.803 ± 0.003 | 0.145 | bell | ≤ 0.116 | under-confident (scale gap) |
| 7-param radius (`wasp39_tier1.hdf`, 3 seeds) | `h9_rad_sigma` | Tier 1 | 0.811 ± 0.016 | 0.158 | bell | ≤ 0.116 | under-confident (scale gap) |
| 7-param radius (`wasp39_tier1.hdf`, 3 seeds) | `h9_rad_nocond` | Tier 1 | 0.834 ± 0.014 | 0.160 | bell | ≤ 0.116 | under-confident (scale gap) |
| **6-param ABC (`wasp39_tier1_abc.hdf`, s42)** | `h9_abc_cov_s42` (untranslated ctrl) | Tier 1 | **0.668** | **0.090** | sloped (mild) | ≤ 0.086 | **nominal** (validates ladder) |
| **6-param ABC (`wasp39_tier1_abc.hdf`, s42)** | `h9_abc_nocond_s42` (untranslated ctrl) | Tier 1 | **0.686** | **0.110** | bell | ≤ 0.086 | **nominal** (validates ladder) |
| **6-param ABC (`wasp39_tier1_abc.hdf`, s42)** | **`cyclegan_posterior_cov` (translated)** | **Tier 1** | **0.483** | **0.335** | **sloped (severe)** | **0.388** | **overconfident & biased high** |

*Per P6-D11/G4b, SBC p-value magnitudes are withheld as comparative statistics because exponents
move across seeds; qualitative rank shape, TARP deviation, coverage@68, and rank skew carry the
evidence. Full per-arm p-values remain logged in P6-D24.*

---

## 3. Why V1b Sharpens Framing B: Translation vs. Conditioning Across Evaluators

Under Framing B, the paper rests on two pillars — **noise conditioning** (Track 1) and
**data-space domain translation** (Track 2). V1b establishes a structural contrast between them
that strengthens the manuscript rather than repeating the same negative twice:

| evaluator | Pillar 1: Noise conditioning (`cov` vs `nocond`) | Pillar 2: Data-space CycleGAN (`translated` vs `untranslated`) |
|---|---|---|
| Distributional (held-out real OOT frames, P6-D13) | — (architecture change, not a spectrum map) | **Harmful** (MMD 0.782 vs 0.288 none / 0.168 rand; 2.7× worse) |
| In-distribution synthetic test (`abc_test.hdf`) | **Null** on loss (P6-D12) and cov@68 (0.698 vs 0.692, P6-D1) | **Harmful** (P6-D3: cov@68 0.557 vs 0.690; width 2.020 vs 2.222; −10 nats logdens) |
| Non-circular Tier-1 real-noise injections (A1/V1b) | **Mildly helpful** (P6-D17: cov 0.803 < σ 0.811 < nocond 0.834; TARP dev 0.145 < 0.158 < 0.160) | **Severely harmful** (P6-D24: cov@68 0.483 vs 0.668/0.686; TARP dev 0.335 vs 0.090/0.110) |
| Real WASP-39b vs TauREx-NS anchor (P6-D2, P6-D26) | **Strongly harmful** (+0.083 nats JS, 8.4× seed spread) | Degraded (fixed-radius 6-param family; see V2 inferential table) |

Three analytical takeaways belong directly in the manuscript's Results and Discussion:

1. **No evaluator rescues unpaired data-space translation.** Noise conditioning exhibits a genuine
   evaluator disagreement — null on held-out synthetic loss, mildly positive on Tier-1 injection
   calibration, and strongly negative on real WASP-39b (confirmed real and non-proxy by A3/P6-D26).
   Data-space CycleGAN translation exhibits **no such tension**: every evaluator — distributional
   MMD, in-distribution synthetic posterior quality (P6-D3), and non-circular real-noise injection
   calibration (P6-D24) — points in the same negative direction.

2. **Opposite failure mode: systematic bias and overconfidence, not under-confidence.** Across all
   20 untranslated 7-parameter A1 cells (P6-D17), MIRAGE fails in one consistent direction:
   **under-confident** posteriors (`cov@68 = 0.704–0.851`, bell-shaped ranks, `|skew| ≤ 0.116`),
   primarily driven by the training-to-real noise-scale mismatch (P6-D7) and localised to planet
   radius (G4b). On the 6-parameter ABC injection ladder, both untranslated controls land right at
   nominal (`0.668` and `0.686`, `|skew| ≤ 0.086`). Only `cyclegan_posterior_cov` breaks into
   **severe overconfidence and directional bias** (`cov@68 = 0.483`, TARP dev `0.335`, strongly
   sloped ranks).

3. **Physical mechanism of the bias (`T` and `log CH₄` pushed systematically high).** On
   `wasp39_tier1_abc.hdf`, rank damage concentrates selectively on temperature (`skew = -0.288`),
   `log CH₄` (`skew = -0.388`), and `log CO₂`, while `log H₂O`, `log CO`, and `log NH₃` remain
   unremarkable. Negative rank skew means the true parameter consistently falls in the lower tail
   of the posterior — i.e., **the translated arm predicts systematically higher `T` and higher
   `log CH₄` than truth**. This follows directly from training an FMPE estimator on
   $(G_{AB}(x), \theta)$ pairs produced by an unpaired translator with no $\theta$-preservation
   objective: `G_AB` distorts per-bin means by `0.308` per-bin $\sigma$ (P5-D11) to satisfy the
   single-planet discriminator, altering continuum slopes and band contrasts. The flow learns to
   invert those distorted spectra; when evaluated on untranslated spectra carrying genuine JWST
   noise, the un-shifted spectral amplitudes and CH₄/CO₂ band depths are misread as larger scale
   height (higher $T$) and stronger methane absorption (higher $\log\text{CH}_4$).

---

## 4. Methodological Safeguards and Honest Limits to Carry into the Paper

1. **Why the 6-parameter ABC controls are essential.** Because `cyclegan_posterior_cov` is a
   6-parameter ABC model while A1's main table is 7-parameter radius, the translated arm cannot
   be compared numerically against the 0.803–0.834 radius rows. Including `h9_abc_cov_s42` (0.668)
   and `h9_abc_nocond_s42` (0.686) on the exact same 1,000 resampled-geometry injections serves as
   a double check: it anchors the harm relative to identical-family controls AND verifies that the
   full-population ABC injection build (`mean -0.044, sd 0.891`) does not artificially penalise a
   well-calibrated model.
2. **Single training seed for V1b.** Unlike the 3-seed H9 radius table, `cyclegan_posterior_cov`
   and its two ABC controls were scored at seed 42 only. Given the gap size (`cov@68`: `0.483` vs
   `0.668–0.686`, a `-0.185` to `-0.203` drop against the 6-param H9 seed spread of `0.011–0.013`;
   TARP dev `0.335` vs `0.090–0.110`), seed noise cannot close the gap, but the single-seed scope
   should be stated explicitly.
3. **Scope boundary (paired with V6, `09_track2_v6_related_work.md`).** V1b refutes **unpaired
   sim-to-real data-space translation** (`G_AB` applied to training simulations without parameter
   supervision). It does not refute latent representation alignment (Swierc et al. 2024) or
   label-preserving real-to-sim transport (SPIN, Jang et al. 2026).

---

## 5. Draft Manuscript Paragraph (Framing B, Track 2 Results Section)

> To verify that our negative translation verdict (P6-D3) is not an artefact of evaluating against
> the nested-sampling anchor — a genuine concern given that noise conditioning flips sign between
> injection labels and the real-data anchor — we score `cyclegan_posterior_cov` on a non-circular
> 6-parameter Tier-1 real-noise injection ladder (`wasp39_tier1_abc.hdf`, 200 prior draws with
> joint ABC planet geometry injected with held-out WASP-39b OOT noise). Unlike noise conditioning,
> the translation verdict is strictly evaluator-independent. Both matched untranslated ABC controls
> achieve nominal calibration on the same injections (`cov@68 = 0.668` and `0.686`, TARP deviation
> `0.090` and `0.110`, max $|\text{skew}| \le 0.086$), validating the ladder. In contrast, the
> CycleGAN-translated arm collapses to `cov@68 = 0.483` and TARP deviation `0.335` — overconfident
> and severely biased in the opposite direction to every untranslated MIRAGE arm (which are
> uniformly under-confident at `0.704–0.851`). Rank skew concentrates on temperature (`-0.288`) and
> $\log\text{CH}_4$ (`-0.388`), pushing retrieved values systematically high because the unpaired
> generator perturbs spectral amplitudes (`0.308` per-bin $\sigma$) without preserving the
> simulator's parameter-to-spectrum map.
