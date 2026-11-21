# v2 manuscript blueprint, framing B

Blueprint only, cover-in-this-order points rather than prose, because Overleaf is canonical
and the prose is yours.

Scope: the whole paper skeleton. The inference-side prose asks are already blueprinted in
`A6_inference_blueprint.md` and are **not** repeated here, only pointed at.

**Framing B locked 2026-11-08** (P6-D10, `06_v2_post_rejection_plan.md:70`). Target TMLR.

**The venue's one risk, stated so it drives the writing** (`07_state_gaps_and_research_agenda.md:649`):
TMLR drops novelty but not evidence, and "single runs, circular calibration and three targets
are exactly what TMLR rejects on". Every section below is arranged so the evidence criterion
is met explicitly rather than hoped for. Where a claim rests on one run, say so in the
sentence that makes the claim.

---

## 0. What v1 was and what has to go

Current file `Paper/neurips_2026 (7).tex`, title *"A Missing Latent, Not a Missing Simulator:
Radius-Augmented Inference for Real JWST Retrieval"*. That title is the retired framing. The
structure is Introduction, Method, Diagnosis, Results, Limitations.

Four things must be removed, not softened.

1. **The 7/7 coverage table** (around line 200). Retired by P5-D5. The intervals were centred
   on the NS anchor and twice its width, so they could not fail to contain it. Replace with
   the A1 and A2 injection numbers.
2. **The radius-latent headline.** Keep the radius story, but as a short motivating case study
   per the standing instruction, not the contribution.
3. **MMD ≈ 1.22 quoted alone** (line 207). A single adapted MMD means nothing without its
   baseline. Both values always.
4. **The single-spectrum sentence** (line 207): the real side "drawn from the single reduced
   WASP-39b spectrum, bootstrap-expanded with 1\% photon-noise jitter to 500 samples". Retired
   by V7. Replace with 3,393 genuine out-of-transit frames across three targets on 26 shared
   bins.

## 1. Title and abstract

- Lead on the transfer failure, not on a method. The agenda's suggestion is the right shape,
  robustness to synthetic misspecification not transferring to real JWST noise.
- Abstract covers, in order: the setup in one sentence; pillar one; pillar two; the
  attribution that rules out noise amplitude and the forward model; the one sentence conceding
  what is a debugging finding rather than a method.
- Do not promise a fix. The paper closes G3 and G9 **by measurement, not by repair**
  (P6-D22). Any sentence implying repair will be read as the v1 claim returning.

## 2. Introduction

- Real JWST retrieval with SBI is established, not novel. Cite Gebhard 2025 (A&A 693 A42),
  Lueber 2025 (ApJL 984 L32), Pagliaro 2026 (A&A 709 A17). Concede novelty up front, which is
  the correct move for TMLR specifically.
- Robust-SBI is crowded and validated mostly on toy simulators. RNPE, RSNL, Huang's MMD
  regulariser, RoPE. One sentence each at most.
- The gap this paper fills: nobody tests whether those robustness gains survive contact with
  real instrument noise, and this paper does, twice, on two independent intervention families.
- The two pillars named explicitly, with a forward reference to each results section.
- State N=3 and the frame counts (3,241 / 56 / 96) in the intro. Do not let a referee find it.

## 3. Method

Keep the v1 structure here, it was not the problem. Four changes.

- Forward model and training data: unchanged in substance.
- Flow-matching posterior with radius: apply `A6_inference_blueprint.md` sections 1 to 3. The
  two deletions are mechanical, "exhaustively" and "unchanged across targets". The
  identifiability rewrite is the substantive one and it got sharper, not weaker.
- Calibration: rewrite entirely. Describe the OT step as what it was, state the circularity
  finding as a result of this paper, and describe A2 as the non-circular rebuild on injection
  labels. Per-target refit, no amortization claim.
- Add a short subsection on the evaluation protocol itself, covering the injection ladder,
  the offset-aware scoring, and the metric-space definition. **This subsection is where the
  evidence criterion is won or lost**, so give it room.

## 4. Results, pillar one, robustness does not transfer

- A1, the Tier 1 injection ladder. 14 cells, 3 training seeds, 2 arms, 2 tiers, 3 targets.
  Lead with seeds, because multiple seeds is the direct answer to the v1 "every headline is a
  single run" criticism.
- A3 / H5, the noise-proxy test, negative and the headline of this pillar (P6-D26).
- H9, the conditioning result and its reversal under injection labels. State the device
  artifact finding plainly, spread collapsing 20x on radius and 114x on abc and reversing
  sign. This is a correction to v1 and it is also evidence of the project's own rigour.
- A2, the non-circular rebuild. Injections demand far less deflation than the real anchor on
  every parameter of every target, median ratio 3.06x / 1.85x / 2.39x, radius worst at 3.83x,
  5.02x, 5.41x. Note the warning was written before the measurement, which is worth one
  clause because it is the strongest possible evidence of a pre-registered prediction.
- The attribution. Tier 0 to Tier 1 to real anchor, 0.916 to 0.712 to 0.225. Noise amplitude
  accounts for under a fifth, and H8 puts the forward model at 0.07 sigma. So the residual is
  real-spectrum information content, reduction systematics, or cloud and vertical structure.
- **The four honest limits from P6-D22 go in this section, not in Limitations.** In particular
  that A2's ESS edge is partly mechanical because its proposals are simply wider.
- ESS: report it beside every interval, and apply the P5-D7 decision that rows below ESS ~20
  get a wider honest interval or no interval. Do not hide the 1.0 to 42.5 range. A reviewer
  who finds it unflagged will discount everything else.

## 5. Results, pillar two, learned translation is worse than hand-designed randomisation

Owner Vedanth. Material is V7, V2, V6, V1b.

- V7 first, because it removes the objection that would otherwise end the section. Real side
  rebuilt on 3,393 genuine frames across three targets, 26 shared bins, balanced by target.
  Translation still 3.71x to 3.92x worse than domain randomisation on the corrected basis.
- Say the pooling caveats in the text, not in a footnote. WASP-39b contributes 3,241 of 3,393
  frames, the balanced sampler reuses WASP-96b's 56 frames 20.2x per epoch, and the three
  targets' noise amplitudes differ by 17x to 26x. Balancing equalises counts, not amplitude.
- V3 / H1 as the mechanism check that refutes the obvious defence. Restricting domain A to the
  real planet's depth neighbourhood moves the mean shift only 0.308 to 0.240, still 7.4x to
  8.1x above unadapted simulation.
- V2 as the positive control that makes the negative meaningful. Quantile mapping reaches
  nominal coverage, 0.677 and 0.950, and is 6.3 percent sharper than structured randomisation
  at width68 2.073 against 2.212. Classifier reweighting takes best log-density at -7.61.
  CycleGAN undercovers at 0.550 and 0.843 and collapses log-density to -17.65.
- **The point of V2 is the dissociation, so state it as such.** Quantile mapping wins on MMD
  *and* on coverage, while CycleGAN loses on both, which is what makes this a retrieval result
  rather than a distributional one. Guard against the P6-D13 caution: a per-bin marginal match
  can be achieved by a map that destroys parameter information, so the coverage column is what
  licenses the claim, not the MMD column.
- V1b, the translated arm on the injection ladder, for evaluator independence.

## 6. The metric correction, and where to put it

P6-D33. This needs one short, confident passage, and the instinct to bury it is wrong. A
paper whose thesis is that evaluation discipline matters gains credibility by reporting a
defect it found in its own evaluator.

- The defect in one sentence. The PCA basis for MMD was fitted on the arm under test, so it
  moved with the checkpoint being scored.
- The consequence that matters: the published tier-2 comparison **reverses**, from 0.57x to
  1.08x of domain randomisation to 1.23x to 1.60x, two arms changing sign, so learned
  translation never beats randomisation anywhere.
- V7's verdict unchanged, 3.71x to 3.92x.
- **The sentence a reader will demand**, and the one worth the most: V2's column barely moved
  under the same fix while tier-2 reversed, and the difference is the number of real frames
  defining domain B, 56 against 3,241. With a well-determined real covariance no single arm
  can drag the retained components. So the sign of basis contamination is not predictable from
  the setup, which is the argument for fixing the basis rather than reasoning about which way
  a contaminated one leans.
- Keep it in the evaluation-protocol subsection of Method, with a pointer from Results. Not an
  appendix.

## 7. Related work

Owner Vedanth. Drafted in `09_track2_v6_related_work.md`, references verified there.

- Add the three reduction-systematics papers from `11_prior_work_reduction_audit.md`, since
  "reduction systematics are first-order" is now part of this paper's attribution argument and
  citing them is required.
- Keep the robust-SBI block short. It is crowded and the point is only that it is toy-validated.

## 8. Limitations and conclusion

- N=3, one simulator, one translator.
- ESS, again, explicitly.
- The radius story as a case study, with the P6-D18 qualification that the 68 percent interval
  contains the literature value on WASP-96b only, one of three.
- No positive method contribution, stated without apology. The contribution is the
  measurement and the audit trail.
- **Do not** reuse the 498 single-core hours figure as ours (P6-D20). It is Exoformer's
  uniform-prior WASP-39b retrieval on one core and quoting it as MIRAGE's cost overstates by
  about 5500x, checkable by anyone in one minute.

## 9. Decisions owed before writing closes

1. **V2's JS_NS column, now unblocked and cheap** (P6-D35). V2's stated acceptance test
   requires MMD *and* posterior JS to NS, and the column reads `pending` for all seven arms.
   The blocker is **not** the uncommitted arm checkpoints, which `--score-ns-only` never
   loads. It is that the two halves sit on different laptops. The seven `v2_*_samples.npz`
   draws are on Vedanth's machine, the 314 KB nested-sampling anchor
   `wasp39b_ns_posterior_fitrad.npz` is on yours, and both are gitignored under `data/*`.
   Send him the anchor, have him pull for the P6-D33 guard and run
   `python scripts/v2_inferential_eval.py --score-ns-only`, and have him send the seven npz
   back so neither artifact stays single-copy. Filling the column from **his** npz rather
   than resampling is what keeps it consistent with the quantiles already beside it in the
   same row. Publishing a table containing "pending" is not an option, and dropping the
   column is no longer the fallback it was.
2. **Author order**, now that an astronomy paper is also in play
   (`12_astronomy_paper_blueprint.md`).
3. **arXiv v2 goes last** (P6-D10), carrying four corrections: the chi2 floor mismatch, the OT
   calibration, the conditioning result, and the P6-D8 reversal of that conditioning result
   under injection labels. P6-D33 adds a fifth, the MMD basis.

## 10. Order of operations

1. Method's evaluation-protocol subsection first. It is new, it is where the evidence
   criterion is met, and both results sections depend on its definitions.
2. Pillar two next. It is finished, it has a clear shape, and it belongs to the co-author, so
   unblocking him early is worth more than drafting in paper order.
3. Pillar one, applying `A6_inference_blueprint.md` as you go.
4. Figures. The v1 pair were called unreadable at print size by T46u, and figs 1 to 5 were
   already rebuilt 2026-11-18 at higher dpi with shared legends.
5. Introduction and abstract last, once the two pillars are on the page and their relative
   weight is visible.
