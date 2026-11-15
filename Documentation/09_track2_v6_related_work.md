# V6: related work (Track 2 draft)

Written 2026-11-15 for framing B (sim-to-real negative results, TMLR). Owner: Vedanth (Track 2).
Work item V6 in `08_work_split_v2.md`, gaps G14 and G15 in `07_state_gaps_and_research_agenda.md`.

**How the citations were checked.** Each entry below was read against the primary record, meaning
the arXiv abstract page or the journal or proceedings landing page. No secondary summary was used.
Where that check found a mistake in our own internal documents, it is listed under "Corrections
to our own docs" at the end, so the manuscript does not inherit it. Where a detail could not be
confirmed from the primary record, the prose says less rather than guessing.

---

## Draft prose

### Simulation-based inference on real JWST spectra

Amortized simulation-based inference has already been run on real JWST transmission spectra, so
we do not claim to be first. Gebhard et al. (2025) introduced flow-matching posterior estimation
(FMPE) for atmospheric retrieval. Their models are conditioned on the assumed noise level, and
importance sampling is used to verify and correct the network's output. FASTER (Lueber et al.
2025) used neural networks for retrieval and Bayesian model comparison on the NIRSpec PRISM
spectrum of WASP-39b. Exoformer (Pagliaro et al. 2026) used a transformer to produce informative
priors for TauREx nested sampling, applied to WASP-39b PRISM and WASP-17b SOSS and speeding the
retrievals by a factor of 3 to 8. These works establish that
amortized retrieval *runs* on real data. They do not ask what happens to its calibration when the
noise model is the thing that is wrong, which is our question.

### Robustness to misspecification in SBI

There is a large body of methods for misspecified simulators. RNPE (Ward et al. 2022) models the
discrepancy between simulated and observed summaries explicitly. Robust SNL (Kelly et al. 2024)
adds adjustment parameters that detect and correct misspecification. Huang et al. (2023) learn
summary statistics with an MMD regulariser that penalises statistics which separate observed data
from the simulator's output. RoPE (Wehenkel et al. 2025) treats the misspecification gap as an
optimal-transport problem between learned representations of real and simulated observations,
and calibrates against a small set of real observations with ground-truth parameters. Most of
these methods are validated on synthetic misspecification or on real problems where labels exist.
JWST retrieval has no ground-truth parameters, which is why our calibration checks inject
prior-draw labels into real noise (Tier 1) rather than borrowing labels from a reference
retrieval. Our retired OT calibration (P5-D5) is a cautionary case of RoPE's recipe applied
without independent labels. The transport target was the nested-sampling posterior, which then
also served as the coverage reference, so the check was circular.

### Domain adaptation for neural posterior estimation

The closest comparison class is Swierc et al. (2024), who combine NPE with unsupervised domain
adaptation for strong-lens modelling. Their source domain is noiseless simulation, their target is
survey-like noisy data, and the adaptation is an MMD loss on the embedding network's latent
features. They report 1 to 2 orders of magnitude better accuracy and much better coverage than NPE
alone. The distinction our paper turns on is *where* the adaptation acts. Swierc et al. align
**representations** inside the inference network. CycleGAN translation, and the cheap baselines we
add (per-bin affine, per-bin quantile mapping, classifier reweighting), all act on the **data**
before the network sees it. Our negative result is about data-space translation. It says nothing
against representation alignment, and the paper should say so explicitly.

Data-space translation has its own lineage. CycleGAN (Zhu et al. 2017) learns unpaired
translation with a cycle-consistency constraint, and it is the learned translator we tested.
Diffusion Schrödinger bridges (De Bortoli et al. 2021) solve the unpaired transport problem as
entropy-regularised optimal transport between two distributions, and they are the natural modern
replacement for adversarial translation. SPIN (Jang et al. 2026, preprint) is the most direct SBI
analogue. It learns a real-to-simulator transport from unlabeled real data, uses the simulator's
labels to preserve parameter-relevant information during training, and maps each real observation
into the simulator domain before posterior evaluation. That is the opposite direction to our
CycleGAN arm, which pushes simulations toward the real domain and retrains on them, and it is the
design a reviewer will ask about. We should cite it as concurrent work and not claim our negative
covers it.

The cheap baselines also come from outside astronomy. Quantile mapping is the standard
bias-correction method for climate-model output (Cannon, Sobie & Murdock 2015). Classifier
reweighting, which uses a classifier separating simulation from data to supply density-ratio
weights, is the basis of OmniFold in particle physics (Andreassen et al. 2020). Either baseline
can move the simulated distribution toward real data without an adversarial training loop. That
is why a learned translator has to beat them before its failure says anything about learned
translation in particular.

### Noise-model adaptation in other amortized pipelines

Gravitational-wave inference is the field where amortized posteriors meet real, non-stationary
detector noise at scale. Wildberger et al. (2023) built a probabilistic model that forecasts the
noise power spectral density, so that a DINGO network trained before an observing run stays
accurate through it. They report accurate inference on 37 real O3 events. This is the precedent
for system-identified randomisation (V4/H2): fit the noise model to the real noise, then
randomise around the fit. It is distinct from both broad hand-chosen randomisation and learned
translation.

### Pipeline-level shift in JWST retrievals

Pipeline choice is itself a domain shift, and so far it has been studied with nested sampling
rather than with SBI. Schleich et al. (2025) ran TauREx on three spectra reduced independently
from the same NIRSpec PRISM transit of WASP-39b. They found that posteriors differ between
reductions, most severely for species constrained only by minor features (SO2, C2H2), while the
planetary radius and p-T profile were stable. Roy-Perez et al. (2026) retrieved six independent
PRISM reductions of WASP-39b using MultiNest with Planetary Spectrum Generator forward models.
They found that the inter-pipeline differences in temperature, abundances and cloud opacity often
exceed an order of magnitude, comparable to the uncertainty from modelling assumptions. These
results motivate H7: if the posterior shift between two reductions of one visit is larger than
any noise-model intervention, then noise-model adaptation is the smaller lever.

---

## References, verified

| key | reference | checked at |
|---|---|---|
| Andreassen 2020 | A. Andreassen, P. T. Komiske, E. M. Metodiev, B. Nachman, J. Thaler, "OmniFold: A Method to Simultaneously Unfold All Observables", PRL 124, 182001 (2020), arXiv:1911.09107 | arXiv 1911.09107 |
| Cannon 2015 | A. J. Cannon, S. R. Sobie, T. Q. Murdock, "Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes?", J. Climate 28(17), 6938-6959 (2015), doi:10.1175/JCLI-D-14-00754.1 | journal record |
| De Bortoli 2021 | V. De Bortoli, J. Thornton, J. Heng, A. Doucet, "Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling", NeurIPS 34 (2021), arXiv:2106.01357 | NeurIPS proceedings |
| Gebhard 2025 | T. D. Gebhard, J. Wildberger, M. Dax, A. Kofler, D. Angerhausen, S. P. Quanz, B. Schölkopf, "Flow matching for atmospheric retrieval of exoplanets: Where reliability meets adaptive noise levels", A&A 693, A42 (2025), arXiv:2410.21477 | arXiv + A&A |
| Huang 2023 | D. Huang, A. Bharti, A. Souza, L. Acerbi, S. Kaski, "Learning Robust Statistics for Simulation-based Inference under Model Misspecification", NeurIPS 36 (2023), arXiv:2305.15871 | arXiv + NeurIPS |
| Jang 2026 | J. Jang, E. Jeong, K. S. Choi, H. Kim, "Information-Preserving Domain Transfer with Unlabeled Data in Misspecified Simulation-Based Inference" (SPIN), arXiv:2605.05652 (2026), **preprint, not peer reviewed** | arXiv |
| Kelly 2024 | R. P. Kelly, D. J. Nott, D. T. Frazier, D. J. Warne, C. Drovandi, "Misspecification-Robust Sequential Neural Likelihood for Simulation-Based Inference", TMLR (2024), arXiv:2301.13368 | arXiv + TMLR listing |
| Lueber 2025 | A. Lueber, K. Karchev, C. Fisher, M. Heim, R. Trotta, K. Heng, "Near-instantaneous Atmospheric Retrievals and Model Comparison with FASTER", ApJL 984, L32 (2025), arXiv:2502.18045 | arXiv |
| Pagliaro 2026 | L. Pagliaro, T. Zingales, G. Piotto, I. Giovannini, G. Mantovan, "Exoformer: Accelerating Bayesian atmospheric retrievals with transformer neural networks", A&A 709, A17 (2026), doi:10.1051/0004-6361/202558264, arXiv:2603.27623 | arXiv full text (Sect. 5.2, Table 3) + A&A listing |
| Roy-Perez 2026 | Roy-Perez, Pérez-Hoyos, Barrado-Izagirre, Chen-Chen, "The effect of JWST/NIRSpec data reduction on the retrieval of WASP-39b atmospheric properties", A&A 707, A297 (2026), arXiv:2602.06722 | arXiv + A&A |
| Schleich 2025 | S. Schleich, S. Boro Saikia, Q. Changeat, M. Güdel, A. Voigt, I. Waldmann, "Knobs and dials of retrieving JWST transmission spectra. II. Impacts of pipeline-level differences on retrieval posteriors", A&A 704, A223 (2025), arXiv:2511.05652 | arXiv + A&A |
| Swierc 2024 | P. Swierc, M. Tamargo-Arizmendi, A. Ćiprijanović, B. D. Nord, "Domain-Adaptive Neural Posterior Estimation for Strong Gravitational Lens Analysis", arXiv:2410.16347 (2024), FERMILAB-CONF-24-0444 | arXiv abstract + HTML full text |
| Ward 2022 | D. Ward, P. Cannon, M. Beaumont, M. Fasiolo, S. Schmon, "Robust Neural Posterior Estimation and Statistical Model Criticism", NeurIPS 35 (2022), arXiv:2210.06564 | NeurIPS proceedings |
| Wehenkel 2025 | A. Wehenkel, J. L. Gamella, O. Sener, J. Behrmann, G. Sapiro, J.-H. Jacobsen, M. Cuturi, "Addressing Misspecification in Simulation-based Inference through Data-driven Calibration", ICML 2025, PMLR 267, arXiv:2405.08719 | arXiv + ICML |
| Wildberger 2023 | J. Wildberger, M. Dax, S. R. Green, J. Gair, M. Pürrer, J. H. Macke, A. Buonanno, B. Schölkopf, "Adapting to noise distribution shifts in flow-based gravitational-wave inference", PRD 107, 084046 (2023), arXiv:2211.08801 | arXiv (DOI listed there) |
| Zhu 2017 | J.-Y. Zhu, T. Park, P. Isola, A. A. Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017, arXiv:1703.10593 | already cited in `03_track2_cyclegan_ablation.md` |

Still open: Swierc 2024 lists no peer-reviewed venue on arXiv beyond a Fermilab conference report
number, so cite it as an arXiv preprint unless the workshop is confirmed. The Roy-Perez first names
are not given on the arXiv listing, so take them from the A&A page.

---

## Corrections to our own docs found during the check

1. **Roy-Perez et al. did not use TauREx and did not report "up to 2 dex".** `07_...md` section 3
   says "found abundances deviate up to 2 dex". The primary abstract says differences "often
   exceed one order of magnitude" across temperature, abundances and cloud opacity, and the
   retrievals use MultiNest with PSG forward models. Quote the abstract's wording, not "2 dex",
   unless someone finds 2 dex in the full text.
2. **"Knobs and dials II" ran three reductions, the Eureka! baseline plus two others, on one PRISM
   transit, and found the radius stable across them.** That last point matters for us: pipeline
   shift does not obviously act on the latent our headline fix frees. Section 3 of `07_...md` is
   otherwise correct.
3. **Exoformer is a prior generator for TauREx nested sampling, not an amortized posterior.**
   `07_...md` lists it beside FASTER as "ran on real WASP-39b PRISM". The targets and instruments
   are right (Sect. 5.2: WASP-39b NIRSpec PRISM, WASP-17b NIRISS SOSS), but its method makes it a
   nested-sampling accelerator. Do not cite it as amortized SBI on real data. The "~498
   single-core hours" figure is **sourced**: Sect. 5.2 and Table 3 give "TauREx required ≈498 h
   (WASP-39b) ... with uniform priors ... on a single CPU core". An earlier version of this note
   called it unsourced, because only the abstract had been checked. That was wrong, and P6-D19
   independently found the same section. It is the cost of Exoformer's own uniform-prior
   retrieval, not of ours, so it may be cited as their number but not as our compute budget.
4. **Robust SNL's published title** has "for Simulation-Based Inference" appended, and its TMLR
   author list has five names. **RNPE's fifth author is Schmon.**
5. **Swierc et al. align latent features with MMD.** This is confirmed from the full text, so the
   representation-versus-data distinction in the work split is accurate and safe to build the
   section on.

---

## G15: the old ablation script relabelled

`scripts/run_cyclegan_ablation.py` is now
`scripts/probe_cyclegan_input_shift.py`, with its docstring rewritten. It never trained a flow on
translated data. It feeds translated *inputs* to flows trained on untranslated simulations, so it
measures how a sim-trained flow tolerates shifted input. That is a robustness probe. The
posterior-level translation result is `configs/cyclegan_posterior_cov` (P6-D3), and only that
result may be cited as "the translation result". The script prints this warning when it runs.
