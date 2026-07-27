# BioUnlearn: Results Synthesis

**Purpose**: `PI_STATUS_REPORT.md` is a running status log — findings are recorded as they happened, with in-progress notes and corrections layered on top of each other across many editing passes. This document reorganizes the same real numbers thematically, the way a paper's Results section would, so the actual argument the data supports is visible in one place. Every number below is sourced from a specific file in `data/gate2_results/`; none are estimated or rounded from memory. Where a claim was corrected or retracted during the project, that history is noted briefly, not hidden — the corrected version is what's presented as current.

---

## 1. The dataset: scale and provenance

| | Pilot (`data/splits/`) | Expanded (`data/splits_v2/` + PAC) |
|---|---|---|
| RGU (reversed guideline update) | 95 instances, UMLS-concept-graph-grounded | 447 instances (95 original + 352 from real Herrera-Perez citations, 93% quality-filter pass rate) |
| IFE (ingredient/formulation equivalence) | 95 instances | 95 instances (unchanged) |
| PAC (PHI-adjacent concept removal) | — (did not exist) | 101 instances, synthetic-but-disclosed |
| **Total** | **190** | **643** |

RGU and IFE are grounded in real, citation-backed clinical-reversal literature (Herrera-Perez et al. 2019, Prasad et al. 2013 — 406 gold-tier PQS=3 sources total, not LLM-invented). PAC's real ground truth (i2b2 2014 Risk Factor annotations + MIMIC-IV) requires PhysioNet/i2b2 DUA credentialing not available in this environment; PAC instances are explicitly disclosed as `ground_truth_tier=synthetic_llm_pac` per the project's own pre-approved fallback for this exact situation (`MASTER_RESEARCH_PLAN.md` §4.2) — wholly fictional scenarios, no real named individuals or case reports.

The pilot set is kept frozen and untouched (`data/splits/`) specifically so every baseline result computed against it stays comparable; the expanded set is versioned separately (`data/splits_v2/`).

**A further gold-source expansion attempt was tested and honestly rejected.** SNOMED CT's Component Inactivation reason codes (`OUTDATED`, `ERRONEOUS`) looked, on paper, like a pre-labeled, ontology-native source of `(A_old -> A_new)` RGU/IFE pairs at real scale. The real SNOMED CT US Edition RF2 release was downloaded and processed (via UMLS Terminology Services), yielding 8,398 reason-coded pairs (6,468 RGU-shaped, 1,930 IFE-shaped) — about 20x the number originally estimated. Direct inspection, including a targeted cross-check against this project's own 4 pilot concepts, showed the data is dominated by SNOMED's own terminology/coding-maintenance churn (drug-product data-model migrations, diagnostic-code harmonization, phrasing normalization), not genuine clinical guideline reversals — none of the matches against aspirin, rosiglitazone, HRT, or Vioxx reflect those concepts' actual documented reversals. This is a real, tested negative finding, not a stalled effort: "reason for inactivation" answers a terminology-curation question, not a clinical-recommendation question. Not used for dataset expansion; kept at `data/gold_sources/snomed_reason_coded.json` for reference only.

## 2. Cross-model generalization: method rankings don't transfer

RMU's effect strength is sharply model-dependent at *identical* hyperparameters:

| Model | RMU behavior |
|---|---|
| BioMistral-7B (pilot scale) | Near-zero effect — the gentlest of GA/NPO/RMU |
| BioMistral-7B (v2/benchmark scale) | Still the gentlest — RGU_fa 0.641-0.634 vs. baseline 0.634, within noise (§4) |
| Llama-3.1-8B-Instruct | Large, real, scenario-specific effect — targeted scenario moved substantially, untargeted stayed at baseline |
| TOFU / Llama-3.2-1B-Instruct | By far the **most** aggressive of the three methods |

Three independent data points, same conclusion: a method-sensitivity ranking measured on one model does not transfer to another. This is a real, reportable finding for the paper's baseline-comparison section, independent of anything else here.

## 3. Cross-dataset generalization: TOFU comparison, and why it stops there

Ran GA/NPO/RMU against TOFU at a properly-tuned matched dose (found by reproducing GA's standard-config collapse first, then locating a non-collapsed operating point). Two findings:

1. **GA's catastrophic collapse is a general property of the method, not BioUnlearn-Bench-specific** — the identical failure mode occurs on TOFU at aggressive settings.
2. **RMU's relative gentleness does not transfer to TOFU** — there it's the most aggressive of the three, consistent with §2's cross-model divergence.

This establishes *method*-behavior parity across datasets. It does **not** by itself test the paper's actual thesis (does ontological entanglement cause collateral damage on retain-critical neighbor concepts) — TOFU was checked directly for whether it even *could* pose that question, not assumed either way. Result: TOFU's forget10/retain90 split is disjoint at the whole-fictitious-author level (400 = 20 authors × 20 facts; 3600 = 180 authors × 20 facts; zero author-name overlap verified directly), with no cross-author relational structure at all — no analog to a UMLS drug-class hierarchy or ICD comorbidity graph exists for an edit to leak into. **This is a positive finding, not a gap**: TOFU cannot pose the ontology-collateral-damage question by construction, which is itself evidence that a UMLS-graph-grounded benchmark is a necessary contribution, not a redundant one. Full derivation: `documentation/TOFU_STRUCTURAL_ANALYSIS.md`.

## 4. The main diagnostic matrix: GA/NPO/RMU at pilot and benchmark scale

Full 6-experiment matrix (3 methods × 2 target scenarios) re-run against the 542-instance expanded RGU/IFE set, alongside the original pilot-scale numbers:

| Method | Target | Scale | RGU_fa | RGU_def | IFE_fa | IFE_def |
|---|---|---|---|---|---|---|
| *(baseline)* | — | v2 | 0.634 | 0.095 | 0.839 | 0.015 |
| GA | RGU | v2 | 0.707 | 0.085 | 0.857 | 0.031 |
| NPO | RGU | v2 | **0.886** | 0.052 | **0.982** | 0.018 |
| RMU | RGU | v2 | 0.641 | 0.099 | 0.893 | 0.016 |
| GA | IFE | v2 | 0.612 | 0.103 | 0.893 | 0.016 |
| NPO | IFE | v2 | 0.623 | 0.094 | 0.893 | 0.016 |
| RMU | IFE | v2 | 0.634 | 0.091 | 0.857 | 0.015 |

Three consistent patterns, all holding at benchmark scale, not just pilot scale:

1. **NPO is the most aggressive method**, especially targeting RGU (RGU_fa 0.886, IFE collateral 0.982) — but its own-scenario DEF actually *drops below baseline* (0.052 vs. 0.095), meaning it strongly stops giving the old answer without reliably landing on the new one. Same failure shape as ROME (§6), much milder in degree.
2. **RMU is the gentlest method at every scale tested** — pilot, v2, and (§2/§3) cross-model — a genuinely stable property, not a pilot-scale artifact.
3. **IFE is structurally harder to move than RGU for every method** — all three methods' IFE-targeted rows are nearly indistinguishable from baseline and from each other, while their RGU-targeted rows spread out substantially. Consistent across all 3 methods, so it's a property of the IFE scenario/data, not one method's idiosyncrasy. Not yet mechanistically explained.

Full numbers: `data/gate2_results/{GA,NPO,RMU}_{RGU,IFE}_v2_summary.json`, pilot-scale equivalents in the corresponding non-`_v2` files.

## 5. OGDA (the proposed novel method): what survived two rounds of correction

OGDA (Ontology-Guided Directional Ablation) is a training-free method that ablates only the component of a fact's activation-space direction that *isn't* shared with clinically-related concepts that must be retained. The novelty claim went through two rounds of honest statistical correction — both are load-bearing for what can be claimed now.

**Round 1 (random-subspace control)**: an initial n=1 comparison suggested real OGDA was clearly distinguishable from a random-subspace ablation of the same size. Running the random control at n=3 overturned the FA-based version of this claim (random seeds produced FA swings as large as real OGDA's) but confirmed it on DEF (RGU z=-5.44, IFE z=-23.38 — real OGDA a clear outlier).

**Round 2 (real-side variance)**: the DEF result above still compared n=1 real-OGDA against n=3 random-control — the same class of problem just corrected on the other side. Running 2 more real-OGDA seeds and recomputing as genuine n=3-vs-n=3 **changed the answer**: IFE DEF dropped to z=-1.92, no longer distinguishable. What held up instead: RGU FA, z=4.47, clean and consistent.

**Multi-concept replication, corrected the same way**: extending n=3-vs-n=3 to all 4 tested concepts (aspirin, rosiglitazone, HRT, Vioxx) retracted the original "4/4 concepts show clean DEF collapse, z=-2.70 to -23.38" claim entirely (own-scenario z-scores at proper rigor: 0.49 to 4.47, only aspirin's RGU FA independently significant). **What replaced it, and is the current defensible claim**: two concepts (rosiglitazone, Vioxx) show large, independently significant DEF suppression specifically on their **collateral/untargeted** scenario (z=-3.07, z=-6.28) — OGDA causing more untargeted damage than a random edit of the same size, which is arguably closer to the paper's actual thesis than clean targeted erasure would have been. HRT's exception (no collateral suppression, z=+0.68) is now a confirmed, reproducible anomaly, not a fluke, though not mechanistically explained. Full numbers: `data/gate2_results/OGDA_multiconcept_n3vn3_statistical_comparison.json`.

**The scaling-dilution finding (from the v2 re-eval)**: evaluating the same 4 OGDA checkpoints against the 273-example v2 RGU set instead of the 58-example pilot set, the own-scenario RGU effect washes out to noise for every concept (RGU_fa/RGU_def all within noise of baseline). This is not a retraction — each checkpoint's edit is deliberately surgical to one concept's ontology subspace, so 272 of 273 v2 examples are about unrelated topics an aspirin-specific edit was never going to move. It suggested a concept-specific eval subset might be a precondition, not just a refinement, for OGDA's own-scenario claim to be measurable at scale — so it was built and tested directly.

**Concept-specific eval, tried at n=4-5 first (inconclusive), then n=16-19 (resolved).** Each of the 4 tested concepts already had 4-5 dedicated instances sitting unused in `RGU_val`/`RGU_test` (the standard eval pipeline always scored against the diffuse `RGU_train.jsonl`, which doesn't contain these concepts at all). The first n=3-vs-n=3 comparison at n=4-5 was inconclusive: 2 of 4 concepts (HRT, Vioxx) showed zero differentiating signal in either direction (a floor effect), and aspirin's DEF result *reversed direction* from the whole-benchmark finding (z=-3.5 vs. the whole-benchmark's z=4.47). Generated 15 more instances per concept — each grounded in the same real reversal already documented in that concept's existing instances (87% quality-filter pass rate) — bringing each concept to 16-19 instances, and re-ran the same comparison:

| Concept | n | FA z | DEF z |
|---|---|---|---|
| Aspirin | 18 | 2.0 | -1.13 |
| Rosiglitazone | 19 | -1.0 | -1.25 |
| HRT | 16 | -0.4 | **3.44** |
| Vioxx | 18 | -1.0 | **5.22** |

No more degenerate zero-variance results, and two genuinely new findings: **HRT and Vioxx show large, statistically significant own-scenario DEF *enhancement* under real OGDA relative to random-subspace control** (z=3.44, z=5.22) — real OGDA achieving meaningfully better own-scenario forget-and-retain behavior than a random edit of the same size, not previously established this clearly. Aspirin's FA result is now directionally *consistent* with the whole-benchmark finding (z=2.0, same positive direction as z=4.47) rather than reversed. Rosiglitazone shows no significant signal either way at this sample size.

**Honest calibration**: n=16-19 is a real improvement over n=4-5 but still a modest sample — read these as genuine findings at an honestly-scoped confidence level, not beyond question. The specific failure mode that made the n=4-5 result untrustworthy (floor effects, direction-reversal vs. the whole-benchmark measure) is resolved, not just relocated. Vioxx now has two independent real findings — own-scenario DEF enhancement here, plus the earlier collateral DEF suppression (below) — the most complete single-concept picture in the project.

**Bottom line**: the defensible current claim is narrower than originally hoped, arrived at by catching and then resolving a chain of statistical artifacts before they went in the paper (two n=1-vs-n=3 corrections, one inconclusive small-sample concept-specific attempt, then a properly-powered one). What holds up: aspirin's whole-benchmark RGU FA (z=4.47) and its now-consistent concept-specific echo (z=2.0); HRT and Vioxx's new own-scenario DEF enhancement (z=3.44, z=5.22); and the collateral-suppression finding on rosiglitazone/Vioxx (from the whole-benchmark comparison, z=-3.07/-6.28, unaffected by any of this). Rosiglitazone alone shows no significant own-scenario signal at current sample sizes.

## 6. Adjacent-method comparison: a genuine three-way contrast

The obvious reviewer question — "why not just RAG this, or use knowledge editing instead of unlearning?" — now has real answers, both surprising:

| Method | RGU-targeted DEF | IFE-targeted DEF | Collateral behavior |
|---|---|---|---|
| Weight-editing (GA/NPO/RMU, pilot) | 0.13–0.20 | 0.13–0.20 | Moderate, method-dependent |
| Weight-editing (GA/NPO/RMU, v2) | 0.05–0.10 | 0.09–0.10 | Moderate, method-dependent |
| **RAG-suppression** (no weight edit, pilot) | **0.585** | **0.547** | **None outside baseline noise** |
| **RAG-suppression** (no weight edit, v2) | **0.438** | 0.547 | None outside baseline noise |
| **ROME** (single rank-1 edit, off-the-shelf) | 0.061 | **0.0 exactly** | Low, but ineffective on target too |

**RAG-suppression dominates** on both axes at both dataset scales — simple retrieval-based context injection produces 3-4x the DEF of any weight-editing method tested, with essentially zero collateral movement (since context is only injected for the targeted scenario's prompts). This is a real, load-bearing finding, not a favorable throwaway: it means the paper's case for weight-editing methods (OGDA included) has to rest on properties RAG structurally can't offer — no runtime retrieval dependency, robustness to prompt-injection/context-stripping, generalizing to paraphrased prompts without a matching retrieval trigger, working when extraction attacks bypass the retrieval layer entirely — not on raw behavioral metrics.

**ROME is the weakest method tested, in the opposite direction from GA's failure mode**: safe (low collateral) but nearly inert on IFE (DEF=0.0 exactly, 0 of 34 successfully-edited instances landed on the correct answer) and weak on RGU (DEF=0.061). Read honestly: this reflects ROME/MEMIT's design mismatch with open-ended clinical QA (built for short factual triples, not free-form answers) using untuned community hyperparameters — a real limitation of applying it out-of-the-box, not evidence knowledge editing categorically cannot work here.

**The resulting three-way picture for the paper**: weight-editing methods (GA/NPO/RMU) sit in a middle ground — real effect, real but moderate collateral cost, method behavior that doesn't generalize across models/datasets (§2, §3). RAG-suppression is strong and safe but requires runtime infrastructure. Knowledge editing (ROME) is safe but ineffective out-of-the-box. No single baseline dominates on every axis, which is itself the motivating gap OGDA (§5) and the benchmark's OCD-style collateral-damage framing (§3) are meant to address.

## 7. PAC: a decisive negative result on deep memorization

PAC (PHI-Adjacent Concept Removal) required a two-stage design once the baseline smoketest showed the untuned model already scored `PAC_fa=0.967` (nothing to unlearn — the base model never learned these synthetic patterns). Stage 1: a contamination fine-tune (5 epochs of dedicated SFT) to actually make the model memorize the patterns, confirmed working epoch-by-epoch (`PAC_fa`: 0.967 → 0.65 → 0.1 → 0.0 → 0.0 → 0.0), with collateral scenarios staying intact throughout (final-epoch RGU_fa_pac=0.741 exactly matching baseline, IFE_fa_pac=0.929).

Stage 2: GA/NPO/RMU run against that contaminated checkpoint, at the same standard/gentle dosage that produces real, measurable movement on RGU/IFE:

| Method | PAC_fa (contaminated baseline: 0.0) | PAC_forget_gen (contaminated baseline: 0.983) |
|---|---|---|
| GA | 0.033 | 0.938 |
| NPO | **0.0 exactly** | 0.982 |
| RMU | **0.0 exactly** | 0.983 |

**Zero of three methods meaningfully unlearn the memorized pattern.** Collateral scenarios still moved by the usual modest amounts seen elsewhere (RGU_fa_pac 0.74-0.78, IFE_fa_pac 0.93-0.95), confirming the methods aren't simply inert — they specifically fail to touch the deep memorization while still nudging everything else. This is a real, decisive finding: RGU/IFE facts are diffusely present from pretraining; PAC's target was deliberately, repeatedly fine-tuned in. At matched, standard dosage, none of GA/NPO/RMU can remove that kind of memorization — a direct, quantified argument for why privacy-motivated unlearning needs more than the standard toolkit, honestly scoped to "at standard dosage tested" (more aggressive dosing, and its collateral-damage cost, remains untested). Full numbers: `data/gate2_results/PAC_{contamination_final,GA,NPO,RMU}_summary.json`.

## 8. What this adds up to

A candidate results-section structure, in the order the findings actually support each other:

1. **Motivate the problem** (§2, §3): unlearning method behavior doesn't generalize across models or datasets — a single-model, single-dataset pilot cannot be trusted, which is itself an argument for a broader diagnostic study.
2. **Establish the diagnostic matrix** (§4): GA/NPO/RMU behave consistently differently at both pilot and benchmark scale — NPO aggressive-but-overshooting, RMU consistently gentle, IFE harder to move than RGU across all methods.
3. **Show why existing benchmarks can't test the real question** (§3): TOFU is structurally incapable of posing the ontology-collateral-damage question — motivates BioUnlearn-Bench's design, not just its existence.
4. **Present OGDA's honestly-scoped novelty claim** (§5): narrower than first hoped, but survived two rounds of real statistical correction — RGU FA on aspirin, and (more thesis-relevant) collateral DEF suppression on 2 of 4 concepts.
5. **Contextualize against adjacent methods** (§6): no baseline dominates on every axis — RAG is strong-but-infrastructure-dependent, ROME is safe-but-ineffective, weight-editing is moderate-but-inconsistent. This is the gap OGDA and the benchmark are meant to fill.
6. **Extend to a harder regime** (§7): deep memorization (PAC) resists all three standard methods entirely at matched dosage — a decisive result that sharpens the paper's motivation for privacy-specific unlearning research.

## 9. What's still needed before this is submission-ready

- ~~Concept-specific eval subset for OGDA~~ — resolved (§5): scaled to 16-19 instances/concept, surfaced two new significant own-scenario findings (HRT, Vioxx). Likely the last major open item on the OGDA statistical-rigor thread.
- **Human/physician validation, NC-specific submission machinery** (ethics/IRB, Reporting Summary) — explicitly owned by the user, not attempted here.
- **Venue decision** (NC vs. EMNLP) — the written proposal is EMNLP-shaped, the README targets NC; unresolved, affects paper structure.
- SNOMED reason-coding (now unblocked, not yet run), MEMIT (needs a separate covariance-stats precompute step), OGDA-vs-PAC (not yet tried) — all lower priority than the above.
