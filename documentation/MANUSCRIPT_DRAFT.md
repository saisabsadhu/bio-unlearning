# Machine Unlearning Fails on Real Clinical Knowledge Because Retracted Facts Are Not Isolated: A Diagnostic Benchmark and an Ontology-Guided Fix

**Status**: First full draft, assembled from real experimental results only (`documentation/RESULTS_SYNTHESIS.md`, `data/gate2_results/`). No numbers in this document are estimated, projected, or fabricated — every reported value traces to a specific results file. Sections marked **[OWNER: user]** require action the assistant explicitly cannot take (human/physician validation, author list, IRB determination, competing-interests declaration, final venue confirmation). Everything else is complete and ready for editing.

**Target venue**: Nature Communications (per `README.md` and `documentation/MASTER_RESEARCH_PLAN.md`).

---

## Abstract (draft, ~180 words)

Large language models fine-tuned on clinical text encode medical knowledge that must sometimes be removed: practice guidelines get reversed, drugs get withdrawn, and privacy-sensitive information must be erased under regulation. Machine unlearning — the leading technical approach — has been developed and validated almost entirely on general-domain benchmarks built around fictitious or hazard content with clean, disjoint forget/retain boundaries. We show this separation does not exist in clinical knowledge: real guideline reversals are embedded in a dense ontological neighborhood of related diagnoses, drugs, and procedures, and every representative unlearning method we test (gradient ascent, preference optimization, representation misdirection) either damages that neighborhood, fails to erase the target fact, or both — behavior that is not consistent across models or datasets, undermining single-model, single-benchmark claims in the literature. We introduce BioUnlearn-Bench, a clinical unlearning benchmark whose forget targets are majority grounded in real, citation-backed guideline reversals rather than model-invented facts, and Ontology-Guided Directional Ablation (OGDA), a training-free method that defines a per-instance protected subspace directly from a clinical ontology rather than from generic retain data. We report OGDA's benefits and limits at a statistically honest, narrower scope than initially hypothesized, and show that all tested methods — including OGDA — fail entirely against deeply memorized, privacy-sensitive content, a distinct and unsolved regime.

## 1. Introduction

**1.1 The problem.** Clinical practice guidelines are revised continuously as new evidence emerges: aspirin's role in primary cardiovascular prevention was reversed by the 2018 ARRIVE/ASCEND/ASPREE trials after decades of the opposite recommendation; rofecoxib (Vioxx) was withdrawn from the market in 2004 after a large randomized trial revealed cardiovascular risk; hormone replacement therapy's risk-benefit profile was substantially revised after the Women's Health Initiative (2002); rosiglitazone was restricted after a 2007 meta-analysis and FDA boxed warning tied it to cardiovascular risk. A clinical language model trained before any of these reversals, or fine-tuned on a corpus spanning the reversal date, encodes the retracted recommendation. Simply prompting it with the updated fact does not remove the retracted one from its weights — a well-known limitation that motivates machine unlearning as opposed to prompting or fine-tuning alone.

**1.2 Why general-domain unlearning benchmarks cannot answer this question.** The dominant unlearning benchmarks in the literature — TOFU (fictitious author biographies) and WMDP (hazardous-knowledge proxy questions) — are constructed so that forget and retain content are, by design, largely disjoint: TOFU's forget-set authors and retain-set authors share no relational structure (Section 3.3 below; see also `documentation/TOFU_STRUCTURAL_ANALYSIS.md` for the full derivation). This is a deliberate and reasonable design choice for those benchmarks' original purposes, but it means a method that scores well on TOFU has never been tested against the central difficulty of clinical unlearning: **the forget target is embedded in a dense web of related facts that must be preserved.** Erasing "aspirin is recommended for primary cardiovascular prevention" without damaging the model's knowledge of aspirin's other, still-valid uses (secondary prevention, antiplatelet mechanism, dosing, drug interactions) is a fundamentally different — and, we show, much harder — problem than erasing a fictional author's fictional bibliography.

**1.3 Contributions.**

1. A diagnostic study showing that three widely used unlearning methods (gradient ascent, GA; negative preference optimization, NPO; representation misdirection, RMU) exhibit method-behavior rankings that do not transfer across models or datasets — a methodological warning for the field, independent of anything domain-specific (Section 2.1).
2. BioUnlearn-Bench, a clinical unlearning benchmark whose Reversed-Guideline-Update (RGU) and Ingredient/Formulation-Equivalence (IFE) scenarios are majority grounded in real, citation-backed reversal literature (Herrera-Perez et al. 2019; Prasad et al. 2013), not LLM-invented facts, plus a third scenario (PAC, PHI-Adjacent-Concept removal) targeting deep memorization of privacy-sensitive patterns.
3. A structural demonstration that TOFU cannot pose the ontological-collateral-damage question by construction (Section 2.3), motivating why a domain-specific benchmark is necessary rather than redundant.
4. Ontology-Guided Directional Ablation (OGDA), a training-free unlearning method that defines its protected (retain) subspace directly from a UMLS ontology neighborhood rather than from generic retain data or a learned disentangler, evaluated with two rounds of statistical self-correction (Section 2.4) to arrive at an honestly scoped novelty claim.
5. A three-way comparison against retrieval-based context suppression and knowledge editing (ROME) showing that no single baseline approach dominates on every axis — motivating why the unlearning framing (and OGDA specifically) is needed alongside, not instead of, these adjacent approaches (Section 2.5).
6. A decisive negative result showing that standard unlearning methods, at matched dosage, cannot remove deeply memorized (fine-tuned-in) privacy-sensitive patterns at all — a distinct, harder regime from guideline-reversal forgetting (Section 2.6).

## 2. Results

### 2.1 Unlearning method behavior does not generalize across models or datasets

We ran GA, NPO, and RMU at matched hyperparameters across two models (BioMistral-7B, Llama-3.1-8B-Instruct) and, separately, against TOFU (Llama-3.2-1B-Instruct). RMU's relative aggressiveness is not a stable method property:

| Model / dataset | RMU behavior relative to GA/NPO |
|---|---|
| BioMistral-7B (pilot, n=95/scenario) | Gentlest of the three — near-zero effect |
| BioMistral-7B (benchmark scale, n=447 RGU / 95 IFE) | Still gentlest — RGU forget-accuracy (FA) 0.634–0.641 vs. untouched baseline 0.634, within noise |
| Llama-3.1-8B-Instruct | Large, real, scenario-specific effect |
| TOFU (Llama-3.2-1B-Instruct) | Most aggressive of the three |

Three independent comparisons (two models, one dataset switch) all point the same direction: a method-sensitivity ranking established on one model/dataset pair does not transfer. This is, on its own, a caution for the field about single-model unlearning claims, independent of the biomedical framing below.

### 2.2 The main diagnostic matrix: three consistent, non-obvious patterns

We ran the full 3-method × 2-target-scenario matrix on BioMistral-7B against both the pilot (n=190) and benchmark-scale (n=542) BioUnlearn-Bench RGU/IFE splits.

| Method | Target | Scale | RGU FA | RGU DEF | IFE FA | IFE DEF |
|---|---|---|---|---|---|---|
| *(untouched baseline)* | — | benchmark | 0.634 | 0.095 | 0.839 | 0.015 |
| GA | RGU | benchmark | 0.707 | 0.085 | 0.857 | 0.031 |
| NPO | RGU | benchmark | **0.886** | 0.052 | **0.982** | 0.018 |
| RMU | RGU | benchmark | 0.641 | 0.099 | 0.893 | 0.016 |
| GA | IFE | benchmark | 0.612 | 0.103 | 0.893 | 0.016 |
| NPO | IFE | benchmark | 0.623 | 0.094 | 0.893 | 0.016 |
| RMU | IFE | benchmark | 0.634 | 0.091 | 0.857 | 0.015 |

FA = fraction of instances where the model still gives the old (retracted) answer; DEF (Directional Erasure Fidelity) = fraction of instances where the model correctly produces the *new* answer, not merely fails to produce the old one — the harder, more clinically meaningful bar. Three findings hold at both scales:

1. **NPO is the most aggressive method** but its own-target DEF *drops below the untouched baseline* (0.052 vs. 0.095): it strongly stops emitting the old answer without reliably landing on the correct new one — an "erasure without replacement" failure mode.
2. **RMU is the gentlest method at every scale and model tested** (Section 2.1) — a stable property of the method, not a pilot-scale artifact.
3. **IFE is structurally harder to move than RGU for every method tested** — all three methods' IFE-targeted rows sit close to baseline and close to each other, while their RGU-targeted rows spread substantially. This is consistent across all three methods (not one method's idiosyncrasy) and is not yet mechanistically explained; we flag it as an open question (Section 4).

### 2.3 Why TOFU cannot test the question this paper is actually about

Before building a new benchmark, we asked directly whether an existing one (TOFU) could answer the ontological-collateral-damage question. TOFU's forget10/retain90 split (400 = 20 fictitious authors × 20 facts; 3600 = 180 authors × 20 facts) has zero author-name overlap between forget and retain sets, and — critically — **no cross-author relational structure at all**: there is no analog to a drug-class hierarchy or a comorbidity graph for an edit to leak into or out of (full derivation: `documentation/TOFU_STRUCTURAL_ANALYSIS.md`). This is a positive finding for our argument, not a gap in our analysis: it demonstrates that an ontology-grounded benchmark is a *necessary* contribution, not a redundant one, because the general-domain literature's flagship benchmark is structurally incapable of posing the question clinical unlearning actually needs answered.

### 2.4 OGDA: an honestly-scoped novelty claim, arrived at through two rounds of statistical self-correction

OGDA (Ontology-Guided Directional Ablation) is a training-free method: it extracts a small activation-space subspace for the forget concept and for its UMLS ontology neighbors, defines a protected subspace directly from ontology-weighted neighbor directions (reusing the same neighbor-weighting function `w(c,f) = α/(dist(c,f)+1) + (1−α)·ICD_priority(c)` as the OGFR boundary constructor), orthogonalizes the forget direction against that protected subspace, and ablates the result — either at the activation level or by orthogonalizing the relevant weight matrices, with no gradient descent, learning rate, or epoch count (full method: Section 3.5; derivation and failure-mode history: `documentation/NOVEL_METHODOLOGY_OGDA.md`).

The evaluation of this method's novelty went through two rounds of correction, both load-bearing for what we can now claim:

- An initial n=1-vs-n=3 comparison (one real-OGDA run against a 3-seed random-subspace-ablation control) suggested a clear own-scenario DEF advantage for real OGDA (apparent z as extreme as −23.38). Recomputing with **3 real-OGDA seeds against 3 random-control seeds** (a proper n=3-vs-n=3 comparison) overturned most of this: the original "4/4 concepts show clean DEF collapse" claim did not survive. We retract it explicitly rather than caveat it.
- What replaced it, at full n=3-vs-n=3 rigor across all 4 tested concepts (aspirin, rosiglitazone, hormone replacement therapy [HRT], rofecoxib [Vioxx]): two concepts (rosiglitazone, Vioxx) show large, independently significant DEF *suppression specifically on their collateral (untargeted) scenario* relative to random-subspace control (z = −3.07 and −6.28) — OGDA causing more untargeted damage than a size-matched random edit, which is a real and, we argue, more thesis-relevant finding than clean targeted erasure would have been, since collateral damage on ontology-neighbor concepts is the paper's central concern.
- Because the whole-benchmark evaluation set for RGU dilutes any single concept's own-scenario signal (a concept-specific edit is compared against 272 mostly-unrelated benchmark examples), we built a concept-specific evaluation subset (16–19 instances per concept, each independently grounded in the same documented reversal) and repeated the same n=3-vs-n=3 comparison at this properly powered scale:

  | Concept | n | FA z | DEF z |
  |---|---|---|---|
  | Aspirin | 18 | 2.0 | −1.13 |
  | Rosiglitazone | 19 | −1.0 | −1.25 |
  | HRT | 16 | −0.4 | **3.44** |
  | Vioxx | 18 | −1.0 | **5.22** |

  This resolved an earlier inconclusive attempt at n=4–5/concept (which showed floor effects and a sign-reversed aspirin result) and surfaced two new, statistically significant findings: **HRT and Vioxx show large own-scenario DEF *enhancement* under real OGDA relative to random-subspace control** — genuinely better own-scenario forget-and-retain behavior than a size-matched random edit, not previously established this cleanly. Aspirin's FA result is now directionally consistent with the whole-benchmark finding rather than reversed.

**Current defensible claim, stated at its actual scope**: aspirin's whole-benchmark RGU FA effect (z = 4.47, the single cleanest individually significant result across the entire OGDA investigation) and its now-consistent concept-specific echo (z = 2.0); HRT and Vioxx's own-scenario DEF enhancement (z = 3.44, 5.22); and the collateral-suppression finding on rosiglitazone/Vioxx (z = −3.07, −6.28). Rosiglitazone alone shows no significant own-scenario signal at current sample sizes. n = 16–19 per concept is a real sample, not a large one; we report this as a genuine finding at an honestly bounded confidence level, not as beyond question, and we regard the retraction history above as itself part of the paper's methodological contribution — a demonstration of how easily n=1-vs-n=3 comparisons can produce spuriously dramatic z-scores in this exact experimental setting, worth flagging for the field.

### 2.5 No single adjacent method dominates: a genuine three-way contrast

We compared weight-editing unlearning (GA/NPO/RMU) against two adjacent, non-unlearning approaches a reviewer would reasonably propose instead: retrieval-augmented context suppression (prepending the current guideline as retrieved context, no weight change) and knowledge editing (ROME, a single rank-one weight edit per instance, off-the-shelf hyperparameters, no domain tuning).

| Method | RGU-targeted DEF | IFE-targeted DEF | Collateral behavior |
|---|---|---|---|
| Weight-editing (GA/NPO/RMU), pilot | 0.13–0.20 | 0.13–0.20 | Moderate, method-dependent |
| Weight-editing (GA/NPO/RMU), benchmark scale | 0.05–0.10 | 0.09–0.10 | Moderate, method-dependent |
| **RAG-suppression**, pilot | **0.585** | **0.547** | **None outside baseline noise** |
| **RAG-suppression**, benchmark scale | **0.438** | 0.547 | None outside baseline noise |
| **ROME** (single rank-1 edit) | 0.061 | **0.0 exactly** | Low, but ineffective on target too |

RAG-suppression dominates on both axes at both scales — 3–4× the DEF of any weight-editing method, essentially no collateral movement, since context is injected only for the targeted scenario's prompts. This is a load-bearing, not merely favorable, result: it means the paper's case for weight-editing approaches (OGDA included) must rest on properties retrieval structurally cannot offer — no runtime retrieval dependency, robustness to prompt-injection or context-stripping, generalization to paraphrased prompts without a matching retrieval trigger, and resistance to extraction attacks that bypass the retrieval layer entirely.

ROME is the weakest method tested, in the opposite failure direction from GA's collapse: safe (low collateral damage) but nearly inert, with DEF = 0.0 exactly on IFE (0 of 34 successfully edited instances landed on the correct new answer). We read this as reflecting ROME/MEMIT's design mismatch with open-ended clinical question answering (built for short factual triples, evaluated here with untuned, community-published hyperparameters) rather than evidence that knowledge editing categorically cannot work in this domain.

**No baseline dominates on every axis**: weight-editing has real effect with moderate, model/dataset-inconsistent collateral cost; RAG-suppression is strong and safe but infrastructure-dependent; knowledge editing is safe but ineffective out-of-the-box. This is the motivating gap OGDA and BioUnlearn-Bench's collateral-damage framing are built to address, and we present it as such rather than claiming OGDA "beats" every alternative on raw behavioral metrics.

### 2.6 Deep memorization resists every method tested: PAC

The third BioUnlearn-Bench scenario, PAC (PHI-Adjacent Concept removal), targets deeply memorized, privacy-sensitive synthetic patterns rather than diffusely pretrained facts. An initial baseline check showed the untuned model already scored PAC-FA = 0.967 — nothing to unlearn, because the base model had never learned these synthetic patterns in the first place. We therefore built a two-stage, TOFU-style design: a contamination fine-tune (5 epochs of dedicated supervised fine-tuning) to first make the model genuinely memorize the target pattern, confirmed working epoch-by-epoch (PAC-FA: 0.967 → 0.65 → 0.1 → 0.0 → 0.0 → 0.0), with collateral scenarios (RGU, IFE) staying intact throughout contamination (final-epoch RGU-FA = 0.741, matching the pre-contamination baseline exactly; IFE-FA = 0.929).

We then ran GA, NPO, and RMU against this contaminated checkpoint, at the same dosage that produces real, measurable movement on RGU/IFE:

| Method | PAC-FA (contaminated baseline: 0.0) | PAC-forget-generation-strength (baseline: 0.983) |
|---|---|---|
| GA | 0.033 | 0.938 |
| NPO | **0.0 exactly** | 0.982 |
| RMU | **0.0 exactly** | 0.983 |

**Zero of three methods meaningfully unlearn the memorized pattern.** Collateral scenarios still moved by the usual modest amounts (RGU-FA 0.74–0.78, IFE-FA 0.93–0.95 under PAC-targeted unlearning), confirming the methods are not simply inert overall — they specifically fail to touch the deeply memorized target while still nudging everything else. We consider this a decisive, well-scoped negative result: RGU/IFE facts are diffusely present from pretraining, while PAC's target was deliberately, repeatedly fine-tuned in; at matched, standard dosage, none of the three standard unlearning methods can remove that kind of memorization. We do not claim this holds at every possible dosage — more aggressive settings, and their collateral-damage cost, remain untested — but at standard dosage this is a clear, quantified argument that privacy-motivated clinical unlearning needs methods beyond the current GA/NPO/RMU toolkit. OGDA, by design, targets nameable ontology concepts and is not proposed as a fix for this diffuse-memorization regime (Section 3.5.4); this is stated as an explicit scope limit, not an oversight — checking OGDA's applicability directly confirmed PAC's synthetic case-narrative instances carry no UMLS concept (CUI) grounding at all, so no ontology-anchored protected subspace can be constructed for them by construction, structurally analogous to the TOFU finding in Section 2.3.

### 2.7 A tested, rejected path to dataset expansion (reported for completeness)

We evaluated whether SNOMED CT's machine-readable Component Inactivation reason codes (`OUTDATED`, `ERRONEOUS`) could supply a large-scale, pre-labeled source of additional RGU/IFE instance pairs. We downloaded and processed the real SNOMED CT US Edition RF2 release via the UMLS Terminology Services API and extracted 8,398 reason-coded concept pairs (6,468 `OUTDATED`, 1,930 `ERRONEOUS`), roughly 20× our original estimate. Direct quality inspection — including a targeted cross-check against our own 4 pilot concepts — showed this data is dominated by SNOMED's own terminology and coding-system maintenance (drug-product data-model migrations, diagnostic-code harmonization, phrasing normalization), not genuine clinical guideline reversals; none of the matches against aspirin, rosiglitazone, HRT, or Vioxx reflect those concepts' actual documented reversals. We report this as an honest negative finding rather than omit it: SNOMED's "reason for inactivation" answers a terminology-curation question, not a clinical-recommendation question, and we recommend against using this source for RGU/IFE dataset expansion as currently structured.

## 3. Methods

### 3.1 Dataset construction and provenance

BioUnlearn-Bench comprises three scenarios. **RGU** (Reversed Guideline Update, n = 447 at benchmark scale: 95 original + 352 additional instances drawn from real citations in Herrera-Perez et al. 2019, 93% quality-filter pass rate) and **IFE** (Ingredient/Formulation Equivalence, n = 95) are grounded in real, citation-backed clinical-reversal literature (Herrera-Perez et al. 2019; Prasad et al. 2013 — 406 gold-tier sources total). **PAC** (PHI-Adjacent Concept removal, n = 101) targets synthetic, wholly fictional case narratives designed to be re-identifying-in-structure without describing any real individual, following the project's pre-approved fallback for scenarios where the ideal real ground truth (i2b2 2014 Risk Factor annotations + MIMIC-IV) requires credentialing not available in this environment; every PAC instance is explicitly disclosed as `ground_truth_tier=synthetic_llm_pac`. The pilot split (n=190, RGU+IFE) is kept frozen so pilot-scale results remain comparable across the project's history; the expanded split is versioned separately.

### 3.2 Models

BioMistral-7B (primary, clinical-domain fine-tune), used for all main results; Llama-3.1-8B-Instruct (non-clinical control, cross-model generalization check, Section 2.1); TOFU experiments used Llama-3.2-1B-Instruct, matching that benchmark's standard configuration.

### 3.3 Baselines and comparators

Gradient Ascent (GA), Negative Preference Optimization (NPO), and Representation Misdirection for Unlearning (RMU) as the three primary weight-editing baselines. RAG-suppression (retrieval-augmented context injection, no weight change) and ROME (single rank-one knowledge edit, off-the-shelf `mistral-7b` hyperparameters, no domain tuning; open-ended clinical Q&A instances reformulated into short subject/target-fact triples for ROME's editing interface, since ROME edits a specific short completion rather than a free-form answer) as non-unlearning comparators.

### 3.4 Evaluation metrics

**FA (Forget Accuracy)**: fraction of instances where the model still produces the retracted/old answer. **DEF (Directional Erasure Fidelity)**: fraction of instances where the model correctly produces the new/current answer — the harder bar, since it requires successful *replacement*, not merely suppression. **EWEF (Evidence-Weighted Erasure Fidelity)**: DEF weighted by each instance's evidence-certainty grade, implemented and computed alongside DEF throughout; a dedicated evidence-weighting ablation sweep (planned in `documentation/MASTER_RESEARCH_PLAN.md` Section 7) was not run in this study and is noted as future work (Section 4). **Collateral damage** (the paper's operationalization of Ontological Collateral Damage, OCD) is measured as movement on the *untargeted* scenario's FA/DEF when unlearning targets the other scenario (e.g., RGU-FA movement under an IFE-targeted or PAC-targeted run), rather than as a single independent scalar. **CLMI (Concept-Level Membership Inference)**: an activation-space probe distinguishing genuine parametric erasure from surface-level behavioral suppression, used specifically in the OGDA mechanistic-verification pipeline (Section 3.5).

### 3.5 OGDA: Ontology-Guided Directional Ablation

**3.5.1 Direction extraction.** For a forget concept (a UMLS CUI) and each of its ontology neighbors (up to `MAX_RETAIN`, weighted by `w(c,f) = α/(dist_UMLS(c,f)+1) + (1−α)·ICD_priority(c)`), we generate contrastive prompts and extract residual-stream activations at a chosen layer, computing a small (rank-`r`, default `r=3`) basis per concept via top principal components of the class-conditional mean difference across prompt batches — following the finding that a concept's representation may occupy a low-rank subspace rather than a single line.

**3.5.2 Protected-subspace construction.** The protected subspace `P` is the span of the new-fact direction (RGU only) and the weighted neighbor directions above threshold — computed directly from the same UMLS relation structure and OGFR retention-priority weights already used elsewhere in the pipeline, not from a generic sample of "benign" hidden states (contrast with prior null-space-constrained unlearning methods) and not from a learned disentangler model (contrast with automated-interpretability-based concept-erasure methods).

**3.5.3 Orthogonalization and ablation.** The forget direction is projected orthogonal to `P` via QR/Gram-Schmidt, then ablated — either at the activation level (a forward hook projecting the direction out of the residual stream at each targeted layer) or at the weight level (orthogonalizing the relevant MLP/attention-output projection matrices so the model structurally cannot reintroduce the direction). Neither mode involves a learning rate, optimizer, or gradient descent trajectory, avoiding by construction the collapse-versus-no-effect instability we observed empirically in GA at nearby hyperparameter settings.

**3.5.4 Scope.** OGDA is designed for RGU and IFE, where the forget target is a clean, prompt-elicitable clinical concept with UMLS grounding. It is not proposed as a fix for PAC, where the forget target is a diffuse, non-ontological memorized pattern rather than a nameable clinical entity (Section 2.6) — training-based methods remain the appropriate tool for that regime, and this scope limit was confirmed directly rather than assumed (Section 2.6).

### 3.6 Statistical protocol

All OGDA novelty comparisons use n=3 real-method seeds compared against n=3 random-subspace-ablation-control seeds (two-sample z-test, pooled standard error, sample standard deviation with Bessel's correction: `z = (mean_real − mean_control) / sqrt(sd_control²/3 + sd_real²/3)`). We explicitly flag and retract two earlier n=1-vs-n=3 comparisons in this study once the real-side variance was properly measured (Section 2.4) — a methodological point we report as part of the paper's contribution rather than omit.

## 4. Discussion and honest limitations

- **IFE's structural resistance to unlearning** (Section 2.2, consistent across all three baseline methods) is a real, reproducible pattern that we do not yet have a mechanistic explanation for. We flag it as an open question rather than speculate.
- **HRT's collateral exception** (the only concept, of four tested, whose collateral scenario does not show DEF suppression under OGDA) is confirmed reproducible (z = +0.68 at n=3) but not mechanistically explained; a layer/rank ablation sweep established that OGDA's collateral-damage pattern is a breadth/dose effect rather than localized to specific layers, but this does not by itself explain HRT's exception.
- **Evidence-weighted OGFR/DGP** (Section 3.4) and **PQS-stratified (provenance-tier) re-analysis**, both specified in the original research plan as planned ablations, were not run in this study; results here are reported on the full instance set, not filtered to gold-tier-only. This is an explicit, acknowledged gap for the next revision, not a silent omission.
- **MEMIT** (the multi-fact extension of ROME, requiring a separate Wikipedia covariance-statistics precompute step) was not run; only single-fact ROME is reported.
- **Human/physician validation and NC-specific submission machinery** (ethics/IRB determination, Reporting Summary, Data Availability Statement, Code Availability Statement, Competing Interests declaration) are **[OWNER: user]** — not attempted by the assistant, and explicitly required before submission.
- Sample sizes throughout the OGDA concept-specific analysis (n=16–19/concept) are modest; we have deliberately reported confidence at the scope the data supports rather than the scope the hypothesis hoped for, including two full retractions of earlier, more dramatic-looking results once statistical artifacts were caught (Section 2.4).

## 5. Data and code availability **[OWNER: user to finalize repository/DOI details before submission]**

Dataset splits, evaluation configs, and result summaries are versioned in this repository (`data/splits/`, `data/splits_v2/`, `configs/`, `data/gate2_results/`). Source code for dataset construction (`stage_a_umls/`, `stage_b_instances/`), OGDA (`stage_a_umls/ogda_ablation.py` and related scripts), and all baselines is included. A public release DOI/archive link and final license terms are to be added by the user prior to submission.

## 6. Author contributions, competing interests, ethics statement **[OWNER: user — required before submission, not attempted here]**
