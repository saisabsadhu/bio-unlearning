# BioUnlearn: Master Research Plan (v1)

**Status**: DRAFT — integrates (1) the original EMNLP-shaped proposal's surviving content, (2) fixes from `critique_nature_communications_fit.md`, (3) the ground-truth research in `groundtruth_sources_dr_vindo_directions.md`, and (4) new methodology extensions. This document is the operational plan going forward — the original `.docx` proposal remains useful as an EMNLP-style background/lit-review reference but should not be treated as the execution plan.

**Target venue**: Nature Communications (per repo README). Framing, rigor bar, and required sections below are written for NC, not for an NLP conference.

**How to use this document**: Sections 1-9 define what the paper claims and why. Sections 10-16 define exactly what to build and run, in order, with explicit decision gates before committing further resources. Section 17 onward is process/logistics. If you are picking this up to start work, skip to Section 21 ("Immediate Next Actions") first, then come back to the relevant numbered section for detail.

---

## 1. Core Thesis (NC framing)

Large language models fine-tuned on clinical text encode medical knowledge that must sometimes be removed: guidelines get retracted, drugs get withdrawn, and privacy-sensitive patterns must be erased under HIPAA/GDPR. Machine unlearning — the leading technical approach to this — has been developed and validated almost entirely on general-domain benchmarks (TOFU, WMDP, MUSE) built around fictitious or hazard content with clean forget/retain separation. We show that this separation does not exist in clinical knowledge, and that every representative unlearning method fails as a result, in three specific, measurable, and previously invisible ways. We fix each failure with a domain-adaptive framework, and — critically — we build the evaluation benchmark and the framework's core mechanism out of **real, independently-documented clinical knowledge changes** (retracted guidelines, withdrawn drugs, reason-coded ontology inactivations) rather than synthetic or LLM-invented facts, because a paper making patient-safety claims should not rest its ground truth on the same class of model it is trying to fix.

That last sentence is the paper's actual novelty axis for an NC audience: not "three new metrics" (that's the EMNLP framing), but **"unlearning evaluated and enabled by verifiable, real-world clinical knowledge provenance, exploiting a structural property — standardized evidence-certainty grading — that only exists in the clinical-guideline domain."**

## 2. Contributions (revised, five named contributions)

1. **Diagnostic study**: six representative unlearning methods (GA, GD, NPO, RMU, SU, WHP), plus two non-unlearning comparators (knowledge editing, RAG-suppression — see Section 8), fail on at least one of three biomedical-specific axes (OCD, DEF, CLMI) on real clinical LLMs, invisible on TOFU/WMDP-Bio.
2. **Provenance-Tiered Ground Truth Construction (PTGC)** — new. A dataset construction methodology that grades every instance by source provenance (gold institutional record / reason-coded ontology diff / LLM-phrasing-only) and reports results stratified by tier, so that dataset reliability is falsifiable rather than asserted. This is itself a contribution independent of BioUnlearn-Bench — reusable by any future clinical-NLP dataset paper.
3. **Three evaluation metrics**: OCD (Ontological Collateral Damage), DEF (Directional Erasure Fidelity), CLMI (Concept-Level Membership Inference) — as before — plus **EWEF (Evidence-Weighted Erasure Fidelity)**, new, which uses each source's certified evidence-certainty grade (GRADE / ACC-AHA-ESC Class-LOE) to weight how much credit a reversal should get, directly addressing "how do you know the new answer won't be reversed too."
4. **BioUnlearn**: the OGFR / DGP / CLMI framework as originally designed, each extended with an **evidence-weighted variant** (Section 7) that is only possible because clinical guidelines carry standardized certainty metadata — general-domain forget targets (TOFU, WMDP) have no analogous field, which is why this is a genuine domain-specific methodological contribution rather than a generic method validated on medical data.
5. **BioUnlearn-Bench**: the first text-only generative clinical unlearning benchmark built with the PTGC methodology — majority of RGU/IFE instances grounded in real institutional records (Section 4), not LLM invention.

## 3. Related Work Positioning (updated)

Keep the existing differentiation table from the proposal doc (MLLMU-Med, MedForget, Hierarchical Dual-Strategy, STEU, DriftMedQA) — it remains accurate. Add:

| Paper | What it is | Differentiator |
|---|---|---|
| PrimeKG-CL (Radwan, Li et al., arXiv:2605.10529, 2026) | Continual graph-learning benchmark on two real PrimeKG snapshots (2021 vs 2023), 889K removed / 5.83M added edges across 9 databases | Different problem (continual *learning* on a KG, not LLM weight *unlearning*); different artifact (graph edges, not clinical QA text). Must cite and explicitly differentiate — read the full paper before submission to confirm no overlap in specific concepts used, and consider citing its edge-diff computation as validation that biomedical KGs really do have this much real churn. |
| ROME / MEMIT (knowledge editing) | Targeted weight edits that directly rewrite a fact (old→new) | Not previously a baseline in this plan — added in Section 8 because it natively matches the RGU scenario's directional (A_old→A_new) structure and is the most obvious reviewer-proposed alternative. |
| RAG-suppression (retrieval-augmented correction) | Prepend retracted/updated guideline text as retrieved context, no weight change | Added as a comparator, not a baseline to "beat" — its expected failure mode (passes DEF behaviorally, fails CLMI because the old fact is still fully present in weights) is itself evidence for Failure Mode 3 (Verification Opacity) and worth a dedicated discussion subsection. |

## 4. Ground-Truth Architecture: Provenance-Tiered Ground Truth Construction (PTGC)

### 4.1 The problem this solves

The original plan used Claude 3.5 Sonnet to both (a) generate BioUnlearn-Bench instances and (b) generate OGFR's retain-expansion instances. Same-oracle-for-benchmark-and-fix is a circularity a reviewer will flag (see critique doc). Separately, Dr. Vindo's objection was that UMLS-style ground truth, where it exists, mostly isn't clinically meaningful (vocabulary housekeeping), and where it is meaningful, it often isn't documented in a structured form at all.

### 4.2 Tier definitions and Provenance Quality Score (PQS)

Every instance in BioUnlearn-Bench gets a **PQS tag**, reported and stratifiable in every results table:

| Tier | PQS | Definition | Used for |
|---|---|---|---|
| **Gold** | 3 | Fact and its reversal/rationale come from a named institutional or peer-reviewed record with a date and citation, independent of any LLM | Primary RGU + IFE instances |
| **Silver** | 2 | Fact comes from a structured ontology diff with an explicit machine-readable reason code (not free LLM invention) | Secondary RGU + IFE instances, and all OCD/OGFR neighbor-boundary instances |
| **Oracle-phrasing** | 1 | The underlying fact is Gold or Silver; an LLM (Claude 3.5 Sonnet) is used *only* to phrase it as a natural clinical question, under a grounding-fidelity constraint (Section 5, Stage C) that rejects any phrasing introducing information not present in the source record | All instances (phrasing step is universal) — PQS 1 is not a separate fact source, it's a phrasing-quality flag layered on top of Gold/Silver |
| **LLM-only** | 0 | Fact invented by the oracle with no traceable external record (the original plan's approach) | **Eliminated as a permitted tier for RGU.** Retained only for PAC scenario synthetic-scenario generation where no real record can exist by construction (see 4.3), and there it is disclosed explicitly as such. |

Every main results table (Section 13) must be reportable filtered to PQS≥2 (gold+silver only) as well as on the full set, so a reviewer can check whether conclusions hold on the highest-provenance subset alone. This replaces "9.1% human-validated, rest silver" as the paper's answer to "how do we know your dataset is right" with something stronger: a majority of RGU/IFE instances are gold by construction, not by spot-check.

### 4.3 Source catalog (concrete, per scenario)

| Scenario | Source | Tier | Est. yield | Access method | Notes |
|---|---|---|---|---|---|
| RGU | Prasad et al. 2013 (Mayo Clinic Proc., 146 reversals) + Herrera-Perez et al. 2019 (eLife, 396 reversals) | Gold | ~400-500 after dedup | Published tables, manual/semi-automated extraction of practice + citation + year | Best single source; peer-reviewed, dated, already published years before this paper |
| RGU | FDA Drug Safety-related Labeling Changes (SrLC) database + Drugs@FDA/DailyMed historical label PDFs | Gold | ~150-250 curated (triaged for clinical prominence) | FDA public database (since 2016) + historical label diff | Gives literal before/after label text, exact date |
| RGU | WHO Model List of Essential Medicines — Expert Committee Technical Reports (biennial, since 1977) | Gold | ~50-100 (removals/rejections across cycles) | WHO Technical Report Series PDFs | Richest prose rationale of any source found |
| RGU | ESC / ACC / AHA cardiology guideline version comparisons (published systematic reviews already track Class/LOE changes) | Gold | ~100-200 | Extract from published comparison papers + guideline PDFs | Concentrated domain, good pilot candidate given cardiology already well-represented in MIMIC-IV |
| RGU | Cochrane review updates where conclusion flipped | Gold | small (~20-40) | Cochrane Library "What's New" sections, filtered to conclusion-change subset | Small but very high quality — use as a "clean gold" stress-test subset |
| RGU / IFE | SNOMED CT Component Inactivation Reference Sets, filtered to reason=`OUTDATED` (RGU) or reason=`ERRONEOUS` (IFE) | Silver | ~300-500 after filtering to CUIs overlapping MIMIC-IV/i2b2 domain | UMLS Metathesaurus license pull, RF2 release files | Also gives a free (A_old→replacement) pair via the historical association link |
| IFE | DrugBank "withdrawn" status field + RxNorm obsolete/deprecated cross-reference | Silver | ~50-100 | Already in your planned stack | Confirms drug-level withdrawal independent of guideline text |
| PAC | i2b2 2014 Risk Factor annotations + MIMIC-IV structured demographics | N/A (not a "reversal," so PQS scheme doesn't apply the same way) | 800 forget / 600 retain as originally planned | As originally planned | PAC is a privacy-removal scenario, not a temporal-reversal scenario — it does not benefit from the WHO/FDA/SNOMED sources above and keeps its original construction path. Do not force PAC into the PTGC reversal framework. |

**Do not fabricate final counts.** The numbers above are harvesting targets, not committed dataset sizes — Stage B0 (Section 5) determines real yield, and the Feasibility Gates (Section 14) block moving forward until real counts are in hand.

### 4.4 Evidence-certainty metadata schema

Normalize every Gold/Silver RGU instance's old and new evidence onto a common 4-point ordinal scale so metrics can compare across sources:

| Common scale | GRADE certainty (WHO/Cochrane/NICE) | ACC/AHA/ESC Level of Evidence | USPSTF grade (approx. mapping) |
|---|---|---|---|
| 3 (High) | High | A | A |
| 2 (Moderate) | Moderate | B | B |
| 1 (Low) | Low | C | C |
| 0 (Very low / expert opinion only) | Very low | C (expert consensus) | I (insufficient) |

Store both `old_certainty` and `new_certainty` per RGU instance. This is the substrate for EWEF and the evidence-weighted OGFR/DGP variants (Section 7).

### 4.5 MIMIC-IV temporal natural-experiment validation layer (new, optional but recommended)

For a subset of RGU instances with a precise change date (FDA warnings are best here — exact dates), bucket MIMIC-IV notes/orders by admission date relative to the change date and check whether real-world documented practice actually shifted (e.g., rosiglitazone: FDA warning May 21, 2007, independently published studies show ~70% prescribing drop within two years). This does not change the forget/retain construction, but for any instance where MIMIC-IV itself spans the relevant dates, it gives an extra, corpus-internal confirmation that the reversal was real and adopted — a distinctive selling point for an EHR-grounded paper. Treat as a bonus analysis (Section 13, secondary table), not a blocking dependency.

## 5. Dataset Construction Pipeline (revised, mapped to repo folders)

Existing repo folders: `stage_a_umls/`, `stage_b_instances/`. Below is the full pipeline with new stages named to fit this convention.

| Stage | Folder (existing / new) | What it does | Depends on |
|---|---|---|---|
| A | `stage_a_umls/` (extend) | UMLS graph traversal (existing) **+ new**: pull SNOMED CT RF2 Component Inactivation + Historical Association reference sets, filter to reason=`OUTDATED`/`ERRONEOUS`, cross-reference CUIs against MIMIC-IV/i2b2 concept frequency to prioritize clinically-relevant ones | UMLS/SNOMED license (already have via NLM UTS) |
| B0 | `stage_b0_gold_sources/` (**new**) | Harvest Gold-tier sources: parse Prasad/Herrera-Perez tables, pull FDA SrLC + Drugs@FDA historical labels for triaged drug list, extract WHO EML Technical Report removal/rejection entries, extract ESC/AHA/ACC version-comparison data from published papers, extract Cochrane conclusion-flip reviews. Output: per-instance JSON `{fact, old_value, new_value, source, citation, date, old_certainty, new_certainty, pqs=3}` | Stage A (for CUI cross-referencing) |
| B | `stage_b_instances/` (extend existing) | LLM-assisted **phrasing only** of Stage A/B0 facts into clinical QA format (Claude 3.5 Sonnet), under a strict grounding prompt that forbids adding unsourced detail. New automated **grounding-fidelity filter**: re-prompt the oracle to extract back out the fact it was given the phrasing for, and reject if the round-tripped fact doesn't match the source (catches hallucinated elaboration) | Stage A, B0 |
| C | `stage_b_instances/filter_instances.py` (extend) | Existing 4 automated filters (consistency, CUI validity, source verification, plausibility) **+ grounding-fidelity filter** (above) **+ PQS tagging** | Stage B |
| D | `stage_c_validation/` (**new — planned in README but not yet created**) | Human validation — **revised for NC rigor** (Section 18): expand beyond 2 MD-2 students; stratify validated sample by PQS tier and report agreement separately per tier | Stage C |
| E | `data/splits/` (existing) | Freeze splits, version with DVC, release prep | Stage D |

### CLMI-specific gate (mandatory, blocks Stage 2 experiments regardless of dataset progress)

Current pilot (`data/clmi_prescreen_v2/confirmed_concepts_v2.json`) shows `clmi_mean: 1.0, std: 0.0` for essentially every concept at n=50/50 — suspiciously perfect. Before any paper-facing CLMI number is trusted:

1. **Label-swap control**: retrain the probe with positive/negative labels randomly permuted. Expect AUROC ≈ 0.5. If it's still high, the probe is keying on a template artifact, not concept identity.
2. **Paraphrase-matched negative control**: regenerate negative-class prompts as paraphrases of the *same* concept mentioned in a different context/sentence structure than the positive class, removing any surface cue (sentence length, phrasing template) that differs systematically between classes.
3. Only if both controls behave as expected (near-chance on label-swap, and AUROC still meaningfully separates true concept-presence vs. absence on the paraphrase-matched control) does CLMI proceed to full-scale use as a paper metric.

This is Gate 0 in Section 14 — do this before scaling CLMI to more concepts, since if the probe methodology needs revision, better to find out on 10 concepts than 60.

## 6. BioUnlearn-Bench Scenario Specs (revised)

Keep the three-scenario structure (PAC / RGU / IFE) and the four design principles from the original proposal (text-only generative, UMLS-anchored boundaries, three clinically grounded scenarios, MIMIC-IV/i2b2 grounding). Changes:

- **RGU**: primary ground truth now Gold-tier (Prasad/Herrera-Perez/FDA SrLC/WHO EML/ESC-AHA-ACC/Cochrane) per Section 4.3, supplemented by Silver-tier SNOMED reason-coded pairs where Gold coverage is thin for a given clinical subdomain. The original "seven guideline reversal categories" taxonomy is kept as the organizing schema for tagging harvested Gold instances, not as a generation prompt template.
- **IFE**: primary ground truth now DrugBank withdrawn-status + SNOMED `ERRONEOUS`-reason concepts + any Gold-tier debunked-claim sources found during Stage B0, supplementing (not replacing) the original DrugBank/RxNorm/USPSTF/IDSA verification approach.
- **PAC**: unchanged from original plan — i2b2 + MIMIC-IV grounded, no PTGC tiering applies (see 4.3 rationale).
- Final scenario sizes are **determined by Stage B0 harvesting yield**, not fixed in advance. Do not lock 800/700/1300 forget counts until real counts are in.

## 7. Methodology: BioUnlearn Framework (recap + evidence-weighted extensions)

### 7.1 OGFR (Ontology-Guided Forget/Retain Boundary Constructor) — as originally specified, plus:

**Evidence-Weighted OGFR (new)**: add evidence-certainty as a third term in the neighbor retention weight:

```
w(c, f) = α·[1/(dist_UMLS(c,f)+1)] + β·ICD_priority(c) + γ·cert(c,f)
```

where `cert(c,f)` is the normalized (0-3) evidence-certainty (Section 4.4) of the relation connecting neighbor `c` to forget concept `f`, and `α+β+γ=1`. Ablation sweeps `γ ∈ {0, 0.2, 0.4}` against the original two-term weighting (`γ=0` recovers the original OGFR exactly, giving a clean nested-model comparison).

### 7.2 DGP (Directional Gradient Projection) — as originally specified, plus:

**Evidence-Weighted DGP (new)**: scale the retain-reinforcement coefficient β by the certainty of the new evidence and the erasure aggressiveness by the certainty gap:

```
β_eff = β_0 · cert(A_new)          # more confident retention push when new evidence is itself high-certainty
η_eff = η_0 · (1 + λ·(cert(A_old) - cert(A_new)))   # more aggressive erasure when old evidence was high-certainty but has been overturned by evidence at least as strong
```

Sweep `λ ∈ {0, 0.25, 0.5}` (λ=0 recovers original DGP). This directly targets the reviewer question "why should a Conditional/Low-certainty update be erased as aggressively as a Strong/High-certainty one" — answer: it shouldn't, and now the method says so explicitly.

### 7.3 CLMI (Concept-Level Membership Inference) — as originally specified, plus the mandatory validity pre-check in Section 5.

### 7.4 EWEF — Evidence-Weighted Erasure Fidelity (new metric)

```
EWEF(θ') = Σ_i  w_i · FA_old_i(θ') · Acc_new_i(θ')   /   Σ_i w_i
```
summed over RGU instances `i`, where `w_i = cert(A_new_i)` (weight by confidence in the replacement fact). Report alongside plain DEF; a large DEF/EWEF divergence indicates the method's apparent success is concentrated on low-certainty (weaker, more contestable) reversals — an important honesty check the plain DEF metric cannot provide.

### 7.5 Provenance stratification (methodological, not a new algorithm)

Every Table 1/2/3-style result (Section 13) must be computable filtered to PQS≥2, per Section 4.2. This is a reporting requirement, not a new metric.

## 8. Baselines (expanded)

| Baseline | Family | Why included |
|---|---|---|
| GA, GD, NPO, RMU, SU, WHP | Gradient/preference/activation/token/RL-based unlearning | As originally planned — unchanged |
| **ROME / MEMIT (new)** | Knowledge editing | Directly matches RGU's (A_old→A_new) directional structure — the single most obvious "why not just edit the fact" reviewer objection. Expected result: strong FA/DEF on RGU (it's designed for exactly this), but high OCD (edits are less surgical about ontological neighbors than claimed) and/or high CLMI (edit may be locally overridable, residual old-fact representation may persist off the edited pathway) — the comparison is the point, not "losing" to it. |
| **RAG-suppression (new)** | Retrieval-augmented correction, no weight change | Prepend retrieved current guideline as context. Expected result: high behavioral DEF (looks like it solves the problem) but CLMI stays high (~parametric presence unchanged) — this is the paper's cleanest illustration of Failure Mode 3 (Verification Opacity): a solution that looks adequate until you check the metric that doesn't rely on model behavior. |

## 9. Models

Unchanged from original plan: BioMistral-7B (primary), Meditron-7B (cross-architecture replication), Llama-3.1-8B-Instruct (non-clinical control). Keep this — it's one of the plan's strongest design elements (Section "Genuine strengths" in the critique doc).

## 10. Statistical Protocol (new — fixes the biggest rigor gap identified in the critique)

- **≥3 random seeds** per (method × model × scenario) configuration for all Table 1/2 results. Report mean ± 95% CI.
- **Paired significance testing**: for every BioUnlearn-vs-best-baseline comparison in Table 1, report a paired bootstrap or Wilcoxon signed-rank test (paired at the instance level) — not just point-estimate deltas.
- **Hyperparameter selection variance**: report the sensitivity of the top-line result to the top-3 hyperparameter configurations from the sweep, not only the single best config, to guard against overfitting the sweep to the benchmark.
- No "Expected Results" table with fabricated precision is to be produced before real experiments run (see Section 13 — templates only, all cells blank/TBD until real data exists).

## 11. Complete Research Questions

| RQ | Question | Primary analysis |
|---|---|---|
| RQ1 | Do all baselines show higher OCD on BioUnlearn-Bench than TOFU/WMDP-Bio? | Diagnostic table, Gold+Silver combined and PQS≥2-only |
| RQ2 | Do all baselines achieve DEF < 0.60 on RGU while BioUnlearn achieves DEF > 0.80? | Diagnostic table |
| RQ3 | Does CLMI reliably distinguish parametric erasure from behavioral suppression, post label-swap/paraphrase validity checks (Gate 0)? | CLMI validity report + main table |
| RQ4 | Does BioUnlearn achieve best FA/OCD/DEF across all scenarios on both clinical models, with significance (Section 10)? | Main results table |
| RQ5 | Is OGFR's OCD advantage from UMLS structure or retain-set size? | OGFR-Random ablation |
| RQ6 | Is DGP's DEF advantage from orthogonal projection or from adding g_retain? | DGP-NaiveSum ablation |
| RQ7 | Does the OCD gap shrink on general-domain Llama-3.1-8B? | Cross-model table |
| RQ8 | Does OGFR transfer to improve OCD on TOFU/WMDP when applied to NPO/RMU? | Transfer table |
| **RQ9 (new)** | Does evidence-weighted DGP/OGFR outperform their unweighted variants, and does the gain scale with the certainty gap? | Evidence-weighting ablation |
| **RQ10 (new)** | Do conclusions (RQ1, RQ2, RQ4) hold when restricted to PQS≥2 (gold+silver) instances only? | PQS-stratified re-analysis |
| **RQ11 (new)** | Do ROME/MEMIT and RAG-suppression pass DEF but fail CLMI, confirming Failure Mode 3 independent of "unlearning" method family? | Comparator table |
| **RQ12 (new)** | For the subset of RGU instances with dates falling inside MIMIC-IV's timespan, does real-world documented practice shift match the guideline reversal date (external validity check)? | MIMIC-IV natural-experiment secondary analysis |

## 12. Complete Ablation Matrix

| Ablation | What it isolates |
|---|---|
| OGFR → Random boundary expansion | UMLS structure vs. retain-set size (original) |
| DGP → Naive sum (no projection) | Orthogonal projection vs. adding g_retain (original) |
| UMLS hop depth k=1 / k=2 / k=3 | Optimal neighborhood radius (original) |
| Scenario isolation (PAC-only / RGU-only / IFE-only) | Per-scenario contribution (original) |
| w/o OGFR, w/o DGP (each held out) | Component-level necessity (original) |
| **Evidence-weight γ (OGFR) ∈ {0, 0.2, 0.4}** | Whether certainty-weighted boundary construction helps (new) |
| **Evidence-weight λ (DGP) ∈ {0, 0.25, 0.5}** | Whether certainty-scaled erasure aggressiveness helps (new) |
| **Gold-only vs. Silver-only vs. Gold+Silver mixed** | Whether conclusions depend on ground-truth provenance tier (new — directly answers "silver labels are unreliable") |
| **CLMI label-swap / paraphrase-matched controls** | Probe validity, not a method ablation, but reported alongside (new, Gate 0) |
| **Knowledge-editing (ROME/MEMIT) and RAG-suppression as non-unlearning comparators** | Whether the failure modes are specific to gradient-based unlearning or general to any correction approach (new) |

## 13. Results Table Plan (templates — no fabricated numbers)

Produce these tables with real data only, once experiments run:

- **Table 1 — Main Results** (BioMistral-7B, all baselines + comparators + BioUnlearn, FA/OCD/DEF/EWEF/CLMI/MedQA, mean±CI over ≥3 seeds, PQS≥2 subset shown alongside full set)
- **Table 2 — Diagnostic** (TOFU/WMDP-Bio vs. BioUnlearn-Bench OCD/DEF/CLMI, all baselines)
- **Table 3 — Ablations** (full matrix from Section 12)
- **Table 4 — Cross-Model Generalization** (BioMistral / Meditron / Llama-3.1-8B)
- **Table 5 — Transfer** (OGFR applied to NPO/RMU on TOFU)
- **Table 6 (new) — Comparator table** (ROME/MEMIT, RAG-suppression vs. BioUnlearn on RGU specifically — DEF vs CLMI divergence is the key column)
- **Table 7 (new) — MIMIC-IV natural-experiment validation** (subset of RGU instances with dates inside MIMIC-IV span, documented real-world adoption vs. model behavior)
- **Supplementary — CLMI validity report** (label-swap AUROC, paraphrase-matched AUROC, before any concept's CLMI number is used elsewhere)

## 14. Feasibility Gates (sequenced — do not skip ahead)

| Gate | Trigger | Pass criterion | If fail |
|---|---|---|---|
| **Gate 0 — CLMI validity** | Before scaling CLMI past current 10 pilot concepts | Label-swap control ≈ chance; paraphrase-matched control still separates concept presence meaningfully | Revise probe methodology (harder negatives, activation normalization, position selection) before proceeding |
| **Gate 1 — Gold-source yield** | After Stage B0 harvesting | ≥400 combined Gold-tier RGU+IFE candidates across all sources in Section 4.3 | Broaden source list (e.g., add NICE surveillance reports, additional Cochrane domains) or accept a smaller, more concentrated benchmark (e.g., cardiology-focused) and adjust scope claims accordingly |
| **Gate 2 — Core diagnostic claim (as in original plan)** | After 50-instance pilot run of GA/NPO/RMU on Gold-tier pilot instances | Bio-Bench OCD > 5% for all baselines AND TOFU OCD < 3% (unchanged decision table from original Section 10) | Investigate which scenario/category drives the gap, narrow paper scope, or reconsider core claim |
| **Gate 3 — Evidence-weighting payoff** | After running evidence-weighted DGP/OGFR ablation | Evidence-weighted variant beats unweighted on at least one axis (DEF-vs-certainty-gap correlation, or OCD on high-certainty-neighbor subset) with a plausible mechanism, not just noise | If no effect, keep unweighted BioUnlearn as the main method and report evidence-weighting as a negative/null result in a supplementary section — still scientifically honest and reportable, just not a headline contribution |
| **Gate 4 — PQS-stratified robustness** | After main experiments | RQ4's conclusions replicate on PQS≥2-only subset | If conclusions weaken substantially on Gold+Silver-only, report both, and lead the paper's claims with the more conservative (PQS≥2) numbers |

## 15. Timeline & Phases

| Phase | Duration | Contents |
|---|---|---|
| 0. CLMI validity fix | 3-5 days | Gate 0 |
| 1. Gold-source harvesting (Stage B0) | 2-3 weeks | Parallelizable across sources; output per-source JSON with PQS=3 tags |
| 2. SNOMED reason-code extraction (Stage A extension) | 1 week | Parallel with Phase 1 |
| 3. Phrasing + grounding-fidelity filter (Stage B/C) | 1-2 weeks | Depends on Phase 1+2 |
| 4. Human validation (Stage D, revised scope per Section 18) | 3-4 weeks | Depends on Phase 3 |
| 5. Splits, freeze, release prep (Stage E) | 3-5 days | Depends on Phase 4 |
| 6. Diagnostic study (Gate 2, existing baselines on real Gold+Silver data) | 1-2 weeks compute | Can start once Phase 3 output exists for a pilot subset, ahead of full Phase 4-5 completion |
| 7. Full baseline + comparator experiments (Section 8, ≥3 seeds, Section 10 stats) | 3-4 weeks compute | Depends on Phase 5 |
| 8. BioUnlearn training + ablation matrix (Section 12) | 2-3 weeks compute | Depends on Phase 7 infra being validated |
| 9. Cross-model + transfer experiments | 1-2 weeks compute | Parallelizable with Phase 8 |
| 10. MIMIC-IV natural-experiment analysis | 1 week | Parallelizable, low compute |
| 11. Writing (mapped to Section 19) | 3-4 weeks, overlapping with late experiment phases | Draft sections as their data becomes final |

Total estimated calendar time: ~14-18 weeks from Gate 0 to submission-ready draft, assuming harvesting (Phase 1) and experiment infrastructure work happen partly in parallel. This is longer than the original plan's ~9 weeks, reflecting the added rigor (real ground-truth harvesting, more baselines, more seeds, expanded human validation) — an honest tradeoff for the NC-grade bar.

## 16. Compute Budget (revised)

| Experiment | Est. GPU-hours | Notes |
|---|---|---|
| Pilot feasibility (Gate 2) | 8h | Unchanged |
| Diagnostic study, 6 baselines + 2 comparators × 3 models × ≥3 seeds | ~270h | Was 90h for 1 seed; ×3 seeds |
| TOFU/WMDP-Bio baselines, ≥3 seeds | ~90h | Was 30h ×3 |
| CLMI probe training + validity controls | ~30h | Was 16h, + label-swap/paraphrase controls |
| BioUnlearn training incl. evidence-weighted variants, ×3 seeds | ~130h | Was 36h ×3 (roughly, plus evidence-weight sweep) |
| Ablation matrix (expanded, Section 12) | ~120h | Was 55h, expanded for evidence-weighting + gold/silver split ablations |
| Cross-model (Llama-3.1-8B) | ~72h | Was 24h ×3 |
| Relearning attack | ~54h | Was 18h ×3 |
| ROME/MEMIT + RAG comparator experiments | ~40h | New |
| DriftMedQA + TOFU transfer | ~45h | Was 15h ×3 |
| **Total** | **≈860 GPU-hours** | ≈3-4 weeks wall-clock on a 4×A100 80GB node, or compressible with a larger allocation |

## 17. Software Stack Additions

On top of the original stack (HuggingFace Transformers, OpenUnlearning, TransformerLens, scikit-learn, lm-evaluation-harness, DVC): add an editing-methods library (e.g., EasyEdit) for ROME/MEMIT, and a minimal retrieval stack (e.g., a simple dense retriever + context-prepending harness) for the RAG-suppression comparator. Both are lightweight additions, not new infrastructure categories.

## 18. Human Validation & Ethics Plan (NC-grade revision)

- **Annotator panel**: expand beyond 2 MD-2 students. Recommend at minimum one board-certified physician reviewer (even part-time/consulting) co-signing the validation protocol and reviewing a stratified sample, given the paper makes patient-safety claims. Dr. Vindo's involvement, if the collaboration proceeds, could plausibly fill or help recruit this role — flag this explicitly as a discussion point with him (Section 22).
- **Validated sample size**: stratify by PQS tier — since Gold-tier instances already carry independent institutional provenance, human validation effort should concentrate on Silver-tier (SNOMED reason-coded) and all phrasing steps (checking the LLM phrasing didn't distort the Gold/Silver fact), rather than spreading validation thinly and uniformly across all tiers as before.
- **NC-required sections** (not present in the current EMNLP-shaped doc): Reporting Summary, Data Availability Statement, Code Availability Statement, Competing Interests declaration, and an ethics statement that goes beyond "PhysioNet/n2c2 DUA" to explicitly address IRB/exempt-status determination for the human annotation component.
- Strip ARR/double-blind/EMNLP-specific language entirely from any NC-facing draft (per critique doc).

## 19. Paper Writing / Submission Plan

Map paper sections to the data/experiments that must be final before drafting:

| Paper section | Depends on |
|---|---|
| Abstract, Intro, Motivation | Can draft early (largely unchanged from proposal doc's Sections 1-3, reframed per Section 1 above) |
| Background (UMLS, SNOMED, GRADE, models) | Can draft early; extend original Section 3 with SNOMED/GRADE/WHO material from Section 4 |
| Related Work | Draft after PrimeKG-CL close-read (Section 3) confirms differentiation |
| Dataset section | Finalize only after Phase 5 (splits frozen) |
| Method section (OGFR/DGP/CLMI/EWEF) | Can draft early (formal definitions are ready now, Section 7) |
| Diagnostic results | After Phase 6 |
| Main results + ablations | After Phases 7-8 |
| Cross-model + transfer | After Phase 9 |
| MIMIC-IV natural-experiment box/vignette | After Phase 10 |
| Limitations, Ethics, Broader Impact | Draft early, revise once real dataset yield (Gate 1) and human-validation results (Section 18) are known |
| Reporting Summary / Data & Code Availability | Last, once release artifacts are final |

## 20. Risk Register

| Risk | Mitigation |
|---|---|
| Gold-source yield too low (Gate 1 fails) | Narrow scope honestly (e.g., cardiology-concentrated benchmark) rather than backfilling with LLM-only instances |
| CLMI probe artifact (Gate 0) | Blocks scaling until resolved; do not report CLMI numbers built on an unvalidated probe |
| Evidence-weighting shows no effect (Gate 3) | Report as null result, keep unweighted method as headline, still publishable |
| PrimeKG-CL overlaps more than expected with your framing | Read it fully before Related Work drafting; pivot differentiation language if needed, possibly cite it as corroborating evidence of real biomedical KG churn rather than treating it purely as competing work |
| ACIP-sourced instances read as politically charged | Restrict to pre-2020, settled ACIP examples only, or omit ACIP entirely if any instance could be read as commentary on active vaccine policy debates |
| Compute budget (860h) exceeds available allocation | Prioritize Phase 6 (diagnostic, the core claim) and Phase 7 (main results) first; ablations and cross-model can be trimmed or run at 1-2 seeds with a caveat if allocation is tight |
| Timeline (14-18 weeks) is longer than original 9-week estimate | This is the honest cost of NC-grade rigor; if a faster submission is needed, consider whether an EMNLP-track submission of the original (thinner) plan is a valid parallel/fallback path — but do not blend the two framings in one draft |

## 21. Immediate Next Actions (given current repo state)

Current state: 192 pilot instances (95 RGU + 95 IFE, no PAC yet), CLMI prescreen v2 done on 10 concepts with unresolved perfect-AUROC concern, splits already generated for RGU/IFE at pilot scale.

1. **Run Gate 0 (CLMI validity check)** on the existing 10 prescreened concepts before generating any more CLMI data — cheapest, highest-priority action, blocks nothing else.
2. **Start Stage B0 gold-source harvesting in parallel** — begin with Prasad (2013) + Herrera-Perez (2019) tables (structured, finite, fastest to extract) and FDA SrLC (structured database, scriptable). These don't depend on Gate 0.
3. **Do not generate more LLM-only RGU/IFE silver instances** in the current style until Stage B0 yield is known (Gate 1) — avoid throwing away work that Gold sources would replace; the existing 192 pilot instances can be retroactively PQS-tagged (likely PQS 0-1) and either kept as an explicit LLM-only comparison arm or superseded.
4. **Begin SNOMED CT reason-code extraction** (Stage A extension) — independent workstream, can run alongside 1-3.
5. Once Gate 0 and initial Stage B0 yield are in hand, re-run the original 3-day feasibility check (now Gate 2) on real Gold-tier pilot data instead of LLM-only pilot data.

## 22. Open Items to Resolve with Dr. Vindo

- Confirm whether he sees the PTGC tiering approach (Section 4) as addressing his original concern, or whether he had a different specific mechanism in mind.
- Confirm his possible role in the expanded human-validation panel (Section 18) — a board-certified physician co-reviewer would materially strengthen the paper.
- Confirm reaction to the evidence-weighted OGFR/DGP proposal (Section 7) as "the" biomedical-specific methodological novelty, or whether he had a different novelty direction in mind when he raised the point.
- Confirm whether UMich has any of its own EHR/guideline-change data infrastructure (given his likely informatics background) that could substitute for or supplement the public sources in Section 4.3.
- Confirm venue: this plan is written for Nature Communications (per README); if EMNLP or a dual-track strategy is still live, that changes Section 18/19 materially (see critique doc).
