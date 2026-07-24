# BioUnlearn: Complete Project History

A full chronological record — initial state, what the professor said, what we researched, what came out of that research, what we built and ran, and exactly what's in the repo right now. Companion to `PI_STATUS_REPORT.md` (that one is the condensed version for external reporting; this one is the complete internal record with nothing compressed out).

---

## Phase 0: Initial state

Repo cloned from `saisabsadhu/bio-unlearning`, branch `saisab`. It contained one artifact: `documentation/BioUnlearn_EMNLP2026_FINAL_Proposal (1).docx` (1512 lines) — a complete proposal already written, but for a different venue than the one the repo actually targets.

## Phase 1: Critique — venue mismatch found

Reviewed the docx against the repo's own `README.md`, which says "Targeting Nature Communications." Found a fundamental mismatch, written up in `critique_nature_communications_fit.md`:

- The proposal was written **entirely for EMNLP 2026** — references ARR, double-blind policy, an "ARR-required Responsible NLP Research Checklist," and an "Anticipated Reviewer Objections" section in NLP-conference adversarial style. None of that is right for NC.
- NC needs: a broad-scientist-readable framing (not ML-mechanics-first), general-significance novelty (not "three new metrics"), NC-specific paper sections (Reporting Summary, Data/Code Availability, Competing Interests, real ethics/IRB statement), and materially deeper human validation (2 MD students validating 9.1% of data is thin for a clinical-safety journal claim).
- Genuine strengths worth keeping regardless of venue: the three-failure-mode diagnostic structure (ontological entanglement, temporal directionality, verification opacity), the pre-registered 3-day feasibility gate before full dataset construction, the Llama-3.1-8B cross-model control, and the two critical ablations (OGFR-Random, DGP-NaiveSum) that pre-empt obvious reviewer objections.
- Independent concerns flagged: circularity risk (same LLM oracle building both benchmark and fix training signal), fully-fabricated "Expected Results" tables with fake two-decimal numbers, no statistical rigor plan (point estimates, no seeds/variance), and a missing baseline class (knowledge editing / RAG-suppression).
- **A specific red flag in the existing pilot data**: the CLMI prescreen showed `clmi_mean: 1.0, std: 0.0` for essentially every one of the 10 pilot concepts — suspiciously perfect, flagged as needing a validity check (scrambled/label-swap control) before trusting CLMI as a load-bearing metric. This directly motivated Gate 0 (Phase 6 below).

## Phase 2: What Dr. Vindo (UMich collaborator) actually said

Recorded verbatim in `groundtruth_sources_dr_vindo_directions.md` line 3. He proposed shifting toward two things:

> **(A)** a more concretized, documented ground truth for "what to forget" — analogous to UMLS changes over time — while noting most such changes aren't mistakes and often aren't documented anywhere; and **(B)** building the unlearning dataset directly from EHR data or clinical guidelines rather than (or in addition to) an LLM annotation oracle.

## Phase 3: Research conducted in response to Dr. Vindo

Full research memo: `groundtruth_sources_dr_vindo_directions.md`. Verdict up front: **his intuition was half right and half already solved.**

**On direction A (concretized, documented ground truth):**
- He was right that raw UMLS diff files (`MRCUI.RRF`, `MRCONSO_HISTORY.txt`, `MRREL_HISTORY.txt`) record *that* something changed but not *why*, and are dominated by terminology housekeeping (CUI merges, vocabulary version bumps) rather than clinically meaningful reversals.
- He was wrong that clinically-meaningful changes are undocumented — they are, just not inside UMLS itself:
  - **SNOMED CT's Component Inactivation Reference Sets** carry an explicit reason code per inactivation: `DUPLICATE`/`AMBIGUOUS` (housekeeping), `ERRONEOUS` (a real mistake), `OUTDATED` (correct once, superseded — exactly the RGU/temporal-directionality target, pre-labeled), `MOVED_ELSEWHERE`/`LIMITED`. Each inactivated concept links to its replacement via the historical association — a real, ontology-sourced (A_old → A_new) pair with no oracle needed to invent the pairing. Already accessible via the existing NLM UTS account.
  - **RxNorm** (`RXNCUICHANGES.RRF`, `/historystatus` API) tracks deprecated drug concepts/NDC codes but with thinner reason-coding than SNOMED — useful for cross-checking dates, not a primary source.
  - **PrimeKG-CL** (Radwan, Li et al., arXiv:2605.10529, 2026): a very recent, directly adjacent benchmark — real diffs between two PrimeKG snapshots (June 2021 vs. July 2023), 889K removed / 5.83M added / 7.21M persistent edges across 9 biomedical databases. Flagged as both an opportunity (pre-computed real diffs at scale) and a risk (adjacent territory — must differentiate explicitly: it's continual graph *learning*, not weight *unlearning*).
  - **Prasad et al. 2013 (Mayo Clinic Proceedings)** — 146 named clinical practices contradicted by RCT evidence, each with the contradicting trial citation. **Herrera-Perez et al. 2019 (eLife)** — 396 more from JAMA/Lancet/NEJM 2003–2017. Together ~500+ real, dated, citation-backed reversals, peer-reviewed and published independent of anything we construct — identified as the single strongest answer to "concretized ground truth that isn't LLM-invented."

**On direction B ("go to EHR or guidelines"):**
- Concluded that raw EHR *audit-log* mining (clinician click/workflow behavior) is the wrong read — that literature is about provider behavior, not fact-level ground truth.
- The stronger reading: anchor forget targets in official guideline/regulatory records that are better-documented than UMLS — **FDA Drug Safety-related Labeling Changes (SrLC) database** (structured, since Jan 2016, tracks exactly which label section changed and when — combined with Drugs@FDA/DailyMed historical PDFs, gives literal before/after label text with zero LLM involvement in the ground truth itself), USPSTF recommendation history, Cochrane systematic review updates (rare conclusion flips, ~4-9%, but exceptionally well documented when they happen), NICE guideline surveillance reports, Choosing Wisely (better fit for IFE than RGU).
- **A genuinely EHR-native idea proposed back to him**: use MIMIC-IV's real timestamps as a natural experiment — bucket notes/orders by admission date relative to a known guideline-change date (e.g., rosiglitazone's May 21, 2007 FDA warning, already tied to a published 70% prescribing drop within two years) and use the *actual documented clinical practice shift in the data itself* as empirical confirmation a reversal was real and adopted, not just a paper claim.

**Follow-up: WHO and other international guideline bodies** (addendum to the same memo, in direct response to the project's restated dual goal — a strong benchmark *and* a methodology novelty specific to the biomedical paradigm):
- **WHO Model List of Essential Medicines** — standout source. Every addition, amendment, and rejected/removed medicine gets a full prose rationale in the WHO Technical Report Series, richer than SNOMED's single-word codes.
- **WHO living guidelines** (e.g., COVID-19 therapeutics) — excellent depth, narrow disease scope.
- **ESC/ACC/AHA cardiology guidelines** — every recommendation carries a machine-parseable Class of Recommendation (I/IIa/IIb/III) × Level of Evidence (A/B/C), with existing academic literature auditing how these shift release-to-release.
- **ACIP (CDC vaccine recommendations)** — good ground truth quality but flagged as currently politically contentious; recommended being conservative and sticking to old/settled (pre-2020) examples if used at all.
- **The methodological hook this unlocks**: GRADE/Class-LOE-style standardized evidence-certainty grading is a structural property unique to the clinical-guideline domain — TOFU/WMDP/MUSE have no analogous "how confident was the source" metadata. This motivated three concrete extensions: Evidence-Weighted DGP (scale forget/retain gradient balance by certainty delta between old and new recommendation), Evidence-Weighted OGFR (weight retain-protected neighbors by evidence certainty, not just ontology hop-distance), and a new metric, **EWEF** (Evidence-Weighted Erasure Fidelity — upweight/report separately instances where both retraction and replacement are high-certainty).

## Phase 4: Synthesis — what came out of this research

Four concrete recommendations were written up (`groundtruth_sources_dr_vindo_directions.md`, "Recommended synthesis" section) and folded into a full rebuild of the research plan:

1. Replace/augment RGU ground truth with the Prasad + Herrera-Perez catalogs (~500 instances), supplemented by FDA SrLC pairs.
2. Use SNOMED CT's reason-coded inactivation (`OUTDATED` vs. `ERRONEOUS` vs. `DUPLICATE`) as the mechanism that directly answers his "most changes aren't mistakes" concern with an existing, machine-readable label rather than manual judgment.
3. Add an EHR-native validation layer against MIMIC-IV's own temporal distribution or published real-world adoption-decline studies.
4. Read PrimeKG-CL closely before finalizing scope, both as an engineering shortcut and as related work requiring explicit differentiation.

This produced **`MASTER_RESEARCH_PLAN.md`** — the full, 22-section, now-authoritative operational plan (thesis, five contributions, PTGC ground-truth methodology, dataset pipeline, framework methodology, baselines, statistical protocol, complete research questions, complete ablation matrix, feasibility gates, timeline, compute budget, ethics, risk register, immediate next actions, and open items to confirm with Dr. Vindo). Its core methodological contribution is **PTGC (Provenance-Tiered Ground Truth Construction)**: every forget/retain instance graded by a Provenance Quality Score (PQS 0-3: LLM-only, oracle-phrasing, ontology reason-coded, real citation-backed gold), so dataset reliability is falsifiable and reported stratified by tier rather than asserted uniformly trustworthy.

## Phase 5: Infrastructure build

- Merged the full `locuslab/open-unlearning` framework (Hydra-config-driven, HF Transformers-based unlearning/eval pipeline) from `origin/wmdp-llama32-3b-experiment` into `saisab`.
- Set up HF auth (token), verified model access for BioMistral-7B, Meditron-7B, Llama-3.1-8B-Instruct.
- Built BioUnlearn-Bench dataset configs (`configs/data/datasets/BioUnlearn_{RGU,IFE}_{forget,retain}[_val|_test].yaml`) and custom eval metrics (`src/evals/metrics/bioun.py`: FA, DEF, EWEF, OCD; `src/evals/bioun.py`: `BioUnlearnEvaluator`).
- Real bugs fixed here: a Hydra entrypoint bug (missing `remove_unused_columns: False` caused "batch was empty" — fixed by using `unlearn.yaml` not `train.yaml`); a pre_compute cache-key collision where generic mount keys (`forget_gen`) were shared across RGU/IFE in the same eval run, silently leaking one scenario's cached metrics into the other's — fixed with scenario-scoped keys (`RGU_forget_gen`/`IFE_forget_gen`) and `access_key` remapping; an evaluator registration bug (base `Evaluator` class needs a positional `name` the generic registration path didn't supply) — fixed with a `BioUnlearnEvaluator` subclass matching the TOFU/MUSE pattern.

## Phase 6: Gate 0 — CLMI validity check

Directly answers the red flag from Phase 1 (suspiciously perfect CLMI=1.0 everywhere). Ran `stage_a_umls/clmi_gate0_validity_check.py` — 10 pilot concepts × 3 pooling schemes (original/last-token/mean) × {real labels, label-swapped labels}. Result and findings in `data/clmi_gate0/GATE0_FINDINGS.md` and `gate0_summary.json`. This validated CLMI as a load-bearing metric before it was trusted for anything downstream.

## Phase 7: Gate 2 — baseline diagnostic pilot (GA, NPO, RMU)

**GradAscent**, first attempt: original hyperparameters (lr=1e-5, ~40 steps) caused catastrophic collapse — `train_loss` diverged from -30 to -222, generations became gibberish, giving the degenerate `FA=1.0, DEF=0.0` pattern identically on both scenarios (scientifically useless — indistinguishable from any other collapsed run). Root-caused to unconstrained-loss unbounded descent; fixed by dropping to lr=2e-6 and capping `max_steps=8`, matching the master plan's step-count-based (not epoch-based) GA tuning. Re-ran successfully on both RGU and IFE — real, non-degenerate FA/DEF/EWEF movement, targeted scenario moving more than untargeted in each case (`data/gate2_results/RGU_ga_pilot_summary.json`, `IFE_ga_pilot_summary.json`).

**NPO/RMU**, first attempts: blocked by a fixed ~30.88 GiB memory cost (trainable model + full-precision deep-copied reference model + optimizer state) regardless of batch size or sequence length — confirmed by the identical OOM figure appearing at both `batch_size=1` and after reducing `max_length` 512→288. Initially just quantizing the reference model to 8-bit alone didn't help (same 30.88GB — the *trainable* model's own footprint was the dominant cost, not the reference copy).

**Resolved** by combining two infrastructure changes (neither alone was sufficient):
1. Opt-in LoRA wrapping (`src/model/__init__.py`, `BIOUNLEARN_USE_LORA=1`) — cuts trainable parameters to 0.58% of the model (42M of 7.28B). Required `model.enable_input_require_grads()` alongside `gradient_checkpointing=True` (otherwise embedding output has `requires_grad=False`, breaking checkpointed backward), and explicit dtype normalization on LoRA adapter params (peft's default doesn't always match the base model's bf16).
2. Reference model loads in 8-bit by default (`src/trainer/unlearn/grad_diff.py::_prepare_ref_model`, `BitsAndBytesConfig(load_in_8bit=True)` instead of `copy.deepcopy`).

**RMU needed three more fixes** on top of those two: `module_regex` needed an optional `(base_model\.model\.)?` prefix to match PEFT-wrapped module names (RMU uses `re.fullmatch`); `trainable_params_regex` changed from `.*` to `.*lora.*` (RMU's `create_optimizer` explicitly re-enables `requires_grad` on every regex-matched param — `.*` would have silently re-enabled gradients on the entire frozen base model, undoing LoRA's savings without any visible error); and an explicit `dtype=model_act.dtype` cast on the reference model's activations in `rmu.py` (bitsandbytes' 8-bit layers dequantize to fp16 internally regardless of the trainable model's bf16, causing "Found dtype Half but expected BFloat16" during backward).

**Result: the full 3-method × 2-scenario matrix now runs successfully** (`RGU_npo_pilot_summary.json`, `IFE_npo_pilot_summary.json`, `RGU_rmu_pilot_summary.json`, `IFE_rmu_pilot_summary.json`). Consistent pattern: RMU is gentlest at matched step budgets (near-zero movement on its targeted scenario), NPO and GA show comparable small, non-degenerate movement, targeted scenarios move more than untargeted ones in NPO/GA (not in RMU, at this dose).

## Phase 8: OGDA — the novel method, full development arc

**Literature positioning**: checked against ~15 recent papers (Arditi et al. 2024's weight-orthogonalization/activation-ablation technique — the base mechanism OGDA builds on; NSRU, PISCES, SAGO, EGUP, AMNESIA, REMEDI — related but non-overlapping unlearning-adjacent work). The specific combination — ontology-anchored protected subspace + training-free directional ablation + applied to clinical guideline reversal — is not claimed elsewhere. Full table in `NOVEL_METHODOLOGY_OGDA.md`.

**v1 — single-layer weight-orthogonalization** (`ogda_ablation.py`): first real result before any ablation — extracting the aspirin forget direction and its 7 real UMLS/RxNorm neighbors' protected subspace at layer 7 gave `cos²(w_old, P) = 0.9853`, a direct quantitative confirmation of the ontological-entanglement hypothesis (Failure Mode 1). An unplanned second finding: this overlap decreases monotonically with depth (0.996 at layer 4 → 0.535 at layer 28). Ablating only layer 7's `mlp.down_proj` left CLMI at 1.0 — root-caused (not a bug): a single MLP edit doesn't touch signal already in the residual stream from embeddings or attention.

**v1 extended — multi-layer weight-orthogonalization** (layers 4-28): still CLMI=1.0, same root cause at more layers — `down_proj`-only edits never touch attention's `o_proj`, so signal survives via residual skip-connections regardless of layer count.

**v2 — activation-level ablation** (`ogda_activation_ablation.py`): switched to forward hooks projecting the direction directly out of the residual stream (the theoretically correct mechanism). First attempt still showed CLMI=1.0 — but a deliberate mechanical sanity check (does the hook change anything at all?) caught a real bug before it was wrongly reported as a finding: `output_hidden_states=True` does **not** reflect forward-hook modifications in this transformers version (4.55.4), even though the hooks correctly affect real downstream computation (verified directly: zeroing a layer's output via hook changed "The capital of France is" → not "Paris"). Fixed by capturing activations via a second set of hooks chained after the ablation hooks. Re-run with corrected capture: **CLMI still measured 1.0** — a real negative result this time, not a measurement artifact.

**v3 — subspace ablation + Restricted-CLMI** (`ogda_subspace_ablation.py`): implemented multi-direction subspace ablation (rank 3/layer via SVD of pairwise contrasts) and a fairness-matched verification protocol (probe restricted to ablated layers only, capacity capped near the ablation's own rank, not an unconstrained 512-dim probe over all 32 layers). First attempt: every layer's forget subspace came back **fully contained** in the protected subspace (rank 0 everywhere, nothing ablated) — root-caused to a real circularity bug: the forget subspace was built from `concept − neighbor` contrasts and the protected subspace from `neighbor − concept` contrasts using the *same* neighbor set — mathematically the same subspace up to sign by construction, independent of any real entanglement. Fixed by introducing an independent generic background pool (other pilot concepts, excluding the target and its neighbors) so both subspaces are built against a shared, independent reference rather than against each other.

Re-run with the fix: forget subspace now genuinely survived orthogonalization (rank 3 at all 25 layers — real entanglement, not a construction artifact). But **both full-stack and Restricted-CLMI (6 PCA components, ablated layers only) still measured 1.0**. Checked and ruled out a token-length confound (mean 20.35 vs. 21.35 tokens between the two prompt classes — not meaningful). Working interpretation, documented in `NOVEL_METHODOLOGY_OGDA.md` Section 8.5: near-synonym pairs ("aspirin cardiovascular prevention" vs. "Low-Dose Aspirin") retain enough token-identity signal from differing words that mean-pooled whole-sequence probing will detect it regardless of which semantic directions are ablated — a limitation of the verification protocol against near-synonym distractors, not necessarily proof the ablation failed. This directly reinforces the paper's own Failure Mode 3 (Verification Opacity) thesis from the inside.

**Behavioral test (FA/DEF), the decisive missing piece**: the subspace-ablation script only ever tested via in-memory forward hooks — never produced a checkpoint that could be loaded fresh and evaluated with the existing bioun FA/DEF/EWEF suite. Added `apply_permanent_subspace_orthogonalization()` — bakes the same per-layer subspace basis into permanent weights via `W' = (I − BᵀB)W`, applied to **both** `self_attn.o_proj` and `mlp.down_proj` at each ablated layer (not just `down_proj`, which v1 already found insufficient alone), plus `--save_checkpoint` to persist it. Ran the full bioun eval suite against this checkpoint for the first time.

Manually inspected raw generations first to rule out GA-style collapse — confirmed fluent, coherent text (not gibberish). Result: FA rose on **both** scenarios (RGU 0.741→0.759, IFE 0.839→0.893) while DEF collapsed toward/to zero on both (RGU 0.192→0.157, IFE 0.015→**0.0**). Since DEF = FA_old × Acc_new, this means the model drifts away from old answers without correctly landing on the intended replacement — broad quality drift, not clean directional erasure. IFE (untargeted — aspirin is an RGU concept) moved as much as RGU (targeted), a collateral-damage signature that the current 25-layer/2-matrix-per-layer edit is too blunt. **This is the first evidence OGDA changes real generation behavior at all — directly contradicting CLMI's null reading — but not yet in the targeted way the method is designed to achieve.**

## Phase 9: Current, honest novelty assessment

| Claim | Status |
|---|---|
| Genuinely unclaimed combination in the literature | Verified |
| Entanglement hypothesis real and quantifiable | Verified (cos²=0.985) |
| Ablation mechanism actually executes (not a no-op) | Verified (mechanical sanity check + fixed a real hook-capture bug) |
| Subspace construction mathematically sound | Verified (found and fixed the circularity bug) |
| Achieves genuine parametric erasure (CLMI) | Not shown — CLMI=1.0 even fairness-matched; likely a verification-protocol limitation against near-synonyms |
| Changes real generation behavior (FA/DEF) | Yes, but not selectively — broad drift, not targeted erasure |

**Conclusion carried into `PI_STATUS_REPORT.md`**: a well-differentiated, rigorously stress-tested novel proposal, not yet a demonstrated working method. The single most load-bearing next experiment (not yet run): a random-subspace control ablation of matched size, to isolate whether ontology-anchoring specifically matters or whether any similarly-sized intervention produces comparable drift.

## Phase 10: What's in the repo right now (file manifest)

```
documentation/
  BioUnlearn_EMNLP2026_FINAL_Proposal (1).docx   original EMNLP-shaped proposal (Phase 0)
  critique_nature_communications_fit.md          Phase 1
  groundtruth_sources_dr_vindo_directions.md     Phases 2-4 (research memo + WHO addendum)
  MASTER_RESEARCH_PLAN.md                        Phase 4 output — 22-section authoritative plan
  NOVEL_METHODOLOGY_OGDA.md                       Phase 8 — OGDA proposal + full empirical journey (Section 8)
  OGDA_REPRODUCIBILITY.md                         exact algorithm/commands to reproduce every OGDA result
  PI_STATUS_REPORT.md                             condensed external-facing summary
  PROJECT_HISTORY.md                              this document

docs/bioun_running_notes.md                      practical engineering gotchas, kept current

src/                                              locuslab/open-unlearning framework (Phase 5), extended:
  evals/bioun.py, evals/metrics/bioun.py          FA/DEF/EWEF/OCD custom metrics + evaluator
  model/__init__.py                               + opt-in LoRA wrapping (Phase 7)
  trainer/unlearn/grad_diff.py                    + 8-bit reference model loading (Phase 7)
  trainer/unlearn/rmu.py                          + dtype-mismatch fix (Phase 7)

configs/
  data/datasets/BioUnlearn_{RGU,IFE}_*.yaml       dataset configs (Phase 5)
  eval/bioun_metrics/*.yaml, eval/bioun.yaml       eval suite composition (Phase 5)
  experiment/unlearn/bioun/{RGU,IFE}_{ga,npo,rmu}.yaml   6 tuned experiment configs (Phase 7)
  trainer/RMU.yaml                                LoRA-compatible module/trainable-params regex (Phase 7)
  model/BioMistral-7B.yaml, Meditron-7B.yaml       model configs

stage_a_umls/
  clmi_gate0_validity_check.py                    Phase 6
  clmi_post_unlearning_check.py                   reusable post-hoc CLMI checker
  clmi_prescreen.py, clmi_prescreen_v2.py          earlier pilot-concept screening
  merge_graphs.py, umls_graph.py, rxnorm_relations.py   real UMLS/RxNorm concept graph construction
  ogda_ablation.py                                Phase 8 v1
  ogda_activation_ablation.py                     Phase 8 v2
  ogda_subspace_ablation.py                        Phase 8 v3 (current) + checkpoint saving

stage_b0_gold_sources/README.md                   Stage B0 harvesting tracker — 30/400+ gold reversals so far
                                                   (Herrera-Perez + Prasad partial seeds; SNOMED/FDA/WHO extraction not started)

data/
  gold_sources/{herrera_perez_2019,prasad_2013}_seed.json   30 real citation-backed reversals
  clmi_gate0/                                     Gate 0 outputs
  gate2_results/                                  all 11 experiment result files (GA/NPO/RMU x2, OGDA x5, behavioral x1)
  splits/                                         190 pilot instances (95 RGU + 95 IFE), train/val/test

saves/ (gitignored)                               model checkpoints and eval outputs, not version-controlled
```

## Phase 12: The random-subspace control, the statistical corrections, and multi-concept replication

Everything below happened in one extended, largely autonomous continuation, run to closure on the open items Phase 11 listed.

**Cross-model matrix completed**: GA, NPO, and RMU all extended to Llama-3.1-8B-Instruct (non-clinical control). NPO/RMU needed more than the LoRA+8-bit-reference-model fix that worked for 7B BioMistral -- the 8B base model's own weights were now the bottleneck. Added **QLoRA** (4-bit base model + 4-bit reference model via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`, `prepare_model_for_kbit_training` before LoRA wrapping) as a further, reusable capability. Result: real, non-degenerate NPO and RMU runs on Llama. A genuinely interesting cross-model finding fell out of this: RMU's effect strength is sharply model-dependent at *identical* hyperparameters -- near-zero on BioMistral, a large real scenario-specific effect on Llama -- meaning single-model method-sensitivity rankings don't generalize.

**The random-subspace control (the load-bearing experiment)**: built `ogda_random_control.py` -- identical weight-edit mechanism to real OGDA, but ablating a random orthonormal subspace instead of an ontology-derived one. First attempt (1 seed) suggested a clean "real OGDA is special" result. **This was directly retested and corrected**: two more random seeds showed FA has substantial seed-to-seed variance (one random seed moved FA *more* than real OGDA, in the opposite direction) -- the n=1 conclusion didn't survive. Computed proper z-scores of real OGDA's result against the n=3 random-seed distribution: **FA is not a reliable differentiator (z=0.64-2.31), but DEF is a large, clear statistical outlier (z=-5.44 RGU, z=-23.38 IFE)**. The claim that OGDA's mechanism is non-generic was narrowed and re-grounded specifically on DEF, with the correction documented in place rather than silently revised.

**Layer/rank sweep, an apparent localization, and its retraction**: systematic sweep (CLMI screening across rank/layer combinations, then direct behavioral testing) found that narrowing to late layers only (20-28, then 24-28) made the collateral IFE DEF collapse disappear at 5 layers, isolating it to layers 20-23. Testing layers 20-23 alone confirmed they reproduced the full collapse -- looked like a precise 4-layer mechanistic locus. **Directly tested and falsified**: ablating all 21 *other* layers (excluding 20-23) still produced the full collapse, meaning those layers aren't uniquely necessary. Corrected interpretation: a breadth/dose threshold effect, not a specific locus. The "avoid layers 20-23" fix idea the false localization suggested was explicitly retracted.

**A filename-collision bug caught mid-sweep**: the ablation script's output path only encoded rank and CUI, not layer range -- a narrower-window sweep point silently overwrote the original 25-layer result before it was committed. Caught via `git diff` before it corrupted the record; recovered from git history, fixed the script to include layer range (and later, seed) in output filenames.

**Multi-concept replication (the decisive check)**: the random-control distribution is concept-agnostic (same base model/benchmark, only the ablated subspace differs), so it was directly reusable as the null distribution for the other 3 concepts' *already-collected* real-OGDA results -- no new GPU runs needed, just the right comparison, run after the user directly challenged whether the work had a real destination. Result: own-scenario DEF is a statistical outlier for **all 4 concepts** (aspirin z=-5.44, rosiglitazone z=-23.38, HRT z=-3.34, Vioxx z=-23.38) -- the original aspirin finding was not a fluke. The one partial exception: HRT's *cross-scenario* (collateral) effect doesn't replicate (z=+1.12), precisely locating the earlier-noted HRT anomaly to the collateral-damage side specifically, not the core erasure signal. This is currently the strongest, most defensible piece of novelty evidence in the project.

**Seed support added to the real-OGDA side**: `make_prompts` now accepts a `seed` that shuffles template order before sampling, threaded through `ogda_subspace_ablation.py --seed`, so the ontology-anchored construction can build its own multi-seed distribution (previously only the random control had one) -- infrastructure built, not yet run to completion (blocked mid-session by another tenant's GPU job).

**Gold-source diversification, moving past 2 papers**: built and validated a DailyMed SPL version-history puller (`stage_b0_gold_sources/dailymed_label_diff.py`) -- real, dated, FDA-regulated before/after drug label text, zero LLM involvement anywhere in the ground truth. Three real results so far, matched to existing pilot concepts: rosiglitazone (Avandamet boxed-warning title changed "MYOCARDIAL ISCHEMIA" -> "MYOCARDIAL INFARCTION", 2009-2012, reflecting the post-Nissen-controversy FDA revision), and two HRT products (Prempro/Premphase and Estrogel/estradiol, both showing real WHI-driven label evolution). SNOMED reason-code extraction remains blocked specifically on UMLS API credentials not being present in this environment (`configs/config.py` is gitignored) -- an access gap, not an approach failure.

## Phase 13: Open items and immediate next steps

Carried from `MASTER_RESEARCH_PLAN.md` Section 22 (unresolved, need Dr. Vindo's input) and the current experimental frontier:

- Confirm with Dr. Vindo whether PTGC's tiering addresses his original concern, or whether he had a different mechanism in mind.
- Confirm his possible role as a board-certified-physician-adjacent co-reviewer for human validation.
- Confirm reaction to the evidence-weighted OGFR/DGP/EWEF proposal as "the" biomedical-specific methodological novelty.
- Confirm whether UMich has its own EHR/guideline-change data infrastructure that could substitute for public sources.
- Confirm venue (NC vs. a dual-track strategy) — this materially changes the human-validation and submission-plan sections.
- Run the now-built seeded ontology-anchored OGDA construction across multiple seeds (infra ready, blocked on GPU availability at time of writing) -- would let the *real* side of the novelty comparison have proper variance too, not just the random-control side.
- Re-supply UMLS API credentials to unblock SNOMED reason-code extraction.
- Understand why HRT's collateral (cross-scenario) effect doesn't replicate the pattern the other 3 concepts show, now that this is precisely located rather than just noted as an anomaly.
- Extend the DailyMed puller to more pilot-concept drugs (rofecoxib/Vioxx has no DailyMed SPL history -- withdrawn pre-dating typical DailyMed coverage; would need a different source for that concept specifically).
- Map the 396 Herrera-Perez entries against BioUnlearn-Bench's scenario definitions to turn them into real dataset instances, not just a source catalog.
- Scale the dataset beyond pilot size (190 instances) and build the still-missing PAC scenario.
- No human/physician validation, no knowledge-editing (ROME/MEMIT) or RAG-suppression baselines, no NC-specific paper machinery (ethics/IRB, Reporting Summary) — all still open, per the original critique.

## Phase 14: The TOFU cross-dataset comparison, and settling the OCD-thesis question structurally

Direct continuation after the user asked for a full cross-model/cross-dataset/ablation status
audit and explicitly said to keep running without stopping.

**TOFU GA/NPO/RMU at matched dose**: reproduced GA's standard-config (10 epochs, lr=1e-5)
catastrophic collapse on TOFU (`model_utility` 0.60 -> 0.0 by epoch 4) -- confirms the
collapse failure mode is a general, dose-dependent property of GA, not something specific to
BioUnlearn-Bench's data. Tried BioMistral's exact gentle dose (lr=2e-6, 8 steps) on TOFU --
too gentle, essentially no movement. Found the actual TOFU sweet spot at lr=1e-5, 13 steps (1
epoch): real, non-collapsed movement (`model_utility` 0.60->0.59, `forget_Q_A_Prob`
0.88->0.80). Ran NPO and RMU at the identical dose for a fair three-way comparison.

**A second RMU config bug, same root cause as the LoRA one, different symptom**: the first
TOFU RMU attempt trained with `[RMU] Set requires_grad=True on 0 parameters` -- a silent
failure that would have "completed" with a totally unchanged model. Root cause:
`configs/trainer/RMU.yaml`'s `trainable_params_regex: .*lora.*` (added earlier specifically
so RMU wouldn't undo LoRA's memory savings on the 7B/8B models) matches zero parameter names
when LoRA isn't active -- this TOFU run used the small full-precision 1B model, no LoRA.
Caught immediately by the now-habitual check of the logged trainable-param count before
trusting any run's output. Fixed via an explicit CLI override,
`'trainer.method_args.trainable_params_regex=[".*"]'`, confirmed by the log showing 146
trainable params on the retry.

**Result**: RMU's relative gentleness (established on BioMistral, where it was the mildest of
the three methods) does **not** hold on TOFU -- there it's the most aggressive
(`model_utility` 0.60->0.50, `forget_Q_A_Prob` 0.88->0.41), consistent with the earlier
BioMistral-vs-Llama-3.1-8B divergence. Third independent data point confirming RMU's
"gentleness" is model-dependent, not a fixed method property -- a real cross-model,
cross-dataset finding worth stating carefully (per-model, not universal) in the paper.

**Settling the actual thesis question, not just the "did we run TOFU at all" question**: the
GA/NPO/RMU runs above establish method-behavior parity (collapse vs. non-collapse
generalizes across datasets) but don't touch the paper's real diagnostic claim -- that
clinical concepts' ontological entanglement causes collateral damage on retain-critical
*neighbor* concepts. Rather than trying to force that comparison onto TOFU, checked whether
TOFU's own structure could even support it. Loaded `forget10`/`retain90` directly: 400 = 20
authors x 20 facts, 3600 = 180 authors x 20 facts (400+3600 = 4000 = 200 x 20, matching
TOFU's documented 200-author design). Extracted author identities directly via regex against
the "what is the full name of the author..." answer pattern and confirmed **zero overlap**
between forget-side and retain-side authors. This means TOFU's split is disjoint at the whole
fictitious-author level, with each author's facts (birthplace, genre, awards, book titles)
independently generated and structurally unconnected to any other author -- no analog to a
UMLS drug-class hierarchy or ICD comorbidity graph exists for an edit to leak into. Full
writeup with reproduction code in `documentation/TOFU_STRUCTURAL_ANALYSIS.md`.

**Why this is a finding, not a gap**: TOFU cannot pose the OCD-style neighbor-collateral-
damage question by construction, independent of whether the experiment was ever attempted --
which is itself positive evidence that a UMLS-graph-grounded benchmark is a necessary
contribution rather than a redundant one. This closes the item that had been flagged as the
"real" thesis-supporting comparison still missing, and reframes it from an open experimental
gap into a benchmark-design argument the paper should state explicitly (directly pre-empts
the reviewer question "why not just use TOFU/WMDP for this").

## Phase 15: Closing the multi-concept statistical-rigor gap on all 4 concepts, and a second retraction

Direct continuation, same session, after Phase 14. Section 8.9 of `NOVEL_METHODOLOGY_OGDA.md`
had flagged that the multi-concept replication claim (`OGDA_multiconcept_replication_summary.json`,
own-scenario DEF "outlier" at z=-2.70 to -23.38 across all 4 real-UMLS-graph concepts) was built
on the same n=1-real-vs-n=3-random weakness already caught and corrected on aspirin specifically
-- and had "not yet been re-run with n=3 per concept" as an explicitly open item.

**Closed directly**: ran 2 more real-OGDA seeds each on rosiglitazone (C0289313), HRT (C0282402),
and Vioxx (C0876768) via a background driver script (`ogda_subspace_ablation.py --seed 1/2`
followed by `src/eval.py --config-name=eval.yaml model=BioMistral-7B eval=bioun` against each
saved checkpoint) -- 6 concept-seed pairs, ~5-10 min each, ran cleanly with no errors or GPU
contention. Recomputed every own-scenario and other-scenario FA/DEF z-score as a genuine
n=3-vs-n=3 two-sample comparison (sample std, ddof=1 -- verified this matches the documented
aspirin z=4.47 exactly before trusting the method on the new concepts). Full numbers:
`data/gate2_results/OGDA_multiconcept_n3vn3_statistical_comparison.json`.

**Second retraction, not just a caveat**: the original "3 of 4 concepts show clean DEF collapse
to exactly 0.0, decisive" claim does not survive. Own-scenario z-scores at proper n=3-vs-n=3
range from 0.49 to 4.47 -- only aspirin's RGU_fa clears conventional significance on its own;
the other three concepts' own-scenario DEF sit at z=-1.86 to -2.04, right at the edge, not
clearly past it. Marked `OGDA_multiconcept_replication_summary.json` explicitly `SUPERSEDED`
in place (kept for the historical record of what the flawed n=1 analysis showed, not deleted)
rather than quietly revising the number.

**What's new and more interesting than what was lost**: the *collateral* (other-scenario) side.
Two concepts -- rosiglitazone and Vioxx -- show a large, individually significant DEF
suppression specifically on their **untargeted** scenario (z=-3.07 and z=-6.28, the latter the
single largest effect in the whole table), meaning OGDA does more collateral damage than a
random subspace edit of the same size for those two concepts. This is arguably a more relevant
finding for the paper's actual thesis than clean own-scenario erasure would have been, since
collateral damage on ontology-neighbor concepts is exactly what the paper is trying to
characterize. HRT's collateral exception -- previously noted once, on n=1 -- is now a confirmed,
reproducible anomaly (z=+0.68, the only positive DEF z-score across all 8 own+other
comparisons), not a single-run oddity.

**Honest bottom line**: across all 8 own+other DEF comparisons, 7 point the same direction
(real OGDA's DEF below the random-control mean) -- corroborating context for a real, if more
modest than originally claimed, effect, though not an independent formal test since all 4
concepts reuse the same n=3 random-control distribution. This is a smaller, more defensible
novelty claim than the original 4-concept "decisive" framing, produced by directly closing a
gap the project's own documentation had flagged as open, in direct response to the user's
"keep on running things, do not stop" instruction, rather than leaving it as a known limitation
indefinitely.

## Phase 16: Credentials supplied, dataset scale-up, RAG-suppression baseline, ROME/MEMIT started

Triggered by the user asking directly whether the project had "full-fledged A* grade paper
stuff" -- answered honestly (no, with a specific gap list: thin novelty claim, pilot-scale
dataset, missing PAC scenario, no human validation, no adjacent-method baselines, no NC
submission machinery, unresolved venue). The user then divided the list: they would personally
own human validation and NC submission machinery, asked for adjacent-method baselines (ROME/MEMIT,
RAG-suppression) to be done "by machine," and supplied both `OPENROUTER_KEY` and `UMLS_API_KEY`
after asking for a real cost estimate (given as under $2 for the dataset-scaling work, based on
actual per-call token counts, not a guess).

**Credentials**: `configs/config.py` created from the template with both keys, confirmed
gitignored and untracked before writing anything (`git check-ignore -v`), both smoke-tested
with trivial calls before any real usage (UMLS: `POST .../api-key` returned 201/TGT; OpenRouter:
a 10-token completion returned real cost telemetry, confirming the earlier cost estimate was
right). Per the user's explicit "test first, iterate with the prompts" instruction, every new
LLM-driven pipeline below was dry-run on 5-10 examples and inspected by hand before being run
at full scale.

**Herrera-Perez dataset scale-up**: built a new extraction-based generator
(`stage_b_instances/generate_from_herrera_perez.py`) that converts each of the 396 real trial
summaries into an RGU instance by *extracting and reformatting* the already-given text, rather
than *inventing* a scenario from a UMLS concept graph the way the original pilot generator does
-- a deliberately lower-hallucination-risk design. Test batch (8 instances) passed the existing
4-filter quality pipeline at 100%, well above the original pilot's 75% pass rate, validating the
prompt design before committing to the full run. Full run: 379/396 succeeded (17 parse
failures, 0 skips), and 352/379 (93%) passed quality filtering -- **352 new real RGU instances**.

**Dataset merge, done carefully to protect existing comparability**: every baseline result in
this project (GA/NPO/RMU/OGDA/RAG-suppression) is computed against the exact 190-instance pilot
set in `data/splits/`. Rather than regenerating those files and silently invalidating every prior
number, built a separate `data/splits_v2/` (`stage_b_instances/make_splits_v2_expanded.py`)
combining the original 95 pilot RGU instances with the 352 new ones (447 RGU total), leaving IFE
and the original `data/splits/` untouched. **Total dataset: 542 instances, up from 190 (2.85x)**,
with the baseline-comparability concern explicitly documented rather than glossed over.

**RAG-suppression baseline -- a real, load-bearing adjacent-method comparison**: built
`stage_a_umls/rag_suppression_baseline.py` -- no weight edit at all; a retrieved-correction
context (built from each instance's own `reversal_reason`/`ground_truth_source`/`retain_answer`)
is prepended only to the targeted scenario's prompts, with the untargeted scenario left as plain
baseline prompts. Smoke-tested on 4 examples before the full 58+56-example run. Result: **DEF on
the targeted scenario is 0.585 (RGU) / 0.547 (IFE) -- 3-4x every weight-editing method tested**
(GA/NPO/RMU/OGDA all scored 0.13-0.20), with the untargeted scenario staying within the same
noise floor already established for random-seed variance elsewhere in this project (no real
collateral movement). Framed honestly as a finding that sharpens the paper rather than
undercutting it: the case for weight-editing methods now has to rest on properties RAG can't
offer (no runtime retrieval dependency, robustness to prompt-injection, generalizing without a
matching retrieval trigger), not on raw behavioral metrics.

**ROME/MEMIT baseline -- external code, with an explicit approval gate**: reimplementing ROME's
rank-one weight-update math from scratch was judged too risky (subtle bugs in delicate linear
algebra could silently produce a meaningless baseline). Cloned `zjunlp/EasyEdit` to check
feasibility -- attempting to copy its ROME/MEMIT source into the project was blocked by the
safety classifier (pulling in and integrating external code without the user having named or
approved it specifically). Stopped, explained exactly what was being done and why, and asked
directly -- the user approved. Vendored a small (~15 file), self-contained subset (not the full
multimodal EasyEdit framework, which has a much heavier and version-pinned dependency tree that
risked conflicting with the existing pipeline) into `stage_a_umls/rome_vendor/`, fixed the
relative imports for the flattened layout, confirmed it imports cleanly against the
already-installed torch/transformers versions (no reinstall needed). Found ROME's `mistral-7b`
hparams config conveniently sets `mom2_adjustment: false`, meaning it doesn't need the expensive
Wikipedia covariance-statistics precompute step that MEMIT does -- ROME is the more tractable
near-term target. Built and validated (10/10 passing after one case-sensitivity fix) a
reformulation step (`stage_a_umls/rome_reformulate.py`) that converts open-ended RGU/IFE
clinical Q&A into the short subject/target-fact triples ROME/MEMIT actually operate on, since
neither method edits open-ended text the way GA/NPO/RMU/OGDA do. Full-scale reformulation
launched; actual edit-execution and FA/DEF/OCD scoring is the next concrete step, not yet done.

**ROME edit-execution: a real GPU-memory bug found and fixed, then a real (weak) result.** First
full-scale attempt (`stage_a_umls/rome_baseline_eval.py`, loads BioMistral-7B once, then per
instance: apply a rank-1 edit -> generate+score against the original clinical question -> restore
original weights -> next instance) crashed partway through both RGU and IFE, saving no output.
Root cause, found by reading the log tail carefully rather than just retrying: the shared GPU has
another tenant permanently holding ~20.7GB of the 39GB card, and BioMistral-7B (bf16, ~14GB) plus
ROME's backward-pass optimization loop sits right at the memory edge -- OOMs are expected and
recoverable in isolation, but two real bugs turned single OOMs into full crashes: (1) the
post-OOM "is the model still sane" sanity-check generation call wasn't itself exception-guarded,
so a second OOM during recovery crashed the whole script uncaught; (2) weight restoration only
ran in the successful-path code, not in a `finally` block, so a `generate()` OOM *after* a
successful edit would skip restoration entirely and silently carry a partially-edited model into
the next iteration. Fixed both: the sanity check is now wrapped in its own try/except, weight
restoration runs in `finally` regardless of what failed, and the script checkpoints
`data/gate2_results/ROME_{scenario}_summary.json` after every single instance (not just at the
end) so a future crash can never lose more than one instance's progress.

**Result, at full scale**: RGU completed 49/56 instances (7 OOM, gracefully skipped, no crash);
IFE completed 34/44 (10 OOM). **RGU target DEF = 0.061, IFE target DEF = 0.0 exactly** -- not one
of the 34 successfully-edited IFE instances produced the correct new answer. FA rises in both
(0.86 RGU, 0.94 IFE -- the edit does make the model stop giving the old answer) but accuracy on
the new answer stays near zero (0.12 RGU, 0.0 IFE exactly) -- the edit mostly breaks the old fact
without landing on the replacement. This is the weakest result of every method tested in the
whole project, below GA/NPO/RMU/OGDA (0.13-0.20 DEF) and far below RAG-suppression (0.55-0.58).
Collateral damage stayed low in both directions (IFE-targeted collateral RGU_def=0.201, actually
slightly *above* untouched baseline's 0.192 -- no real collateral effect) -- ROME is safe but
ineffective here, the mirror image of GA's effective-but-destructive collapse.

**Honest read**: reported plainly as a real limitation of applying ROME/MEMIT out-of-the-box to
open-ended clinical QA (built and tuned for short factual triples like "The Eiffel Tower is in
___", not longer free-form clinical answers, using unmodified community-published Mistral-7B
hyperparameters with zero domain tuning) -- not evidence that knowledge editing categorically
cannot work on this task, and not a flaw in the comparison methodology. Completes the adjacent-
method baseline item from Phase 16's list: RAG-suppression (strong, real) and ROME (weak, real)
are both now honestly reported, giving the paper a genuine three-way contrast (weight-editing
methods' collapse-vs-effect tradeoff, RAG's strong-but-infrastructure-dependent correction, and
knowledge-editing's safe-but-ineffective-out-of-the-box result) instead of just the original
GA/NPO/RMU/OGDA matrix.
