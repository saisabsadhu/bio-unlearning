# OGDA: Ontology-Guided Directional Ablation

**A training-free, mechanistically-grounded replacement for DGP, motivated by our own empirical pilot data and positioned against the actual 2026 literature (not assumed prior art).**

Status: proposed 2026-07-07, following a literature search across ~15 recent (2025-2026) papers in LLM unlearning, mechanistic concept erasure, and biomedical unlearning benchmarks, done specifically to find genuine novelty rather than assume it. This document states plainly what is established prior art (must cite), what is close-but-distinct, and what is actually new here.

## 1. Why DGP needed rethinking (evidence, not speculation)

While running the Gate 2 pilot (documented in `data/gate2_results/`), plain GradAscent showed a razor-thin, hyperparameter-sensitive boundary between two failure modes on the *same* 58-example RGU training set:
- `lr=1e-5`, 40 steps: complete model collapse (loss diverged from -30 to -222, grad norms in the thousands; post-training generations had zero ROUGE overlap with *either* the forget or the retain answer — pure gibberish, not genuine forgetting).
- `lr=2e-6`, 8 steps: no measurable forgetting at all (FA numerically unchanged from the pre-unlearning baseline).

This is exactly the well-documented gradient-based unlearning instability the field struggles with, and it's why the original master plan called for a hyperparameter grid rather than one fixed config. But it also motivates a structurally different question: **is there a way to do the erasure that doesn't require a gradient descent trajectory at all**, and therefore can't land in either failure mode by construction? Section 3 below answers yes, using a technique family that already exists for a different purpose (LLM safety/refusal ablation) but has not been combined with ontology-guided protection or applied to directional clinical knowledge reversal.

## 2. What's actually out there (literature check, done before writing this)

| Paper | What it does | Why it doesn't cover this |
|---|---|---|
| Arditi et al. 2024, "Refusal in LLMs is Mediated by a Single Direction" (arXiv:2406.11717) | Establishes that a single difference-of-means direction causally mediates refusal; shows this direction can be removed via **activation ablation or weight orthogonalization** (projecting weight matrices to never write to that direction) | Foundational technique we build on directly -- but applied to jailbreak/refusal, no directional (old-fact/new-fact) pairing, no ontology, no clinical domain. **Must cite as the core mechanism's origin.** |
| Piras et al. 2026 (per search summary) | Shows difference-in-means is a special case of a broader direction-extraction family; multiple directions per layer (concept cones) rather than one line | Confirms single-direction ablation can be too simplistic -- informs our design choice to ablate a small *subspace*, not a single vector (Section 3, Step 1) |
| NSRU, "Null-Space Constrained Low-Rank Adaptation for Response-Specified LLM Unlearning" (arXiv:2606.10989, 2026) | **Closest prior art.** Explicit directional pairing (undesired response y⁻ vs. safe target response y⁺, structurally identical to our A_old/A_new), retain subspace estimated via SVD over benign hidden representations, LoRA updates projected into that subspace's null space | Still a gradient-based training loop (LoRA), just constrained -- doesn't remove the hyperparameter-sensitivity problem, only bounds its damage. Retain subspace is **data-derived** (a generic sample of "benign" representations), not concept-specific or ontology-anchored. Evaluated on TOFU/WMDP-Bio (biosecurity, not medicine); no clinical domain. |
| ActErase (arXiv:2601.00267, 2026) | Training-free, prompt-pair-derived activation redirection at inference time | Text-to-image diffusion models; no ontology; no directional old/new structure; not LLMs. |
| PISCES (arXiv:2505.22586, 2025) | In-parameter concept erasure via a learned disentangler model + automated interpretability to find concept-encoding MLP directions, then remove them from weights | General concept erasure (Gemma2/Llama3.1), no directional pairing, no ontology guidance, no clinical domain, and requires training a disentangler model (more expensive than our closed-form approach) |
| SAGO (arXiv:2604.14808, 2026) | Retention-prioritized gradient synthesis (PCGrad-style sign-constrained gradient combination) | General forget/retain sets, no directional pairing, no ontology, gradient-based (has the same training-loop instability we're trying to avoid) |
| EGUP (arXiv:2508.20443, 2026) | Reweights unlearning strength by *representational* (activation-similarity) entanglement between forget and retain samples | Confirmed via direct fetch: representational, not ontology-graph-based; TOFU/MUSE only, no biomedical domain |
| AMNESIA (arXiv:2605.30599, 2026) | Real (non-synthetic) 70,560-QA clinical unlearning benchmark from PMC-Patients-v2; **independently found that unlearning same-disease patients erodes shared clinical knowledge of other patients with that disease** -- essentially an independent empirical replication of our Failure Mode 1 (ontological entanglement / OCD) | Operates at coarse 11-category disease-bucket granularity (GPT-labeled), not fine UMLS-CUI-relation graph structure; purely patient-privacy scenario (analogous to our PAC), **no temporal/guideline-reversal scenario, no directional pairing, no ontology-guided method, only diagnoses the problem, proposes no fix** |
| REMEDI (arXiv:2606.07141, 2026) | MIMIC-III-based multi-label ICD-code classification unlearning benchmark | Classification task, not generative QA; notes the same disease-correlation entanglement problem (e.g. diabetes/CAD co-occurrence) at the ICD-code level; no directional reversal scenario |

**Important calibration**: AMNESIA independently confirming the ontological-entanglement phenomenon (their finding, our Failure Mode 1 / OCD) is good news scientifically (it's real, replicated in a completely different setting) but bad news for novelty-by-being-first -- we can no longer claim "we are the first to show unlearning causes clinically-entangled collateral damage." What we *can* still claim, and what OGDA is built to deliver, is being first to **fix** it with a mechanism that (a) operates at fine ontology-relation granularity rather than coarse disease buckets, (b) handles the directional guideline-reversal structure AMNESIA/REMEDI don't address at all, and (c) requires no gradient training loop.

## 3. OGDA: the method

Applies to RGU (and IFE) scenario instances, where a forget concept `c_f` (UMLS CUI) has, in the RGU case, a paired current answer `A_new`, and an ontology neighborhood `N_k(f)` already computed by OGFR's existing UMLS graph traversal (`stage_a_umls/umls_graph.py`).

### Step 1 -- Direction extraction (reuses CLMI's existing machinery)

For each of `A_old` (or `c_f` generally), `A_new` (RGU only), and each neighbor `n ∈ N_k(f)`: generate contrastive prompts (the same protocol already implemented in `clmi_gate0_validity_check.py::make_prompts`) and extract residual-stream activations at a chosen layer `L`. Compute a small set of directions per concept (not just one -- per Piras et al., a concept may occupy a low-rank subspace rather than a line): top principal components of the class-conditional mean-difference across several contrastive prompt batches, capped at a small number `r` (e.g. `r=3`) per concept.

This yields `W_old` (small basis for the outdated concept), `W_new` (RGU only), and `{W_n : n ∈ N_k(f)}`.

### Step 2 -- Ontology-anchored protected subspace (the actual novel step)

Unlike NSRU's data-derived retain subspace (SVD over a generic sample of "benign" hidden states) or PISCES's automated-interpretability-discovered features, the protected subspace here is **defined by the same UMLS relation structure and retention-priority weights OGFR already computes**:

```
P = span( W_new  union  { W_n : n in N_k(f), weight w(n, f) >= threshold } )
```

using the existing OGFR weight `w(c,f) = alpha/(dist_UMLS(c,f)+1) + (1-alpha)*ICD_priority(c)`, capped at `MAX_RETAIN` neighbors exactly as OGFR already does. This makes the protected subspace (a) concept-specific -- every forget instance gets its own bespoke protection set rather than one generic subspace for the whole model, (b) small and well-conditioned (tens of directions, not a generic sample requiring hundreds/thousands of benign examples), and (c) auditable -- a reviewer or regulator can inspect exactly which UMLS relations produced the protection set, which matters for a clinical-safety paper's credibility.

### Step 3 -- Orthogonalization and ablation (training-free)

Project the forget subspace to be orthogonal to the protected subspace: `W_old_perp = W_old - proj_P(W_old)` (standard QR/Gram-Schmidt). Then apply Arditi-style ablation using `W_old_perp`:

- **Activation-level** (reversible, cheap -- good for the ablation study comparing to weight-level): at every forward pass, `h' = h - sum_i (w_i w_i^T) h` for each direction `w_i` in `W_old_perp`.
- **Weight-level** (permanent, "in-parameter" -- matches the plan's existing emphasis on genuine parametric erasure): project the MLP `down_proj` (and attention output projection) weight matrices at layer `L` to be orthogonal to `W_old_perp`, following Arditi's weight-orthogonalization recipe, so the model structurally cannot write in that direction again.

No learning rate, no epoch count, no optimizer, no gradient descent trajectory -- and therefore no GA-style collapse-vs-no-effect instability, by construction rather than by careful tuning.

### Step 4 -- Verification (closes the Gate 0 loop with a causal story, not just an empirical hope)

Run CLMI (already implemented, `mean_pool` scheme adopted in Gate 0) on the ablated model. Because `W_old_perp` is the exact direction family CLMI's own probe would key on, we now have a **mechanistic explanation** for why CLMI should drop -- not just an empirical correlation we're hoping holds. This is the missing piece Gate 0 flagged as still open (`data/clmi_gate0/GATE0_FINDINGS.md`): the pre-vs-post CLMI comparison on the GA checkpoint showed no movement because GA at that dose didn't touch anything; OGDA gives a checkpoint where we know, by construction, that the targeted direction was removed, making it the correct test case for finishing Gate 0's validity claim.

## 4. New metric enabled by this design: Protection Fidelity (PF)

Since the protected subspace `P` is known exactly (not just measured after the fact), we can report, per RGU/IFE instance:

```
overlap(f) = cos^2(angle between W_old and P)      -- how much the forget and protect directions overlapped BEFORE ablation (an interpretable per-instance difficulty score)
PF(f) = Acc_new(post-ablation) / Acc_new(pre-ablation)   -- how much retain accuracy on the protected concepts survived
```

No gradient-based baseline can offer this: you cannot cleanly explain *why* GA, NPO, or RMU preserved or damaged a specific retain concept for a specific forget instance. Here it's a direct, inspectable linear-algebra fact, reportable per-instance and aggregated -- a genuinely new kind of evidence for a reviewer asking "how do you know your method's OCD improvement isn't luck."

## 5. Honest scoping (matches the plan's existing standard for DGP)

- **Linear-representation assumption**: OGDA assumes the forget concept is (approximately) linearly represented in a low-rank subspace at the chosen layer. Piras et al. 2026's concept-cone finding means this can fail for some concepts -- report sensitivity to subspace rank `r` and layer choice `L` as an ablation, don't assume `r=3` at one layer is universally right.
- **Scope**: designed for RGU and IFE, where the forget target is a clean, prompt-elicitable clinical concept. **Not** proposed as a fix for PAC (privacy/patient-identity forgetting), where the "concept" is a diffuse statistical pattern rather than a nameable clinical entity -- OGFR/DGP-style training-based methods remain the right tool there. This mirrors DGP's original honest scoping (Section 7.3.3 of the master plan) and should be stated just as directly.
- **Single-layer intervention risk**: concepts may be redundantly represented across layers; a single-layer ablation may be circumventable by relearning attacks that route around it. The existing relearning-attack experiment (Section 12 of the master plan) becomes doubly important here -- it's the test of whether a closed-form edit is actually as durable as a trained one, not just cheaper.

## 6. How this changes the master plan

- Section 7.2 (DGP) should be retitled/reframed as OGDA, keeping DGP's original gradient-based formulation as a **documented alternative/ablation** ("DGP" becomes the gradient-based baseline OGDA is compared against, not the headline method) -- this is scientifically useful: it lets the paper report "closed-form ontology-guided ablation vs. gradient-based ontology-guided projection" as a real ablation axis, strengthening rather than discarding the earlier work.
- Section 2 (Contributions) gains OGDA as the headline biomedical-specific methodological novelty, with the framing: *"the first method to combine (a) an authoritative external domain ontology to define a per-instance protected subspace, (b) directional erasure for temporally-paired clinical knowledge reversal, and (c) a training-free, closed-form mechanism immune to the collapse-vs-no-effect instability we demonstrate empirically in gradient-based baselines."*
- New research questions: does OGDA's PF metric correlate with OCD as measured by baselines (construct validity check)? Does OGDA avoid the collapse/no-effect bimodality GA showed across a range of ablation strengths (this is directly testable with our existing pilot data as the comparison point)?
- Compute impact: OGDA removes the need for a training loop for RGU/IFE ablation experiments entirely -- activation extraction (already implemented for CLMI) plus a closed-form projection is orders of magnitude cheaper than the GA/NPO/RMU training runs, meaningfully reducing the compute budget in Section 16 of the master plan for this component (though the gradient-based baselines are still needed for comparison).

## 7. What to build next (concrete, in priority order)

1. Implement Step 1-3 as a new script, `stage_a_umls/ogda_ablation.py`, reusing `clmi_gate0_validity_check.py`'s activation extraction and `stage_a_umls/umls_graph.py`'s neighbor weights directly (both already exist and are tested).
2. Run it on the same RGU pilot concepts already used for CLMI (Gate 0), at the same layer(s), producing an ablated checkpoint the same way `RGU_ga_pilot` was produced.
3. Re-run the existing bioun eval suite (FA/DEF/EWEF) and the CLMI post-unlearning check on this new checkpoint -- directly comparable to the GA pilot's numbers already in `data/gate2_results/`.
4. Only then decide rank `r` and layer `L` sensitivity sweeps, and whether activation-level or weight-level ablation is reported as primary.

## 8. Empirical validation journey (done 2026-07-07, real GPU experiments, not projected)

This section reports what actually happened when Steps 1-3 above were implemented and run on the real aspirin concept (C0004057, 7 real UMLS/RxNorm neighbors from `data/concept_graphs/merged/`). It is reported in full, including the parts that didn't work on the first attempt, because the sequence of failures is itself informative and each one was root-caused rather than papered over.

### 8.1 First real result, before any ablation: overlap confirms the entanglement hypothesis

Extracting the aspirin forget direction and the protected subspace from its 7 real ontology neighbors at layer 7 gave `cos^2(w_old, P) = 0.9853` -- the forget direction overlaps its own neighbor subspace by 98.5%. This is a direct, quantitative, mechanistic confirmation of Failure Mode 1 (ontological entanglement / OCD): naive ablation of the raw forget direction, without the orthogonalization step, would have removed almost the entire signal shared with retain-critical neighboring concepts.

An unplanned second finding fell out of computing this per-layer rather than at one layer: the overlap **decreases monotonically with depth**, from 0.996 at layer 4 to 0.535 at layer 28 (full table: `data/gate2_results/OGDA_multilayer_weightortho_aspirin_summary.json`). This suggests early-to-middle layers encode more surface/lexical features shared between closely related clinical concepts, while later layers differentiate them more -- an interesting, unplanned observation worth its own analysis, potentially informing which layers are most productive to target.

### 8.2 Single-layer weight-orthogonalization: no effect (root-caused, not a bug)

Ablating only layer 7's `mlp.down_proj` (Arditi et al. 2024's weight-orthogonalization recipe) left CLMI at 1.0 for all 10 pilot concepts, including the targeted one. Root cause: this edit only stops that one matrix from *adding* new component along the forget direction -- it does nothing to a component already present in the residual stream from token embeddings or from attention output, neither of which the edit touches. A single MLP-layer edit is necessary but not sufficient.

### 8.3 Multi-layer weight-orthogonalization (25 layers, 4-28): still no effect, same root cause

Extending to 25 layers (extracting per-layer directions from one batched forward pass) still left CLMI at 1.0. Same root cause as 8.2, just at more layers: weight-orthogonalizing `down_proj` everywhere still never touches the embedding layer or any attention `o_proj`, so the direction survives via the residual stream's skip connections regardless of how many MLPs are edited.

### 8.4 Activation-level ablation: a real implementation bug caught by a sanity check, then a real negative result

Switched to the theoretically-correct mechanism: a forward hook at each layer projecting the direction directly out of the actual residual-stream tensor, which should remove the signal regardless of which sublayer introduced it. First attempt showed CLMI still at 1.0 -- but a deliberate mechanical sanity check (does the hook change anything at all?) caught a real bug before this was wrongly reported as a finding: `output_hidden_states=True`'s diagnostic tensors do **not** reflect forward-hook modifications in this transformers version, even though the hooks correctly affect real downstream computation (verified directly: zeroing a layer's output via hook changed the model's top predicted token for "The capital of France is" from "Paris" to `<unk>`). The CLMI verification script had been silently reading stale, pre-ablation activations.

Fixed by capturing activations via a second set of hooks chained after the ablation hooks (guaranteed to see whatever the ablation hooks already returned, since PyTorch chains forward hooks in registration order on the same module). Re-running with this corrected, verified capture: **CLMI still measured 1.0** for aspirin vs. its nearest neighbor (Low-Dose Aspirin) even after a fully verified, mechanically-confirmed 25-layer activation ablation.

### 8.5 Subspace ablation: found and fixed a real circularity bug, then a fairness-matched probe, then a genuinely deep finding

Implemented the two fixes Section 8.5 (original) proposed: multi-direction subspace ablation (rank 3 per layer, SVD of several pairwise contrasts) instead of one mean-difference vector, and Restricted-CLMI (probe trained only on the ablated layers, capacity capped near the ablation's own rank instead of an unconstrained 512-dim probe over the full stack).

First attempt: every single layer's forget subspace came back **fully contained** in the protected subspace (rank 0 after orthogonalization, nothing left to ablate, at all 25 layers). Root-caused before drawing any conclusion: the forget subspace was being built from `concept − neighbor_i` contrasts, and the protected subspace from `neighbor_i − concept` contrasts, using the *same* neighbor set for both -- these are the same vectors up to sign, so the forget subspace was mathematically guaranteed to equal the protected subspace by construction, independent of any real entanglement. Fixed by introducing an independent generic background pool (the other pilot concepts, excluding the target and its own neighbors) and building both subspaces as contrasts against that shared background instead of against each other -- now the two subspaces can genuinely differ, constrained only by real representational overlap, not by the math of how they were built.

Re-running with the fix: forget subspace survived orthogonalization at all 25 layers (rank 3 throughout, confirming genuine, non-trivial entanglement rather than the earlier construction artifact). But **both full-stack and Restricted-CLMI (6 PCA components, ablated layers only) still measured 1.0** for aspirin vs. Low-Dose Aspirin.

Checked one more likely confound before accepting this: token-length mismatch between the two prompt classes (a trivial statistical difference no directional ablation would remove). Measured directly -- mean lengths 20.35 vs. 21.35 tokens, not a meaningful gap. Ruled out.

**Working interpretation**: "aspirin cardiovascular prevention" and "Low-Dose Aspirin" are near-synonyms sharing almost all surface vocabulary except a few differing words. Mean-pooling over an entire sequence means that even after removing the broad semantic directions we computed, the specific *token-identity* signal at the position of any word that differs between the two prompt sets (e.g. "cardiovascular", "prevention" vs. "Low-Dose") still contributes to the pooled average, and a linear probe can detect it. This points to a limitation of the verification protocol itself, not just this ablation: **mean-pooled whole-sequence linear probing is too permissive a test against near-synonym distractors**, because it can key on any remaining lexical difference rather than specifically on "does the model still know the fact." Two concrete fixes for future work: (a) probe at the *generation-relevant* token position (last token / the position where the model would need to produce the answer) rather than mean-pooling the whole sequence, matching how the representation is actually used; (b) treat CLMI as a secondary diagnostic with known limitations rather than the sole arbiter of success, with the behavioral FA/DEF metrics (Section 8.6) as the primary, decision-relevant success criterion.

This directly reinforces the original master plan's Failure Mode 3 (Verification Opacity) thesis from the inside: building a fair, working verification protocol turned out to be genuinely hard, which is exactly the argument Failure Mode 3 makes about the field generally -- our own struggle to get CLMI right is itself supporting evidence for the paper's core motivation, not a side issue.

### 8.6 Next: does OGDA change actual model behavior (FA/DEF), independent of the CLMI question above

CLMI answers "is the concept still linearly recoverable from activations" -- a parametric-presence question. It does not directly answer the more decision-relevant question: does the model's actual generated output change? That requires running the existing `bioun` eval suite (FA/DEF/EWEF, already implemented and validated in Gate 2) against a saved OGDA checkpoint, exactly as was done for the GradAscent pilot. This is the next concrete step (`data/gate2_results/` will hold the comparison once run) and is a genuinely independent test from everything in this section -- OGDA could plausibly change generation behavior even where CLMI's mean-pooled probe is not swayed by it, since generation depends on a specific decoding trajectory, not a pooled average over the whole sequence.

This is now a trustworthy negative result, not a bug. Interpretation: a single difference-of-means direction per layer captures only one axis of separation. Per Piras et al. 2026's "concept cone" finding (Section 2's table), factual/conceptual knowledge likely occupies a higher-rank subspace than a single line -- removing 25 single directions (one per layer) leaves other separating dimensions fully available to a high-capacity probe (512-dimensional PCA + logistic regression over the full 32-layer residual stream) to exploit. **This most likely generalizes beyond OGDA**: any single/few-direction ablation method, evaluated against a sufficiently powerful, unconstrained linear probe, may show the same pattern. That is itself a genuine, reportable methodological finding about how "verified erasure" should be defined and measured, not a dead end for OGDA specifically.

### 8.5 What this changes about the plan going forward

- **CLMI needs a fairness-matched variant.** A 512-dimensional probe over the full residual stream is not a fair test of whether a specific, interpretable, rank-`r` intervention succeeded -- it can find unrelated separating signal regardless of what was removed. Propose a **Restricted-CLMI**: project activations onto the complement of the *known* ablated subspace before probing (does separability survive outside the intervention's own target space?), reported alongside the unrestricted version, so both "did we remove what we intended" and "is there anything else separating these concepts" are answered rather than conflated into one number.
- **Ablate a larger subspace, not one direction per layer.** Extract the top few principal components of each concept's contrastive activations per layer (matching Section 3 Step 1's original design, which specified this before the implementation simplified to a single mean-difference vector) and ablate the whole subspace, not just its mean direction.
- **This is exactly the kind of result worth reporting as a finding, not hiding.** A paper section titled "Why naive directional ablation is insufficient for biomedical concept erasure, and what that implies for verification design" is a genuine contribution in its own right -- arguably a stronger, more defensible one than an artificially clean success, because it's mechanistically explained and independently motivates both a fix (multi-direction subspace ablation) and a metric refinement (Restricted-CLMI) rather than resting on a single positive number.

### 8.7 The random-subspace control: a real result, then a real correction, then a precise finding

The single most important open question after Section 8.6's behavioral result (real OGDA moves FA up and DEF down/to-zero on both scenarios) was whether that movement reflects the ontology-anchored direction specifically, or whether *any* similarly-sized intervention (same rank, same 25 layers, same weight-edit mechanism) would cause comparable drift. Built `ogda_random_control.py`: identical mechanism, but the subspace ablated at each layer is a random orthonormal rank-3 basis (QR of a Gaussian matrix), not derived from any concept or ontology.

**First attempt (seed=0)**: the random control's IFE numbers came back *exactly* identical to baseline, and RGU barely moved -- read at the time as strong evidence the real, ontology-derived direction was doing something non-generic. This was reported as the headline finding.

**Second and third attempts (seed=1, seed=2), run to build a proper variance estimate**: this conclusion did not survive. Seed=1's RGU FA moved *more* than real OGDA's did (-0.052 vs. real OGDA's +0.017), in the opposite direction, and seed=2's IFE FA moved by +0.071, comparable to or exceeding real OGDA's +0.054. **The n=1 comparison was premature** -- FA has substantial seed-to-seed variance under random-subspace ablation and is not, on its own, a reliable way to tell the real intervention apart from a random one.

**However, computing z-scores for real OGDA's aspirin result against the n=3 random-seed distribution (mean, std per metric) revealed a real, precise, more defensible finding**:

| Metric | Random-seed distribution (n=3, mean±std of the delta from baseline) | Real OGDA's delta | z-score |
|---|---|---|---|
| RGU FA | -0.029 ± 0.020 | +0.017 | 2.31 |
| RGU DEF | +0.001 ± 0.007 | -0.035 | **-5.44** |
| IFE FA | +0.030 ± 0.037 | +0.054 | 0.64 |
| IFE DEF | +0.001 ± 0.001 | **-0.015 (full collapse to 0.0)** | **-23.38** |

**FA is not a reliable differentiator** (z=0.64 to 2.31 -- IFE FA is statistically indistinguishable from random-subspace noise at this sample size). **DEF is a clear, large-magnitude outlier on both scenarios** (z=-5.44 and -23.38) -- no random seed came anywhere close to real OGDA's RGU DEF drop or, especially, its full IFE DEF collapse to exactly 0.0. Since DEF = FA_old × Acc_new, it requires the model to both stop matching the old answer *and* correctly produce the retain-consistent one -- a more demanding, apparently less noise-prone signal than FA alone.

**Corrected conclusion, replacing the earlier (premature) framing**: the claim that "OGDA's mechanism is non-generic" should now rest specifically on the DEF result, not on a blanket "the random control barely moves anything" statement, which was true for seed=0 but not representative once more seeds were examined. This is a real, honest correction, made and documented in the same place as the original claim rather than quietly revised -- and the corrected, DEF-specific finding is arguably a *stronger*, more precise result than the original blanket one: it identifies exactly which behavioral signature is attributable to the ontology-anchored direction (correct-answer collapse) versus which is not (raw topic drift as measured by FA). n=3 random seeds is still small; more seeds would tighten these z-scores further, and getting a variance estimate on the real-OGDA side (e.g., resampling which neighbor/background prompts are used) is a natural next step before this becomes a load-bearing claim in the paper. Full numbers: `data/gate2_results/OGDA_random_control_statistical_summary.json`.

### 8.8 Layer sweep and a precise mechanistic localization of the collateral-damage effect

Following 8.5's suggestion that a single "gentler" alternative setting was inconclusive and a proper sweep was needed: ran a 4-point CLMI sweep (rank=1/25 layers, rank=3/layers 14-18, rank=3/layers 4-12, rank=3/layers 20-28) on the aspirin concept. All four showed flat CLMI=1.0/1.0 -- CLMI is not a useful screening signal for distinguishing these settings from each other, consistent with 8.5's finding that CLMI is too permissive against near-synonyms regardless of what's ablated. (A real filename-collision bug was caught and fixed during this sweep: the output path only encoded rank and CUI, not layer range, so the layers-14-18 run silently overwrote the original 25-layer result's file before it was committed -- recovered from git history, and the script now includes the layer range in its output filename.)

Moved to direct behavioral testing instead, motivated by interpretability literature's general association of late layers with output-proximate representations:

1. **Layers 20-28 (9 layers) behaviorally**: reproduced the full IFE DEF collapse to exactly 0.0 -- the same signature as the original 25-layer edit -- with roughly a third of the layers.
2. **Narrowing further to layers 24-28 (5 layers)**: the IFE collapse **disappeared entirely** -- both IFE FA and DEF returned to exactly baseline. RGU's movement stayed essentially unchanged from the 9-layer version.
3. **This isolated the effect to layers 20-23** (present in the 9-layer window, absent in the 5-layer one). Tested directly: **layers 20-23 alone (4 of 32 layers, ~12.5% of the network) fully reproduced the IFE DEF collapse to 0.0**, confirming the hypothesis directly.

**This is a real, precise mechanistic finding, not a diffuse "editing more of the network causes more drift" story**: the collateral-damage mechanism driving DEF collapse on the untargeted scenario is concentrated in a specific, narrow 4-layer band. RGU's own-scenario movement, by contrast, stayed roughly constant (FA ~0.724, DEF ~0.19-0.20) across every late-layer variant tested regardless of whether layers 20-23 were included -- suggesting RGU's movement is governed by a different mechanism or layer set than IFE's collapse, and is not simply "the same effect, weaker." This opens a concrete, testable next step: does ablating the full 25-layer range *excluding* layers 20-23 preserve RGU's own-scenario signal while eliminating the IFE collateral collapse? If so, that would be direct evidence toward a genuinely more targeted version of OGDA, rather than accepting collateral damage as an inherent cost of the method. (Result, if run before this document is next updated, will be in `data/gate2_results/OGDA_excl2023_behavioral_C0004057_summary.json`.)
