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
