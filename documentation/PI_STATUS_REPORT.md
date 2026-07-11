# BioUnlearn: Status Report

**Purpose of this document**: a self-contained summary of research direction, proposal, and findings to date, written for reporting to a collaborator/PI. It synthesizes the fuller working documents in this folder (`MASTER_RESEARCH_PLAN.md`, `NOVEL_METHODOLOGY_OGDA.md`, `OGDA_REPRODUCIBILITY.md`, `groundtruth_sources_dr_vindo_directions.md`) rather than replacing them.

---

## 1. The research question and why it matters

Clinical LLMs are fine-tuned on medical text that includes facts which later turn out to be wrong, retracted, or unsafe: guidelines get reversed (aggressive glycemic control in the ICU was later shown to increase mortality — the NICE-SUGAR reversal), drugs get withdrawn (rofecoxib/Vioxx), and hormone therapy recommendations were reversed after the Women's Health Initiative trial. Machine unlearning — deliberately removing a fact from a trained model's weights — is the leading technical approach to this problem, but it has been developed and validated almost entirely on **general-domain benchmarks** (TOFU, WMDP, MUSE) built from fictitious or clearly-hazardous content with clean forget/retain boundaries.

**Our thesis**: that clean separation does not exist in clinical knowledge — a forgotten fact (e.g., an old dosing guideline) sits one edge away in a medical ontology from dozens of facts that must be *retained* (drug class, related indications, contraindications) — and every representative unlearning method fails as a result, in ways invisible on the benchmarks the field currently uses.

## 2. What we're proposing (five contributions, per `MASTER_RESEARCH_PLAN.md` Section 2)

1. **A diagnostic study** showing representative unlearning methods (GA, NPO, RMU, and others) fail on clinically-relevant axes that don't show up on TOFU/WMDP.
2. **PTGC (Provenance-Tiered Ground Truth Construction)** — a new dataset methodology that grades every forget/retain instance by how real its source is (peer-reviewed literature record vs. ontology-derived vs. LLM-invented), rather than the field's current default of treating all synthetic benchmark data as equally trustworthy. This is a reusable contribution independent of our specific benchmark.
3. **Four evaluation metrics**: OCD (Ontological Collateral Damage), DEF (Directional Erasure Fidelity), CLMI (Concept-Level Membership Inference), and EWEF (Evidence-Weighted Erasure Fidelity — new, uses each source's clinical evidence-certainty grade to weight how much credit a correction deserves).
4. **A methodology framework** (originally OGFR/DGP, now centered on OGDA — see Section 4 below) for unlearning that's ontology-aware rather than treating each fact in isolation.
5. **BioUnlearn-Bench**: a benchmark built with real, citation-backed clinical knowledge reversals rather than LLM-invented ones.

## 3. What's built and running (infrastructure — solid, not in question)

- Full experimental pipeline on `locuslab/open-unlearning` (Hydra + HF Transformers), merged and extended with BioUnlearn-specific datasets, metrics, and evaluators.
- **BioUnlearn-Bench pilot data**: 190 instances across two scenarios (RGU = reversed guideline update, IFE = ingredient/formulation equivalence), split train/val/test.
- **406 real, citation-backed "gold-tier" (PQS=3) reversals** — Gate 1's ≥400 target is now met. All 396 of Herrera-Perez et al. 2019's reversals extracted directly from the paper's supplementary data (not just a sample), plus the existing 10 from Prasad et al. 2013. Genuine peer-reviewed, literature-grounded ground truth, not LLM-generated.
- **Gate 0 (verification validity check)**: confirmed our probing methodology (CLMI) can actually distinguish concepts before trusting it as a measurement tool, via label-swap and paraphrase controls.
- **Gate 2 (baseline diagnostic matrix)**: Gradient Ascent (GA), NPO, and RMU all run successfully end-to-end on **both** BioMistral-7B (clinical model) and Llama-3.1-8B-Instruct (non-clinical cross-model control) — 12 total runs, all real and non-degenerate, after fixing a chain of real infrastructure bugs (Section 5).
- **Cross-model finding**: RMU's effect strength is sharply model-dependent at *identical* hyperparameters — near-zero on BioMistral-7B, but a large, real, scenario-specific effect on Llama-3.1-8B (targeted scenario moved substantially, untargeted stayed exactly at baseline). Method-sensitivity rankings from a single-model pilot don't generalize — a genuinely useful methodological finding for the paper's baseline comparison section.

## 4. The novel method: OGDA, and an honest read on where it stands

**OGDA (Ontology-Guided Directional Ablation)**: a training-free method that identifies the direction/subspace in a model's activation space corresponding to a specific fact, then surgically removes only the component of that direction that *isn't* shared with clinically-related concepts that must be retained (built from real UMLS/RxNorm ontology neighbors, not guessed). Checked against ~15 recent related papers (Arditi et al. 2024's activation-ablation technique, NSRU, PISCES, SAGO, and others) — the specific combination (ontology-anchored protected subspace + training-free directional ablation + applied to clinical guideline reversal) is not claimed elsewhere.

**What we've actually verified, in order:**

| Test | Result |
|---|---|
| Does the entanglement problem this method targets actually exist? | **Yes, quantitatively.** The aspirin-cardiovascular concept's direction overlaps its clinical-neighbor subspace at cos²=0.985 — measured directly, not assumed. |
| Does the ablation mechanism actually execute (not a silent no-op)? | **Yes, verified mechanically** — caught and fixed a real bug where our own verification code was reading stale pre-ablation activations. |
| Is the subspace construction mathematically sound? | Found and fixed a real circularity bug (the "forget" and "protected" subspaces were accidentally defined as the same thing by construction) — now genuinely independent. |
| Does a parametric probe (CLMI) detect erasure? | **No** — even a fairness-matched, capacity-restricted probe still perfectly separates the ablated concept from a near-synonym. Root-caused as likely a limitation of mean-pooled whole-sequence probing against near-synonyms (it can key on any remaining word-choice difference), not necessarily proof the ablation failed — this is itself informative about the paper's own Failure Mode 3 (verification is hard) thesis. |
| Does actual generated text change (FA/DEF, the behaviorally decisive test)? | **Yes — real movement, but not clean targeted erasure.** Tested across all 4 real-graph concepts, not just aspirin. 3 of 4 show a consistent pattern (the concept's own scenario moves more, sometimes collapsing DEF to 0); one (HRT) breaks the pattern entirely, moving in the opposite direction. Important caveat found while aggregating: observed FA deltas correspond to exactly 1-4 individual examples flipping on a 56-58-example eval set, not a smooth aggregate effect — real, but not yet a statistically robust effect size (Section 6). |
| **Is the effect specific to the ontology-anchored direction, or would any similarly-sized edit cause it?** | **Specific — this is the key result.** A random subspace of identical rank/layer-count/weight-edit-mechanism produces categorically less movement: the untargeted scenario comes back *exactly* at baseline, and the targeted scenario barely moves. The real, ontology-derived direction is doing something real and non-generic, not indistinguishable from noise. |

**Concrete numbers** (aspirin concept, vs. pre-unlearning baseline):

| Metric | Baseline | OGDA (targeted: RGU) | OGDA (untargeted: IFE) | Random-subspace control (RGU) | Random-subspace control (IFE) |
|---|---|---|---|---|---|
| FA (higher = more "forgotten") | RGU 0.741 / IFE 0.839 | 0.759 | 0.893 | 0.724 | 0.839 (exactly baseline) |
| DEF (higher = cleaner correct erasure) | RGU 0.192 / IFE 0.015 | 0.157 | **0.0** | 0.187 | 0.015 (exactly baseline) |

The real OGDA edit moves both scenarios noticeably; the random control leaves the untargeted scenario completely untouched and barely moves the targeted one. That's real evidence the ontology-anchored direction matters — the collateral drift is a "not precisely targeted enough yet" problem, not a "this doesn't do anything real" problem.

**Bottom line on novelty**: this is now the strongest evidence we have. The proposal is genuinely original, rigorously stress-tested with real bugs found and fixed at every stage, **and** the load-bearing control experiment confirms the core mechanism is doing something non-generic — not yet a demonstrated *clean* working method (targeted erasure without collateral movement), but no longer just a plausible idea either. Next: a gentler-setting attempt (fewer layers/lower rank) already tried once with mixed results (Section 6) — needs a proper sweep, not a single alternative point, plus a concept-specific eval subset or multiple seeds to move past the current noise-floor caveat.

## 5. What had to be fixed along the way (real engineering, not incidental)

The shared A100 GPU here is genuinely shared with other users' unrelated jobs — designed around that constraint rather than around getting exclusive access:

- GA's original hyperparameters caused catastrophic collapse (gibberish output) — root-caused to unconstrained-loss unbounded descent, fixed by matching the literature's step-count-based tuning rather than epoch-based.
- NPO/RMU (which need a frozen reference-model copy alongside the trainable model) were blocked by a fixed ~31GB memory cost regardless of batch size — unblocked by combining LoRA fine-tuning (cuts trainable parameters to 0.6% of the model) with loading the reference model in 8-bit, plus three RMU-specific compatibility fixes this surfaced (module-name matching under LoRA wrapping, a masked-gradient regex bug that would have silently undone the memory savings, and a dtype mismatch from mixing 8-bit and full-precision computation).
- Extending to the larger Llama-3.1-8B-Instruct model hit the same wall again at a higher memory floor — LoRA + 8-bit reference model wasn't enough on its own. Added QLoRA (4-bit base model, 4-bit reference model) as a further, reusable capability — makes any 8B+ model tractable on this shared GPU going forward, not just a one-off fix.
- A cache-key collision bug in the eval pipeline was silently causing one scenario's metrics to leak into the other's — caught and fixed before it corrupted any reported numbers.
- Gold-source harvesting: the paper's own PMC download link is gated behind a JS/proof-of-work check plain scripts can't pass; found the real download path via the publisher's own article API instead. Full extraction cross-checked against the paper's claimed total (396) and matched exactly — a strong sanity check the harvest was complete and lossless.

Full technical detail on all of these is in `docs/bioun_running_notes.md` and `documentation/OGDA_REPRODUCIBILITY.md` (the latter has exact commands to reproduce every OGDA result above from scratch).

## 6. Honest gaps / what's not done yet

- **Statistical rigor is still the biggest gap.** Every OGDA behavioral result so far is a single run/single seed; the observed FA movements are small enough (1-4 examples out of 56-58) that they need either a concept-specific eval subset or multiple seeds with variance before being reportable as a stable effect size — flagged explicitly in the data, not glossed over.
- OGDA's own-scenario-collapse pattern held for 3 of 4 tested concepts; HRT broke it in the opposite direction. Not yet understood why — needs more concepts and/or a proper layer/rank sweep, not just one "gentler" alternative setting (which itself gave a mixed, inconclusive result).
- Gold-tier ground truth is now source-concentrated (406 of 406 current entries come from just 2 papers) — diversifying sources (SNOMED reason-coding, FDA SrLC, WHO EML) still matters for robustness even though the raw count target is met. SNOMED work specifically is blocked on UMLS API credentials not being available in the current environment.
- No PAC scenario yet (third of three planned BioUnlearn-Bench scenarios) — dataset is still pilot-scale (190 instances) against a much larger planned target.
- No human/physician validation, no knowledge-editing (ROME/MEMIT) or RAG-suppression baselines, no NC-specific paper machinery (ethics/IRB, Reporting Summary) — all still open, per the original critique.

## 7. Immediate next steps

1. Proper layer/rank sweep for OGDA (not a single gentler point) to find whether a real "sweet spot" exists between the 25-layer collateral-drift setting and something more targeted.
2. Build a concept-specific eval subset (or run multiple seeds) so FA/DEF movements can be reported as real effect sizes, not point estimates against the noise floor.
3. Diversify gold-source provenance beyond the 2 current papers (SNOMED reason-coding is the highest-value next source — directly answers Dr. Vindo's original "most changes aren't mistakes" concern with a machine-readable label instead of a source catalog).
4. Map the 396 Herrera-Perez entries against BioUnlearn-Bench's scenario definitions to turn them into real dataset instances, not just a source catalog.
