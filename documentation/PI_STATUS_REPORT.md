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
- **30 real, citation-backed "gold-tier" reversals** harvested so far from peer-reviewed sources (Herrera-Perez et al. 2019, Prasad et al. 2013) against a target of 400+ — genuine literature-grounded ground truth, not LLM-generated, though this harvesting is still early (Section 6 below).
- **Gate 0 (verification validity check)**: confirmed our probing methodology (CLMI) can actually distinguish concepts before trusting it as a measurement tool, via label-swap and paraphrase controls.
- **Gate 2 (baseline diagnostic matrix)**: the three most important comparison methods — Gradient Ascent (GA), NPO, and RMU — all now run successfully end-to-end on BioMistral-7B across both scenarios (6 total runs), producing real, non-degenerate, interpretable results after fixing a chain of real infrastructure bugs (see Section 5).

## 4. The novel method: OGDA, and an honest read on where it stands

**OGDA (Ontology-Guided Directional Ablation)**: a training-free method that identifies the direction/subspace in a model's activation space corresponding to a specific fact, then surgically removes only the component of that direction that *isn't* shared with clinically-related concepts that must be retained (built from real UMLS/RxNorm ontology neighbors, not guessed). Checked against ~15 recent related papers (Arditi et al. 2024's activation-ablation technique, NSRU, PISCES, SAGO, and others) — the specific combination (ontology-anchored protected subspace + training-free directional ablation + applied to clinical guideline reversal) is not claimed elsewhere.

**What we've actually verified, in order:**

| Test | Result |
|---|---|
| Does the entanglement problem this method targets actually exist? | **Yes, quantitatively.** The aspirin-cardiovascular concept's direction overlaps its clinical-neighbor subspace at cos²=0.985 — measured directly, not assumed. |
| Does the ablation mechanism actually execute (not a silent no-op)? | **Yes, verified mechanically** — caught and fixed a real bug where our own verification code was reading stale pre-ablation activations. |
| Is the subspace construction mathematically sound? | Found and fixed a real circularity bug (the "forget" and "protected" subspaces were accidentally defined as the same thing by construction) — now genuinely independent. |
| Does a parametric probe (CLMI) detect erasure? | **No** — even a fairness-matched, capacity-restricted probe still perfectly separates the ablated concept from a near-synonym. Root-caused as likely a limitation of mean-pooled whole-sequence probing against near-synonyms (it can key on any remaining word-choice difference), not necessarily proof the ablation failed — this is itself informative about the paper's own Failure Mode 3 (verification is hard) thesis. |
| Does actual generated text change (FA/DEF, the behaviorally decisive test)? | **Yes — real movement, but not the targeted kind.** The model's answers do shift measurably, but shift roughly as much on the *untargeted* scenario as the targeted one, and don't cleanly land on the correct replacement fact (see numbers below). This reads as broad quality drift from editing a large fraction of the network (25 of 32 layers), not concept-specific erasure yet. |

**Concrete numbers** (aspirin concept, vs. pre-unlearning baseline):

| Metric | Baseline | OGDA (targeted: RGU) | OGDA (untargeted: IFE) |
|---|---|---|---|
| FA (higher = more "forgotten") | RGU 0.741 / IFE 0.839 | 0.759 | 0.893 |
| DEF (higher = cleaner correct erasure) | RGU 0.192 / IFE 0.015 | 0.157 | **0.0** |

Both scenarios moved in the same direction by similar magnitude — the untargeted one moved *more*. That's the collateral-damage signature of an intervention that's currently too blunt (editing too much of the network), not evidence of a working, selective mechanism.

**Bottom line on novelty**: we have a genuinely original, well-motivated proposal, rigorously stress-tested with real bugs found and fixed at every stage — but not yet a demonstrated *working* method. The next concrete tests (already identified, not yet run): (a) a much gentler edit (fewer layers, lower rank) to see if targeted movement survives without the collateral drift, (b) a random-subspace control ablation of matched size, to isolate whether the ontology-anchoring specifically matters or whether any comparably-sized intervention would produce similar drift — this second test is the one that would most directly prove or disprove the paper's central claim.

## 5. What had to be fixed along the way (real engineering, not incidental)

The shared A100 GPU here is genuinely shared with other users' unrelated jobs — designed around that constraint rather than around getting exclusive access:

- GA's original hyperparameters caused catastrophic collapse (gibberish output) — root-caused to unconstrained-loss unbounded descent, fixed by matching the literature's step-count-based tuning rather than epoch-based.
- NPO/RMU (which need a frozen reference-model copy alongside the trainable model) were blocked by a fixed ~31GB memory cost regardless of batch size — unblocked by combining LoRA fine-tuning (cuts trainable parameters to 0.6% of the model) with loading the reference model in 8-bit, plus three RMU-specific compatibility fixes this surfaced (module-name matching under LoRA wrapping, a masked-gradient regex bug that would have silently undone the memory savings, and a dtype mismatch from mixing 8-bit and full-precision computation).
- A cache-key collision bug in the eval pipeline was silently causing one scenario's metrics to leak into the other's — caught and fixed before it corrupted any reported numbers.

Full technical detail on all of these is in `docs/bioun_running_notes.md` and `documentation/OGDA_REPRODUCIBILITY.md` (the latter has exact commands to reproduce every OGDA result above from scratch).

## 6. Honest gaps / what's not done yet

- Gold-tier ground truth harvesting is at 30 of a 400+ target — real progress, but early.
- OGDA has only been tested on one concept (aspirin); three more real UMLS-graph concepts (HRT, rosiglitazone, Vioxx) are ready to test once the layer-range/rank question above is resolved.
- No cross-model check yet (Llama-3.1-8B-Instruct as a non-clinical control) to see whether the entanglement problem is BioMistral-specific or general.
- The random-subspace control ablation (the single most important next experiment for the paper's central claim) hasn't been run yet.

## 7. Immediate next steps

1. Run OGDA at a gentler setting (fewer layers / lower rank) and re-test FA/DEF.
2. Run the random-subspace control ablation at matched size — this is the load-bearing experiment for "does ontology-anchoring specifically matter."
3. Continue gold-source harvesting toward the 400+ target.
4. Extend the working GA/NPO/RMU baseline matrix to more concepts and to the cross-model control.
