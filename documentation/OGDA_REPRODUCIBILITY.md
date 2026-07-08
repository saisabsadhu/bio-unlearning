# OGDA: Full Reproducibility Documentation

This document is the complete, step-by-step record of the OGDA (Ontology-Guided Directional Ablation) method: exact algorithm, exact hyperparameters, exact commands, exact data, and exact results, so any part of this work can be reproduced or extended without re-deriving anything from the narrative in `NOVEL_METHODOLOGY_OGDA.md`. Read that document first for motivation and literature positioning; read this one to actually run it.

## 1. Environment

- Machine: single NVIDIA A100-PCIE-40GB, shared with other users (see `docs/bioun_running_notes.md` for shared-GPU handling).
- Python: `~/py10` venv (Python 3.10.18), inherits an existing ML stack (torch 2.7.0+cu126, transformers 4.55.4, accelerate 1.11.0, bitsandbytes 0.48.1, scikit-learn) plus `hydra-core`, `deepspeed`, `wandb`, `lm-eval` installed on top (see repo commit history for exact install commands).
- Model: `BioMistral/BioMistral-7B` (Apache 2.0, ungated), loaded via `AutoModelForCausalLM.from_pretrained`.
- HF auth: token stored at `~/.cache/huggingface/token`.

## 2. Data dependencies

- `data/concept_graphs/merged/{CUI}_merged.json`: real UMLS + RxNorm + FDA-merged concept graphs, each containing a `neighbors` dict keyed by neighbor CUI with fields `name`, `relation`, `hop`, `icd_priority`, `weight`, `source`. Only 4 of the 10 pilot concepts currently have these: `C0004057` (aspirin, 7 neighbors), `C0282402` (hormone replacement therapy, 9 neighbors), `C0289313` (rosiglitazone, 7 neighbors), `C0876768` (Vioxx, 1 neighbor). OGDA as implemented requires this file to exist for a given CUI -- it will not run on concepts without one.
- `stage_a_umls/clmi_gate0_validity_check.py::PROBE_CONCEPTS`: the 10 pilot concepts (name, CUI, hand-picked neighbor, scenario) reused as both the CLMI concept list and, for OGDA, the source of the independent background pool (Section 4.3).

## 3. The OGDA algorithm, exactly as implemented (v3, current)

Implementation: `stage_a_umls/ogda_subspace_ablation.py`. This is the third iteration; v1 (`ogda_ablation.py`, weight-orthogonalization) and v2 (`ogda_activation_ablation.py`, single-direction activation hooks) are kept in the repo as documented negative-result baselines -- see Section 6.

### 3.1 Inputs

- `cui`: UMLS CUI of the forget concept (must have a `data/concept_graphs/merged/{cui}_merged.json` file).
- `concept_name`: the human-readable concept name used to generate contrastive prompts.
- `layers`: which decoder layers to ablate (default `4-28` of BioMistral-7B's 32 layers; layers 0-3 and 29-31 are left untouched -- no principled reason yet for this exact range beyond "avoid the very first few layers, which are dominated by token-level/positional features per general transformer-interpretability findings, and the very last few, which are close to the unembedding and more behavior-specific").
- `rank`: forget subspace rank per layer (default 3).

### 3.2 Step 1 -- Independent background pool (fixes the circularity bug in Section 6.3)

```
background_acts = []
for (bg_concept, bg_cui, _, _) in PROBE_CONCEPTS:
    if bg_cui == cui or bg_concept.lower() in {neighbor names for cui}:
        continue
    background_acts.append(get_activations_all_layers(model, tokenizer, bg_concept, n_prompts=8))
```

This must be a set of concepts *different from both* the forget concept and its own neighbors -- using the neighbor set itself as background is the exact bug documented in Section 6.3, and reintroducing it silently makes the forget and protected subspaces identical up to sign by construction.

### 3.3 Step 2 -- Activation extraction

For the forget concept, each neighbor, and each background concept: generate contrastive prompts via `make_prompts(concept_name, n)` (template-filling, no LLM oracle -- see `clmi_gate0_validity_check.py::TEMPLATES`, 20 fixed sentence templates cycled deterministically), run one batched forward pass with `output_hidden_states=True`, and mean-pool (over non-padding tokens) at every layer to get a `(n_prompts, n_layers, hidden_dim)` array. `N_PROMPTS_CONCEPT=40` for the forget concept, `N_PROMPTS_PER_NEIGHBOR=12` per neighbor, `8` per background concept.

### 3.4 Step 3 -- Per-layer forget subspace

```
for each layer L in layers:
    directions = []
    for each background_batch b:
        d = mean(concept_acts[:, L, :]) - mean(b[:, L, :])
        directions.append(d / norm(d))
    D = stack(directions)              # (n_backgrounds, hidden)
    U, S, Vt = svd(D, full_matrices=False)
    F_basis[L] = Vt[:rank]             # (rank, hidden) -- top principal directions
```

### 3.5 Step 4 -- Per-layer protected subspace (OGFR-consistent weighting)

```
bg_mean[L] = mean over all background batches of mean(b[:, L, :])
for each layer L in layers:
    directions = []
    for each neighbor n (sorted by OGFR weight w(n,f) = alpha/(dist+1) + (1-alpha)*icd_priority(n), alpha=0.6, capped at MAX_RETAIN=25):
        d = mean(neighbor_acts[n][:, L, :]) - bg_mean[L]
        directions.append(d / norm(d))
    D = stack(directions)
    Q, _ = QR(D.T)
    P_basis[L] = Q[:, :rank(D)].T      # orthonormal basis spanning the neighbor subspace
```

### 3.6 Step 5 -- Orthogonalization

```
for each layer L:
    F_perp = []
    for f in F_basis[L]:
        v = f
        for p in P_basis[L]:
            v = v - dot(v, p) * p      # Gram-Schmidt against the protected subspace
        if norm(v) > 1e-6:
            F_perp.append(v / norm(v))
    F_perp_basis[L] = orthonormalize(F_perp)   # re-orthonormalize what survives
```

If `F_perp_basis[L]` is empty (forget subspace fully contained in protected subspace at that layer), that layer is skipped -- ablating would necessarily damage retain-critical neighbor concepts, so OGDA declines to intervene there rather than force an unsafe edit.

### 3.7 Step 6 -- Ablation (activation-level, via forward hooks)

```
class SubspaceAblationHook:
    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        coeffs = einsum("...h,rh->...r", h, F_perp_basis[L])   # projection coefficients
        proj = einsum("...r,rh->...h", coeffs, F_perp_basis[L])  # reconstruct projection
        return h - proj   # remove the whole subspace component
```

Registered via `model.model.layers[L].register_forward_hook(hook)` for every layer with a non-empty `F_perp_basis[L]`.

**Critical implementation warning, found the hard way (Section 6.4)**: `output_hidden_states=True`'s returned tensors do **not** reflect forward-hook modifications in the transformers version used here (4.55.4), even though the hooks correctly affect real downstream computation. Any verification code must capture activations via its own forward hooks chained *after* the ablation hooks (registration order determines hook chaining on the same module), not via `outputs.hidden_states`.

## 4. Verification protocols

### 4.1 CLMI (full-stack, original protocol)

Concatenate mean-pooled activations across all 32 layers (`32 * 4096 = 131072` raw features), PCA to `min(512, n-1, n_features)` components, 5-fold stratified cross-validated `LogisticRegression(C=1.0)`, report mean/std AUROC. This is the *original* CLMI protocol from Gate 0 -- known (Section 6.5) to be too permissive against near-synonym distractors, kept for continuity/comparison.

### 4.2 Restricted-CLMI (fairness-matched, new)

Same probe, but activations are pooled only over the *ablated* layers (not all 32), and PCA components are capped at `max(2 * rank, 4)` rather than 512 -- a capacity roughly matched to what the ablation itself touched, rather than an unconstrained probe.

### 4.3 Mechanical sanity check (do this before trusting any negative result)

Before trusting a "no effect" result from any hook-based intervention, run a minimal, fast, drastic-intervention check: register a hook that zeroes a layer's entire output, and confirm the model's top predicted token on an unrelated factual prompt changes (e.g. "The capital of France is" -> not "Paris"). This is a ~10-second check and it is what caught the `output_hidden_states` staleness bug in Section 6.4 -- always run it after touching hook-registration code, before running a full CLMI pass.

## 5. Exact commands to reproduce every result in `NOVEL_METHODOLOGY_OGDA.md`

```bash
cd /NFSDISK/saisab/biounlearn

# Gate 0 (CLMI validity pre-check, all 10 concepts, base model, no ablation)
python3 stage_a_umls/clmi_gate0_validity_check.py
# -> data/clmi_gate0/gate0_summary.json

# v1: single-layer weight-orthogonalization (documented negative result, Section 6.2)
python3 stage_a_umls/ogda_ablation.py C0004057 "aspirin cardiovascular prevention" --layers 7
python3 stage_a_umls/clmi_post_unlearning_check.py OGDA_aspirin ./saves/unlearn/OGDA_C0004057
# -> CLMI = 1.0000 for all concepts (weight-orthogonalization doesn't touch
#    embeddings/attention, signal survives via residual skip-connections)

# v1: multi-layer weight-orthogonalization (Section 6.3)
python3 stage_a_umls/ogda_ablation.py C0004057 "aspirin cardiovascular prevention" --layers 4-28
python3 stage_a_umls/clmi_post_unlearning_check.py OGDA_aspirin_multilayer ./saves/unlearn/OGDA_C0004057
# -> CLMI = 1.0000, same root cause, at 25 layers

# v2: activation-level, single direction per layer (Section 6.4)
python3 stage_a_umls/ogda_activation_ablation.py C0004057 "aspirin cardiovascular prevention" --layers 4-28
# -> mechanically verified hooks work (sanity check), CLMI = 1.0000 -- real negative result

# v3: subspace ablation + Restricted-CLMI (Section 6.5, current)
python3 stage_a_umls/ogda_subspace_ablation.py C0004057 "aspirin cardiovascular prevention" --layers 4-28 --rank 3
# -> data/gate2_results/OGDA_subspace_rank3_C0004057_summary.json
# -> forget subspace survives orthogonalization (rank 3 at all 25 layers -- real
#    entanglement, not the earlier construction artifact)
# -> CLMI(full) = 1.0000, CLMI(restricted, 6 PCA comps) = 1.0000
# -> token-length confound checked and ruled out (20.35 vs 21.35 mean tokens)
```

Every command above is idempotent and safe to re-run; `ogda_ablation.py` and `ogda_subspace_ablation.py` overwrite `saves/unlearn/OGDA_{cui}/` and `data/gate2_results/OGDA_*` respectively.

## 6. Result manifest (what's saved where)

| File | Contents |
|---|---|
| `data/clmi_gate0/gate0_summary.json` | Base-model CLMI, 10 concepts x 3 pooling schemes x {real, label-swapped} |
| `data/clmi_gate0/GATE0_FINDINGS.md` | Narrative interpretation of the above |
| `data/gate2_results/OGDA_single_layer7_aspirin_summary.json` | v1 single-layer result + CLMI |
| `data/gate2_results/OGDA_multilayer_weightortho_aspirin_summary.json` | v1 multi-layer result + CLMI + root-cause note |
| `data/gate2_results/OGDA_activation_verified_aspirin_summary.json` | v2 result, hook-verified |
| `data/gate2_results/OGDA_subspace_rank3_C0004057_summary.json` | v3 result, full + restricted CLMI |
| `data/gate2_results/RGU_ga_pilot_summary.json`, `IFE_ga_pilot_summary.json` | GradAscent pilot (gradient-based comparison baseline), pre/post FA/DEF/EWEF |

## 7. Known limitations of this documentation (be honest about what's not yet done)

- Only tested on one concept (aspirin, C0004057) with a full run; HRT (C0282402), rosiglitazone (C0289313), and Vioxx (C0876768) have real merged graphs and are the next concepts to run (Vioxx has only 1 real neighbor, likely too thin for a meaningful protected subspace -- worth noting if it behaves differently).
- No FA/DEF/EWEF (behavioral, not just CLMI/parametric) result yet for any OGDA checkpoint -- Section 8.6 of the methodology doc flags this as the next concrete step; a saved-to-disk OGDA checkpoint (not just in-memory hooks) is needed for that, which the current `ogda_subspace_ablation.py` does not yet do (it only tests via in-process hooks, doesn't call `model.save_pretrained`).
- Layer range `4-28` and rank `3` were reasonable starting choices, not the result of a sweep -- a layer/rank sensitivity study is still open.
- No comparison yet against a non-ontology-anchored ablation baseline (e.g. protected subspace = random directions, or = a generic "any other topic" subspace instead of the specific UMLS neighbors) -- this is the OGDA-equivalent of the master plan's OGFR-Random ablation, and would be the direct evidence that the *ontology structure specifically* (not just "some" protection) is doing the work.
