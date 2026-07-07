"""
OGDA v2: activation-level ablation (the theoretically correct mechanism).

FINDING from ogda_ablation.py (weight-orthogonalization on mlp.down_proj only, single-
and multi-layer): CLMI stayed at 1.0 for the targeted concept even after ablating 25
layers. Reason (confirmed by re-deriving the math): projecting mlp.down_proj to be
orthogonal to direction d only stops THAT SPECIFIC matrix from ADDING new component
along d to the residual stream. It does nothing to a component along d that is already
present in the residual stream from the token embeddings or from attention output
(o_proj) at any layer -- both untouched by that edit. Since our contrastive prompts
literally contain the concept's name as tokens, the embedding layer alone can inject
most of the "this text is about aspirin" signal, and every later layer's residual
stream just carries it forward via the skip connection regardless of what the MLPs
do. Weight-orthogonalizing down_proj was necessary-but-not-sufficient.

This version ablates the ACTUAL residual-stream activation at every layer via a forward
hook: h' = h - d(d^T h), for the layer-specific direction d (orthogonalized against that
layer's ontology-protected subspace, exactly as before). This removes the component
along d regardless of which sublayer (embedding, attention, or MLP) put it there --
guaranteed correct by construction, at the cost of being a runtime intervention rather
than a permanent weight edit (see documentation/NOVEL_METHODOLOGY_OGDA.md Section 3 for
the activation-level vs weight-level distinction; this establishes whether the concept
works at all before worrying about making it permanent).

Usage:
    python stage_a_umls/ogda_activation_ablation.py C0004057 "aspirin cardiovascular prevention" \
        --layers 4-28
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clmi_gate0_validity_check import make_prompts, get_hidden_states_batched
from ogda_ablation import (
    load_merged_graph, neighbor_weight, get_activations_all_layers,
    build_protected_subspace_per_layer, orthogonalize_against_subspace, parse_layers,
    MODEL_ID, N_PROMPTS_CONCEPT, N_PROMPTS_PER_NEIGHBOR, MAX_RETAIN,
)


class DirectionAblationHook:
    """Registered on a decoder layer's forward; projects `direction` out of the
    layer's OUTPUT hidden state (the residual stream after that layer)."""
    def __init__(self, direction):
        self.direction = direction  # (hidden,) torch tensor, unit norm

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        d = self.direction.to(h.dtype).to(h.device)
        proj = torch.einsum("...h,h->...", h, d).unsqueeze(-1) * d
        h_new = h - proj
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cui")
    parser.add_argument("concept_name")
    parser.add_argument("--layers", default="4-28")
    args = parser.parse_args()

    graph = load_merged_graph(args.cui)
    neighbors = graph.get("neighbors", {})
    print(f"Concept: {graph['name']} ({args.cui}), {len(neighbors)} real UMLS/RxNorm neighbors")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda", output_hidden_states=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = parse_layers(args.layers, n_layers)
    print(f"Ablating layers (activation-level): {layers}")

    print("Extracting forget-concept activations (all layers, one pass)...")
    concept_acts = get_activations_all_layers(model, tokenizer, args.concept_name, N_PROMPTS_CONCEPT)

    weighted = sorted(
        ((neighbor_weight(n), cui, n) for cui, n in neighbors.items()),
        key=lambda x: -x[0],
    )[:MAX_RETAIN]

    print("Extracting neighbor (protected subspace) activations...")
    neighbor_acts_by_name = {}
    neighbor_weights = []
    for w, cui, n in weighted:
        name = n.get("name", cui)
        acts = get_activations_all_layers(model, tokenizer, name, N_PROMPTS_PER_NEIGHBOR)
        neighbor_acts_by_name[name] = acts
        neighbor_weights.append(w)
        print(f"    neighbor '{name}' (weight={w:.3f}) -> activations extracted")

    directions_per_layer = {}
    for layer_idx in layers:
        w_old_raw = concept_acts[:, layer_idx, :].mean(axis=0)
        w_old_raw = w_old_raw / (np.linalg.norm(w_old_raw) + 1e-8)
        P_basis = build_protected_subspace_per_layer(neighbor_acts_by_name, concept_acts, layer_idx, neighbor_weights)
        overlap = float(np.sum([np.dot(w_old_raw, row) ** 2 for row in P_basis])) if len(P_basis) else 0.0
        w_old_perp = orthogonalize_against_subspace(w_old_raw, P_basis) if len(P_basis) else w_old_raw
        if w_old_perp is None:
            print(f"  layer {layer_idx}: overlap~1.0, SKIPPED")
            continue
        directions_per_layer[layer_idx] = torch.tensor(w_old_perp, dtype=torch.float32)
        print(f"  layer {layer_idx}: overlap_cos2={overlap:.4f}, direction ready")

    # IMPORTANT (found by direct verification, see docs): `output_hidden_states=True`'s
    # diagnostic tensors do NOT reflect forward-hook modifications in this transformers
    # version, even though the hooks DO correctly affect real downstream computation
    # (confirmed: zeroing a layer's output via hook changes the model's actual top
    # predicted token). So we must capture activations via our OWN hooks, not via
    # outputs.hidden_states, or the "verification" silently checks stale pre-ablation
    # values regardless of what the ablation actually did.
    ablation_hooks = []
    for layer_idx, direction in directions_per_layer.items():
        hook_fn = DirectionAblationHook(direction)
        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        ablation_hooks.append(handle)
    print(f"Registered {len(ablation_hooks)} activation-ablation hooks.")

    captured = {}

    def make_capture_hook(layer_idx):
        def _hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach()
            return output
        return _hook

    capture_hooks = []
    for layer_idx in range(model.config.num_hidden_layers):
        handle = model.model.layers[layer_idx].register_forward_hook(make_capture_hook(layer_idx), prepend=False)
        capture_hooks.append(handle)

    def get_pooled_activations_via_hooks(prompts):
        """Runs each prompt through the (possibly ablated) model and pools the
        POST-ablation activations captured by capture_hooks -- guaranteed correct
        because capture_hooks are registered after ablation_hooks on the same layer,
        so they see whatever ablation_hooks already returned."""
        vecs = []
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                model(**enc)
            mask = enc["attention_mask"][0].bool()
            layer_vecs = []
            for layer_idx in range(model.config.num_hidden_layers):
                h = captured[layer_idx][0]  # (seq, hidden)
                pooled = h[mask].float().mean(dim=0)
                layer_vecs.append(pooled.cpu().numpy())
            vecs.append(np.concatenate(layer_vecs))
        return np.stack(vecs)

    from clmi_gate0_validity_check import compute_cv_auroc
    print("\nRunning in-process CLMI check via verified hook-capture (not output_hidden_states)...")
    neg_concept = list(neighbors.values())[0].get("name", "an unrelated topic") if neighbors else "an unrelated topic"
    pos_prompts = make_prompts(args.concept_name, 40)
    neg_prompts = make_prompts(neg_concept, 40)
    all_prompts = pos_prompts + neg_prompts
    y = np.array([1] * len(pos_prompts) + [0] * len(neg_prompts))
    X = get_pooled_activations_via_hooks(all_prompts)
    mean_auroc, std_auroc = compute_cv_auroc(X, y)
    print(f"  Post-activation-ablation CLMI (verified hook-capture) for '{args.concept_name}' vs '{neg_concept}': {mean_auroc:.4f} +/- {std_auroc:.4f}")

    for h in ablation_hooks + capture_hooks:
        h.remove()


if __name__ == "__main__":
    main()
