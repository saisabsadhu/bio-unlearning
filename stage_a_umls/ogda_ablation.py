"""
OGDA: Ontology-Guided Directional Ablation (documentation/NOVEL_METHODOLOGY_OGDA.md)

Training-free, closed-form alternative to gradient-based unlearning (GA/NPO/RMU/DGP).
For a forget concept c_f with real UMLS/RxNorm neighbors (data/concept_graphs/merged/),
extracts a difference-of-means "forget direction" from contrastive prompts (reusing
clmi_gate0_validity_check.py's exact prompt-generation and mean-pooling machinery),
builds a protected subspace from the concept's real ontology neighbors (weighted
exactly as OGFR already weights them: w(c,f) = alpha/(dist+1) + (1-alpha)*icd_priority),
orthogonalizes the forget direction against that subspace, and applies the ablation by
projecting the model's down_proj weight matrices to be orthogonal to the resulting
direction (Arditi et al. 2024 weight-orthogonalization recipe) -- a permanent,
in-parameter edit requiring no gradient descent loop.

IMPORTANT finding from the first (single-layer) version of this script: ablating only
one layer (e.g. layer 7) left CLMI unchanged (1.0) even for the targeted concept --
the residual stream carries redundant, multi-layer representations of factual concepts
(consistent with the "concept cone" / multi-directional finding in recent literature,
not the single shallow direction seen for behavioral triggers like refusal). This
version ablates across a RANGE of layers, extracting per-layer directions from a single
batched forward pass, matching how the underlying weight-orthogonalization technique is
actually applied in the literature it's drawn from.

Usage:
    python stage_a_umls/ogda_ablation.py C0004057 "aspirin cardiovascular prevention" \
        --layers 4-28 --out_dir saves/unlearn/OGDA_C0004057
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

MODEL_ID = "BioMistral/BioMistral-7B"
N_PROMPTS_CONCEPT = 40
N_PROMPTS_PER_NEIGHBOR = 12
ALPHA = 0.6  # matches configs/config_template.py ALPHA default
MAX_RETAIN = 25  # matches configs/config_template.py MAX_RETAIN default


def load_merged_graph(cui):
    path = f"data/concept_graphs/merged/{cui}_merged.json"
    with open(path) as f:
        return json.load(f)


def neighbor_weight(neighbor):
    """Matches OGFR's existing weighting: w(c,f) = alpha/(dist+1) + (1-alpha)*icd_priority."""
    dist = neighbor.get("hop", 1)
    icd_priority = neighbor.get("icd_priority", 0.5)
    return ALPHA / (dist + 1) + (1 - ALPHA) * icd_priority


def get_activations_all_layers(model, tokenizer, concept_name, n_prompts, seed=None):
    """One batched forward pass; returns (n_prompts, n_layers, hidden) mean-pooled
    (over non-pad tokens) activations for every layer at once. seed shuffles
    which templates are used (see make_prompts) -- lets the same concept's
    direction be extracted from a genuinely different prompt sample per seed,
    for building a real multi-seed distribution on the ontology-anchored side."""
    prompts = make_prompts(concept_name, n_prompts, seed=seed)
    hidden_list, ids_list, attn_list = get_hidden_states_batched(model, tokenizer, prompts)
    out = []
    for i in range(len(prompts)):
        layer_stack = hidden_list[i]  # (n_layers, seq, hidden)
        mask = attn_list[i].bool()
        pooled = layer_stack[:, mask, :].mean(dim=1)  # (n_layers, hidden)
        out.append(pooled.numpy())
    return np.stack(out)  # (n_prompts, n_layers, hidden)


def difference_of_means_direction(concept_acts, background_acts):
    """Arditi-style: direction = mean(concept) - mean(background). concept_acts,
    background_acts: (n_prompts, hidden) for a single layer."""
    d = concept_acts.mean(axis=0) - background_acts.mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-8)


def build_protected_subspace_per_layer(neighbor_acts_by_name, background_acts, layer_idx, weights_by_name):
    """neighbor_acts_by_name: {name: (n_prompts, n_layers, hidden)}. Returns (rank, hidden)
    orthonormal basis for layer `layer_idx`."""
    directions = []
    for name, acts in neighbor_acts_by_name.items():
        d = difference_of_means_direction(acts[:, layer_idx, :], background_acts[:, layer_idx, :])
        directions.append(d)
    if not directions:
        return np.zeros((0, background_acts.shape[2]))
    D = np.stack(directions)
    Q, _ = np.linalg.qr(D.T)
    rank = min(len(directions), Q.shape[1])
    return Q[:, :rank].T


def orthogonalize_against_subspace(w_old, P_basis):
    w = w_old.copy()
    for row in P_basis:
        w = w - np.dot(w, row) * row
    norm = np.linalg.norm(w)
    if norm < 1e-8:
        return None  # fully contained in protected subspace at this layer -- skip it
    return w / norm


def apply_weight_orthogonalization(model, layer, direction):
    """Arditi et al. 2024 weight-orthogonalization: project the MLP down_proj weight
    matrix at `layer` to be orthogonal to `direction`, so the layer can never write in
    that direction to the residual stream again. Permanent, in-parameter, no gradients."""
    down_proj = model.model.layers[layer].mlp.down_proj
    W = down_proj.weight.data.to(torch.float32)  # (hidden_out, intermediate)
    device = W.device
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    direction_t = direction_t / direction_t.norm()

    I_minus_dd = torch.eye(W.shape[0], dtype=torch.float32, device=device) - torch.outer(direction_t, direction_t)
    W_new = I_minus_dd @ W
    down_proj.weight.data = W_new.to(down_proj.weight.dtype)


def parse_layers(spec, n_layers):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), min(int(hi) + 1, n_layers)))
    return [int(x) for x in spec.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cui")
    parser.add_argument("concept_name")
    parser.add_argument("--layers", default="4-28", help="e.g. '4-28' or '7' or '5,10,15'")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or f"saves/unlearn/OGDA_{args.cui}"
    os.makedirs(out_dir, exist_ok=True)

    graph = load_merged_graph(args.cui)
    neighbors = graph.get("neighbors", {})
    print(f"Concept: {graph['name']} ({args.cui}), {len(neighbors)} real UMLS/RxNorm neighbors")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cuda", output_hidden_states=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = parse_layers(args.layers, n_layers)
    print(f"Ablating layers: {layers} (model has {n_layers} layers total)")

    print("Extracting forget-concept activations (all layers, one pass)...")
    concept_acts = get_activations_all_layers(model, tokenizer, args.concept_name, N_PROMPTS_CONCEPT)

    weighted = sorted(
        ((neighbor_weight(n), cui, n) for cui, n in neighbors.items()),
        key=lambda x: -x[0],
    )[:MAX_RETAIN]

    print("Extracting neighbor (protected subspace) activations (all layers, one pass each)...")
    neighbor_acts_by_name = {}
    neighbor_weights = []
    for w, cui, n in weighted:
        name = n.get("name", cui)
        acts = get_activations_all_layers(model, tokenizer, name, N_PROMPTS_PER_NEIGHBOR)
        neighbor_acts_by_name[name] = acts
        neighbor_weights.append(w)
        print(f"    neighbor '{name}' (hop={n.get('hop')}, weight={w:.3f}) -> activations extracted")

    per_layer_results = {}
    for layer_idx in layers:
        w_old_raw = concept_acts[:, layer_idx, :].mean(axis=0)
        w_old_raw = w_old_raw / (np.linalg.norm(w_old_raw) + 1e-8)

        P_basis = build_protected_subspace_per_layer(neighbor_acts_by_name, concept_acts, layer_idx, neighbor_weights)
        overlap = float(np.sum([np.dot(w_old_raw, row) ** 2 for row in P_basis])) if len(P_basis) else 0.0

        w_old_perp = orthogonalize_against_subspace(w_old_raw, P_basis) if len(P_basis) else w_old_raw
        if w_old_perp is None:
            print(f"  layer {layer_idx}: forget direction fully contained in protected subspace (overlap~1.0) -- SKIPPED to avoid retain damage")
            per_layer_results[layer_idx] = {"overlap_cos2": overlap, "ablated": False}
            continue

        apply_weight_orthogonalization(model, layer_idx, w_old_perp)
        per_layer_results[layer_idx] = {"overlap_cos2": overlap, "ablated": True}
        print(f"  layer {layer_idx}: overlap_cos2={overlap:.4f}, ablated=True")

    print(f"Saving ablated model to {out_dir}...")
    model = model.to(torch.bfloat16)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "cui": args.cui,
        "concept": args.concept_name,
        "layers_targeted": layers,
        "n_layers_ablated": sum(1 for r in per_layer_results.values() if r["ablated"]),
        "n_neighbors_used": len(neighbor_weights),
        "per_layer": per_layer_results,
        "neighbor_weights": neighbor_weights,
    }
    with open(os.path.join(out_dir, "ogda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Done.")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
