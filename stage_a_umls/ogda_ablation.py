"""
OGDA: Ontology-Guided Directional Ablation (documentation/NOVEL_METHODOLOGY_OGDA.md)

Training-free, closed-form alternative to gradient-based unlearning (GA/NPO/RMU/DGP).
For a forget concept c_f with real UMLS/RxNorm neighbors (data/concept_graphs/merged/),
extracts a difference-of-means "forget direction" from contrastive prompts (reusing
clmi_gate0_validity_check.py's exact prompt-generation and mean-pooling machinery),
builds a protected subspace from the concept's real ontology neighbors (weighted
exactly as OGFR already weights them: w(c,f) = alpha/(dist+1) + (1-alpha)*icd_priority),
orthogonalizes the forget direction against that subspace, and applies the ablation by
projecting the model's down_proj weight matrices at a chosen layer to be orthogonal to
the resulting direction (Arditi et al. 2024 weight-orthogonalization recipe) -- a
permanent, in-parameter edit requiring no gradient descent loop.

Usage:
    python stage_a_umls/ogda_ablation.py C0004057 "aspirin cardiovascular prevention" \
        --layer 7 --out_dir saves/unlearn/OGDA_C0004057
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clmi_gate0_validity_check import make_prompts, get_hidden_states_batched, pool_mean

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


def get_layer_activations_for_concept(model, tokenizer, concept_name, n_prompts, layer):
    """Returns (n_prompts, hidden_dim) mean-pooled activations at a single layer."""
    prompts = make_prompts(concept_name, n_prompts)
    hidden_list, ids_list, attn_list = get_hidden_states_batched(model, tokenizer, prompts)
    # hidden_list[i] is (n_layers, seq, hidden); pool_mean expects the full stack and
    # does mean-pooling over tokens for ALL layers -- we just index out our layer after.
    vecs = []
    for i in range(len(prompts)):
        layer_stack = hidden_list[i][layer:layer + 1]  # (1, seq, hidden)
        pooled = pool_mean(layer_stack, ids_list[i], attn_list[i], tokenizer, None)  # (hidden,)
        vecs.append(pooled)
    return np.stack(vecs)


def difference_of_means_direction(concept_acts, background_acts):
    """Arditi-style: direction = mean(concept) - mean(background)."""
    d = concept_acts.mean(axis=0) - background_acts.mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-8)


def build_protected_subspace(model, tokenizer, neighbors, layer, background_acts):
    """Returns (r, hidden_dim) orthonormal basis for the weighted neighbor directions,
    capped at MAX_RETAIN neighbors by descending weight (matches OGFR's MAX_RETAIN cap)."""
    weighted = sorted(
        ((neighbor_weight(n), cui, n) for cui, n in neighbors.items()),
        key=lambda x: -x[0],
    )[:MAX_RETAIN]

    directions = []
    weights = []
    for w, cui, n in weighted:
        name = n.get("name", cui)
        acts = get_layer_activations_for_concept(model, tokenizer, name, N_PROMPTS_PER_NEIGHBOR, layer)
        d = difference_of_means_direction(acts, background_acts)
        directions.append(d)
        weights.append(w)
        print(f"    neighbor '{name}' (hop={n.get('hop')}, weight={w:.3f}) -> direction extracted")

    if not directions:
        return np.zeros((0, background_acts.shape[1])), []

    D = np.stack(directions)  # (n_neighbors, hidden)
    # QR on the weighted, stacked directions gives an orthonormal basis spanning P
    Q, _ = np.linalg.qr(D.T)
    rank = min(len(directions), Q.shape[1])
    return Q[:, :rank].T, weights  # (rank, hidden)


def orthogonalize_against_subspace(w_old, P_basis):
    """w_old_perp = w_old - proj_P(w_old), P_basis rows are an orthonormal basis for P."""
    w = w_old.copy()
    for row in P_basis:
        w = w - np.dot(w, row) * row
    norm = np.linalg.norm(w)
    if norm < 1e-8:
        raise ValueError("Forget direction fully contained in protected subspace -- cannot ablate without damaging retain concepts. Consider a different layer or expanding rank.")
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

    # down_proj writes to the residual stream as W @ x; direction d lives in that
    # output (residual-stream) space. Zero d's contribution from every column:
    # W' = (I - d d^T) W, so the layer can no longer write along d at all.
    I_minus_dd = torch.eye(W.shape[0], dtype=torch.float32, device=device) - torch.outer(direction_t, direction_t)
    W_new = I_minus_dd @ W
    down_proj.weight.data = W_new.to(down_proj.weight.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cui")
    parser.add_argument("concept_name")
    parser.add_argument("--layer", type=int, default=7)
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

    print(f"Extracting forget-concept direction at layer {args.layer}...")
    concept_acts = get_layer_activations_for_concept(model, tokenizer, args.concept_name, N_PROMPTS_CONCEPT, args.layer)

    print("Extracting neighbor (protected subspace) directions...")
    P_basis, neighbor_weights = build_protected_subspace(model, tokenizer, neighbors, args.layer, concept_acts)
    print(f"  Protected subspace rank: {P_basis.shape[0]}")

    w_old_raw = concept_acts.mean(axis=0)
    w_old_raw = w_old_raw / (np.linalg.norm(w_old_raw) + 1e-8)

    overlap = float(np.sum([np.dot(w_old_raw, row) ** 2 for row in P_basis])) if len(P_basis) else 0.0
    print(f"  cos^2 overlap of forget direction with protected subspace (pre-ablation difficulty score): {overlap:.4f}")

    w_old_perp = orthogonalize_against_subspace(w_old_raw, P_basis) if len(P_basis) else w_old_raw
    print(f"Applying weight-orthogonalization ablation at layer {args.layer}...")
    apply_weight_orthogonalization(model, args.layer, w_old_perp)

    print(f"Saving ablated model to {out_dir}...")
    model = model.to(torch.bfloat16)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "cui": args.cui,
        "concept": args.concept_name,
        "layer": args.layer,
        "n_neighbors_used": len(neighbor_weights),
        "protected_subspace_rank": int(P_basis.shape[0]),
        "pre_ablation_overlap_cos2": overlap,
        "neighbor_weights": neighbor_weights,
    }
    with open(os.path.join(out_dir, "ogda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
