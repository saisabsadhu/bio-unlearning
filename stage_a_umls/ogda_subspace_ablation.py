"""
OGDA v3: subspace (multi-direction) ablation + Restricted-CLMI fair verification.

Two fixes motivated by documentation/NOVEL_METHODOLOGY_OGDA.md Section 8.5:

1. SUBSPACE ABLATION: v1/v2 ablated a single mean-difference direction per layer.
   Per Piras et al. 2026's "concept cone" finding, factual concepts likely occupy a
   higher-rank subspace than one line. This version extracts MULTIPLE discriminative
   directions per layer -- one difference-of-means vector per (concept, each individual
   neighbor) pair, then SVD's the resulting set down to the top-r principal directions --
   giving a genuine rank-r forget subspace, still orthogonalized against the ontology-
   protected subspace exactly as before.

2. RESTRICTED-CLMI: the original CLMI probe (512-dim PCA over the full 32-layer,
   4096-dim-per-layer residual stream = 131072 raw features) has enough capacity to
   separate almost any two distinct clinical topics regardless of what a small ablation
   removed -- that's not a fair test of whether the ablation worked, it's closer to a
   generic "are these two texts about different things" test. Restricted-CLMI instead
   trains the probe ONLY on the ablated layers' activations, capped at a small number of
   PCA components roughly matched to the ablation's own rank -- a fairer, matched-
   capacity test of whether separability survives specifically where we intervened.

Usage:
    python stage_a_umls/ogda_subspace_ablation.py C0004057 "aspirin cardiovascular prevention" \
        --layers 4-28 --rank 3
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clmi_gate0_validity_check import make_prompts, PROBE_CONCEPTS
from ogda_ablation import (
    load_merged_graph, neighbor_weight, get_activations_all_layers,
    parse_layers, MODEL_ID, N_PROMPTS_CONCEPT, N_PROMPTS_PER_NEIGHBOR, MAX_RETAIN,
)


def build_forget_subspace_per_layer(concept_acts, background_acts, layer_idx, rank):
    """Difference-of-means direction(s) of the forget concept against a GENERIC
    background (not the ontology neighbors -- see module docstring on the circularity
    bug this replaces), SVD'd to the top-`rank` principal directions across several
    background batches for robustness."""
    c = concept_acts[:, layer_idx, :]
    directions = []
    for bg_acts in background_acts:  # list of (n_prompts, n_layers, hidden) arrays
        b = bg_acts[:, layer_idx, :]
        d = c.mean(axis=0) - b.mean(axis=0)
        norm = np.linalg.norm(d)
        if norm > 1e-8:
            directions.append(d / norm)
    if not directions:
        return np.zeros((0, c.shape[1]))
    D = np.stack(directions)
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    r = min(rank, len(directions), Vt.shape[0])
    return Vt[:r]  # (r, hidden)


def build_protected_subspace_per_layer(neighbor_acts_by_name, background_acts, layer_idx, neighbor_weights):
    """Each neighbor's direction relative to the SAME generic background used for the
    forget subspace -- not relative to the forget concept itself, so this and the
    forget subspace are not forced to be identical/anti-parallel by construction."""
    directions = []
    bg_mean = np.mean([b[:, layer_idx, :].mean(axis=0) for b in background_acts], axis=0)
    for name, acts in neighbor_acts_by_name.items():
        d = acts[:, layer_idx, :].mean(axis=0) - bg_mean
        norm = np.linalg.norm(d)
        if norm > 1e-8:
            directions.append(d / norm)
    if not directions:
        return np.zeros((0, bg_mean.shape[0]))
    D = np.stack(directions)
    Q, _ = np.linalg.qr(D.T)
    rank = min(len(directions), Q.shape[1])
    return Q[:, :rank].T


def orthogonalize_subspace_against_subspace(F_basis, P_basis):
    """Remove the component of each forget-subspace basis vector that lies in the
    protected subspace, then re-orthonormalize what remains."""
    if len(P_basis) == 0:
        return F_basis
    F_perp = []
    for f in F_basis:
        v = f.copy()
        for p in P_basis:
            v = v - np.dot(v, p) * p
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            F_perp.append(v / norm)
    if not F_perp:
        return np.zeros((0, F_basis.shape[1]))
    M = np.stack(F_perp)
    Q, _ = np.linalg.qr(M.T)
    rank = min(len(F_perp), Q.shape[1])
    return Q[:, :rank].T


class SubspaceAblationHook:
    """Projects a whole subspace (multiple directions) out of the residual stream,
    not just one vector."""
    def __init__(self, basis):
        self.basis = basis  # (r, hidden) torch tensor, orthonormal rows

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        basis = self.basis.to(h.dtype).to(h.device)
        # h: (..., hidden); coeffs: (..., r) = h @ basis^T; proj = coeffs @ basis
        coeffs = torch.einsum("...h,rh->...r", h, basis)
        proj = torch.einsum("...r,rh->...h", coeffs, basis)
        h_new = h - proj
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


def apply_permanent_subspace_orthogonalization(model, basis_per_layer):
    """Bakes the subspace ablation into weights so it survives save_pretrained ->
    from_pretrained in a fresh process, unlike the forward-hook version which is
    runtime-only. Applied to BOTH self_attn.o_proj and mlp.down_proj at each
    ablated layer (the two components that write into the residual stream at
    that layer) via W' = (I - B^T B) @ W, where B has orthonormal rows spanning
    the subspace to remove. This is an approximation of the hook-verified
    version: it prevents ablated layers from *adding* new component along the
    subspace, but doesn't retroactively clean component already written into
    the residual stream by earlier (unablated) layers -- see
    documentation/OGDA_REPRODUCIBILITY.md Section 3.3 for why weight-edits
    alone were previously found insufficient without also covering o_proj."""
    for layer_idx, basis in basis_per_layer.items():
        layer = model.model.layers[layer_idx]
        B = basis.to(model.device)
        for module in [layer.self_attn.o_proj, layer.mlp.down_proj]:
            W = module.weight.data
            B = B.to(W.dtype)
            P = B.T @ B  # (hidden, hidden) projection onto the subspace
            I = torch.eye(P.shape[0], dtype=W.dtype, device=W.device)
            module.weight.data = (I - P) @ W


def compute_cv_auroc(X, y, n_components, seed=42):
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1]) if X.shape[1] > 0 else 0
    if n_comp <= 0:
        return 0.5, 0.0
    pca = PCA(n_components=n_comp)
    X_r = pca.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aurocs = []
    for train_idx, test_idx in skf.split(X_r, y):
        if len(np.unique(y[test_idx])) < 2:
            continue
        probe = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        probe.fit(X_r[train_idx], y[train_idx])
        proba = probe.predict_proba(X_r[test_idx])[:, 1]
        aurocs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aurocs)), float(np.std(aurocs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cui")
    parser.add_argument("concept_name")
    parser.add_argument("--layers", default="4-28")
    parser.add_argument("--rank", type=int, default=3, help="forget subspace rank per layer")
    parser.add_argument("--save_checkpoint", default=None, help="if set, bake the ablation into permanent weights and save here for behavioral (FA/DEF) eval")
    parser.add_argument("--seed", type=int, default=None, help="shuffles which prompt templates are sampled, for building a real multi-seed distribution on the ontology-anchored side (matches the random-control's seeded variation)")
    args = parser.parse_args()

    graph = load_merged_graph(args.cui)
    neighbors = graph.get("neighbors", {})
    print(f"Concept: {graph['name']} ({args.cui}), {len(neighbors)} real UMLS/RxNorm neighbors, subspace rank={args.rank}")

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

    print("Extracting forget-concept activations (all layers, one pass)...")
    concept_acts = get_activations_all_layers(model, tokenizer, args.concept_name, N_PROMPTS_CONCEPT, seed=args.seed)

    weighted = sorted(
        ((neighbor_weight(n), cui, n) for cui, n in neighbors.items()),
        key=lambda x: -x[0],
    )[:MAX_RETAIN]

    print("Extracting neighbor (protected subspace) activations...")
    neighbor_acts_by_name = {}
    neighbor_weights = []
    for w, cui, n in weighted:
        name = n.get("name", cui)
        acts = get_activations_all_layers(model, tokenizer, name, N_PROMPTS_PER_NEIGHBOR, seed=args.seed)
        neighbor_acts_by_name[name] = acts
        neighbor_weights.append(w)
        print(f"    neighbor '{name}' (weight={w:.3f}) -> activations extracted")

    # Independent generic background pool (other pilot concepts, unrelated to this one
    # or its neighbors) -- used as the reference class for BOTH the forget and the
    # protected subspace, so the two are not forced to be identical/anti-parallel by
    # construction (the bug this replaces: using concept-vs-neighbor for forget AND
    # neighbor-vs-concept for protection makes them the same subspace by definition).
    print("Extracting generic background activations (other pilot concepts)...")
    background_acts = []
    neighbor_name_set = {n.get("name", "").lower() for n in neighbors.values()}
    for bg_concept, bg_cui, _, _ in PROBE_CONCEPTS:
        if bg_cui == args.cui or bg_concept.lower() in neighbor_name_set:
            continue
        acts = get_activations_all_layers(model, tokenizer, bg_concept, 8, seed=args.seed)
        background_acts.append(acts)
        print(f"    background '{bg_concept}' -> activations extracted")

    basis_per_layer = {}
    for layer_idx in layers:
        F_basis = build_forget_subspace_per_layer(concept_acts, background_acts, layer_idx, args.rank)
        P_basis = build_protected_subspace_per_layer(neighbor_acts_by_name, background_acts, layer_idx, neighbor_weights)
        F_perp = orthogonalize_subspace_against_subspace(F_basis, P_basis)
        if len(F_perp) == 0:
            print(f"  layer {layer_idx}: forget subspace fully contained in protected subspace, SKIPPED")
            continue
        basis_per_layer[layer_idx] = torch.tensor(F_perp, dtype=torch.float32)
        print(f"  layer {layer_idx}: forget subspace rank={F_basis.shape[0]}, protected rank={P_basis.shape[0]}, post-orthogonalization rank={F_perp.shape[0]}")

    ablation_hooks = []
    for layer_idx, basis in basis_per_layer.items():
        handle = model.model.layers[layer_idx].register_forward_hook(SubspaceAblationHook(basis))
        ablation_hooks.append(handle)
    print(f"Registered {len(ablation_hooks)} subspace-ablation hooks.")

    captured = {}

    def make_capture_hook(layer_idx):
        def _hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach()
            return output
        return _hook

    capture_hooks = [
        model.model.layers[i].register_forward_hook(make_capture_hook(i), prepend=False)
        for i in range(n_layers)
    ]

    def get_pooled_activations_via_hooks(prompts, layer_subset=None):
        vecs = []
        layer_range = layer_subset if layer_subset is not None else range(n_layers)
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                model(**enc)
            mask = enc["attention_mask"][0].bool()
            layer_vecs = []
            for layer_idx in layer_range:
                h = captured[layer_idx][0]
                pooled = h[mask].float().mean(dim=0)
                layer_vecs.append(pooled.cpu().numpy())
            vecs.append(np.concatenate(layer_vecs))
        return np.stack(vecs)

    neg_concept = list(neighbors.values())[0].get("name", "an unrelated topic") if neighbors else "an unrelated topic"
    pos_prompts = make_prompts(args.concept_name, 40)
    neg_prompts = make_prompts(neg_concept, 40)
    all_prompts = pos_prompts + neg_prompts
    y = np.array([1] * len(pos_prompts) + [0] * len(neg_prompts))

    print("\n=== Full-stack CLMI (original protocol: all 32 layers, up to 512 PCA components) ===")
    X_full = get_pooled_activations_via_hooks(all_prompts)
    full_auroc, full_std = compute_cv_auroc(X_full, y, n_components=512)
    print(f"  CLMI(full) for '{args.concept_name}' vs '{neg_concept}': {full_auroc:.4f} +/- {full_std:.4f}")

    print("\n=== Restricted-CLMI (ablated layers only, capacity matched to ablation rank) ===")
    ablated_layers = sorted(basis_per_layer.keys())
    if not ablated_layers:
        print("  No layers were ablated (forget subspace fully contained in protected subspace everywhere) -- skipping.")
        restricted_auroc, restricted_std, matched_components = None, None, None
    else:
        X_restricted = get_pooled_activations_via_hooks(all_prompts, layer_subset=ablated_layers)
        matched_components = max(args.rank * 2, 4)  # modest capacity, not 512
        restricted_auroc, restricted_std = compute_cv_auroc(X_restricted, y, n_components=matched_components)
        print(f"  CLMI(restricted, {len(ablated_layers)} layers, {matched_components} PCA comps) for '{args.concept_name}' vs '{neg_concept}': {restricted_auroc:.4f} +/- {restricted_std:.4f}")

    for h in ablation_hooks + capture_hooks:
        h.remove()

    if args.save_checkpoint:
        print(f"\nBaking subspace ablation into permanent weights (o_proj + down_proj, {len(basis_per_layer)} layers)...")
        apply_permanent_subspace_orthogonalization(model, basis_per_layer)
        model.save_pretrained(args.save_checkpoint)
        tokenizer.save_pretrained(args.save_checkpoint)
        print(f"Saved OGDA checkpoint to {args.save_checkpoint}")

    summary = {
        "cui": args.cui, "concept": args.concept_name, "rank": args.rank,
        "layers_ablated": ablated_layers,
        "clmi_full": {"mean": full_auroc, "std": full_std},
        "clmi_restricted": {"mean": restricted_auroc, "std": restricted_std, "n_pca_components": matched_components},
    }
    layers_tag = args.layers.replace("-", "to")
    seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    out_path = f"data/gate2_results/OGDA_subspace_rank{args.rank}_layers{layers_tag}_{args.cui}{seed_tag}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
