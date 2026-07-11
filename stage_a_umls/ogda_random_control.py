"""
OGDA random-subspace control — the load-bearing experiment for the paper's central
claim. Ablates a RANDOM rank-r subspace at the same layers, via the exact same
permanent weight-orthogonalization mechanism (W' = (I - B^T B) W on both
self_attn.o_proj and mlp.down_proj) used for the real, ontology-anchored OGDA edit
in ogda_subspace_ablation.py. Nothing about the mechanism differs -- only the
subspace being removed is now unrelated to any concept, ontology, or forget target.

Why this matters: the real OGDA behavioral result (data/gate2_results/
OGDA_behavioral_C0004057_summary.json) showed FA rising and DEF collapsing on
BOTH the targeted (RGU) and untargeted (IFE) scenarios -- a pattern equally
consistent with "the ontology-anchored aspirin direction specifically matters"
and "editing this many layers/matrices causes generic drift regardless of what
subspace is removed." This script produces the comparison needed to tell those
apart: if a random subspace of the same rank/layer-count produces similar
FA/DEF movement, the drift is generic, not ontology-specific. If it produces
much less movement, that's evidence the real OGDA edit is doing something
non-generic.

Usage:
    python stage_a_umls/ogda_random_control.py --layers 4-28 --rank 3 --seed 0 \
        --save_checkpoint saves/unlearn/OGDA_random_control_checkpoint
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ogda_ablation import MODEL_ID, parse_layers
from ogda_subspace_ablation import apply_permanent_subspace_orthogonalization


def random_orthonormal_basis(rank, hidden, seed, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    M = torch.randn(hidden, rank, generator=g)
    Q, _ = torch.linalg.qr(M)  # (hidden, rank), orthonormal columns
    return Q.T.to(device=device, dtype=dtype)  # (rank, hidden), orthonormal rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="4-28")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_checkpoint", required=True)
    args = parser.parse_args()

    print(f"Random-subspace control: layers={args.layers}, rank={args.rank}, seed={args.seed}")
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    n_layers = model.config.num_hidden_layers
    layers = parse_layers(args.layers, n_layers)
    hidden = model.config.hidden_size

    basis_per_layer = {}
    for i, layer_idx in enumerate(layers):
        basis_per_layer[layer_idx] = random_orthonormal_basis(
            args.rank, hidden, seed=args.seed * 1000 + layer_idx, device=model.device, dtype=torch.float32
        )
    print(f"Built {len(basis_per_layer)} random rank-{args.rank} bases (layers {layers[0]}-{layers[-1]}).")

    print("Baking random-subspace ablation into permanent weights (o_proj + down_proj)...")
    apply_permanent_subspace_orthogonalization(model, basis_per_layer)
    model.save_pretrained(args.save_checkpoint)
    tokenizer.save_pretrained(args.save_checkpoint)
    print(f"Saved random-control checkpoint to {args.save_checkpoint}")


if __name__ == "__main__":
    main()
