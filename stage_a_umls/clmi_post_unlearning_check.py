"""
Gate 0, part 2 (MASTER_RESEARCH_PLAN.md Section 5): does CLMI actually drop
after genuine unlearning, using the mean_pool scheme adopted from Gate 0?

Compares CLMI (5-fold CV AUROC, mean_pool activations) on the base
BioMistral-7B model vs. the RGU GradAscent pilot checkpoint
(saves/unlearn/RGU_ga_pilot), for the same 10 pilot concepts. RGU-scenario
concepts were targeted by training; IFE-scenario concepts were not, and
serve as a same-run specificity control (their CLMI should move less, if
the effect measured on RGU concepts is real and not just generic drift
from any training at all).
"""

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
from clmi_gate0_validity_check import PROBE_CONCEPTS, N_PER_CLASS, make_prompts, get_hidden_states_batched, pool_mean

OUT_DIR = "data/clmi_gate0"
os.makedirs(OUT_DIR, exist_ok=True)


def compute_cv_auroc(X, y, n_components=512, seed=42):
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
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


def run(model_path, label):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", output_hidden_states=True
    )
    model.eval()

    results = {}
    for concept, cui, neighbor, scenario in PROBE_CONCEPTS:
        pos_prompts = make_prompts(concept, N_PER_CLASS)
        neg_prompts = make_prompts(neighbor, N_PER_CLASS)
        all_prompts = pos_prompts + neg_prompts
        y = np.array([1] * len(pos_prompts) + [0] * len(neg_prompts))

        hidden_list, ids_list, attn_list = get_hidden_states_batched(model, tokenizer, all_prompts)
        X = np.stack([
            pool_mean(hidden_list[i], ids_list[i], attn_list[i], tokenizer, None)
            for i in range(len(all_prompts))
        ])
        mean_auroc, std_auroc = compute_cv_auroc(X, y)
        results[cui] = {
            "concept": concept, "scenario": scenario,
            "mean_pool_auroc_mean": round(mean_auroc, 4),
            "mean_pool_auroc_std": round(std_auroc, 4),
        }
        print(f"  [{scenario}] {concept:40s} CLMI(mean_pool) = {mean_auroc:.4f} +/- {std_auroc:.4f}")

    del model
    torch.cuda.empty_cache()

    out_path = os.path.join(OUT_DIR, f"clmi_{label}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    label = sys.argv[1]
    model_path = sys.argv[2]
    run(model_path, label)
