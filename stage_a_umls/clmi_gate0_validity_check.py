"""
Gate 0 (MASTER_RESEARCH_PLAN.md Section 5): CLMI validity pre-check.

The existing clmi_prescreen_v2.py showed clmi_mean=1.0, std=0.0 for
essentially every concept -- suspiciously perfect. This script diagnoses
why, and tests whether a fixed pooling scheme resolves it.

DIAGNOSIS (found by reading clmi_prescreen_v2.py's get_activations_batched):
Positive-class prompts are LLM-instructed to mention the concept term, and
activations are pooled at the token position where the concept's first word
literally appears. Negative-class prompts are LLM-instructed to AVOID the
concept term -- so that token search almost always fails, and the code
falls back to pooling the LAST token of an unrelated sentence instead. The
probe is very likely separating "activation at the token identical to the
concept word" from "activation at the last token of a different sentence"
-- a lexical/positional shortcut, not a parametric-knowledge test.

This script re-runs the same 10 pilot concepts with THREE pooling schemes
on template-based prompts (no LLM oracle needed -- OPENROUTER_KEY isn't
configured in this checkout, and a validity check benefits from controlled,
inspectable prompts anyway):
  (a) original  -- reproduce the exact original method for comparison
  (b) last_token -- last non-pad token, same rule for pos AND neg classes
  (c) mean_pool  -- mean over all non-pad tokens, same rule for both classes

For each scheme, reports:
  - real AUROC (5-fold CV, matching original protocol)
  - label-swap AUROC (labels permuted; expect ~0.5 for a valid probe)

Pass criterion (MASTER_RESEARCH_PLAN.md Gate 0): a pooling scheme is
considered valid if its label-swap AUROC is close to chance (~0.5) while
its real AUROC still meaningfully exceeds chance -- unlike the original
scheme, which is expected to show high AUROC even under label-swap for
some diagnostic variants, or to owe its real AUROC entirely to the
pos/neg extraction asymmetry rather than genuine content.
"""

import json
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = "data/clmi_gate0"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_ID = "BioMistral/BioMistral-7B"

PROBE_CONCEPTS = [
    ("aspirin cardiovascular prevention", "C0004057", "platelet aggregation inhibitor", "RGU"),
    ("hormone replacement therapy", "C0282402", "estrogen receptor modulator", "RGU"),
    ("rosiglitazone diabetes", "C0289313", "thiazolidinedione class drug", "IFE"),
    ("rofecoxib Vioxx withdrawal", "C0876768", "COX-2 selective inhibitor", "IFE"),
    ("beta-carotene lung cancer risk", "C0053275", "antioxidant vitamin supplement", "RGU"),
    ("metformin renal contraindication", "C0025598", "biguanide blood glucose lowering", "IFE"),
    ("warfarin vitamin K interaction", "C0043031", "anticoagulant drug monitoring", "IFE"),
    ("digoxin hypokalemia toxicity", "C0012265", "cardiac glycoside mechanism", "IFE"),
    ("tight glycemic control ICU harm", "C0017725", "insulin infusion intensive care", "RGU"),
    ("fluoroquinolone tendon rupture", "C0016403", "antibiotic musculoskeletal effect", "IFE"),
]

N_PER_CLASS = 40

TEMPLATES = [
    "A patient asks their physician about {term}.",
    "The clinical guideline discusses {term} in detail.",
    "During rounds, the resident mentioned {term} as relevant to the case.",
    "A 55-year-old patient with multiple comorbidities was evaluated for {term}.",
    "The nursing staff documented {term} in the patient's chart.",
    "Recent literature has examined {term} across several patient populations.",
    "The attending physician explained {term} to the medical student.",
    "A pharmacist reviewed the patient's regimen with attention to {term}.",
    "The case report described a patient presenting with {term}.",
    "Clinical decision support flagged {term} for further review.",
    "The elderly patient's family asked about {term} during the consultation.",
    "A quality improvement committee reviewed protocols related to {term}.",
    "The emergency department physician considered {term} in the differential.",
    "Grand rounds this week focused on {term} and its clinical implications.",
    "The patient's electronic health record noted a history involving {term}.",
    "A second-year resident presented a case involving {term}.",
    "The specialist consult addressed questions about {term}.",
    "Nursing education materials were updated to reflect {term}.",
    "The discharge summary referenced {term} as part of the treatment plan.",
    "A multidisciplinary team discussed {term} at the weekly conference.",
]


def make_prompts(term, n):
    prompts = []
    i = 0
    while len(prompts) < n:
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        suffix = "" if i < len(TEMPLATES) else f" (variant {i // len(TEMPLATES) + 1})"
        prompts.append(tmpl.format(term=term) + suffix)
        i += 1
    return prompts


def get_hidden_states_batched(model, tokenizer, prompts, batch_size=16):
    """Returns list of (hidden_states_tuple, input_ids, attention_mask) per batch,
    concatenated per-prompt as (n_layers, seq_len, hidden) is too big to keep all;
    instead directly returns three pooled representations per prompt:
    concept-token-search pooling (needs concept_term), last-token pooling,
    mean pooling -- computed by caller since concept_term differs per class.
    Here we just return raw hidden_states + input_ids + attention_mask per prompt.
    """
    all_hidden = []  # list of (n_layers, hidden) per pooling done by caller
    all_input_ids = []
    all_attn = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        encoded = tokenizer(
            batch_prompts, return_tensors="pt", truncation=True, max_length=128, padding=True
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]  # (n_layers) x (batch, seq, hidden)
        bsz = encoded["input_ids"].shape[0]
        for b in range(bsz):
            layer_stack = torch.stack([h[b] for h in hidden_states], dim=0)  # (n_layers, seq, hidden)
            all_hidden.append(layer_stack.cpu().float())
            all_input_ids.append(encoded["input_ids"][b].cpu())
            all_attn.append(encoded["attention_mask"][b].cpu())
        del outputs, encoded
        torch.cuda.empty_cache()
    return all_hidden, all_input_ids, all_attn


def pool_original(layer_stack, input_ids, attn_mask, tokenizer, concept_term):
    """Reproduce original clmi_prescreen_v2 logic: search for concept's first
    word token; fall back to last non-pad token if not found."""
    concept_first_word = concept_term.split()[0]
    concept_token_ids = set(tokenizer.encode(" " + concept_first_word, add_special_tokens=False))
    ids_list = input_ids.tolist()
    mask_list = attn_mask.tolist()
    positions = [p for p, tid in enumerate(ids_list) if tid in concept_token_ids and mask_list[p] == 1]
    if not positions:
        positions = [mask_list.index(0) - 1 if 0 in mask_list else len(mask_list) - 1]
    pooled = layer_stack[:, positions, :].mean(dim=1)  # (n_layers, hidden)
    return pooled.flatten().numpy()


def pool_last_token(layer_stack, input_ids, attn_mask, tokenizer, concept_term):
    mask_list = attn_mask.tolist()
    last_pos = mask_list.index(0) - 1 if 0 in mask_list else len(mask_list) - 1
    pooled = layer_stack[:, last_pos, :]  # (n_layers, hidden)
    return pooled.flatten().numpy()


def pool_mean(layer_stack, input_ids, attn_mask, tokenizer, concept_term):
    mask = attn_mask.bool()
    pooled = layer_stack[:, mask, :].mean(dim=1)  # (n_layers, hidden)
    return pooled.flatten().numpy()


POOLERS = {
    "original": pool_original,
    "last_token": pool_last_token,
    "mean_pool": pool_mean,
}


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


def main():
    print("Loading BioMistral-7B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", output_hidden_states=True
    )
    model.eval()
    print("Model loaded.\n")

    all_results = []

    for concept, cui, neighbor, scenario in PROBE_CONCEPTS:
        print(f"--- {concept} ({cui}) vs {neighbor} [{scenario}] ---")
        pos_prompts = make_prompts(concept, N_PER_CLASS)
        neg_prompts = make_prompts(neighbor, N_PER_CLASS)
        all_prompts = pos_prompts + neg_prompts
        y = np.array([1] * len(pos_prompts) + [0] * len(neg_prompts))

        hidden_list, ids_list, attn_list = get_hidden_states_batched(model, tokenizer, all_prompts)

        concept_result = {"cui": cui, "concept": concept, "neighbor": neighbor, "scenario": scenario}
        for pool_name, pool_fn in POOLERS.items():
            X = np.stack([
                pool_fn(
                    hidden_list[i], ids_list[i], attn_list[i], tokenizer,
                    concept if y[i] == 1 else neighbor,
                )
                for i in range(len(all_prompts))
            ])
            real_mean, real_std = compute_cv_auroc(X, y)
            y_shuffled = np.random.RandomState(0).permutation(y)
            swap_mean, swap_std = compute_cv_auroc(X, y_shuffled)
            concept_result[pool_name] = {
                "real_auroc_mean": round(real_mean, 4),
                "real_auroc_std": round(real_std, 4),
                "label_swap_auroc_mean": round(swap_mean, 4),
                "label_swap_auroc_std": round(swap_std, 4),
            }
            print(
                f"  [{pool_name:10s}] real={real_mean:.4f}±{real_std:.4f}  "
                f"label_swap={swap_mean:.4f}±{swap_std:.4f}"
            )

        all_results.append(concept_result)
        with open(os.path.join(OUT_DIR, f"{cui}_gate0.json"), "w") as f:
            json.dump(concept_result, f, indent=2)
        print()

    # Aggregate summary across concepts
    summary = {}
    for pool_name in POOLERS:
        real_vals = [r[pool_name]["real_auroc_mean"] for r in all_results]
        swap_vals = [r[pool_name]["label_swap_auroc_mean"] for r in all_results]
        summary[pool_name] = {
            "mean_real_auroc": round(float(np.mean(real_vals)), 4),
            "mean_label_swap_auroc": round(float(np.mean(swap_vals)), 4),
        }

    print("=" * 70)
    print("GATE 0 SUMMARY (averaged over 10 pilot concepts)")
    print("=" * 70)
    for pool_name, s in summary.items():
        print(f"{pool_name:12s}  real_auroc={s['mean_real_auroc']:.4f}  label_swap_auroc={s['mean_label_swap_auroc']:.4f}")

    with open(os.path.join(OUT_DIR, "gate0_summary.json"), "w") as f:
        json.dump({"per_concept": all_results, "summary": summary}, f, indent=2)
    print(f"\nWritten to {OUT_DIR}/gate0_summary.json")


if __name__ == "__main__":
    main()
