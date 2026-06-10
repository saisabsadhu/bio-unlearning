"""
CLMI Pre-screening v2
- 50 positive + 50 negative prompts per concept (was 20+20)
- Full GPU utilization via batch inference
- PCA to 512 components (matches proposal spec)
- 5-fold cross-validated AUROC for reliability
"""

import torch
import json, os, sys, time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    OPENROUTER_KEY, OPENROUTER_BASE,
    OPENROUTER_MODEL, SITE_URL, SITE_NAME
)

llm_client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_KEY)
OUT_DIR    = "data/clmi_prescreen_v2"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_ID   = "BioMistral/BioMistral-7B"
BATCH_SIZE = 16     # process 16 prompts at once — uses ~60-80GB across GPUs

# ── Load model ────────────────────────────────────────────────────────

print("Loading BioMistral-7B (full precision for better activations)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True,
)
model.eval()

# Report actual GPU usage
for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"GPU {i}: {alloc:.1f}GB allocated / {total:.1f}GB total")

print()

# ── Concepts ──────────────────────────────────────────────────────────

PROBE_CONCEPTS = [
    ("aspirin cardiovascular prevention", "C0004057", "platelet aggregation inhibitor",     "RGU"),
    ("hormone replacement therapy",       "C0282402", "estrogen receptor modulator",        "RGU"),
    ("rosiglitazone diabetes",            "C0289313", "thiazolidinedione class drug",       "IFE"),
    ("rofecoxib Vioxx withdrawal",        "C0876768", "COX-2 selective inhibitor",          "IFE"),
    ("beta-carotene lung cancer risk",    "C0053275", "antioxidant vitamin supplement",     "RGU"),
    ("metformin renal contraindication",  "C0025598", "biguanide blood glucose lowering",   "IFE"),
    ("warfarin vitamin K interaction",    "C0043031", "anticoagulant drug monitoring",      "IFE"),
    ("digoxin hypokalemia toxicity",      "C0012265", "cardiac glycoside mechanism",        "IFE"),
    ("tight glycemic control ICU harm",   "C0017725", "insulin infusion intensive care",    "RGU"),
    ("fluoroquinolone tendon rupture",    "C0016403", "antibiotic musculoskeletal effect",  "IFE"),
]

# ── Prompt generation ─────────────────────────────────────────────────

def generate_prompts(concept, neighbor, n=50):
    import re

    system = """You are generating probe prompts for a clinical NLP evaluation.
Output ONLY a valid JSON array of strings. No explanation, no markdown fences.
Each string is a short clinical sentence or question (1-2 sentences).
Vary phrasing, clinical context, and patient demographics significantly."""

    def call(user):
        for attempt in range(3):
            try:
                r = llm_client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": SITE_URL,
                        "X-Title": SITE_NAME
                    },
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=0.9,
                    max_tokens=3000,
                )
                text = r.choices[0].message.content.strip()
                text = re.sub(r'^```[a-z]*\n?', '', text)
                text = re.sub(r'\n?```$', '', text).strip()
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [p for p in parsed if isinstance(p, str)][:n]
            except Exception as e:
                print(f"      [WARN] attempt {attempt+1}: {e}")
                time.sleep(2)
        return []

    pos = call(
        f"Generate {n} varied clinical sentences that naturally mention "
        f"'{concept}' in different clinical contexts, patient types, "
        f"and question formats."
    )
    time.sleep(0.5)
    neg = call(
        f"Generate {n} varied clinical sentences that naturally mention "
        f"'{neighbor}' WITHOUT mentioning '{concept}'. "
        f"Use different clinical contexts and patient types."
    )
    return pos, neg

# ── Batch activation extraction ───────────────────────────────────────

def get_activations_batched(prompts, concept_term, batch_size=BATCH_SIZE):
    """
    Batch inference for GPU efficiency.
    Extracts residual stream activations at concept token positions.
    Returns (n_prompts, n_layers * hidden_size) array.
    """
    all_activations = []

    # Tokenize concept term to find its tokens in sequences
    concept_first_word = concept_term.split()[0]
    concept_token_ids  = set(
        tokenizer.encode(" " + concept_first_word, add_special_tokens=False)
    )

    # Process in batches
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start: batch_start + batch_size]

        # Tokenize batch with padding
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors (batch, seq, hidden)
        hidden_states = outputs.hidden_states[1:]  # skip embedding layer

        batch_size_actual = encoded["input_ids"].shape[0]

        for b in range(batch_size_actual):
            input_ids = encoded["input_ids"][b].tolist()

            # Find concept token positions
            concept_pos = [
                pos for pos, tid in enumerate(input_ids)
                if tid in concept_token_ids
                and encoded["attention_mask"][b][pos].item() == 1
            ]
            if not concept_pos:
                # Fall back to last non-padding token
                mask    = encoded["attention_mask"][b].tolist()
                concept_pos = [mask.index(0) - 1 if 0 in mask
                               else len(mask) - 1]

            # Extract and concatenate layer activations
            layer_vecs = []
            for layer_hidden in hidden_states:
                pos_vecs = layer_hidden[b, concept_pos, :]   # (n_pos, hidden)
                avg_vec  = pos_vecs.mean(dim=0)              # (hidden,)
                layer_vecs.append(avg_vec.cpu().float().numpy())

            full_vec = np.concatenate(layer_vecs, axis=0)
            all_activations.append(full_vec)

        # Free GPU memory between batches
        del outputs, encoded
        torch.cuda.empty_cache()

        print(f"      Batch {batch_start//batch_size + 1}/"
              f"{(len(prompts)-1)//batch_size + 1} done", end="\r")

    print()
    return np.array(all_activations)

# ── CLMI with 5-fold CV ───────────────────────────────────────────────

def compute_clmi_cv(X, y, n_components=512):
    """
    5-fold stratified cross-validated AUROC.
    More reliable than single train/test split on small N.
    """
    # PCA reduction
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca    = PCA(n_components=n_comp)
    X_r    = pca.fit_transform(X)

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []

    for train_idx, test_idx in skf.split(X_r, y):
        X_tr, X_te = X_r[train_idx], X_r[test_idx]
        y_tr, y_te = y[train_idx],   y[test_idx]

        if len(np.unique(y_te)) < 2:
            continue

        probe = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        probe.fit(X_tr, y_tr)
        proba = probe.predict_proba(X_te)[:, 1]
        aurocs.append(roc_auc_score(y_te, proba))

    return float(np.mean(aurocs)), float(np.std(aurocs))

# ── Main per-concept pipeline ─────────────────────────────────────────

def run_concept(concept, cui, neighbor, scenario, n_prompts=50):
    print(f"\n{'─'*60}")
    print(f"  Concept  : {concept}")
    print(f"  CUI      : {cui}  |  Scenario: {scenario}")
    print(f"  Neighbor : {neighbor}")
    print(f"{'─'*60}")

    # Step 1 — generate prompts
    print(f"  Generating {n_prompts} pos + {n_prompts} neg prompts...")
    pos_prompts, neg_prompts = generate_prompts(concept, neighbor, n=n_prompts)
    print(f"  Got: {len(pos_prompts)} positive, {len(neg_prompts)} negative")

    if len(pos_prompts) < 20 or len(neg_prompts) < 20:
        print(f"  [SKIP] Insufficient prompts")
        return None

    # Step 2 — batch activation extraction
    print(f"  Extracting activations (batch_size={BATCH_SIZE})...")
    all_prompts = pos_prompts + neg_prompts
    all_acts    = get_activations_batched(all_prompts, concept)

    pos_acts = all_acts[:len(pos_prompts)]
    neg_acts = all_acts[len(pos_prompts):]
    print(f"  Activation matrix: {all_acts.shape}")

    X = np.vstack([pos_acts, neg_acts])
    y = np.array([1]*len(pos_acts) + [0]*len(neg_acts))

    # Step 3 — 5-fold CV AUROC
    print(f"  Running 5-fold cross-validated probe...")
    clmi_mean, clmi_std = compute_clmi_cv(X, y, n_components=512)

    status = "CONFIRMED" if clmi_mean >= 0.70 else "WEAK"
    print(f"\n  CLMI = {clmi_mean:.4f} ± {clmi_std:.4f}  →  {status}")

    result = {
        "cui":           cui,
        "concept":       concept,
        "neighbor":      neighbor,
        "scenario":      scenario,
        "clmi_mean":     round(clmi_mean, 4),
        "clmi_std":      round(clmi_std,  4),
        "status":        status,
        "n_pos":         len(pos_prompts),
        "n_neg":         len(neg_prompts),
        "act_shape":     list(all_acts.shape),
    }

    path = os.path.join(OUT_DIR, f"{cui}_clmi_v2.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    return result

# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("BioUnlearn — CLMI Pre-screening v2")
    print("50+50 prompts, batch inference, 5-fold CV AUROC")
    print("="*60 + "\n")

    results   = []
    confirmed = []
    weak      = []

    for concept, cui, neighbor, scenario in PROBE_CONCEPTS:
        result = run_concept(concept, cui, neighbor, scenario, n_prompts=50)
        if result:
            results.append(result)
            if result["status"] == "CONFIRMED":
                confirmed.append(result)
            else:
                weak.append(result)
        time.sleep(1)

    # Summary table
    print("\n" + "="*65)
    print("CLMI PRE-SCREENING v2 RESULTS")
    print("="*65)
    print(f"{'Concept':<38} {'CLMI':>8}  {'±':>6}  Status")
    print("─"*65)
    for r in sorted(results, key=lambda x: x["clmi_mean"], reverse=True):
        marker = "✓" if r["status"] == "CONFIRMED" else "✗"
        print(f"{marker} {r['concept'][:36]:<36} "
              f"{r['clmi_mean']:>8.4f}  "
              f"{r['clmi_std']:>6.4f}  "
              f"{r['status']}")

    print("─"*65)
    print(f"\nCONFIRMED (CLMI ≥ 0.70): {len(confirmed)}/{len(results)}")
    print(f"WEAK     (CLMI < 0.70): {len(weak)}/{len(results)}")

    confirmed_path = os.path.join(OUT_DIR, "confirmed_concepts_v2.json")
    with open(confirmed_path, "w") as f:
        json.dump(confirmed, f, indent=2)
    print(f"\nConfirmed → {confirmed_path}")
