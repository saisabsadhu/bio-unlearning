"""
CLMI Pre-screening — verify which concepts BioMistral has
parametrically encoded before running unlearning experiments.

For each concept:
1. Generate 20 positive prompts (mention concept in forget context)
2. Generate 20 negative prompts (mention 1-hop UMLS neighbor)
3. Extract residual stream activations at concept token positions
4. Train linear probe, compute AUROC
5. Flag concept as CONFIRMED (CLMI >= 0.70) or WEAK (CLMI < 0.70)

Only CONFIRMED concepts go into the final benchmark.
"""

import torch
import json
import os
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    OPENROUTER_KEY, OPENROUTER_BASE,
    OPENROUTER_MODEL, SITE_URL, SITE_NAME
)

llm_client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_KEY)

MODEL_ID   = "BioMistral/BioMistral-7B"
OUT_DIR    = "data/clmi_prescreen"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────

print("Loading BioMistral-7B for CLMI pre-screening...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True,   # need this for activation extraction
)
model.eval()
print("Model loaded.\n")

# ── Probe concepts ────────────────────────────────────────────────────

# Sample from both scenarios — 10 concepts total for pre-screening
PROBE_CONCEPTS = [
    # (concept_name,         cui,        neighbor_name,              scenario)
    ("aspirin cardiovascular prevention", "C0004057", "platelet aggregation",    "RGU"),
    ("hormone replacement therapy",       "C0282402", "estrogen therapy",         "RGU"),
    ("rosiglitazone diabetes",            "C0289313", "thiazolidinedione class",  "IFE"),
    ("rofecoxib Vioxx withdrawal",        "C0876768", "COX-2 inhibitor",          "IFE"),
    ("beta-carotene lung cancer risk",    "C0053275", "antioxidant supplement",   "RGU"),
    ("metformin renal contraindication",  "C0025598", "biguanide diabetes drug",  "IFE"),
    ("warfarin vitamin K interaction",    "C0043031", "anticoagulant therapy",    "IFE"),
    ("digoxin hypokalemia toxicity",      "C0012265", "cardiac glycoside",        "IFE"),
    ("tight glycemic control ICU harm",   "C0017725", "insulin infusion protocol","RGU"),
    ("fluoroquinolone tendon rupture",    "C0016403", "antibiotic adverse effect","IFE"),
]

# ── Prompt generation ─────────────────────────────────────────────────

def generate_probe_prompts(concept, neighbor, n=20):
    """Generate positive and negative probe prompts via LLM."""

    pos_system = """Generate clinical prompts that mention a specific medical concept.
Output a JSON array of exactly {n} short clinical sentences or questions (1-2 sentences each)
that naturally mention the concept. Vary the phrasing significantly.
Output ONLY the JSON array, no explanation."""

    neg_system = """Generate clinical prompts about a related but DIFFERENT medical concept.
Output a JSON array of exactly {n} short clinical sentences or questions (1-2 sentences each)
that naturally mention the neighbor concept WITHOUT mentioning the target concept.
Output ONLY the JSON array, no explanation."""

    import time, re

    def call(system, user):
        for _ in range(3):
            try:
                r = llm_client.chat.completions.create(
                    extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_NAME},
                    model=OPENROUTER_MODEL,
                    messages=[{"role":"system","content":system.format(n=n)},
                              {"role":"user","content":user}],
                    temperature=0.8, max_tokens=1200,
                )
                text = r.choices[0].message.content.strip()
                text = re.sub(r'^```[a-z]*\n?','',text)
                text = re.sub(r'\n?```$','',text)
                return json.loads(text)
            except Exception as e:
                print(f"    [WARN] {e}")
                time.sleep(2)
        return []

    pos_prompts = call(
        pos_system,
        f"Concept to mention: {concept}\nGenerate {n} varied clinical prompts:"
    )
    neg_prompts = call(
        neg_system,
        f"Target concept to AVOID: {concept}\n"
        f"Neighbor concept to mention instead: {neighbor}\n"
        f"Generate {n} varied clinical prompts about {neighbor}:"
    )
    return pos_prompts[:n], neg_prompts[:n]

# ── Activation extraction ─────────────────────────────────────────────

def get_activations(prompts, concept_term):
    """
    For each prompt, extract residual stream activations at the token
    positions corresponding to concept_term. Average across concept tokens
    and across all layers.
    Returns array of shape (n_prompts, n_layers * hidden_size)
    """
    all_activations = []

    concept_tokens = tokenizer.encode(
        " " + concept_term.split()[0],   # first content word
        add_special_tokens=False
    )

    for prompt in prompts:
        if not isinstance(prompt, str):
            continue

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden)
        hidden_states = outputs.hidden_states  # all layers incl. embedding

        # Find concept token positions in input
        input_ids = inputs["input_ids"][0].tolist()
        concept_positions = []
        for pos, tok_id in enumerate(input_ids):
            if tok_id in concept_tokens:
                concept_positions.append(pos)

        # Fall back to middle token if concept not found
        if not concept_positions:
            concept_positions = [len(input_ids) // 2]

        # Extract and average activations across concept positions and layers
        layer_vecs = []
        for layer_hidden in hidden_states[1:]:   # skip embedding layer
            # layer_hidden: (1, seq_len, hidden_size)
            pos_vecs = layer_hidden[0, concept_positions, :]  # (n_pos, hidden)
            avg_vec  = pos_vecs.mean(dim=0)                   # (hidden,)
            layer_vecs.append(avg_vec.cpu().float().numpy())

        # Concatenate all layers → (n_layers * hidden_size,)
        full_vec = np.concatenate(layer_vecs, axis=0)
        all_activations.append(full_vec)

    return np.array(all_activations) if all_activations else None

# ── CLMI computation ──────────────────────────────────────────────────

def compute_clmi(concept, cui, neighbor, scenario):
    print(f"\n  {'─'*55}")
    print(f"  Concept  : {concept}")
    print(f"  CUI      : {cui}  |  Scenario: {scenario}")
    print(f"  Neighbor : {neighbor}")
    print(f"  {'─'*55}")

    # Step 1 — generate prompts
    print(f"  Generating probe prompts (20 pos + 20 neg)...")
    pos_prompts, neg_prompts = generate_probe_prompts(concept, neighbor, n=20)
    print(f"  Got {len(pos_prompts)} positive, {len(neg_prompts)} negative")

    if len(pos_prompts) < 10 or len(neg_prompts) < 10:
        print(f"  [SKIP] Not enough prompts")
        return None

    # Step 2 — extract activations
    print(f"  Extracting activations...")
    pos_acts = get_activations(pos_prompts, concept)
    neg_acts = get_activations(neg_prompts, neighbor)

    if pos_acts is None or neg_acts is None:
        print(f"  [SKIP] Activation extraction failed")
        return None

    print(f"  Activation shape: {pos_acts.shape}")

    # Step 3 — PCA reduction
    X = np.vstack([pos_acts, neg_acts])
    y = np.array([1]*len(pos_acts) + [0]*len(neg_acts))

    n_components = min(64, X.shape[0]-1, X.shape[1])
    pca = PCA(n_components=n_components)
    X_r = pca.fit_transform(X)

    # Step 4 — train probe and compute AUROC
    # Use leave-one-out style: train on 80%, test on 20%
    n        = len(X_r)
    n_train  = int(n * 0.8)
    idx      = np.random.permutation(n)

    X_train, y_train = X_r[idx[:n_train]], y[idx[:n_train]]
    X_test,  y_test  = X_r[idx[n_train:]], y[idx[n_train:]]

    if len(np.unique(y_test)) < 2:
        # Ensure both classes in test set
        X_test  = X_r
        y_test  = y

    probe = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    probe.fit(X_train, y_train)

    proba = probe.predict_proba(X_test)[:, 1]
    clmi  = roc_auc_score(y_test, proba)

    status = "CONFIRMED" if clmi >= 0.70 else "WEAK"
    print(f"\n  CLMI = {clmi:.4f}  →  {status}")

    result = {
        "cui":          cui,
        "concept":      concept,
        "neighbor":     neighbor,
        "scenario":     scenario,
        "clmi_pretrain":round(float(clmi), 4),
        "status":       status,
        "n_pos":        len(pos_prompts),
        "n_neg":        len(neg_prompts),
    }

    # Save
    path = os.path.join(OUT_DIR, f"{cui}_clmi.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    return result

# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("="*60)
    print("BioUnlearn — CLMI Pre-screening on BioMistral-7B")
    print("Verifying parametric encoding before unlearning")
    print("="*60)

    results  = []
    confirmed = []
    weak      = []

    for concept, cui, neighbor, scenario in PROBE_CONCEPTS:
        result = compute_clmi(concept, cui, neighbor, scenario)
        if result:
            results.append(result)
            if result["status"] == "CONFIRMED":
                confirmed.append(result)
            else:
                weak.append(result)
        time.sleep(1)

    # Summary
    print("\n" + "="*65)
    print("CLMI PRE-SCREENING RESULTS")
    print("="*65)
    print(f"{'Concept':<40} {'CLMI':>6}  {'Status'}")
    print("─"*65)
    for r in sorted(results, key=lambda x: x["clmi_pretrain"], reverse=True):
        marker = "✓" if r["status"] == "CONFIRMED" else "✗"
        print(f"{marker} {r['concept'][:38]:<38} "
              f"{r['clmi_pretrain']:>6.4f}  {r['status']}")

    print("─"*65)
    print(f"\nCONFIRMED (CLMI ≥ 0.70): {len(confirmed)}/{len(results)} concepts")
    print(f"WEAK     (CLMI < 0.70): {len(weak)}/{len(results)} concepts")

    # Save confirmed list for Stage B to use
    confirmed_path = os.path.join(OUT_DIR, "confirmed_concepts.json")
    with open(confirmed_path, "w") as f:
        json.dump(confirmed, f, indent=2)
    print(f"\nConfirmed concepts → {confirmed_path}")
    print("These are the only concepts that go into the final benchmark.")
