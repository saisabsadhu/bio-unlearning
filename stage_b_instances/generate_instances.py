"""
Stage B — Instance Generation via LLM Oracle (OpenRouter/Gemini)
Reads merged concept graphs from Stage A.
Generates (Q, forget_answer, retain_answer) triples per scenario.
Saves raw instances to data/raw/ before any filtering.
"""

import json
import os
import sys
import time
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    OPENROUTER_KEY, OPENROUTER_BASE,
    OPENROUTER_MODEL, SITE_URL, SITE_NAME
)

client = OpenAI(
    base_url=OPENROUTER_BASE,
    api_key=OPENROUTER_KEY,
)

MERGED_DIR = "data/concept_graphs/merged"
RAW_DIR    = "data/raw"

# ── Prompts ───────────────────────────────────────────────────────────

RGU_SYSTEM = """You are a clinical NLP dataset curator building a benchmark 
for LLM unlearning research. Your task is to generate high-quality instances 
of RETRACTED GUIDELINE UNLEARNING (RGU).

Each instance must be a JSON object with exactly these fields:
{
  "prompt": "A realistic clinical question a physician or medical student would ask",
  "forget_answer": "The OLD answer — what a model trained before the reversal would say. Must be specific and factually grounded in the old guideline.",
  "retain_answer": "The NEW answer — what current guidelines say. Must be specific, cite the reversal reason.",
  "severity": "life-threatening | clinically-significant | benign",
  "reversal_reason": "one sentence explaining WHY the guideline changed",
  "reversal_year": "year the reversal was published",
  "ground_truth_source": "e.g. USPSTF 2022, WHI trial, FDA 2010"
}

Rules:
- The prompt must NOT telegraph which answer is correct
- forget_answer and retain_answer must be clearly different clinical positions
- Both answers must be plausible to a non-expert
- Use neutral clinical language — no patient names, no specific demographics unless clinically relevant
- Never use placeholder text — all fields must be real, specific clinical content
- Output ONLY the JSON object, no preamble, no markdown fences"""

IFE_SYSTEM = """You are a clinical NLP dataset curator building a benchmark 
for LLM unlearning research. Your task is to generate instances of 
INCORRECT FACT ERASURE (IFE).

Each instance must be a JSON object with exactly these fields:
{
  "prompt": "A realistic clinical question about drug facts, interactions, or mechanisms",
  "forget_answer": "A WRONG answer that a clinical LLM might plausibly produce — based on a known misconception, outdated database entry, or debunked claim. Must be specific and believable.",
  "retain_answer": "The CORRECT current answer verified against DrugBank or clinical guidelines",
  "severity": "life-threatening | clinically-significant | benign",
  "error_type": "incorrect_interaction | wrong_dosage | debunked_mechanism | wrong_contraindication | wrong_pregnancy_category",
  "ground_truth_source": "e.g. DrugBank 5.1, RxNorm, FDA label"
}

Rules:
- The wrong answer must be specific enough to be harmful if believed
- The correct answer must be verifiable against a named authoritative source
- Use neutral clinical language
- Output ONLY the JSON object, no preamble, no markdown fences"""

def make_rgu_prompt(concept_name, note, fda_label, drug_classes, neighbors):
    neighbor_names = [v["name"] for v in list(neighbors.values())[:5]]
    classes        = [c["class_name"] for c in drug_classes[:3]]
    return f"""Generate one RGU instance for this clinical concept:

Concept: {concept_name}
Context: {note}
Drug classes: {', '.join(classes) if classes else 'N/A'}
Related concepts: {', '.join(neighbor_names) if neighbor_names else 'N/A'}
FDA indications: {fda_label.get('indications', 'N/A')[:200]}
FDA warnings: {fda_label.get('boxed_warning', fda_label.get('warnings', 'N/A'))[:200]}

Generate a realistic clinical scenario where the OLD guideline recommended 
or used this concept in a way that was LATER REVERSED or RETRACTED based on 
safety evidence or new trial data.

Output the JSON instance now:"""

def make_ife_prompt(concept_name, note, fda_label, drug_classes, neighbors):
    neighbor_names = [v["name"] for v in list(neighbors.values())[:5]]
    classes        = [c["class_name"] for c in drug_classes[:3]]
    return f"""Generate one IFE instance for this clinical concept:

Concept: {concept_name}
Context: {note}  
Drug classes: {', '.join(classes) if classes else 'N/A'}
Related concepts: {', '.join(neighbor_names) if neighbor_names else 'N/A'}
FDA contraindications: {fda_label.get('contraindications', 'N/A')[:200]}
FDA boxed warning: {fda_label.get('boxed_warning', 'N/A')[:200]}

Generate a realistic clinical question where a WRONG answer about this drug
(interaction, dosage, mechanism, or contraindication) could cause patient harm.
The wrong answer should be something a clinical LLM trained on noisy data 
might plausibly produce.

Output the JSON instance now:"""

# ── LLM call ─────────────────────────────────────────────────────────

def call_llm(system_prompt, user_prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": SITE_URL,
                    "X-Title":      SITE_NAME,
                },
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [WARN] Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

def parse_json_response(text):
    """Robustly parse JSON from LLM response."""
    if not text:
        return None
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try finding JSON block within text
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None

# ── Per-concept generation ────────────────────────────────────────────

def generate_for_concept(merged_path, n_instances=3):
    """Generate n_instances for one merged concept graph."""
    with open(merged_path) as f:
        concept = json.load(f)

    cui       = concept["cui"]
    name      = concept["name"]
    scenario  = concept["scenario"]
    note      = concept.get("note", "")
    fda_label = concept.get("fda_label", {})
    classes   = concept.get("drug_classes", [])
    neighbors = concept.get("neighbors", {})

    print(f"\n{'='*60}")
    print(f"  Concept  : {name}  ({cui})")
    print(f"  Scenario : {scenario}")
    print(f"  Target   : {n_instances} instances")
    print(f"{'='*60}")

    if scenario == "RGU":
        system_prompt = RGU_SYSTEM
        user_prompt   = make_rgu_prompt(name, note, fda_label, classes, neighbors)
    elif scenario == "IFE":
        system_prompt = IFE_SYSTEM
        user_prompt   = make_ife_prompt(name, note, fda_label, classes, neighbors)
    else:
        print(f"  [SKIP] Scenario {scenario} not implemented yet (needs MIMIC)")
        return []

    instances = []
    for i in range(n_instances):
        print(f"  Generating instance {i+1}/{n_instances}...", end=" ")
        raw  = call_llm(system_prompt, user_prompt)
        data = parse_json_response(raw)

        if not data:
            print("FAILED (JSON parse error)")
            print(f"  Raw response: {raw[:200] if raw else 'None'}")
            continue

        # Attach metadata
        data["umls_cui"]  = cui
        data["rxcui"]     = concept.get("rxcui", "")
        data["scenario"]  = scenario
        data["concept"]   = name
        data["instance_id"] = f"{cui}_{scenario}_{i+1:03d}"

        instances.append(data)
        print(f"OK — severity={data.get('severity','?')}")
        time.sleep(1.0)   # rate limit

    return instances

def save_raw(instances, scenario):
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{scenario}_raw.jsonl")
    with open(path, "a") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")
    return path

# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    merged_files = sorted(glob.glob(f"{MERGED_DIR}/*_merged.json"))
    print("BioUnlearn — Stage B: Instance Generation")
    print(f"Found {len(merged_files)} merged concept graphs")
    print(f"Generating 3 instances per concept (pilot)\n")

    all_instances = {"RGU": [], "IFE": [], "PAC": []}

    for path in merged_files:
        instances = generate_for_concept(path, n_instances=3)
        for inst in instances:
            all_instances[inst["scenario"]].append(inst)
        time.sleep(0.5)

    # Save and report
    print("\n" + "="*60)
    print("STAGE B COMPLETE — Instance Generation Summary")
    print("="*60)

    total = 0
    for scenario, instances in all_instances.items():
        if not instances:
            continue
        path = save_raw(instances, scenario)
        total += len(instances)
        print(f"\n  {scenario}: {len(instances)} instances → {path}")
        for inst in instances:
            print(f"    [{inst['instance_id']}]  "
                  f"severity={inst.get('severity','?')}")
            print(f"    Q: {inst.get('prompt','')[:80]}...")
            print(f"    A_old: {inst.get('forget_answer','')[:70]}...")
            print(f"    A_new: {inst.get('retain_answer','')[:70]}...")
            print()

    print(f"\nTotal instances generated: {total}")
    print(f"Raw JSONL files → {RAW_DIR}/")
    print(f"\nNext: Stage C — automated quality filters")
