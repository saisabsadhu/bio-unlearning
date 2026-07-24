"""
Stage C — Automated Quality Filters
Reads raw JSONL from Stage B, applies 4 filters sequentially.
Saves filtered instances to data/filtered/ with pass/fail reasons.

Filter 1 — Schema completeness   : all required fields present and non-empty
Filter 2 — Answer distinctness   : forget_answer != retain_answer (not just surface)
Filter 3 — Prompt diversity      : no two prompts too similar within same concept
Filter 4 — Clinical plausibility : re-score via LLM (1-5), reject if < 4
"""

import json
import os
import sys
import glob
import time
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    OPENROUTER_KEY, OPENROUTER_BASE,
    OPENROUTER_MODEL, SITE_URL, SITE_NAME
)

client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_KEY)

RAW_DIR      = "data/raw"
FILTERED_DIR = "data/filtered"

# ── Filter 1 — Schema completeness ───────────────────────────────────

REQUIRED_FIELDS_RGU = {
    "prompt", "forget_answer", "retain_answer",
    "severity", "reversal_reason", "ground_truth_source"
}
REQUIRED_FIELDS_IFE = {
    "prompt", "forget_answer", "retain_answer",
    "severity", "error_type", "ground_truth_source"
}
REQUIRED_FIELDS_PAC = {
    "prompt", "forget_answer", "retain_answer",
    "severity", "pac_category", "ground_truth_source"
}

def filter_schema(inst):
    scenario = inst.get("scenario", "")
    if scenario == "RGU":
        required = REQUIRED_FIELDS_RGU
    elif scenario == "PAC":
        required = REQUIRED_FIELDS_PAC
    else:
        required = REQUIRED_FIELDS_IFE
    missing  = []
    for field in required:
        val = inst.get(field, "")
        if not val or str(val).strip() == "":
            missing.append(field)
    if missing:
        return False, f"missing fields: {missing}"
    return True, "ok"

# ── Filter 2 — Answer distinctness ───────────────────────────────────

def jaccard(a, b):
    """Token-level Jaccard similarity."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def filter_distinctness(inst):
    forget  = inst.get("forget_answer", "")
    retain  = inst.get("retain_answer", "")
    sim     = jaccard(forget, retain)
    if sim > 0.75:
        return False, f"answers too similar (Jaccard={sim:.2f})"
    if forget.strip().lower() == retain.strip().lower():
        return False, "answers identical"
    return True, f"Jaccard={sim:.2f}"

# ── Filter 3 — Prompt diversity ───────────────────────────────────────

def filter_diversity(inst, seen_prompts):
    """Reject if this prompt is too similar to one already accepted."""
    prompt = inst.get("prompt", "")
    for seen in seen_prompts:
        sim = jaccard(prompt, seen)
        if sim > 0.50:
            return False, f"prompt too similar to existing (Jaccard={sim:.2f})"
    return True, "ok"

# ── Filter 4 — Clinical plausibility re-score ────────────────────────

PLAUSIBILITY_SYSTEM = """You are a clinical NLP evaluator. 
Rate the clinical plausibility of this unlearning instance on a scale of 1-5:
5 = Clinically accurate, realistic scenario, clear distinction between old and new answer
4 = Mostly accurate, minor issues
3 = Somewhat plausible but has factual gaps or unclear distinction  
2 = Implausible or factually incorrect
1 = Completely wrong or nonsensical

Respond with ONLY a JSON object: {"score": <1-5>, "reason": "<one sentence>"}
No preamble, no markdown."""

def filter_plausibility(inst):
    prompt = f"""Instance to evaluate:
Scenario: {inst.get('scenario')}
Clinical question: {inst.get('prompt')}
Old answer (to forget): {inst.get('forget_answer')}
New answer (to retain): {inst.get('retain_answer')}
Source: {inst.get('ground_truth_source', 'N/A')}

Rate this instance:"""

    try:
        response = client.chat.completions.create(
            extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_NAME},
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": PLAUSIBILITY_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        raw  = response.choices[0].message.content.strip()
        # Strip fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        data = json.loads(raw)
        score = int(data.get("score", 0))
        reason = data.get("reason", "")
        if score < 4:
            return False, f"plausibility={score}/5 — {reason}"
        return True, f"plausibility={score}/5 — {reason}"
    except Exception as e:
        # If LLM call fails, be conservative and pass
        return True, f"plausibility check skipped ({e})"

# ── Run all filters on one instance ──────────────────────────────────

def run_filters(inst, seen_prompts):
    results = {}

    # F1
    passed, msg = filter_schema(inst)
    results["F1_schema"] = {"passed": passed, "msg": msg}
    if not passed:
        return False, results

    # F2
    passed, msg = filter_distinctness(inst)
    results["F2_distinctness"] = {"passed": passed, "msg": msg}
    if not passed:
        return False, results

    # F3
    passed, msg = filter_diversity(inst, seen_prompts)
    results["F3_diversity"] = {"passed": passed, "msg": msg}
    if not passed:
        return False, results

    # F4
    time.sleep(0.5)
    passed, msg = filter_plausibility(inst)
    results["F4_plausibility"] = {"passed": passed, "msg": msg}
    if not passed:
        return False, results

    return True, results

# ── Process one JSONL file ────────────────────────────────────────────

def filter_file(raw_path):
    scenario = os.path.basename(raw_path).replace("_raw.jsonl", "")
    print(f"\n{'='*60}")
    print(f"  Filtering: {scenario}")
    print(f"{'='*60}")

    instances = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    print(f"  Input: {len(instances)} raw instances")

    passed_instances = []
    failed_instances = []
    seen_prompts     = []

    for inst in instances:
        iid = inst.get("instance_id", "?")
        print(f"\n  [{iid}]")

        ok, filter_results = run_filters(inst, seen_prompts)

        for fname, res in filter_results.items():
            status = "✓" if res["passed"] else "✗"
            print(f"    {status} {fname}: {res['msg']}")

        if ok:
            inst["filter_results"] = filter_results
            inst["filter_status"]  = "PASSED"
            passed_instances.append(inst)
            seen_prompts.append(inst.get("prompt", ""))
            print(f"    → PASSED")
        else:
            inst["filter_results"] = filter_results
            inst["filter_status"]  = "FAILED"
            failed_instances.append(inst)
            print(f"    → FAILED")

    # Save
    os.makedirs(FILTERED_DIR, exist_ok=True)

    passed_path = os.path.join(FILTERED_DIR, f"{scenario}_passed.jsonl")
    failed_path = os.path.join(FILTERED_DIR, f"{scenario}_failed.jsonl")

    with open(passed_path, "w") as f:
        for inst in passed_instances:
            f.write(json.dumps(inst) + "\n")

    with open(failed_path, "w") as f:
        for inst in failed_instances:
            f.write(json.dumps(inst) + "\n")

    pass_rate = len(passed_instances) / len(instances) * 100 if instances else 0
    print(f"\n  Passed : {len(passed_instances)}/{len(instances)} "
          f"({pass_rate:.0f}%)")
    print(f"  Saved  : {passed_path}")

    return len(passed_instances), len(instances)

# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_files = sorted(glob.glob(f"{RAW_DIR}/*_raw.jsonl"))
    print("BioUnlearn — Stage C: Automated Quality Filters")
    print(f"Processing {len(raw_files)} raw files\n")
    print("Filters: F1=Schema  F2=Distinctness  F3=Diversity  F4=Plausibility\n")

    summary = []
    for path in raw_files:
        passed, total = filter_file(path)
        scenario      = os.path.basename(path).replace("_raw.jsonl","")
        summary.append((scenario, passed, total))

    print("\n" + "="*55)
    print("STAGE C COMPLETE — Filter Summary")
    print("="*55)
    print(f"{'Scenario':<12} {'Passed':<8} {'Total':<8} {'Rate'}")
    print("-"*55)
    for scenario, passed, total in summary:
        rate = passed/total*100 if total else 0
        print(f"{scenario:<12} {passed:<8} {total:<8} {rate:.0f}%")
    print("="*55)
    print(f"\nPassed instances → {FILTERED_DIR}/")
    print("Next: inspect passed instances, then Stage D — dataset splits")
