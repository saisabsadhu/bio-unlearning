"""
Generates MORE concept-specific RGU instances for the 4 OGDA-tested concepts
(aspirin C0004057, rosiglitazone C0289313, HRT C0282402, Vioxx C0876768),
since the concept-specific eval (see documentation/NOVEL_METHODOLOGY_OGDA.md
Section 8.11) found the existing 4-5 instances per concept too few for a
stable n=3-vs-n=3 comparison -- 2 of 4 concepts showed zero differentiating
signal in either direction (a floor effect), and the concept where there
WAS a signal (aspirin) disagreed in direction with the whole-benchmark
result.

Each new instance is grounded in the SAME real reversal event already
documented in that concept's existing instances (same ground_truth_source,
same underlying FDA/USPSTF/WHI/Merck event) -- only the clinical "angle"
(patient demographics, comorbidities, presenting question) varies, keeping
ground truth consistent with what's already in the dataset rather than
inventing a new reversal.

Usage:
    python stage_b_instances/generate_concept_specific_instances.py --cui C0004057 --n 15 --dry-run
    python stage_b_instances/generate_concept_specific_instances.py --n 15
"""

import argparse
import json
import os
import sys
import time
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import OPENROUTER_KEY, OPENROUTER_BASE, OPENROUTER_MODEL, SITE_URL, SITE_NAME

client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_KEY)

RAW_DIR = "data/raw"

CONCEPTS = {
    "C0004057": {"concept": "Aspirin primary prevention"},
    "C0289313": {"concept": "Rosiglitazone"},
    "C0282402": {"concept": "Hormone Replacement Therapy"},
    "C0876768": {"concept": "Vioxx rofecoxib"},
}

ANGLES = [
    "a specific age/sex patient with no comorbidities",
    "a patient with a relevant comorbidity that changes the risk calculus",
    "a patient asking about starting the therapy for the first time",
    "a patient currently on the therapy asking whether to continue",
    "a question from a medical student/trainee about the general guideline",
    "a patient with a family history relevant to the condition",
    "a question comparing this therapy to an alternative",
    "a question about the specific mechanism behind the reversal",
]

SYSTEM_PROMPT = """You are a clinical NLP dataset curator building the RGU
(Retracted Guideline Unlearning) scenario for an LLM unlearning benchmark.

You will be given a concept, its already-documented real guideline reversal
(reversal reason, year, and source -- a REAL FDA/USPSTF/WHI/trial event, not
invented), and a specific clinical "angle" to write a NEW question from.

Ground every claim in the given reversal information -- do not invent new
facts, numbers, or claims beyond what's given. Output exactly this JSON:
{
  "prompt": "A realistic clinical question from the given angle, phrased neutrally (must not telegraph which answer is correct)",
  "forget_answer": "The OLD answer/belief -- what was practiced BEFORE the given reversal",
  "retain_answer": "The NEW, evidence-based answer -- what current guidance supports, per the given reversal",
  "severity": "life-threatening | clinically-significant | benign"
}

Rules:
- forget_answer and retain_answer must be clearly distinct clinical positions, not paraphrases
- Use neutral clinical language, no patient names
- This must be a genuinely different question from a typical generic one -- follow the given angle specifically
- Output ONLY the JSON object, no preamble, no markdown fences"""


def make_user_prompt(concept, reversal_reason, reversal_year, ground_truth_source, angle):
    return f"""Concept: {concept}
Documented reversal: {reversal_reason}
Reversal year: {reversal_year}
Source: {ground_truth_source}
Angle for this question: {angle}

Generate the instance now:"""


def call_llm(user_prompt, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_NAME},
                model=OPENROUTER_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                temperature=0.8,
                max_tokens=500,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [WARN] attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def parse_json(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def load_reversal_info(cui):
    path = f"data/splits_concept/{cui}_instances.jsonl"
    with open(path) as f:
        inst = json.loads(f.readline())
    return inst["reversal_reason"], inst["reversal_year"], inst["ground_truth_source"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cui", default=None, help="single CUI, omit to run all 4")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=f"{RAW_DIR}/RGU_concept_specific_raw.jsonl")
    args = parser.parse_args()

    cuis = [args.cui] if args.cui else list(CONCEPTS.keys())
    results = []
    n_fail = 0

    for cui in cuis:
        concept = CONCEPTS[cui]["concept"]
        reversal_reason, reversal_year, ground_truth_source = load_reversal_info(cui)
        print(f"\n=== {concept} ({cui}) === grounded in: {ground_truth_source}")

        # existing instance_ids to continue numbering from
        existing_path = f"data/splits_concept/{cui}_instances.jsonl"
        existing_ids = [json.loads(l)["instance_id"] for l in open(existing_path)]
        start_idx = len(existing_ids) + 1

        for i in range(args.n):
            angle = ANGLES[i % len(ANGLES)]
            print(f"  [{i}] angle={angle[:50]}")
            raw = call_llm(make_user_prompt(concept, reversal_reason, reversal_year, ground_truth_source, angle))
            data = parse_json(raw)
            if data is None:
                print(f"    -> PARSE FAILURE")
                n_fail += 1
                continue

            data["reversal_reason"] = reversal_reason
            data["reversal_year"] = reversal_year
            data["ground_truth_source"] = ground_truth_source
            data["umls_cui"] = cui
            data["scenario"] = "RGU"
            data["concept"] = concept
            data["instance_id"] = f"{cui}_RGU_{start_idx + i:03d}"
            data["angle"] = angle
            results.append(data)
            print(f"    -> OK. {data.get('prompt', '')[:80]}")
            time.sleep(0.3)

    print(f"\n{'='*60}\nok={len(results)} parse_fail={n_fail}")

    if not args.dry_run and results:
        os.makedirs(RAW_DIR, exist_ok=True)
        mode = "a" if os.path.exists(args.out) else "w"
        with open(args.out, mode) as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Appended {len(results)} instances -> {args.out}")


if __name__ == "__main__":
    main()
