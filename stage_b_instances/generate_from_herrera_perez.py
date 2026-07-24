"""
Stage B (Herrera-Perez track) -- converts the 396-entry gold-tier reversal
catalog (data/gold_sources/herrera_perez_2019_full.json) into real RGU
dataset instances.

Unlike generate_instances.py (which asks an LLM to *invent* a realistic
clinical scenario from a UMLS concept graph), this script asks the LLM to
*extract and reformat* a scenario from a real, citation-backed trial summary
that is already given in full -- a lower-hallucination-risk task, since the
old belief and the reversal finding are both already present in the input
text, not generated from scratch.

Usage:
    python stage_b_instances/generate_from_herrera_perez.py --limit 5 --dry-run
    python stage_b_instances/generate_from_herrera_perez.py --start 0 --end 396
"""

import argparse
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

client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_KEY)

SOURCE_PATH = "data/gold_sources/herrera_perez_2019_full.json"
RAW_DIR = "data/raw"

SYSTEM_PROMPT = """You are a clinical NLP dataset curator building a benchmark
for LLM unlearning research (RGU = Retracted Guideline Unlearning).

You will be given a REAL summary of a randomized controlled trial that
reversed a prior clinical practice. Your job is to EXTRACT AND REFORMAT the
information already present in the summary into a structured instance --
do NOT invent facts, numbers, or claims that are not supported by the given
text. If the summary does not clearly support a distinct "old belief" vs.
"new evidence-based answer" pair, or is too vague/procedural to make a clean
clinical question, output exactly: {"skip": true, "reason": "<why>"}

Otherwise output exactly this JSON object:
{
  "prompt": "A realistic clinical question a physician or medical student would ask about this practice, phrased neutrally (must not telegraph which answer is correct)",
  "forget_answer": "The OLD answer/belief -- what was assumed or practiced BEFORE this trial's finding, grounded in the summary's description of the prior practice/rationale",
  "retain_answer": "The NEW, evidence-based answer -- what the trial's actual finding supports, grounded directly in the summary's reported result and numbers",
  "severity": "life-threatening | clinically-significant | benign",
  "reversal_reason": "one sentence, grounded in the summary, explaining why the practice was reversed",
  "reversal_year": <integer year from the input>,
  "ground_truth_source": "<copy the source field from the input exactly>"
}

Rules:
- forget_answer and retain_answer must be clearly distinct clinical positions, not paraphrases of each other
- Use neutral clinical language, no patient names
- Ground every claim in the given summary text -- do not add outside knowledge
- Output ONLY the JSON object, no preamble, no markdown fences"""


def make_user_prompt(entry):
    return f"""Trial summary to extract from:

Practice: {entry['practice']}
Specialty: {entry.get('specialty', 'N/A')}
Year: {entry.get('year', 'N/A')}
Journal: {entry.get('journal', 'N/A')}
Summary: {entry['summary']}
Prior systematic-review context: {entry.get('systematic_review_context', 'None found')}
Source (copy exactly into ground_truth_source): {entry['source']}

Extract the RGU instance now:"""


def call_llm(user_prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_NAME},
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [WARN] attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def parse_json_response(text):
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
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap total processed, for a quick test batch")
    parser.add_argument("--dry-run", action="store_true", help="print results, do not write to data/raw/")
    parser.add_argument("--out", default=f"{RAW_DIR}/RGU_herrera_perez_raw.jsonl")
    args = parser.parse_args()

    with open(SOURCE_PATH) as f:
        entries = json.load(f)

    end = args.end if args.end is not None else len(entries)
    batch = entries[args.start:end]
    if args.limit:
        batch = batch[: args.limit]

    print(f"Processing {len(batch)} entries (indices {args.start}-{args.start+len(batch)})")

    results = []
    n_skip, n_fail, n_ok = 0, 0, 0
    total_cost = 0.0

    for i, entry in enumerate(batch):
        idx = args.start + i
        print(f"\n[{idx}] {entry['practice'][:70]}")
        user_prompt = make_user_prompt(entry)
        raw = call_llm(user_prompt)
        data = parse_json_response(raw)

        if data is None:
            print(f"    -> PARSE FAILURE. Raw: {raw[:150] if raw else None}")
            n_fail += 1
            continue
        if data.get("skip"):
            print(f"    -> SKIPPED: {data.get('reason')}")
            n_skip += 1
            continue

        data["scenario"] = "RGU"
        data["instance_id"] = f"herrera_perez_{idx}"
        data["source_entry"] = {
            "practice": entry["practice"],
            "specialty": entry.get("specialty"),
            "pqs": entry.get("pqs"),
        }
        results.append(data)
        n_ok += 1
        print(f"    -> OK. prompt: {data.get('prompt', '')[:90]}")
        print(f"       forget: {data.get('forget_answer', '')[:90]}")
        print(f"       retain: {data.get('retain_answer', '')[:90]}")
        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"Done. ok={n_ok} skip={n_skip} parse_fail={n_fail} / {len(batch)}")

    if not args.dry_run and results:
        os.makedirs(RAW_DIR, exist_ok=True)
        mode = "a" if os.path.exists(args.out) else "w"
        with open(args.out, mode) as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Appended {len(results)} instances -> {args.out}")
    elif args.dry_run:
        print("(dry run -- nothing written)")


if __name__ == "__main__":
    main()
