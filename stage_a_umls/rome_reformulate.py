"""
Reformulates BioUnlearn-Bench RGU/IFE instances (open-ended clinical Q&A)
into ROME/MEMIT-style factual-edit requests: {prompt (with {} slot),
subject, target_true, target_new}. ROME/MEMIT edit a specific short
completion given a subject, not open-ended Q&A, so this translation step is
necessary before either method can be applied at all.

Usage:
    python stage_a_umls/rome_reformulate.py --scenario RGU --limit 5 --dry-run
    python stage_a_umls/rome_reformulate.py --scenario RGU --out data/gate2_results/rome_requests_RGU.json
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

SYSTEM_PROMPT = """You convert a clinical Q&A unlearning instance into a ROME/MEMIT-style
factual-edit request. ROME/MEMIT edit a model's completion of a SHORT factual
statement about a SUBJECT -- not open-ended Q&A -- so you must extract:

1. "subject": a short noun phrase (2-6 words) naming the clinical practice/drug/concept
2. "prompt_template": a short declarative sentence containing the subject phrase
   VERBATIM, with a completion slot at the end (do not include the answer).
   Example shape: "The current guidance on <subject> is that it"
3. "target_true": a short phrase (3-12 words) completing prompt_template with the
   OLD (forget_answer) claim, grounded in the given forget_answer
4. "target_new": a short phrase (3-12 words) completing prompt_template with the
   NEW (retain_answer) claim, grounded in the given retain_answer

Rules:
- prompt_template must contain the exact subject string, and reads naturally when
  subject is substituted in and either target is appended at the end
- target_true and target_new must be genuinely different short completions of the
  SAME prompt_template (not full sentences, not restating the subject)
- If this instance cannot be cleanly reduced to a single short subject + fact,
  output exactly: {"skip": true, "reason": "<why>"}

Output ONLY the JSON object, no preamble, no markdown fences."""


def make_user_prompt(inst):
    return f"""Instance to convert:

Concept: {inst.get('concept', '')}
Clinical question: {inst['prompt']}
Old answer (forget_answer): {inst['forget_answer']}
New answer (retain_answer): {inst['retain_answer']}

Convert now:"""


def call_llm(user_prompt, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_NAME},
                model=OPENROUTER_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                temperature=0.2,
                max_tokens=400,
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


def validate(data):
    """Subject must appear verbatim in prompt_template, targets must differ."""
    if not data or data.get("skip"):
        return False, data.get("reason") if data else "parse failure"
    subj = data.get("subject", "")
    tmpl = data.get("prompt_template", "")
    if subj.lower() not in tmpl.lower():
        return False, f"subject '{subj}' not found verbatim in template '{tmpl}'"
    # normalize casing so the literal string substitution in rome_main.py's
    # request['prompt'].replace(subject, '{}') works regardless of sentence-initial capitalization
    idx = tmpl.lower().find(subj.lower())
    data["subject"] = tmpl[idx: idx + len(subj)]
    tt, tn = data.get("target_true", "").strip().lower(), data.get("target_new", "").strip().lower()
    if not tt or not tn or tt == tn:
        return False, "target_true/target_new missing or identical"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["RGU", "IFE"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = f"data/splits/{args.scenario}_train.jsonl"
    with open(path) as f:
        instances = [json.loads(l) for l in f]
    if args.limit:
        instances = instances[: args.limit]

    print(f"Reformulating {len(instances)} {args.scenario} instances")
    results = []
    n_ok, n_skip, n_invalid = 0, 0, 0

    for i, inst in enumerate(instances):
        print(f"\n[{i}] {inst.get('concept', '')[:60]}")
        raw = call_llm(make_user_prompt(inst))
        data = parse_json(raw)
        ok, msg = validate(data)
        if not ok:
            print(f"    -> REJECTED: {msg}")
            if data and data.get("skip"):
                n_skip += 1
            else:
                n_invalid += 1
            continue
        data["instance_id"] = inst.get("instance_id")
        data["umls_cui"] = inst.get("umls_cui")
        results.append(data)
        n_ok += 1
        print(f"    -> OK. subject='{data['subject']}'")
        print(f"       template: {data['prompt_template']}")
        print(f"       true: {data['target_true']}  |  new: {data['target_new']}")
        time.sleep(0.3)

    print(f"\n{'='*60}\nok={n_ok} skip={n_skip} invalid={n_invalid} / {len(instances)}")

    if not args.dry_run and results:
        out_path = args.out or f"data/gate2_results/rome_requests_{args.scenario}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} requests -> {out_path}")


if __name__ == "__main__":
    main()
