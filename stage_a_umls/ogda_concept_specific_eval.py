"""
Concept-specific FA/DEF eval for OGDA: measures each checkpoint's effect
against ONLY the instances actually about the concept it was built to
ablate, instead of the diffuse 56-58-example whole-scenario benchmark.

Why this exists: OGDA's own-scenario effect was real at pilot scale (the
58-example RGU_train.jsonl) but washed out to noise when re-evaluated
against the 273-example v2 set (see documentation/RESULTS_SYNTHESIS.md
Section 5) -- expected, since each checkpoint's edit is deliberately
surgical to one concept's ontology subspace, and the vast majority of a
whole-scenario benchmark is about unrelated topics. This script instead
evaluates against the concept's own dedicated instances (found sitting
unused in RGU_val/RGU_test -- the existing OGDA eval pipeline has always
scored against RGU_train.jsonl, which doesn't contain these 4 concepts'
instances at all).

Usage:
    python stage_a_umls/ogda_concept_specific_eval.py --cui C0004057 \\
        --checkpoint saves/unlearn/OGDA_C0004057_checkpoint --out data/gate2_results/OGDA_concept_specific_C0004057.json
    python stage_a_umls/ogda_concept_specific_eval.py --cui C0004057 --baseline \\
        --out data/gate2_results/OGDA_concept_specific_C0004057_baseline.json
"""

import argparse
import json

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "BioMistral/BioMistral-7B"
ROUGE_THRESHOLD = 0.5
MAX_NEW_TOKENS = 200
USER_START, USER_END = "[INST] ", " [/INST]"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def rouge_recall(scorer, gen, target):
    return scorer.score(target, gen)["rougeL"].recall


def generate_one(model, tokenizer, question):
    prompt = f"{USER_START}{question}{USER_END}"
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                              use_cache=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def score_examples(model, tokenizer, examples, scorer):
    fa_flags, acc_flags, gens = [], [], []
    for ex in examples:
        gen = generate_one(model, tokenizer, ex["prompt"])
        gens.append(gen)
        fa_flags.append(1.0 if rouge_recall(scorer, gen, ex["forget_answer"]) < ROUGE_THRESHOLD else 0.0)
        acc_flags.append(1.0 if rouge_recall(scorer, gen, ex["retain_answer"]) >= ROUGE_THRESHOLD else 0.0)
    fa_old = sum(fa_flags) / len(fa_flags)
    acc_new = sum(acc_flags) / len(acc_flags)
    return fa_old, acc_new, fa_old * acc_new, gens, fa_flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cui", required=True)
    parser.add_argument("--checkpoint", default=None, help="path to OGDA checkpoint; omit with --baseline for untouched model")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not args.baseline and not args.checkpoint:
        raise ValueError("must pass --checkpoint or --baseline")

    instances = load_jsonl(f"data/splits_concept/{args.cui}_instances.jsonl")
    print(f"Loaded {len(instances)} concept-specific instances for {args.cui}")

    model_path = MODEL_NAME if args.baseline else args.checkpoint
    print(f"Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    fa, acc_new, def_score, gens, fa_flags = score_examples(model, tokenizer, instances, scorer)

    result = {
        "cui": args.cui,
        "model_path": model_path,
        "n_instances": len(instances),
        "fa": fa,
        "acc_new": acc_new,
        "def": def_score,
        "per_instance_generations": [
            {"prompt": i["prompt"][:100], "generation": g[:200], "fa_flag": f}
            for i, g, f in zip(instances, gens, fa_flags)
        ],
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_instance_generations"}, indent=2))

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
