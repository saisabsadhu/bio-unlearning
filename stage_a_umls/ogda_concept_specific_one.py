"""
Single (checkpoint, concept) evaluation for the concept-specific OGDA
comparison -- run as its own OS process (not looped in-process) so CUDA
memory is guaranteed fully released on exit. Appends its one result into
the shared batch JSON under the given category/key path.

Usage:
    python stage_a_umls/ogda_concept_specific_one.py --checkpoint saves/unlearn/OGDA_seed2_C0004057_checkpoint \
        --cui C0004057 --category real_ogda --key C0004057.2
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
OUT_PATH = "data/gate2_results/OGDA_concept_specific_batch_summary.json"


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
    fa_flags, acc_flags = [], []
    for ex in examples:
        gen = generate_one(model, tokenizer, ex["prompt"])
        fa_flags.append(1.0 if rouge_recall(scorer, gen, ex["forget_answer"]) < ROUGE_THRESHOLD else 0.0)
        acc_flags.append(1.0 if rouge_recall(scorer, gen, ex["retain_answer"]) >= ROUGE_THRESHOLD else 0.0)
    fa_old = sum(fa_flags) / len(fa_flags)
    acc_new = sum(acc_flags) / len(acc_flags)
    return fa_old, acc_new, fa_old * acc_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cui", required=True)
    parser.add_argument("--category", required=True, choices=["baseline", "random_control", "real_ogda"])
    parser.add_argument("--seed", default=None, help="required for random_control/real_ogda")
    args = parser.parse_args()

    instances = load_jsonl(f"data/splits_concept/{args.cui}_instances.jsonl")
    print(f"{args.cui}: {len(instances)} instances, checkpoint={args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    fa, acc, defv = score_examples(model, tokenizer, instances, scorer)
    print(f"RESULT {args.category} {args.cui} seed={args.seed}: FA={fa:.3f} acc_new={acc:.3f} DEF={defv:.3f}")

    try:
        with open(OUT_PATH) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {"baseline": {}, "random_control": {}, "real_ogda": {}}

    entry = {"fa": fa, "acc_new": acc, "def": defv}
    if args.category == "baseline":
        results["baseline"][args.cui] = entry
    elif args.category == "random_control":
        results["random_control"].setdefault(args.seed, {})[args.cui] = entry
    else:
        results["real_ogda"].setdefault(args.cui, {})[args.seed] = entry

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
