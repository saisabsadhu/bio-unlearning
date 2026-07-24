"""
Batch version of ogda_concept_specific_eval.py: minimizes model loads by
looping concepts within a single model load, rather than reloading per
(checkpoint, concept) pair.

Loads: base model once (eval'd against all 4 concepts), each of the 3
random-control checkpoints once (eval'd against all 4 concepts each, since
the random-control null distribution is concept-agnostic -- established
methodology, see documentation/NOVEL_METHODOLOGY_OGDA.md), and each of the
4 real-OGDA checkpoints once (eval'd only against its own concept, since
those edits are concept-specific by construction).

Usage:
    python stage_a_umls/ogda_concept_specific_batch.py
"""

import json

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "BioMistral/BioMistral-7B"
ROUGE_THRESHOLD = 0.5
MAX_NEW_TOKENS = 200
USER_START, USER_END = "[INST] ", " [/INST]"
OUT_DIR = "data/gate2_results"

CONCEPTS = ["C0004057", "C0289313", "C0282402", "C0876768"]
RANDOM_CONTROL_CHECKPOINTS = {
    0: "saves/unlearn/OGDA_random_control_checkpoint",
    1: "saves/unlearn/OGDA_random_control_seed1_checkpoint",
    2: "saves/unlearn/OGDA_random_control_seed2_checkpoint",
}
REAL_OGDA_CHECKPOINTS = {
    "C0004057": {0: "saves/unlearn/OGDA_C0004057_checkpoint", 1: "saves/unlearn/OGDA_seed1_C0004057_checkpoint", 2: "saves/unlearn/OGDA_seed2_C0004057_checkpoint"},
    "C0289313": {0: "saves/unlearn/OGDA_C0289313_checkpoint", 1: "saves/unlearn/OGDA_seed1_C0289313_checkpoint", 2: "saves/unlearn/OGDA_seed2_C0289313_checkpoint"},
    "C0282402": {0: "saves/unlearn/OGDA_C0282402_checkpoint", 1: "saves/unlearn/OGDA_seed1_C0282402_checkpoint", 2: "saves/unlearn/OGDA_seed2_C0282402_checkpoint"},
    "C0876768": {0: "saves/unlearn/OGDA_C0876768_checkpoint", 1: "saves/unlearn/OGDA_seed1_C0876768_checkpoint", 2: "saves/unlearn/OGDA_seed2_C0876768_checkpoint"},
}


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


def load_model(path):
    print(f"Loading {path}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tokenizer


def main():
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    instances_by_cui = {c: load_jsonl(f"data/splits_concept/{c}_instances.jsonl") for c in CONCEPTS}
    for c, insts in instances_by_cui.items():
        print(f"{c}: {len(insts)} concept-specific instances")

    out_path = f"{OUT_DIR}/OGDA_concept_specific_batch_summary.json"
    try:
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming from existing {out_path}")
    except FileNotFoundError:
        results = {"baseline": {}, "random_control": {}, "real_ogda": {}}

    def save():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    def load_with_retry(path, attempts=6, wait_s=180):
        import time
        for attempt in range(1, attempts + 1):
            try:
                return load_model(path)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                print(f"  OOM loading {path} (attempt {attempt}/{attempts}): {e}")
                if attempt == attempts:
                    raise
                print(f"  waiting {wait_s}s before retry...")
                time.sleep(wait_s)

    # --- Baseline (untouched model), all 4 concepts ---
    if "C0004057" not in results["baseline"]:
        model, tokenizer = load_with_retry(MODEL_NAME)
        for cui in CONCEPTS:
            fa, acc, defv = score_examples(model, tokenizer, instances_by_cui[cui], scorer)
            results["baseline"][cui] = {"fa": fa, "acc_new": acc, "def": defv}
            print(f"  [baseline] {cui}: FA={fa:.3f} acc_new={acc:.3f} DEF={defv:.3f}")
        del model
        torch.cuda.empty_cache()
        save()
    else:
        print("baseline already done, skipping")

    # --- Random-control seeds, all 4 concepts each ---
    for seed, ckpt in RANDOM_CONTROL_CHECKPOINTS.items():
        skey = str(seed)
        if skey in results["random_control"] and len(results["random_control"][skey]) == len(CONCEPTS):
            print(f"random_control seed={seed} already done, skipping")
            continue
        model, tokenizer = load_with_retry(ckpt)
        results["random_control"].setdefault(skey, {})
        for cui in CONCEPTS:
            fa, acc, defv = score_examples(model, tokenizer, instances_by_cui[cui], scorer)
            results["random_control"][skey][cui] = {"fa": fa, "acc_new": acc, "def": defv}
            print(f"  [random_control seed={seed}] {cui}: FA={fa:.3f} acc_new={acc:.3f} DEF={defv:.3f}")
        del model
        torch.cuda.empty_cache()
        save()

    # --- Real OGDA checkpoints, own concept only, all 3 seeds each ---
    for cui, seed_ckpts in REAL_OGDA_CHECKPOINTS.items():
        results["real_ogda"].setdefault(cui, {})
        for seed, ckpt in seed_ckpts.items():
            skey = str(seed)
            if skey in results["real_ogda"][cui]:
                print(f"real_ogda {cui} seed={seed} already done, skipping")
                continue
            try:
                model, tokenizer = load_with_retry(ckpt)
            except torch.cuda.OutOfMemoryError:
                print(f"  GAVE UP on real_ogda {cui} seed={seed} after all retries -- skipping, will retry on next run")
                continue
            fa, acc, defv = score_examples(model, tokenizer, instances_by_cui[cui], scorer)
            results["real_ogda"][cui][skey] = {"fa": fa, "acc_new": acc, "def": defv}
            print(f"  [real_ogda seed={seed}] {cui}: FA={fa:.3f} acc_new={acc:.3f} DEF={defv:.3f}")
            del model
            torch.cuda.empty_cache()
            save()

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
