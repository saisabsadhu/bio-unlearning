"""
RAG-suppression baseline: instead of editing model weights (GA/NPO/RMU/OGDA),
suppress the outdated answer purely at inference time by prepending a
retrieved-correction context to the target scenario's prompts. The model's
weights are never touched.

Rationale: this is the most obvious alternative to weight-based unlearning a
reviewer will ask about ("why not just RAG this?"). It's included as a real
comparison point, not a strawman -- the context text is built directly from
each instance's own reversal_reason/ground_truth_source (RGU) or
error_type/ground_truth_source (IFE) fields, i.e. what a real retriever
would plausibly surface for that specific question.

The untargeted scenario gets NO injected context (nothing relevant would be
retrieved for an unrelated question) -- this is exactly the comparison point:
does RAG suppression achieve FA/DEF on its target without the collateral
damage weight-editing methods showed on the untargeted scenario?

Usage:
    python stage_a_umls/rag_suppression_baseline.py --target RGU
    python stage_a_umls/rag_suppression_baseline.py --target IFE
"""

import argparse
import json

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "BioMistral/BioMistral-7B"
RGU_PATH = "data/splits/RGU_train.jsonl"
IFE_PATH = "data/splits/IFE_train.jsonl"
ROUGE_THRESHOLD = 0.5
MAX_NEW_TOKENS = 200

USER_START, USER_END = "[INST] ", " [/INST]"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_context_rgu(inst):
    return (
        f"Retrieved context: A subsequent evidence review found the following: "
        f"{inst.get('reversal_reason', '')} "
        f"Current guidance per {inst.get('ground_truth_source', 'the current clinical literature')}: "
        f"{inst.get('retain_answer', '')}\n\n"
        f"Using the retrieved context above, answer the following question:\n"
    )


def build_context_ife(inst):
    return (
        f"Retrieved context: This claim has been identified as a "
        f"{inst.get('error_type', 'factual error')} and corrected. "
        f"Verified answer per {inst.get('ground_truth_source', 'the current clinical literature')}: "
        f"{inst.get('retain_answer', '')}\n\n"
        f"Using the retrieved context above, answer the following question:\n"
    )


def make_prompt(question, context=None):
    body = (context or "") + question
    return f"{USER_START}{body}{USER_END}"


def generate_batch(model, tokenizer, prompts, batch_size=8):
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[-1] :], skip_special_tokens=True)
        outputs.extend([g.strip() for g in gen])
        print(f"  generated {min(i+batch_size, len(prompts))}/{len(prompts)}")
    return outputs


def rouge_recall(scorer, gen, target):
    return scorer.score(target, gen)["rougeL"].recall


def compute_fa_def(gens, forget_answers, retain_answers, scorer):
    fa_flags, acc_flags = [], []
    for gen, fa, ra in zip(gens, forget_answers, retain_answers):
        fa_flags.append(1.0 if rouge_recall(scorer, gen, fa) < ROUGE_THRESHOLD else 0.0)
        acc_flags.append(1.0 if rouge_recall(scorer, gen, ra) >= ROUGE_THRESHOLD else 0.0)
    fa_old = sum(fa_flags) / len(fa_flags)
    acc_new = sum(acc_flags) / len(acc_flags)
    return fa_old, acc_new, fa_old * acc_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["RGU", "IFE"], required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap examples per scenario, for a quick smoke test")
    args = parser.parse_args()

    rgu = load_jsonl(RGU_PATH)
    ife = load_jsonl(IFE_PATH)
    if args.limit:
        rgu = rgu[: args.limit]
        ife = ife[: args.limit]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    print(f"Loading {MODEL_NAME} (base, untouched weights)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # Build prompts: target scenario gets injected context, other scenario is plain (baseline behavior)
    if args.target == "RGU":
        rgu_prompts = [make_prompt(i["prompt"], build_context_rgu(i)) for i in rgu]
        ife_prompts = [make_prompt(i["prompt"], None) for i in ife]
    else:
        rgu_prompts = [make_prompt(i["prompt"], None) for i in rgu]
        ife_prompts = [make_prompt(i["prompt"], build_context_ife(i)) for i in ife]

    print(f"\nGenerating RGU ({len(rgu_prompts)} examples, "
          f"{'TARGETED' if args.target=='RGU' else 'untargeted/collateral check'})...")
    rgu_gens = generate_batch(model, tokenizer, rgu_prompts, args.batch_size)

    print(f"\nGenerating IFE ({len(ife_prompts)} examples, "
          f"{'TARGETED' if args.target=='IFE' else 'untargeted/collateral check'})...")
    ife_gens = generate_batch(model, tokenizer, ife_prompts, args.batch_size)

    rgu_fa, rgu_acc, rgu_def = compute_fa_def(
        rgu_gens, [i["forget_answer"] for i in rgu], [i["retain_answer"] for i in rgu], scorer
    )
    ife_fa, ife_acc, ife_def = compute_fa_def(
        ife_gens, [i["forget_answer"] for i in ife], [i["retain_answer"] for i in ife], scorer
    )

    result = {
        "method": f"RAG-suppression (target={args.target}, no weight edit)",
        "target_scenario": args.target,
        "RGU_fa": rgu_fa, "RGU_acc_new": rgu_acc, "RGU_def": rgu_def,
        "IFE_fa": ife_fa, "IFE_acc_new": ife_acc, "IFE_def": ife_def,
        "n_rgu": len(rgu), "n_ife": len(ife),
    }
    print("\n" + json.dumps(result, indent=2))

    out_path = args.out or f"data/gate2_results/RAG_suppression_{args.target}_summary.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
