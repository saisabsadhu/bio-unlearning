"""
ROME knowledge-editing baseline: for each RGU/IFE instance with a valid
ROME request (data/gate2_results/rome_requests_{scenario}.json), apply a
single rank-one edit to BioMistral-7B, then score FA/DEF the same way as
every other method (GA/NPO/RMU/OGDA/RAG-suppression) -- generate on the
ORIGINAL clinical question (not the ROME template) and rougeL-threshold
against forget_answer/retain_answer.

Since ROME edits are targeted (one fact per instance) rather than a single
model-wide checkpoint, this loads BioMistral-7B ONCE and for each instance:
apply edit -> generate+score on that instance's own question (targeted) and
a fixed sample of untargeted-scenario questions (collateral check) -> restore
original weights -> next instance. This avoids reloading the 7B model per
instance while still measuring both targeted and collateral effects per edit.

Usage:
    python stage_a_umls/rome_baseline_eval.py --scenario RGU --limit 5 --dry-run
    python stage_a_umls/rome_baseline_eval.py --scenario RGU
"""

import argparse
import gc
import json
import random

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage_a_umls.rome_vendor.rome import apply_rome_to_model, ROMEHyperParams

MODEL_NAME = "BioMistral/BioMistral-7B"
HPARAMS_PATH = "stage_a_umls/rome_vendor/hparams/BioMistral-7B_ROME.yaml"
ROUGE_THRESHOLD = 0.5
MAX_NEW_TOKENS = 200
N_COLLATERAL_SAMPLE = 10  # fixed sample of the OTHER scenario, checked after every edit


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def rouge_recall(scorer, gen, target):
    return scorer.score(target, gen)["rougeL"].recall


def generate_one(model, tokenizer, question):
    prompt = f"[INST] {question} [/INST]"
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
    parser.add_argument("--scenario", choices=["RGU", "IFE"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    other = "IFE" if args.scenario == "RGU" else "RGU"
    requests = json.load(open(f"data/gate2_results/rome_requests_{args.scenario}.json"))
    train_by_id = {i["instance_id"]: i for i in load_jsonl(f"data/splits/{args.scenario}_train.jsonl")}
    other_pool = load_jsonl(f"data/splits/{other}_train.jsonl")

    random.seed(42)
    collateral_sample = random.sample(other_pool, min(N_COLLATERAL_SAMPLE, len(other_pool)))

    if args.limit:
        requests = requests[: args.limit]

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    out_path = args.out or f"data/gate2_results/ROME_{args.scenario}_summary.json"
    per_instance = []
    n_oom, n_other_fail = 0, 0

    def safe_sanity_check():
        """Never let the post-failure sanity check itself crash the run."""
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            s = generate_one(model, tokenizer, "The sky is")
            print(f"    post-failure sanity check generation: {s[:60]!r}")
        except Exception as e2:
            print(f"    sanity check itself failed ({e2}); GPU likely still under memory pressure, continuing anyway")

    def save_progress():
        n = len(per_instance)
        agg = {
            "method": f"ROME (single-fact rank-1 edit, layer {hparams.layers})",
            "scenario": args.scenario,
            "n_requested": len(requests),
            "n_edits_scored": n,
            "n_oom_failures": n_oom,
            "n_other_failures": n_other_fail,
            "target_fa_mean": sum(x["target_fa"] for x in per_instance) / n if n else None,
            "target_acc_new_mean": sum(x["target_acc_new"] for x in per_instance) / n if n else None,
            "target_def_mean": sum(x["target_def"] for x in per_instance) / n if n else None,
            "collateral_fa_mean": sum(x["collateral_fa"] for x in per_instance) / n if n else None,
            "collateral_acc_new_mean": sum(x["collateral_acc_new"] for x in per_instance) / n if n else None,
            "collateral_def_mean": sum(x["collateral_def"] for x in per_instance) / n if n else None,
            "per_instance": per_instance,
        }
        if not args.dry_run:
            with open(out_path, "w") as f:
                json.dump(agg, f, indent=2)
        return agg

    for i, req in enumerate(requests):
        inst = train_by_id.get(req["instance_id"])
        if inst is None:
            print(f"[{i}] no matching train instance for {req['instance_id']}, skipping")
            continue

        print(f"\n[{i}] {req['instance_id']} :: {req['subject'][:60]}")
        request = [{"prompt": req["prompt_template"], "subject": req["subject"], "target_new": req["target_new"]}]

        edited_model, weights_copy = None, None
        try:
            edited_model, weights_copy = apply_rome_to_model(
                model, tokenizer, request, hparams, return_orig_weights=True
            )

            with torch.no_grad():
                target_gen = generate_one(edited_model, tokenizer, inst["prompt"])
                target_fa = 1.0 if rouge_recall(scorer, target_gen, inst["forget_answer"]) < ROUGE_THRESHOLD else 0.0
                target_acc = 1.0 if rouge_recall(scorer, target_gen, inst["retain_answer"]) >= ROUGE_THRESHOLD else 0.0
                target_def = target_fa * target_acc
                collateral_fa, collateral_acc, collateral_def = score_examples(
                    edited_model, tokenizer, collateral_sample, scorer
                )

            print(f"    target: FA={target_fa} acc_new={target_acc} DEF={target_def}  |  gen: {target_gen[:80]}")
            print(f"    collateral ({other}, n={len(collateral_sample)}): FA={collateral_fa:.2f} DEF={collateral_def:.2f}")

            per_instance.append({
                "instance_id": req["instance_id"],
                "target_fa": target_fa, "target_acc_new": target_acc, "target_def": target_def,
                "collateral_fa": collateral_fa, "collateral_acc_new": collateral_acc, "collateral_def": collateral_def,
            })

        except torch.cuda.OutOfMemoryError as e:
            print(f"    FAILED (OOM): {e}")
            n_oom += 1
            safe_sanity_check()
        except Exception as e:
            print(f"    FAILED: {e}")
            n_other_fail += 1
        finally:
            # ALWAYS restore original weights if the edit itself succeeded, regardless
            # of what happened afterwards (a generate() OOM after a successful edit
            # must not leave the model edited going into the next iteration)
            if weights_copy:
                try:
                    with torch.no_grad():
                        named = dict(model.named_parameters())
                        for name, orig in weights_copy.items():
                            named[name][...] = orig
                except Exception as e3:
                    print(f"    WARNING: weight restore failed ({e3}) -- model may be corrupted for subsequent iterations")
            del edited_model, weights_copy
            gc.collect()
            torch.cuda.empty_cache()

        # checkpoint after every instance so a later crash can't lose prior progress
        save_progress()

    agg = save_progress()
    print("\n" + json.dumps({k: v for k, v in agg.items() if k != "per_instance"}, indent=2))
    if not args.dry_run:
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
