"""BioUnlearn-specific evaluation metrics: FA, DEF, EWEF, OCD.

Formal definitions follow documentation/MASTER_RESEARCH_PLAN.md Sections 7-8
(originally specified in the BioUnlearn EMNLP proposal, Sections 4/8/9).

These metrics build on the framework's existing `rouge` generation metric
(evals/metrics/memorization.py): a prompt is generated once against a
dataset config carrying a target answer text (forget_answer or
retain_answer), and rougeL_recall between the generation and that target
text is used as the text-match signal. A generation "matches" a target
answer when rougeL_recall exceeds `rouge_threshold` (default 0.5).
"""

import numpy as np

from evals.metrics.base import unlearning_metric


def _binary_from_rouge(value_by_index, threshold, direction):
    """Threshold per-instance rougeL_recall into a binary match indicator.

    direction="below": 1.0 if score < threshold (used for "no longer produces
        the forget answer" — FA_old).
    direction="above": 1.0 if score >= threshold (used for "produces the
        target answer" — Acc_new / neighbor accuracy).
    """
    flags = {}
    for idx, entry in value_by_index.items():
        score = entry.get("rougeL_recall") if entry is not None else None
        if score is None:
            continue
        if direction == "below":
            flags[idx] = 1.0 if score < threshold else 0.0
        else:
            flags[idx] = 1.0 if score >= threshold else 0.0
    return flags


@unlearning_metric(name="fa_score")
def fa_score(model, **kwargs):
    """Forget Accuracy (FA): fraction of forget-set prompts where the model's
    generation no longer matches the forget_answer text."""
    threshold = kwargs.get("rouge_threshold", 0.5)
    forget_gen = kwargs["pre_compute"]["forget_gen"]["value_by_index"]
    flags = _binary_from_rouge(forget_gen, threshold, direction="below")
    agg = float(np.mean(list(flags.values()))) if flags else None
    return {"agg_value": agg, "value_by_index": flags}


@unlearning_metric(name="def_score")
def def_score(model, **kwargs):
    """Directional Erasure Fidelity: DEF = FA_old x Acc_new.

    Self-correcting against output suppression: a model that refuses every
    query (or says "I don't know") gets FA_old=1.0 but Acc_new=0.0, so
    DEF=0.0 rather than rewarding blanket refusal.
    """
    threshold = kwargs.get("rouge_threshold", 0.5)
    forget_gen = kwargs["pre_compute"]["forget_gen"]["value_by_index"]
    retain_gen = kwargs["pre_compute"]["retain_gen"]["value_by_index"]

    forget_flags = _binary_from_rouge(forget_gen, threshold, direction="below")
    retain_flags = _binary_from_rouge(retain_gen, threshold, direction="above")
    common_idx = sorted(set(forget_flags) & set(retain_flags))

    if not common_idx:
        return {"agg_value": None, "fa_old": None, "acc_new": None, "value_by_index": {}}

    fa_old = float(np.mean([forget_flags[i] for i in common_idx]))
    acc_new = float(np.mean([retain_flags[i] for i in common_idx]))
    def_value = fa_old * acc_new

    per_instance = {
        i: {
            "fa_old": forget_flags[i],
            "acc_new": retain_flags[i],
            "def": forget_flags[i] * retain_flags[i],
        }
        for i in common_idx
    }
    return {
        "agg_value": def_value,
        "fa_old": fa_old,
        "acc_new": acc_new,
        "value_by_index": per_instance,
    }


@unlearning_metric(name="ewef_score")
def ewef_score(model, **kwargs):
    """Evidence-Weighted Erasure Fidelity: DEF weighted per-instance by the
    evidence-certainty of the replacement fact (MASTER_RESEARCH_PLAN.md Sec
    7.4). `certainty_by_index` should map dataset index -> certainty in
    [0,3] (Section 4.4 ordinal scale). Falls back to uniform weight=1.0
    (equivalent to plain DEF) when certainty metadata isn't available for an
    instance -- this is the expected behavior on current LLM-oracle pilot
    data (PQS<3), which has no GRADE/Class-LOE field yet.
    """
    threshold = kwargs.get("rouge_threshold", 0.5)
    forget_gen = kwargs["pre_compute"]["forget_gen"]["value_by_index"]
    retain_gen = kwargs["pre_compute"]["retain_gen"]["value_by_index"]
    certainty_by_index = kwargs.get("certainty_by_index", {})

    forget_flags = _binary_from_rouge(forget_gen, threshold, direction="below")
    retain_flags = _binary_from_rouge(retain_gen, threshold, direction="above")
    common_idx = sorted(set(forget_flags) & set(retain_flags))

    if not common_idx:
        return {"agg_value": None, "value_by_index": {}}

    weights = np.array([certainty_by_index.get(i, 1.0) for i in common_idx])
    per_instance_def = np.array(
        [forget_flags[i] * retain_flags[i] for i in common_idx]
    )

    weight_sum = weights.sum()
    agg = float(np.sum(weights * per_instance_def) / weight_sum) if weight_sum > 0 else None

    return {
        "agg_value": agg,
        "used_uniform_fallback": not bool(certainty_by_index),
        "value_by_index": {i: float(v) for i, v in zip(common_idx, per_instance_def)},
    }


@unlearning_metric(name="ocd_score")
def ocd_score(model, **kwargs):
    """Ontological Collateral Damage: mean accuracy DROP on UMLS k-hop
    neighbor concepts of the forget target, comparing pre- vs
    post-unlearning accuracy on neighbor QA (original proposal Sec 4.1.2;
    MASTER_RESEARCH_PLAN.md Sec 7).

    Requires a `reference_logs.pre_unlearning.ocd_neighbor_acc` entry
    (per-instance pre-unlearning neighbor accuracy, keyed by dataset index)
    to compute the actual drop. Without it, reports post-unlearning neighbor
    accuracy only, with an explicit note -- this is the expected state until
    a pre-unlearning baseline eval has been run and logged for comparison.
    """
    threshold = kwargs.get("rouge_threshold", 0.5)
    neighbor_gen = kwargs["pre_compute"]["neighbor_gen"]["value_by_index"]
    post_acc_by_concept = _binary_from_rouge(neighbor_gen, threshold, direction="above")

    pre_ref = (
        kwargs.get("reference_logs", {})
        .get("pre_unlearning", {})
        .get("ocd_neighbor_acc", None)
    )
    if not pre_ref:
        agg_post = (
            float(np.mean(list(post_acc_by_concept.values())))
            if post_acc_by_concept
            else None
        )
        return {
            "agg_value": None,
            "post_unlearning_neighbor_acc": agg_post,
            "note": (
                "No pre-unlearning reference_logs supplied; reporting "
                "post-only neighbor accuracy. Set eval.bioun.pre_unlearning_logs_path "
                "to a base-model eval log to compute the actual OCD drop."
            ),
            "value_by_index": post_acc_by_concept,
        }

    common_idx = sorted(set(post_acc_by_concept) & set(pre_ref))
    if not common_idx:
        return {"agg_value": None, "value_by_index": {}}
    drops = [max(0.0, pre_ref[i] - post_acc_by_concept[i]) for i in common_idx]
    agg = float(np.mean(drops))
    return {"agg_value": agg, "value_by_index": dict(zip(common_idx, drops))}
